import asyncio
import io
import os
import time
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import jwt
import uvicorn
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from project.backtesting.backtester import Backtester
from project.config.configuration import get_config
from project.data.database import Database
from project.exchange.exchange_manager import get_exchange_manager
from project.trading.strategy_base import StrategyRegistry
from project.trading.trading_bot import TradingBot
from project.utils.logging_utils import setup_logger
from project.utils.notify import NotificationManager
from project.visualization.data_visualizer import DataVisualizer
from pydantic import BaseModel

logger = setup_logger("api_server")

# Модели данных для API


class UserCredentials(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: int


class UserInfo(BaseModel):
    username: str
    email: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False


class StrategyParameters(BaseModel):
    parameters: Dict[str, Any]
    description: Optional[str] = None


class BotConfig(BaseModel):
    bot_id: Optional[str] = None
    symbol: str
    exchange_id: str
    timeframe: str
    strategy_id: str
    leverage: float = 1.0
    margin_type: str = "isolated"
    position_size: float = 0.01
    is_position_size_percentage: bool = True
    max_positions: int = 1
    allow_shorts: bool = False
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[Dict[str, Any]] = None
    order_type: str = "market"
    post_only: bool = False
    reduce_only: bool = False
    time_in_force: str = "GTC"
    paper_trading: bool = True
    update_interval: int = 60
    warmup_bars: int = 100
    custom_settings: Optional[Dict[str, Any]] = None


class BacktestRequest(BaseModel):
    strategy_id: str
    symbol: str
    exchange_id: str
    timeframe: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    strategy_parameters: Dict[str, Any]
    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0001
    position_size_pct: float = 1.0
    enable_fractional: bool = True
    enable_shorting: bool = False
    custom_settings: Optional[Dict[str, Any]] = None


class OptimizationRequest(BaseModel):
    strategy_id: str
    symbol: str
    exchange_id: str
    timeframe: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    parameter_ranges: Dict[str, Any]
    population_size: int = 50
    generations: int = 10
    crossover_rate: float = 0.7
    mutation_rate: float = 0.2
    elitism_rate: float = 0.1
    optimization_method: str = "genetic"
    max_workers: int = 4
    fitness_metrics: Dict[str, float] = None


class APIServer:
    """Класс для REST API сервера"""

    def __init__(self, config: Dict = None):
        """
        Инициализирует API сервер

        Args:
            config: Конфигурация сервера
        """
        self.config = config or get_config().get("api", {})

        # Настройки сервера
        self.host = self.config.get("host", "0.0.0.0")
        self.port = self.config.get("port", 8000)
        self.debug = self.config.get("debug", False)
        self.workers = self.config.get("workers", 1)
        self.reload = self.config.get("reload", False)

        # Настройки безопасности
        self.secret_key = self.config.get("secret_key", "YOUR_SECRET_KEY_HERE")
        self.token_expire_minutes = self.config.get(
            "token_expire_minutes", 60 * 24
        )  # 24 часа
        self.cors_origins = self.config.get("cors_origins", ["*"])

        # Создаем FastAPI приложение
        self.app = FastAPI(
            title="Trading System API",
            description="API для взаимодействия с торговой системой",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Настраиваем CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Аутентификация
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")

        # База данных
        self.database = None

        # Экземпляр менеджера бирж
        self.exchange_manager = None

        # Словарь ботов
        self.bots = {}

        # Визуализатор
        self.visualizer = DataVisualizer()

        # Менеджер уведомлений
        self.notification_manager = NotificationManager()

        # Задача обновления ботов
        self._bots_update_task = None

        # Бэктестер
        self.backtester = Backtester()

        # Активные задачи оптимизации
        self.optimization_tasks = {}

        # Настраиваем маршруты
        self._setup_routes()

        logger.info("API server initialized")

    def _setup_routes(self):
        """Настраивает маршруты API"""
        # Создаем роутеры
        auth_router = APIRouter(prefix="/api/auth", tags=["Authentication"])
        user_router = APIRouter(prefix="/api/users", tags=["Users"])
        exchange_router = APIRouter(prefix="/api/exchanges", tags=["Exchanges"])
        market_router = APIRouter(prefix="/api/markets", tags=["Market Data"])
        strategy_router = APIRouter(prefix="/api/strategies", tags=["Strategies"])
        bot_router = APIRouter(prefix="/api/bots", tags=["Trading Bots"])
        backtest_router = APIRouter(prefix="/api/backtest", tags=["Backtesting"])
        optimization_router = APIRouter(
            prefix="/api/optimization", tags=["Optimization"]
        )
        chart_router = APIRouter(prefix="/api/charts", tags=["Charts"])
        system_router = APIRouter(prefix="/api/system", tags=["System"])

        # Маршруты аутентификации
        auth_router.post("/token", response_model=Token)(self.login)
        auth_router.post("/refresh", response_model=Token)(self.refresh_token)

        # Маршруты пользователей
        user_router.get("/me", response_model=UserInfo)(self.get_current_user_info)
        user_router.post("/change-password")(self.change_password)

        # Маршруты для работы с биржами
        exchange_router.get("/")(self.get_exchanges)
        exchange_router.get("/{exchange_id}/info")(self.get_exchange_info)
        exchange_router.get("/{exchange_id}/balance")(self.get_balance)
        exchange_router.get("/{exchange_id}/positions")(self.get_positions)
        exchange_router.get("/{exchange_id}/orders")(self.get_orders)
        exchange_router.post("/{exchange_id}/order")(self.create_order)
        exchange_router.delete("/{exchange_id}/order/{order_id}")(self.cancel_order)

        # Маршруты для работы с рыночными данными
        market_router.get("/{exchange_id}/symbols")(self.get_symbols)
        market_router.get("/{exchange_id}/{symbol}/ticker")(self.get_ticker)
        market_router.get("/{exchange_id}/{symbol}/orderbook")(self.get_orderbook)
        market_router.get("/{exchange_id}/{symbol}/trades")(self.get_recent_trades)
        market_router.get("/{exchange_id}/{symbol}/ohlcv")(self.get_ohlcv)

        # Исправляем ошибку в строке 266
        @market_router.get("/ticker/{symbol}")
        async def get_ticker(symbol: str, exchange: str = "binance"):
            """
            Получает текущие данные тикера для указанного символа.

            Args:
                symbol: Торговая пара (например, BTC/USDT)
                exchange: Идентификатор биржи

            Returns:
                Данные тикера
            """
            try:
                ticker = await market_data.get_ticker(exchange, symbol)
                return {"status": "success", "data": ticker}
            except Exception as e:
                logger.error(f"Ошибка при получении тикера: {str(e)}")
                raise HTTPException(
                    status_code=500, detail="Ошибка при получении данных тикера"
                )

        # Маршруты для работы со стратегиями
        strategy_router.get("/")(self.get_strategies)
        strategy_router.get("/{strategy_id}/info")(self.get_strategy_info)
        strategy_router.get("/{strategy_id}/parameters")(self.get_strategy_parameters)
        strategy_router.post("/{strategy_id}/parameters")(self.save_strategy_parameters)

        # Маршруты для работы с ботами
        bot_router.get("/")(self.get_bots)
        bot_router.post("/")(self.create_bot)
        bot_router.get("/{bot_id}")(self.get_bot)
        bot_router.delete("/{bot_id}")(self.delete_bot)
        bot_router.post("/{bot_id}/start")(self.start_bot)
        bot_router.post("/{bot_id}/stop")(self.stop_bot)
        bot_router.post("/{bot_id}/pause")(self.pause_bot)
        bot_router.post("/{bot_id}/resume")(self.resume_bot)
        bot_router.get("/{bot_id}/status")(self.get_bot_status)

        @self.app.route("/api/stats/performance", methods=["GET"])
        async def get_performance_stats(request):
            try:
                # Получаем статистику производительности
                stats = {
                    "cpu_usage": system_stats.get_cpu_usage(),
                    "memory_usage": self.get_memory_usage(),
                    "disk_usage": system_stats.get_disk_usage(),
                    "uptime": system_stats.get_uptime(),
                    "network": system_stats.get_network_stats(),
                }

                return {"status": "success", "data": stats}
            except Exception as e:
                logger.error("Error getting performance stats: {e}")
                return json_response({"success": False, "error": str(e)}, status=500)

        # Маршруты для бэктестинга
        backtest_router.post("/")(self.run_backtest)
        backtest_router.get("/results")(self.get_backtest_results)
        backtest_router.get("/results/{backtest_id}")(self.get_backtest_result)

        # Маршруты для оптимизации
        optimization_router.post("/")(self.run_optimization)
        optimization_router.get("/tasks")(self.get_optimization_tasks)
        optimization_router.get("/tasks/{task_id}")(self.get_optimization_task)
        optimization_router.delete("/tasks/{task_id}")(self.cancel_optimization_task)
        optimization_router.get("/results")(self.get_optimization_results)
        optimization_router.get("/results/{optimization_id}")(
            self.get_optimization_result
        )

        # Маршруты для графиков
        chart_router.get("/ohlc")(self.get_ohlc_chart)
        chart_router.get("/equity")(self.get_equity_chart)
        chart_router.get("/drawdown")(self.get_drawdown_chart)
        chart_router.get("/monthly-returns")(self.get_monthly_returns_chart)
        chart_router.get("/optimization")(self.get_optimization_chart)

        # Маршруты для системных функций
        system_router.get("/status")(self.get_system_status)
        system_router.get("/logs")(self.get_logs)
        system_router.post("/notification")(self.send_notification)

        # Добавляем роутеры в приложение
        self.app.include_router(auth_router)
        self.app.include_router(user_router)
        self.app.include_router(exchange_router)
        self.app.include_router(market_router)
        self.app.include_router(strategy_router)
        self.app.include_router(bot_router)
        self.app.include_router(backtest_router)
        self.app.include_router(optimization_router)
        self.app.include_router(chart_router)
        self.app.include_router(system_router)

        # Добавляем корневой маршрут
        @self.app.get("/", tags=["Root"])
        async def root():
            return {"message": "Trading System API"}

        # Добавляем обработчики событий

        # Вспомогательные методы для статистики системы
        def get_memory_usage(self):
            import psutil

            return psutil.virtual_memory().percent

        @self.app.on_event("startup")
        async def startup_event():
            await self._startup()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self._shutdown()

    async def _startup(self):
        """Инициализирует ресурсы при запуске сервера"""
        # Инициализируем базу данных
        self.database = Database()
        await self.database.connect()

        # Инициализируем менеджер бирж
        self.exchange_manager = await get_exchange_manager()

        # Запускаем задачу обновления ботов
        self._bots_update_task = asyncio.create_task(self._update_bots_task())

        # Загружаем активных ботов из базы данных
        await self._load_active_bots()

        logger.info("API server startup completed")

    async def _shutdown(self):
        """Освобождает ресурсы при остановке сервера"""
        # Останавливаем всех ботов
        for bot_id, bot in list(self.bots.items()):
            try:
                await bot.stop()
            except:
                pass

        # Отменяем задачу обновления ботов
        if self._bots_update_task:
            self._bots_update_task.cancel()
            try:
                await self._bots_update_task
            except asyncio.CancelledError:
                pass

        # Закрываем соединение с базой данных
        if self.database:
            await self.database.disconnect()

        logger.info("API server shutdown completed")

    async def _load_active_bots(self):
        """Загружает активных ботов из базы данных"""
        try:
            # Получаем список активных ботов
            bot_states = await self.database.get_bot_states(is_active=True)

            # Создаем и запускаем ботов
            for bot_state in bot_states:
                try:
                    # Создаем конфигурацию бота
                    bot_config = bot_state.get("config", {})

                    # Создаем бота
                    bot = TradingBot(
                        config=bot_config,
                        database=self.database,
                        notification_manager=self.notification_manager,
                    )

                    # Сохраняем бота
                    self.bots[bot_state.get("bot_id")] = bot

                    # Запускаем бота, если он активен
                    if bot_state.get("is_active", False):
                        await bot.start()

                    logger.info("Loaded bot: {bot_state.get('bot_id')}")

                except Exception as e:
                    logger.error(
                        "Error loading bot {bot_state.get('bot_id')}: {str(e)}"
                    )

            logger.info("Loaded {len(self.bots)} active bots")

        except Exception as e:
            logger.error("Error loading active bots: {str(e)}")

    async def _update_bots_task(self):
        """Задача для периодического обновления состояния ботов"""
        try:
            while True:
                # Сохраняем состояние ботов
                for bot_id, bot in self.bots.items():
                    try:
                        if bot.state != "stopped":
                            # Сохраняем состояние бота
                            await bot._save_state()
                    except Exception as e:
                        logger.error("Error updating bot {bot_id}: {str(e)}")

                # Ждем перед следующим обновлением
                await asyncio.sleep(60)  # Обновляем каждую минуту

        except asyncio.CancelledError:
            logger.info("Bot update task cancelled")
            raise
        except Exception as e:
            logger.error("Error in bot update task: {str(e)}")

    async def verify_token(
        self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="api/auth/token"))
    ) -> Dict:
        """
        Верифицирует JWT токен

        Args:
            token: JWT токен

        Returns:
            Dict: Полезная нагрузка токена

        Raises:
            HTTPException: Если токен недействителен или просрочен
        """
        try:
            # Декодируем токен
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            # Проверяем срок действия
            if payload.get("exp") < time.time():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return payload

        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def create_access_token(self, data: Dict) -> str:
        """
        Создает JWT токен

        Args:
            data: Данные для включения в токен

        Returns:
            str: JWT токен
        """
        # Создаем копию данных
        to_encode = data.copy()

        # Добавляем срок действия
        expire = datetime.now() + timedelta(minutes=self.token_expire_minutes)
        to_encode.update({"exp": expire.timestamp()})

        # Создаем JWT токен
        return jwt.encode(to_encode, self.secret_key, algorithm="HS256")

    async def login(self, form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
        """
        Endpoint для аутентификации и получения токена

        Args:
            form_data: Данные формы аутентификации

        Returns:
            Token: Токен доступа

        Raises:
            HTTPException: Если учетные данные неверны
        """
        # В реальной системе здесь будет проверка учетных данных из базы данных
        # Для примера просто проверяем захардкоженные значения

        # Получаем пользователей из конфигурации
        users = get_config().get("users", [])

        # Ищем пользователя по имени
        user = next((u for u in users if u.get("username") == form_data.username), None)

        if not user or user.get("password") != form_data.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Создаем данные пользователя для токена
        user_data = {
            "sub": user.get("username"),
            "username": user.get("username"),
            "is_admin": user.get("is_admin", False),
        }

        # Создаем токен
        access_token = self.create_access_token(user_data)

        # Вычисляем время истечения
        expires_at = int(
            (datetime.now() + timedelta(minutes=self.token_expire_minutes)).timestamp()
        )

        return Token(
            access_token=access_token, token_type="bearer", expires_at=expires_at
        )

    async def refresh_token(
        self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="api/auth/token"))
    ) -> Token:
        """
        Endpoint для обновления токена

        Args:
            token: Текущий токен

        Returns:
            Token: Новый токен
        """
        # Верифицируем текущий токен
        payload = await self.verify_token(token)

        # Создаем новый токен с теми же данными, но с новым сроком действия
        new_token = self.create_access_token(
            {
                "sub": payload.get("sub"),
                "username": payload.get("username"),
                "is_admin": payload.get("is_admin", False),
            }
        )

        # Вычисляем время истечения
        expires_at = int(
            (datetime.now() + timedelta(minutes=self.token_expire_minutes)).timestamp()
        )

        return Token(access_token=new_token, token_type="bearer", expires_at=expires_at)

    async def get_current_user_info(
        self, token: Dict = Depends(verify_token)
    ) -> UserInfo:
        """
        Endpoint для получения информации о текущем пользователе

        Args:
            token: Верифицированный токен

        Returns:
            UserInfo: Информация о пользователе
        """
        return UserInfo(
            username=token.get("username"), is_admin=token.get("is_admin", False)
        )

    async def change_password(
        self, user_credentials: UserCredentials, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для изменения пароля

        Args:
            user_credentials: Новые учетные данные
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        # В реальной системе здесь будет изменение пароля в базе данных
        # Для примера просто возвращаем успешный результат
        return {"status": "success", "message": "Password changed"}

    async def get_exchanges(self, token: Dict = Depends(verify_token)) -> List[Dict]:
        """
        Endpoint для получения списка бирж

        Args:
            token: Верифицированный токен

        Returns:
            List[Dict]: Список бирж
        """
        # Получаем список доступных бирж
        exchanges = self.exchange_manager.get_exchanges_info()

        return exchanges

    async def get_exchange_info(
        self, exchange_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения информации о бирже

        Args:
            exchange_id: ID биржи
            token: Верифицированный токен

        Returns:
            Dict: Информация о бирже
        """
        # Получаем информацию о бирже
        exchange_info = await self.exchange_manager.get_exchange_info(exchange_id)

        if not exchange_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exchange {exchange_id} not found",
            )

        return exchange_info

    async def get_balance(
        self, exchange_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения баланса

        Args:
            exchange_id: ID биржи
            token: Верифицированный токен

        Returns:
            Dict: Баланс
        """
        # Получаем баланс
        balance = await self.exchange_manager.fetch_balance(exchange_id)

        if not balance:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch balance for exchange {exchange_id}",
            )

        return balance

    async def get_positions(
        self,
        exchange_id: str,
        symbol: Optional[str] = None,
        token: Dict = Depends(verify_token),
    ) -> List[Dict]:
        """
        Endpoint для получения позиций

        Args:
            exchange_id: ID биржи
            symbol: Символ торговой пары (опционально)
            token: Верифицированный токен

        Returns:
            List[Dict]: Список позиций
        """
        # Получаем позиции
        positions = await self.exchange_manager.get_positions(symbol, exchange_id)

        if positions is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch positions for exchange {exchange_id}",
            )

        return positions

    async def get_orders(
        self,
        exchange_id: str,
        symbol: Optional[str] = None,
        token: Dict = Depends(verify_token),
    ) -> List[Dict]:
        """
        Endpoint для получения ордеров

        Args:
            exchange_id: ID биржи
            symbol: Символ торговой пары (опционально)
            token: Верифицированный токен

        Returns:
            List[Dict]: Список ордеров
        """
        # Получаем ордера
        orders = await self.exchange_manager.fetch_open_orders(symbol, exchange_id)

        if orders is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch orders for exchange {exchange_id}",
            )

        return orders

    async def create_order(
        self, exchange_id: str, order_data: Dict, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для создания ордера

        Args:
            exchange_id: ID биржи
            order_data: Данные ордера
            token: Верифицированный токен

        Returns:
            Dict: Созданный ордер
        """
        # Проверяем наличие необходимых полей
        required_fields = ["symbol", "type", "side", "amount"]
        for field in required_fields:
            if field not in order_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}",
                )

        # Создаем ордер
        order = await self.exchange_manager.create_order(
            symbol=order_data["symbol"],
            order_type=order_data["type"],
            side=order_data["side"],
            amount=order_data["amount"],
            price=order_data.get("price"),
            exchange_id=exchange_id,
            params=order_data.get("params", {}),
        )

        if not order:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not create order",
            )

        # Сохраняем ордер в базу данных
        if self.database:
            await self.database.save_order(order)

        return order

    async def cancel_order(
        self,
        exchange_id: str,
        order_id: str,
        symbol: Optional[str] = None,
        token: Dict = Depends(verify_token),
    ) -> Dict:
        """
        Endpoint для отмены ордера

        Args:
            exchange_id: ID биржи
            order_id: ID ордера
            symbol: Символ торговой пары (опционально)
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        # Отменяем ордер
        result = await self.exchange_manager.cancel_order(order_id, symbol, exchange_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not cancel order {order_id}",
            )

        return {"status": "success", "order_id": order_id}

    async def get_symbols(
        self, exchange_id: str, token: Dict = Depends(verify_token)
    ) -> List[str]:
        """
        Endpoint для получения списка торговых пар

        Args:
            exchange_id: ID биржи
            token: Верифицированный токен

        Returns:
            List[str]: Список торговых пар
        """
        # Получаем список торговых пар
        markets = await self.exchange_manager.get_markets(exchange_id)

        if not markets:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch symbols for exchange {exchange_id}",
            )

        # Извлекаем только символы
        symbols = list(markets.keys())

        return symbols

    async def get_ticker(
        self, exchange_id: str, symbol: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения тикера

        Args:
            exchange_id: ID биржи
            symbol: Символ торговой пары
            token: Верифицированный токен

        Returns:
            Dict: Тикер
        """
        # Получаем тикер
        ticker = await self.exchange_manager.fetch_ticker(symbol, exchange_id)

        if not ticker:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch ticker for {symbol} on {exchange_id}",
            )

        return ticker

    async def get_orderbook(
        self,
        exchange_id: str,
        symbol: str,
        limit: int = 10,
        token: Dict = Depends(verify_token),
    ) -> Dict:
        """
        Endpoint для получения книги ордеров

        Args:
            exchange_id: ID биржи
            symbol: Символ торговой пары
            limit: Глубина книги ордеров
            token: Верифицированный токен

        Returns:
            Dict: Книга ордеров
        """
        # Получаем книгу ордеров
        orderbook = await self.exchange_manager.fetch_order_book(
            symbol, exchange_id, limit
        )

        if not orderbook:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch orderbook for {symbol} on {exchange_id}",
            )

        return orderbook

    async def get_recent_trades(
        self,
        exchange_id: str,
        symbol: str,
        limit: int = 50,
        token: Dict = Depends(verify_token),
    ) -> List[Dict]:
        """
        Endpoint для получения недавних сделок

        Args:
            exchange_id: ID биржи
            symbol: Символ торговой пары
            limit: Количество сделок
            token: Верифицированный токен

        Returns:
            List[Dict]: Список сделок
        """
        # Получаем сделки
        trades = await self.exchange_manager.fetch_trades(symbol, exchange_id, limit)

        if trades is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch trades for {symbol} on {exchange_id}",
            )

        return trades

    async def get_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[int] = None,
        token: Dict = Depends(verify_token),
    ) -> List[List]:
        """
        Endpoint для получения OHLCV данных

        Args:
            exchange_id: ID биржи
            symbol: Символ торговой пары
            timeframe: Временной интервал
            limit: Количество свечей
            since: Начальная временная метка (опционально)
            token: Верифицированный токен

        Returns:
            List[List]: OHLCV данные
        """
        # Получаем OHLCV данные
        ohlcv = await self.exchange_manager.fetch_ohlcv(
            symbol, exchange_id, timeframe, limit, since
        )

        if ohlcv is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not fetch OHLCV data for {symbol} on {exchange_id}",
            )

        return ohlcv

    async def get_strategies(self, token: Dict = Depends(verify_token)) -> List[Dict]:
        """
        Endpoint для получения списка стратегий

        Args:
            token: Верифицированный токен

        Returns:
            List[Dict]: Список стратегий
        """
        # Получаем список зарегистрированных стратегий
        strategies = StrategyRegistry.get_all_strategies()

        # Преобразуем в список словарей с информацией о стратегиях
        result = []
        for strategy_name, strategy_class in strategies.items():
            # Получаем параметры по умолчанию
            default_params = strategy_class.get_default_parameters()

            # Получаем дополнительную информацию о стратегии
            strategy_info = {
                "id": strategy_name,
                "name": strategy_name,
                "description": strategy_class.__doc__ or "",
                "default_parameters": default_params,
            }

            result.append(strategy_info)

        return result

    async def get_strategy_info(
        self, strategy_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения информации о стратегии

        Args:
            strategy_id: ID стратегии
            token: Верифицированный токен

        Returns:
            Dict: Информация о стратегии
        """
        try:
            # Получаем класс стратегии
            strategy_class = StrategyRegistry.get_strategy_class(strategy_id)

            # Получаем параметры по умолчанию
            default_params = strategy_class.get_default_parameters()

            # Получаем дополнительную информацию о стратегии
            strategy_info = {
                "id": strategy_id,
                "name": strategy_id,
                "description": strategy_class.__doc__ or "",
                "default_parameters": default_params,
            }

            # Если есть база данных, получаем сохраненные параметры
            if self.database:
                saved_params = await self.database.get_strategy_parameters(strategy_id)
                if saved_params:
                    strategy_info["saved_parameters"] = saved_params

            return strategy_info

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy {strategy_id} not found: {str(e)}",
            )

    async def get_strategy_parameters(
        self, strategy_id: str, token: Dict = Depends(verify_token)
    ) -> List[Dict]:
        """
        Endpoint для получения параметров стратегии

        Args:
            strategy_id: ID стратегии
            token: Верифицированный токен

        Returns:
            List[Dict]: Список наборов параметров
        """
        try:
            # Если есть база данных, получаем сохраненные параметры
            if self.database:
                saved_params = await self.database.get_strategy_parameters(strategy_id)
                return saved_params

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No saved parameters found for strategy {strategy_id}",
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting strategy parameters: {str(e)}",
            )

    async def save_strategy_parameters(
        self,
        strategy_id: str,
        parameters: StrategyParameters,
        token: Dict = Depends(verify_token),
    ) -> Dict:
        """
        Endpoint для сохранения параметров стратегии

        Args:
            strategy_id: ID стратегии
            parameters: Параметры стратегии
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        try:
            # Проверяем, существует ли стратегия
            StrategyRegistry.get_strategy_class(strategy_id)

            # Если есть база данных, сохраняем параметры
            if self.database:
                await self.database.save_strategy_parameters(
                    strategy_id=strategy_id,
                    parameters=parameters.parameters,
                    description=parameters.description,
                )

                return {
                    "status": "success",
                    "message": f"Parameters saved for strategy {strategy_id}",
                }

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database not available",
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving strategy parameters: {str(e)}",
            )

    async def get_bots(self, token: Dict = Depends(verify_token)) -> List[Dict]:
        """
        Endpoint для получения списка ботов

        Args:
            token: Верифицированный токен

        Returns:
            List[Dict]: Список ботов
        """
        # Получаем информацию о ботах
        bots_info = []

        for bot_id, bot in self.bots.items():
            bots_info.append(
                {
                    "bot_id": bot_id,
                    "symbol": bot.symbol,
                    "exchange_id": bot.exchange_id,
                    "strategy_id": bot.strategy_id,
                    "state": bot.state.value,
                    "paper_trading": bot.paper_trading,
                }
            )

        return bots_info

    async def create_bot(
        self, bot_config: BotConfig, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для создания бота

        Args:
            bot_config: Конфигурация бота
            token: Верифицированный токен

        Returns:
            Dict: Информация о созданном боте
        """
        try:
            # Создаем бота
            config = bot_config.dict()

            # Генерируем ID бота, если не указан
            if not config.get("bot_id"):
                config["bot_id"] = f"bot_{uuid.uuid4().hex[:8]}"

            # Создаем экземпляр бота
            bot = TradingBot(
                config=config,
                database=self.database,
                notification_manager=self.notification_manager,
            )

            # Сохраняем бота
            self.bots[config["bot_id"]] = bot

            return {
                "status": "success",
                "bot_id": config["bot_id"],
                "message": f"Bot created successfully",
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating bot: {str(e)}",
            )

    async def get_bot(self, bot_id: str, token: Dict = Depends(verify_token)) -> Dict:
        """
        Endpoint для получения информации о боте

        Args:
            bot_id: ID бота
            token: Верифицированный токен

        Returns:
            Dict: Информация о боте
        """
        # Проверяем существование бота
        if bot_id not in self.bots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Получаем информацию о боте
        bot = self.bots[bot_id]
        return bot.get_info()

    async def delete_bot(
        self, bot_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для удаления бота

        Args:
            bot_id: ID бота
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        # Проверяем существование бота
        if bot_id not in self.bots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Останавливаем бота
        bot = self.bots[bot_id]
        await bot.stop()

        # Удаляем бота
        del self.bots[bot_id]

        return {"status": "success", "message": f"Bot {bot_id} deleted"}

    async def start_bot(self, bot_id: str, token: Dict = Depends(verify_token)) -> Dict:
        """
        Endpoint для запуска бота

        Args:
            bot_id: ID бота
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        # Проверяем существование бота
        if bot_id not in self.bots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Запускаем бота
        bot = self.bots[bot_id]
        await bot.start()

        return {"status": "success", "message": f"Bot {bot_id} started"}

    async def stop_bot(self, bot_id: str, token: Dict = Depends(verify_token)) -> Dict:
        """
        Endpoint для остановки бота

        Args:
            bot_id: ID бота
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        # Проверяем существование бота
        if bot_id not in self.bots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Останавливаем бота
        bot = self.bots[bot_id]
        await bot.stop()

        return {"status": "success", "message": f"Bot {bot_id} stopped"}

    async def pause_bot(self, bot_id: str, token: Dict = Depends(verify_token)) -> Dict:
        """
        Endpoint для приостановки бота

        Args:
            bot_id: ID бота
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        # Проверяем существование бота
        if bot_id not in self.bots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Приостанавливаем бота
        bot = self.bots[bot_id]
        await bot.pause()

        return {"status": "success", "message": f"Bot {bot_id} paused"}

    async def resume_bot(
        self, bot_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для возобновления работы бота

        Args:
            bot_id: ID бота
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        # Проверяем существование бота
        if bot_id not in self.bots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Возобновляем работу бота
        bot = self.bots[bot_id]
        await bot.resume()

        return {"status": "success", "message": f"Bot {bot_id} resumed"}

    async def get_bot_status(
        self, bot_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения статуса бота

        Args:
            bot_id: ID бота
            token: Верифицированный токен

        Returns:
            Dict: Статус бота
        """
        # Проверяем существование бота
        if bot_id not in self.bots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Bot {bot_id} not found"
            )

        # Получаем информацию о боте
        bot = self.bots[bot_id]

        return {
            "bot_id": bot_id,
            "state": bot.state.value,
            "last_update_time": (
                bot.last_update_time.isoformat() if bot.last_update_time else None
            ),
            "last_signal_time": (
                bot.last_signal_time.isoformat() if bot.last_signal_time else None
            ),
            "last_trade_time": (
                bot.last_trade_time.isoformat() if bot.last_trade_time else None
            ),
            "error_count": bot.error_count,
            "positions": len([p for p in bot.positions.values() if p.is_open()]),
            "signals": len(bot.signals),
        }

    async def run_backtest(
        self,
        backtest_request: BacktestRequest,
        background_tasks: BackgroundTasks,
        token: Dict = Depends(verify_token),
    ) -> Dict:
        """
        Endpoint для запуска бэктестирования

        Args:
            backtest_request: Параметры бэктеста
            background_tasks: Объект для выполнения фоновых задач
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        try:
            # Создаем уникальный ID для бэктеста
            backtest_id = f"backtest_{uuid.uuid4().hex}"

            # Запускаем бэктест в фоновом режиме
            background_tasks.add_task(
                self._run_backtest_task,
                backtest_id=backtest_id,
                request=backtest_request,
            )

            return {
                "status": "success",
                "message": f"Backtest started",
                "backtest_id": backtest_id,
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error starting backtest: {str(e)}",
            )

    async def _run_backtest_task(self, backtest_id: str, request: BacktestRequest):
        """
        Фоновая задача для выполнения бэктеста

        Args:
            backtest_id: ID бэктеста
            request: Параметры бэктеста
        """
        try:
            # Получаем класс стратегии
            strategy_class = StrategyRegistry.get_strategy_class(request.strategy_id)

            # Создаем экземпляр стратегии с указанными параметрами
            strategy = strategy_class(parameters=request.strategy_parameters)

            # Загружаем исторические данные
            ohlcv = await self.exchange_manager.fetch_ohlcv(
                symbol=request.symbol,
                exchange_id=request.exchange_id,
                timeframe=request.timeframe,
                limit=1000,
            )

            if not ohlcv:
                logger.error(
                    f"No historical data available for {request.symbol} on {request.exchange_id}"
                )
                return

            # Преобразуем в pandas DataFrame
            import pandas as pd

            # Создаем DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Преобразуем timestamp в datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Устанавливаем timestamp в качестве индекса
            df.set_index("timestamp", inplace=True)

            # Добавляем атрибуты
            df.attrs["symbol"] = request.symbol
            df.attrs["exchange"] = request.exchange_id
            df.attrs["timeframe"] = request.timeframe

            # Фильтруем данные по дате, если указана
            if request.start_date:
                start_date = pd.to_datetime(request.start_date)
                df = df[df.index >= start_date]

            if request.end_date:
                end_date = pd.to_datetime(request.end_date)
                df = df[df.index <= end_date]

            # Настройки бэктеста
            backtest_settings = {
                "initial_balance": request.initial_balance,
                "commission": request.commission,
                "slippage": request.slippage,
                "position_size_pct": request.position_size_pct,
                "enable_fractional": request.enable_fractional,
                "enable_shorting": request.enable_shorting,
                "start_date": request.start_date,
                "end_date": request.end_date,
            }

            # Если есть дополнительные настройки, добавляем их
            if request.custom_settings:
                backtest_settings.update(request.custom_settings)

            # Выполняем бэктест
            result = await self.backtester.backtest(strategy, df, backtest_settings)

            # Сохраняем результаты в базу данных
            if self.database:
                # Добавляем информацию о бэктесте
                result_data = {
                    "backtest_id": backtest_id,
                    "strategy_id": request.strategy_id,
                    "symbol": request.symbol,
                    "exchange": request.exchange_id,
                    "timeframe": request.timeframe,
                    "start_timestamp": (
                        int(df.index[0].timestamp() * 1000) if len(df) > 0 else None
                    ),
                    "end_timestamp": (
                        int(df.index[-1].timestamp() * 1000) if len(df) > 0 else None
                    ),
                    "parameters": request.strategy_parameters,
                    "metrics": result.get("metrics", {}),
                    "trades": result.get("trades", []),
                    "equity_curve": result.get("equity_curve", []),
                }

                await self.database.save_backtest(result_data)

                logger.info("Backtest {backtest_id} completed and saved")

        except Exception as e:
            logger.error("Error during backtest: {str(e)}")
            logger.error(traceback.format_exc())

    async def get_backtest_results(
        self, limit: int = 10, token: Dict = Depends(verify_token)
    ) -> List[Dict]:
        """
        Endpoint для получения результатов бэктестов

        Args:
            limit: Максимальное количество результатов
            token: Верифицированный токен

        Returns:
            List[Dict]: Список результатов бэктестов
        """
        try:
            # Если база данных недоступна, возвращаем пустой список
            if not self.database:
                return []

            # Получаем результаты бэктестов
            backtests = await self.database.get_backtests(limit=limit)

            return backtests

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting backtest results: {str(e)}",
            )

    async def get_backtest_result(
        self, backtest_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения результата конкретного бэктеста

        Args:
            backtest_id: ID бэктеста
            token: Верифицированный токен

        Returns:
            Dict: Результат бэктеста
        """
        try:
            # Если база данных недоступна, возвращаем ошибку
            if not self.database:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database not available",
                )

            # Получаем результаты бэктеста
            backtests = await self.database.get_backtests()

            # Ищем нужный бэктест
            for backtest in backtests:
                if backtest.get("backtest_id") == backtest_id:
                    return backtest

            # Если бэктест не найден, возвращаем ошибку
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting backtest result: {str(e)}",
            )

    async def run_optimization(
        self,
        optimization_request: OptimizationRequest,
        background_tasks: BackgroundTasks,
        token: Dict = Depends(verify_token),
    ) -> Dict:
        """
        Endpoint для запуска оптимизации

        Args:
            optimization_request: Параметры оптимизации
            background_tasks: Объект для выполнения фоновых задач
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        try:
            # Создаем уникальный ID для задачи оптимизации
            task_id = f"optimization_{uuid.uuid4().hex}"

            # Запускаем оптимизацию в фоновом режиме
            background_tasks.add_task(
                self._run_optimization_task,
                task_id=task_id,
                request=optimization_request,
            )

            # Добавляем задачу в список активных
            self.optimization_tasks[task_id] = {
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "request": optimization_request.dict(),
                "progress": 0.0,
            }

            return {
                "status": "success",
                "message": f"Optimization started",
                "task_id": task_id,
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error starting optimization: {str(e)}",
            )

    async def _run_optimization_task(self, task_id: str, request: OptimizationRequest):
        """
        Фоновая задача для выполнения оптимизации

        Args:
            task_id: ID задачи оптимизации
            request: Параметры оптимизации
        """
        try:
            # Получаем класс стратегии
            strategy_class = StrategyRegistry.get_strategy_class(request.strategy_id)

            # Загружаем исторические данные
            ohlcv = await self.exchange_manager.fetch_ohlcv(
                symbol=request.symbol,
                exchange_id=request.exchange_id,
                timeframe=request.timeframe,
                limit=1000,
            )

            if not ohlcv:
                logger.error(
                    f"No historical data available for {request.symbol} on {request.exchange_id}"
                )
                self.optimization_tasks[task_id]["status"] = "error"
                self.optimization_tasks[task_id][
                    "error"
                ] = "No historical data available"
                return

            # Преобразуем в pandas DataFrame
            import pandas as pd

            # Создаем DataFrame
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Преобразуем timestamp в datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Устанавливаем timestamp в качестве индекса
            df.set_index("timestamp", inplace=True)

            # Добавляем атрибуты
            df.attrs["symbol"] = request.symbol
            df.attrs["exchange"] = request.exchange_id
            df.attrs["timeframe"] = request.timeframe

            # Фильтруем данные по дате, если указана
            if request.start_date:
                start_date = pd.to_datetime(request.start_date)
                df = df[df.index >= start_date]

            if request.end_date:
                end_date = pd.to_datetime(request.end_date)
                df = df[df.index <= end_date]

            # Настройки оптимизации
            from project.optimization.genetic_optimizer import GeneticOptimizer

            # Настройки генетического алгоритма
            genetic_config = {
                "population_size": request.population_size,
                "generations": request.generations,
                "crossover_rate": request.crossover_rate,
                "mutation_rate": request.mutation_rate,
                "elitism_rate": request.elitism_rate,
                "max_workers": request.max_workers,
                "fitness_metrics": request.fitness_metrics
                or {
                    "total_return": 1.0,
                    "sharpe_ratio": 1.0,
                    "max_drawdown": -0.5,
                    "win_rate": 0.3,
                },
            }

            # Создаем оптимизатор
            optimizer = GeneticOptimizer(genetic_config)

            # Определяем функцию обратного вызова для обновления прогресса
            async def progress_callback(progress: float, stats: Dict):
                # Обновляем прогресс в задаче
                if task_id in self.optimization_tasks:
                    self.optimization_tasks[task_id]["progress"] = progress
                    self.optimization_tasks[task_id]["stats"] = stats

            # Подключаем базу данных
            optimizer.set_database(self.database)

            # Запускаем оптимизацию
            optimization_result = await optimizer.optimize(
                strategy_class=strategy_class,
                parameter_ranges=request.parameter_ranges,
                data=df,
                progress_callback=progress_callback,
            )

            response = {
                "status": "success",
                "message": "Operation completed",
                "data": optimization_result,
            }

            # Обновляем статус задачи
            if task_id in self.optimization_tasks:
                self.optimization_tasks[task_id]["status"] = "completed"
                self.optimization_tasks[task_id][
                    "end_time"
                ] = datetime.now().isoformat()
                self.optimization_tasks[task_id]["result"] = optimization_result

            logger.info("Optimization {task_id} completed")

        except Exception as e:
            logger.error("Error during optimization: {str(e)}")
            logger.error(traceback.format_exc())

            # Обновляем статус задачи
            if task_id in self.optimization_tasks:
                self.optimization_tasks[task_id]["status"] = "error"
                self.optimization_tasks[task_id]["error"] = str(e)
                self.optimization_tasks[task_id][
                    "end_time"
                ] = datetime.now().isoformat()

    async def get_optimization_tasks(
        self, token: Dict = Depends(verify_token)
    ) -> List[Dict]:
        """
        Endpoint для получения списка задач оптимизации

        Args:
            token: Верифицированный токен

        Returns:
            List[Dict]: Список задач оптимизации
        """
        try:
            # Преобразуем словарь задач в список
            tasks = []
            for task_id, task_info in self.optimization_tasks.items():
                task = {
                    "task_id": task_id,
                    "status": task_info["status"],
                    "start_time": task_info["start_time"],
                    "progress": task_info["progress"],
                    "request": task_info["request"],
                }
                if "end_time" in task_info:
                    task["end_time"] = task_info["end_time"]
                if "result" in task_info:
                    task["result"] = task_info["result"]
                if "error" in task_info:
                    task["error"] = task_info["error"]
                tasks.append(task)

            return tasks

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting optimization tasks: {str(e)}",
            )

    async def get_optimization_task(
        self, task_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения информации о задаче оптимизации

        Args:
            task_id: ID задачи
            token: Верифицированный токен

        Returns:
            Dict: Информация о задаче
        """
        try:
            # Проверяем существование задачи
            if task_id not in self.optimization_tasks:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task {task_id} not found",
                )

            return self.optimization_tasks[task_id]

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting optimization task: {str(e)}",
            )

    async def cancel_optimization_task(
        self, task_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для отмены задачи оптимизации

        Args:
            task_id: ID задачи
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        try:
            # Проверяем существование задачи
            if task_id not in self.optimization_tasks:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Task {task_id} not found",
                )

            # Обновляем статус задачи
            self.optimization_tasks[task_id]["status"] = "cancelled"
            self.optimization_tasks[task_id]["end_time"] = datetime.now().isoformat()

            return {"status": "success", "message": f"Task {task_id} cancelled"}

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error cancelling optimization task: {str(e)}",
            )

    async def get_optimization_results(
        self, limit: int = 10, token: Dict = Depends(verify_token)
    ) -> List[Dict]:
        """
        Endpoint для получения результатов оптимизаций

        Args:
            limit: Максимальное количество результатов
            token: Верифицированный токен

        Returns:
            List[Dict]: Список результатов оптимизаций
        """
        try:
            # Если база данных недоступна, возвращаем пустой список
            if not self.database:
                return []

            # Получаем результаты оптимизаций
            optimizations = await self.database.get_optimizations(limit=limit)

            return optimizations

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting optimization results: {str(e)}",
            )

    async def get_optimization_result(
        self, optimization_id: str, token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для получения результата конкретной оптимизации

        Args:
            optimization_id: ID оптимизации
            token: Верифицированный токен

        Returns:
            Dict: Результат оптимизации
        """
        try:
            # Если база данных недоступна, возвращаем ошибку
            if not self.database:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database not available",
                )

            # Получаем результаты оптимизации
            optimizations = await self.database.get_optimizations()

            # Ищем нужную оптимизацию
            for optimization in optimizations:
                if optimization.get("optimization_id") == optimization_id:
                    return optimization

            # Если оптимизация не найдена, возвращаем ошибку
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization {optimization_id} not found",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting optimization result: {str(e)}",
            )

    async def get_ohlc_chart(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        token: Dict = Depends(verify_token),
    ) -> StreamingResponse:
        """
        Endpoint для получения графика OHLC

        Args:
            exchange_id: ID биржи
            symbol: Символ торговой пары
            timeframe: Временной интервал
            limit: Количество свечей
            token: Верифицированный токен

        Returns:
            StreamingResponse: График OHLC
        """
        try:
            # Получаем OHLCV данные
            ohlcv = await self.exchange_manager.fetch_ohlcv(
                symbol, exchange_id, timeframe, limit
            )

            if ohlcv is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Could not fetch OHLCV data for {symbol} on {exchange_id}",
                )

            # Преобразуем в pandas DataFrame
            import pandas as pd

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Создаем график
            fig = self.visualizer.plot_ohlc(df)

            # Преобразуем график в изображение
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)

            return StreamingResponse(buf, media_type="image/png")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting OHLC chart: {str(e)}",
            )

    async def get_equity_chart(
        self, backtest_id: str, token: Dict = Depends(verify_token)
    ) -> StreamingResponse:
        """
        Endpoint для получения графика equity

        Args:
            backtest_id: ID бэктеста
            token: Верифицированный токен

        Returns:
            StreamingResponse: График equity
        """
        try:
            # Если база данных недоступна, возвращаем ошибку
            if not self.database:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database not available",
                )

            # Получаем результаты бэктеста
            backtests = await self.database.get_backtests()

            # Ищем нужный бэктест
            for backtest in backtests:
                if backtest.get("backtest_id") == backtest_id:
                    equity_curve = backtest.get("equity_curve", [])

                    # Преобразуем в pandas DataFrame
                    import pandas as pd

                    df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)

                    # Создаем график
                    fig = self.visualizer.plot_equity_curve(df)

                    # Преобразуем график в изображение
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)

                    return StreamingResponse(buf, media_type="image/png")

            # Если бэктест не найден, возвращаем ошибку
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting equity chart: {str(e)}",
            )

    async def get_drawdown_chart(
        self, backtest_id: str, token: Dict = Depends(verify_token)
    ) -> StreamingResponse:
        """
        Endpoint для получения графика drawdown

        Args:
            backtest_id: ID бэктеста
            token: Верифицированный токен

        Returns:
            StreamingResponse: График drawdown
        """
        try:
            # Если база данных недоступна, возвращаем ошибку
            if not self.database:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database not available",
                )

            # Получаем результаты бэктеста
            backtests = await self.database.get_backtests()

            # Ищем нужный бэктест
            for backtest in backtests:
                if backtest.get("backtest_id") == backtest_id:
                    equity_curve = backtest.get("equity_curve", [])

                    # Преобразуем в pandas DataFrame
                    import pandas as pd

                    df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)

                    # Создаем график
                    fig = self.visualizer.plot_drawdown(df)

                    # Преобразуем график в изображение
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)

                    return StreamingResponse(buf, media_type="image/png")

            # Если бэктест не найден, возвращаем ошибку
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting drawdown chart: {str(e)}",
            )

    async def get_monthly_returns_chart(
        self, backtest_id: str, token: Dict = Depends(verify_token)
    ) -> StreamingResponse:
        """
        Endpoint для получения графика monthly returns

        Args:
            backtest_id: ID бэктеста
            token: Верифицированный токен

        Returns:
            StreamingResponse: График monthly returns
        """
        try:
            # Если база данных недоступна, возвращаем ошибку
            if not self.database:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database not available",
                )

            # Получаем результаты бэктеста
            backtests = await self.database.get_backtests()

            # Ищем нужный бэктест
            for backtest in backtests:
                if backtest.get("backtest_id") == backtest_id:
                    equity_curve = backtest.get("equity_curve", [])

                    # Преобразуем в pandas DataFrame
                    import pandas as pd

                    df = pd.DataFrame(equity_curve, columns=["timestamp", "equity"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)

                    # Создаем график
                    fig = self.visualizer.plot_monthly_returns(df)

                    # Преобразуем график в изображение
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)

                    return StreamingResponse(buf, media_type="image/png")

            # Если бэктест не найден, возвращаем ошибку
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest {backtest_id} not found",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting monthly returns chart: {str(e)}",
            )

    async def get_optimization_chart(
        self, optimization_id: str, token: Dict = Depends(verify_token)
    ) -> StreamingResponse:
        """
        Endpoint для получения графика оптимизации

        Args:
            optimization_id: ID оптимизации
            token: Верифицированный токен

        Returns:
            StreamingResponse: График оптимизации
        """
        try:
            # Если база данных недоступна, возвращаем ошибку
            if not self.database:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database not available",
                )

            # Получаем результаты оптимизации
            optimizations = await self.database.get_optimizations()

            # Ищем нужную оптимизацию
            for optimization in optimizations:
                if optimization.get("optimization_id") == optimization_id:
                    results = optimization.get("results", [])

                    # Преобразуем в pandas DataFrame
                    import pandas as pd

                    df = pd.DataFrame(results)

                    # Создаем график
                    fig = self.visualizer.plot_optimization_results(df)

                    # Преобразуем график в изображение
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)

                    return StreamingResponse(buf, media_type="image/png")

            # Если оптимизация не найдена, возвращаем ошибку
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization {optimization_id} not found",
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting optimization chart: {str(e)}",
            )

    async def get_system_status(self, token: Dict = Depends(verify_token)) -> Dict:
        """
        Endpoint для получения статуса системы

        Args:
            token: Верифицированный токен

        Returns:
            Dict: Статус системы
        """
        try:
            # Получаем статус системы
            status = {
                "uptime": time.time() - self.start_time,
                "active_bots": len(self.bots),
                "database_connected": (
                    self.database.is_connected() if self.database else False
                ),
                "exchange_manager_connected": (
                    self.exchange_manager.is_connected()
                    if self.exchange_manager
                    else False
                ),
            }

            return status

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting system status: {str(e)}",
            )

    async def get_logs(self, token: Dict = Depends(verify_token)) -> FileResponse:
        """
        Endpoint для получения логов

        Args:
            token: Верифицированный токен

        Returns:
            FileResponse: Логи
        """
        try:
            # Путь к файлу логов
            log_file_path = os.path.join(
                os.path.dirname(__file__), "logs", "api_server.log"
            )

            # Проверяем существование файла
            if not os.path.exists(log_file_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Log file not found"
                )

            return FileResponse(
                log_file_path, media_type="text/plain", filename="api_server.log"
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting logs: {str(e)}",
            )

    async def send_notification(
        self, message: str = Body(...), token: Dict = Depends(verify_token)
    ) -> Dict:
        """
        Endpoint для отправки уведомления

        Args:
            message: Сообщение уведомления
            token: Верифицированный токен

        Returns:
            Dict: Результат операции
        """
        try:
            # Отправляем уведомление
            await self.notification_manager.send_notification(message)

            return {"status": "success", "message": "Notification sent"}

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error sending notification: {str(e)}",
            )


if __name__ == "__main__":
    config = get_config()
    api_server = APIServer(config=config.get("api", {}))
    uvicorn.run(
        api_server.app,
        host=api_server.host,
        port=api_server.port,
        reload=api_server.reload,
        workers=api_server.workers,
    )
