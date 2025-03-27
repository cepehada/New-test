"""
Модуль REST API для внешнего взаимодействия с торговым ботом.
Предоставляет HTTP-интерфейс для управления стратегиями и мониторинга.
"""

import json
import time

import jwt
from aiohttp import web
from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RestAPI:
    """
    REST API для управления торговым ботом.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Инициализирует REST API.

        Args:
            host: Хост для запуска сервера API
            port: Порт для запуска сервера API
        """
        self.config = get_config()
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.site = None
        self.strategy_manager = None
        self.secret_key = self.config.ENCRYPTION_KEY

        # Настраиваем маршруты
        self._setup_routes()
        logger.debug("REST API настроен на {host}:{port}" %)

    def _setup_routes(self) -> None:
        """
        Настраивает маршруты API.
        """
        # Роутеры для разных групп эндпоинтов
        self.app.router.add_get("/api/health", self.health_check)
        self.app.router.add_get("/api/version", self.get_version)

        # Маршруты для стратегий
        self.app.router.add_get("/api/strategies", self.get_strategies)
        self.app.router.add_get("/api/strategies/{strategy_id}", self.get_strategy)
        self.app.router.add_post("/api/strategies", self.create_strategy)
        self.app.router.add_delete(
            "/api/strategies/{strategy_id}", self.delete_strategy
        )

        # Маршруты для ордеров
        self.app.router.add_get("/api/orders", self.get_orders)
        self.app.router.add_post("/api/orders", self.create_order)
        self.app.router.add_delete("/api/orders/{order_id}", self.cancel_order)

        # Маршруты для рыночных данных
        self.app.router.add_get("/api/market/ticker/{symbol}", self.get_ticker)
        self.app.router.add_get("/api/market/ohlcv/{symbol}", self.get_ohlcv)

        # Добавляем промежуточное ПО для аутентификации
        self.app.middlewares.append(self._auth_middleware)

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        """
        Промежуточное ПО для аутентификации запросов.

        Args:
            request: HTTP-запрос
            handler: Обработчик запроса

        Returns:
            HTTP-ответ
        """
        # Пропускаем аутентификацию для некоторых маршрутов
        if request.path in ["/api/health", "/api/version"]:
            return await handler(request)

        # Проверяем наличие токена в заголовке Authorization
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return web.json_response(
                {
                    "error": "Unauthorized",
                    "message": "Missing or invalid authorization token",
                },
                status=401,
            )

        token = auth_header.replace("Bearer ", "")

        try:
            # Валидируем JWT-токен
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])

            # Проверяем срок действия токена
            if "exp" in payload and payload["exp"] < time.time():
                return web.json_response(
                    {"error": "Unauthorized", "message": "Token expired"}, status=401
                )

            # Добавляем данные пользователя в запрос
            request["user"] = payload

            # Продолжаем выполнение запроса
            return await handler(request)

        except jwt.InvalidTokenError:
            return web.json_response(
                {"error": "Unauthorized", "message": "Invalid token"}, status=401
            )

    async def initialize(self) -> None:
        """
        Инициализирует REST API и зависимые компоненты.
        """
        # Получаем экземпляр менеджера стратегий
        from project.bots.strategies.strategy_manager import StrategyManager

        self.strategy_manager = StrategyManager.get_instance()

        logger.info("REST API инициализирован")

    @async_handle_error
    async def start(self) -> None:
        """
        Запускает сервер REST API.
        """
        if self.runner:
            logger.warning("REST API уже запущен")
            return

        await self.initialize()

        # Запускаем сервер
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        logger.info("REST API запущен на http://{self.host}:{self.port}" %)

    async def stop(self) -> None:
        """
        Останавливает сервер REST API.
        """
        if self.site:
            await self.site.stop()
            self.site = None

        if self.runner:
            await self.runner.cleanup()
            self.runner = None

        logger.info("REST API остановлен")

    # Обработчики маршрутов

    async def health_check(self, request: web.Request) -> web.Response:
        """Проверка работоспособности API"""
        return web.json_response({"status": "ok", "timestamp": time.time()})

    async def get_version(self, request: web.Request) -> web.Response:
        """Получение версии API"""
        from project import __version__

        return web.json_response(
            {"version": __version__, "application": "Trading Bot API"}
        )

    async def get_strategies(self, request: web.Request) -> web.Response:
        """Получение списка стратегий"""
        if not self.strategy_manager:
            return web.json_response({"error": "Service unavailable"}, status=503)

        strategies = []
        for strategy_id, strategy in self.strategy_manager.running_strategies.items():
            strategies.append(
                {
                    "id": strategy_id,
                    "name": strategy.__class__.__name__,
                    "status": "running" if not strategy.is_stopped() else "stopped",
                    "config": strategy.config,
                }
            )

        return web.json_response({"strategies": strategies})

    async def get_strategy(self, request: web.Request) -> web.Response:
        """Получение информации о стратегии по ID"""
        if not self.strategy_manager:
            return web.json_response({"error": "Service unavailable"}, status=503)

        strategy_id = request.match_info.get("strategy_id")

        if strategy_id not in self.strategy_manager.running_strategies:
            return web.json_response(
                {
                    "error": "Not found",
                    "message": f"Strategy with ID {strategy_id} not found",
                },
                status=404,
            )

        strategy = self.strategy_manager.running_strategies[strategy_id]

        return web.json_response(
            {
                "id": strategy_id,
                "name": strategy.__class__.__name__,
                "status": "running" if not strategy.is_stopped() else "stopped",
                "config": strategy.config,
                "state": strategy.state,
            }
        )

    async def create_strategy(self, request: web.Request) -> web.Response:
        """Создание новой стратегии"""
        if not self.strategy_manager:
            return web.json_response({"error": "Service unavailable"}, status=503)

        try:
            data = await request.json()

            if "name" not in data:
                return web.json_response(
                    {"error": "Bad request", "message": "Strategy name is required"},
                    status=400,
                )

            # Получаем имя стратегии и параметры
            strategy_name = data["name"]
            params = data.get("params", {})

            # Проверяем, существует ли такая стратегия
            if strategy_name not in self.strategy_manager.strategies:
                return web.json_response(
                    {
                        "error": "Bad request",
                        "message": f"Strategy {strategy_name} not found",
                    },
                    status=400,
                )

            # Запускаем стратегию
            strategy_id = await self.strategy_manager.start_strategy(
                strategy_name, **params
            )

            return web.json_response(
                {
                    "id": strategy_id,
                    "name": strategy_name,
                    "status": "running",
                    "message": f"Strategy {strategy_name} started with ID {strategy_id}",
                },
                status=201,
            )

        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Bad request", "message": "Invalid JSON"}, status=400
            )
        except Exception as e:
            logger.error("Error creating strategy: {str(e)}" %)
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )

    async def delete_strategy(self, request: web.Request) -> web.Response:
        """Остановка и удаление стратегии"""
        if not self.strategy_manager:
            return web.json_response({"error": "Service unavailable"}, status=503)

        strategy_id = request.match_info.get("strategy_id")

        if strategy_id not in self.strategy_manager.running_strategies:
            return web.json_response(
                {
                    "error": "Not found",
                    "message": f"Strategy with ID {strategy_id} not found",
                },
                status=404,
            )

        try:
            await self.strategy_manager.stop_strategy(strategy_id)

            return web.json_response(
                {
                    "id": strategy_id,
                    "status": "stopped",
                    "message": f"Strategy with ID {strategy_id} stopped",
                }
            )

        except Exception as e:
            logger.error("Error stopping strategy {strategy_id}: {str(e)}" %)
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )

    async def get_orders(self, request: web.Request) -> web.Response:
        """Получение списка ордеров"""
        try:
            # Получаем параметры запроса
            exchange_id = request.query.get("exchange", "binance")
            symbol = request.query.get("symbol")
            status = request.query.get("status", "open")  # open, closed, all

            # Импортируем модуль для работы с биржами
            from project.utils.ccxt_exchanges import (
                connect_exchange,
                fetch_closed_orders,
                fetch_open_orders,
            )

            exchange = await connect_exchange(exchange_id)

            if status == "open":
                orders = await fetch_open_orders(exchange_id, symbol)
            elif status == "closed":
                orders = await fetch_closed_orders(exchange_id, symbol)
            else:  # all
                open_orders = await fetch_open_orders(exchange_id, symbol)
                closed_orders = await fetch_closed_orders(exchange_id, symbol)
                orders = open_orders + closed_orders

            return web.json_response({"orders": orders})

        except Exception as e:
            logger.error("Error getting orders: {str(e)}" %)
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )

    async def create_order(self, request: web.Request) -> web.Response:
        """Создание нового ордера"""
        try:
            data = await request.json()

            # Проверяем наличие обязательных полей
            required_fields = ["exchange", "symbol", "type", "side", "amount"]
            for field in required_fields:
                if field not in data:
                    return web.json_response(
                        {
                            "error": "Bad request",
                            "message": f"Field {field} is required",
                        },
                        status=400,
                    )

            # Получаем параметры ордера
            exchange_id = data["exchange"]
            symbol = data["symbol"]
            order_type = data["type"]
            side = data["side"]
            amount = float(data["amount"])
            price = float(data.get("price", 0)) if "price" in data else None
            params = data.get("params", {})

            # Создаем ордер через модуль исполнения ордеров
            from project.trade_executor.order_executor import OrderExecutor

            executor = OrderExecutor()
            result = await executor.execute_order(
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price,
                exchange_id=exchange_id,
                **params,
            )

            if not result.success:
                return web.json_response(
                    {"error": "Order execution failed", "message": result.error},
                    status=500,
                )

            return web.json_response(
                {
                    "order_id": result.order_id,
                    "filled_quantity": result.filled_quantity,
                    "average_price": result.average_price,
                    "fees": result.fees,
                    "status": "success",
                },
                status=201,
            )

        except json.JSONDecodeError:
            return web.json_response(
                {"error": "Bad request", "message": "Invalid JSON"}, status=400
            )
        except Exception as e:
            logger.error("Error creating order: {str(e)}" %)
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )

    async def cancel_order(self, request: web.Request) -> web.Response:
        """Отмена ордера"""
        try:
            order_id = request.match_info.get("order_id")
            exchange_id = request.query.get("exchange", "binance")
            symbol = request.query.get("symbol")

            if not symbol:
                return web.json_response(
                    {"error": "Bad request", "message": "Symbol is required"},
                    status=400,
                )

            # Отменяем ордер через CCXT
            from project.utils.ccxt_exchanges import cancel_order

            result = await cancel_order(exchange_id, order_id, symbol)

            return web.json_response(
                {"order_id": order_id, "status": "cancelled", "result": result}
            )

        except Exception as e:
            logger.error("Error cancelling order: {str(e)}" %)
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )

    async def get_ticker(self, request: web.Request) -> web.Response:
        """Получение тикера для символа"""
        try:
            symbol = request.match_info.get("symbol")
            exchange_id = request.query.get("exchange", "binance")

            # Получаем тикер через CCXT
            from project.utils.ccxt_exchanges import fetch_ticker

            ticker = await fetch_ticker(exchange_id, symbol)

            return web.json_response(ticker)

        except Exception as e:
            logger.error("Error getting ticker: {str(e)}" %)
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )

    async def get_ohlcv(self, request: web.Request) -> web.Response:
        """Получение OHLCV данных для символа"""
        try:
            symbol = request.match_info.get("symbol")
            exchange_id = request.query.get("exchange", "binance")
            timeframe = request.query.get("timeframe", "1h")
            limit = int(request.query.get("limit", 100))

            # Получаем OHLCV через CCXT
            from project.utils.ccxt_exchanges import fetch_ohlcv

            ohlcv = await fetch_ohlcv(exchange_id, symbol, timeframe, limit=limit)

            # Преобразуем данные в формат [timestamp, open, high, low, close, volume]
            formatted_ohlcv = []
            for candle in ohlcv:
                formatted_ohlcv.append(
                    {
                        "timestamp": candle[0],
                        "open": candle[1],
                        "high": candle[2],
                        "low": candle[3],
                        "close": candle[4],
                        "volume": candle[5],
                    }
                )

            return web.json_response({"ohlcv": formatted_ohlcv})

        except Exception as e:
            logger.error("Error getting OHLCV data: {str(e)}" %)
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )
