"""
Базовый класс для всех торговых ботов системы.
Предоставляет основные функции и интерфейс для создания конкретных ботов.
"""

# Стандартные импорты
import asyncio
import uuid
from typing import Dict, List, Any

# Внутренние импорты
from project.config import get_config
from project.data.market_data import MarketData
from project.trade_executor.order_executor import OrderExecutor
from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)


class BaseBot:
    """
    Базовый класс для всех торговых ботов.

    Предоставляет общую функциональность для мониторинга рынка,
    выполнения сделок и управления состоянием бота.
    """

    def __init__(self, config=None, name=None, exchange_id="binance", symbols=None):
        """
        Инициализирует базовый бот.

        Args:
            config: Конфигурация бота
            name: Имя бота
            exchange_id: ID биржи для использования
            symbols: Список символов для мониторинга
        """
        # Основные параметры
        self.bot_id = str(uuid.uuid4())
        self.name = name or f"bot_{self.bot_id[:8]}"
        self.exchange_id = exchange_id
        self.symbols = symbols or ["BTC/USDT"]

        # Загрузка конфигурации
        self.config = config or get_config()

        # Инициализация компонентов
        self.market_data = MarketData.get_instance()
        self.order_executor = OrderExecutor.get_instance()

        # Состояние
        self.running = False
        self.status = "initialized"
        self.error = None
        self.last_run_time = 0
        self.stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": 0.0,
            "start_time": 0,
        }

        # Мониторинг задачи
        self.task = None
        self.check_interval = 60  # в секундах

        logger.info(
            "Инициализирован бот %s для %s на %s",
            self.name,
            self.symbols,
            self.exchange_id,
        )

    @async_handle_error
    async def start(self):
        """
        Запускает бота.

        Returns:
            bool: Успешность запуска
        """
        if self.running:
            logger.info("Бот %s уже запущен", self.name)
            return True

        try:
            self.running = True
            self.status = "running"

            # Запуск задачи мониторинга
            self.task = asyncio.create_task(self._run_loop())
            logger.info("Бот %s запущен", self.name)
            return True

        except Exception as e:
            self.running = False
            self.error = str(e)
            self.status = "error"
            logger.error("Ошибка запуска бота %s: %s", self.name, str(e))
            return False

    @async_handle_error
    async def stop(self):
        """
        Останавливает бота.

        Returns:
            bool: Успешность остановки
        """
        if not self.running:
            logger.info("Бот %s уже остановлен", self.name)
            return True

        logger.info("Останавливаем бота %s", self.name)
        self.running = False

        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.status = "stopped"
        return True

    @async_handle_error
    async def restart(self):
        """
        Перезапускает бота.

        Returns:
            bool: Успешность перезапуска
        """
        logger.info("Перезапуск бота %s", self.name)

        # Останавливаем бота, если запущен
        if self.running:
            await self.stop()

        # Запускаем снова
        success = await self.start()
        return success

    @async_handle_error
    async def get_status(self):
        """
        Получает текущий статус бота.

        Returns:
            str: Статус бота
        """
        return self.status

    @async_handle_error
    async def get_state(self):
        """
        Получает полное состояние бота.

        Returns:
            Dict: Состояние бота
        """
        return {
            "bot_id": self.bot_id,
            "name": self.name,
            "exchange": self.exchange_id,
            "symbols": self.symbols,
            "status": self.status,
            "error": self.error,
            "running": self.running,
            "stats": self.stats,
        }

    @async_handle_error
    async def _run_loop(self):
        """
        Основной цикл работы бота.
        """
        try:
            logger.info("Начало мониторинга рынка ботом %s", self.name)
            self.stats["start_time"] = asyncio.get_event_loop().time()

            while self.running:
                start_time = asyncio.get_event_loop().time()

                # Запускаем основную логику
                await self._process_market_data()

                # Вычисляем время ожидания до следующего цикла
                elapsed = asyncio.get_event_loop().time() - start_time
                wait_time = max(0, self.check_interval - elapsed)

                # Ждем следующий цикл
                await asyncio.sleep(wait_time)

        except asyncio.CancelledError:
            logger.info("Задача бота %s отменена", self.name)

        except Exception as e:
            self.error = str(e)
            self.status = "error"
            logger.error("Ошибка в основном цикле бота %s: %s", self.name, str(e))

        finally:
            self.running = False
            logger.info("Основной цикл бота %s завершен", self.name)

    async def _process_market_data(self):
        """
        Обрабатывает рыночные данные (метод для переопределения).
        """
        # Реализация в наследниках

    async def _process_signals(self, signals):
        """
        Обрабатывает сигналы (метод для переопределения).
        """
        # Реализация в наследниках

    @async_handle_error
    async def execute_order(
        self, symbol, side, amount, order_type="market", price=None
    ):
        """
        Выполняет ордер на бирже.

        Args:
            symbol: Торговая пара
            side: Сторона (buy или sell)
            amount: Объем
            order_type: Тип ордера (market или limit)
            price: Цена (для limit ордеров)

        Returns:
            Dict: Результат выполнения ордера
        """
        try:
            logger.info(
                "Выполнение ордера: %s %s %s, объем: %.8f, цена: %s",
                self.exchange_id,
                symbol,
                side,
                amount,
                str(price) if price else "рыночная",
            )

            # Выполняем ордер через исполнитель
            result = await self.order_executor.execute_order(
                exchange_id=self.exchange_id,
                symbol=symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price,
            )

            # Обновляем статистику
            self.stats["total_trades"] += 1
            if result.success:
                self.stats["successful_trades"] += 1

                # Отправляем уведомление
                await send_trading_signal(
                    f"Ордер выполнен: {side.upper()} {symbol} на {self.exchange_id}, "
                    f"объем: {amount}, стоимость: {result.cost:.2f} USD"
                )
            else:
                self.stats["failed_trades"] += 1
                logger.error("Ошибка выполнения ордера: %s", result.error)

            return result

        except Exception as e:
            logger.error("Ошибка при выполнении ордера: %s", str(e))
            self.stats["failed_trades"] += 1
            return None

    @async_handle_error
    async def get_balance(self, currency=None):
        """
        Получает баланс на бирже.

        Args:
            currency: Валюта для получения баланса

        Returns:
            Dict или float: Баланс для указанной валюты или для всех валют
        """
        try:
            # Получаем баланс через исполнитель
            balance = await self.order_executor.get_balance(self.exchange_id)

            if currency:
                return balance.get("free", {}).get(currency, 0.0)
            return balance.get("free", {})

        except Exception as e:
            logger.error("Ошибка при получении баланса: %s", str(e))
            return {} if currency is None else 0.0

    @async_handle_error
    async def get_market_data(self, symbol):
        """
        Получает рыночные данные для символа.

        Args:
            symbol: Торговая пара

        Returns:
            Dict: Рыночные данные
        """
        try:
            # Получаем данные через MarketData
            ticker = await self.market_data.get_ticker(self.exchange_id, symbol)
            orderbook = await self.market_data.get_orderbook(self.exchange_id, symbol)

            return {"ticker": ticker, "orderbook": orderbook}

        except Exception as e:
            logger.error(
                "Ошибка при получении рыночных данных для %s: %s", symbol, str(e)
            )
            return {}

    async def send_notification(self, message):
        """
        Отправляет уведомление.

        Args:
            message: Сообщение для отправки
        """
        await send_trading_signal(message)
