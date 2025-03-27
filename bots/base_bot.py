"""
Базовый класс для всех торговых ботов.
Предоставляет общую функциональность для всех типов ботов.
"""

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Dict, List

from project.config import get_config
from project.data.market_data import MarketData
from project.risk_management.position_sizer import PositionSizer
from project.trade_executor.order_executor import OrderExecutor
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)


class BotStatus(Enum):
    """
    Статусы работы бота.
    """

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class BaseBot:
    """
    Базовый класс для всех торговых ботов.
    """

    def __init__(
        self,
        name: str = "BaseBot",
        exchange_id: str = "binance",
        symbols: List[str] = None,
    ):
        """
        Инициализирует базовый торговый бот.

        Args:
            name: Имя бота
            exchange_id: Идентификатор биржи
            symbols: Список символов для торговли
        """
        self.config = get_config()
        self.name = name
        self.bot_id = str(uuid.uuid4())
        self.exchange_id = exchange_id
        self.symbols = symbols or []
        self.status = BotStatus.STOPPED
        self.task = None
        self.start_time = 0
        self.last_update_time = 0
        self.market_data = MarketData.get_instance()
        self.order_executor = OrderExecutor.get_instance()
        self.position_sizer = PositionSizer()
        self.update_interval = 10.0  # секунды
        self.stats = {
            "trades_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
        }
        logger.debug("Создан бот {self.name} (id: {self.bot_id}) для {exchange_id}" %)

    @async_handle_error
    async def start(self) -> bool:
        """
        Запускает бота.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status != BotStatus.STOPPED:
            logger.warning("Бот {self.name} уже запущен или в процессе запуска" %)
            return False

        logger.info("Запуск бота {self.name} (id: {self.bot_id})" %)
        self.status = BotStatus.STARTING

        try:
            # Инициализируем бота
            await self._initialize()

            # Запускаем основную задачу бота
            self.task = asyncio.create_task(self._run())
            self.start_time = time.time()
            self.status = BotStatus.RUNNING

            logger.info("Бот {self.name} успешно запущен" %)
            await send_trading_signal(f"Бот {self.name} запущен")

            return True

        except Exception as e:
            logger.error("Ошибка при запуске бота {self.name}: {str(e)}" %)
            self.status = BotStatus.ERROR
            return False

    async def stop(self) -> bool:
        """
        Останавливает бота.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status == BotStatus.STOPPED:
            logger.warning("Бот {self.name} уже остановлен" %)
            return False

        logger.info("Остановка бота {self.name} (id: {self.bot_id})" %)
        self.status = BotStatus.STOPPING

        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

        # Вызываем метод очистки
        try:
            await self._cleanup()
        except Exception as e:
            logger.error("Ошибка при очистке бота {self.name}: {str(e)}" %)

        self.status = BotStatus.STOPPED
        logger.info("Бот {self.name} успешно остановлен" %)
        await send_trading_signal(f"Бот {self.name} остановлен")

        return True

    async def pause(self) -> bool:
        """
        Приостанавливает работу бота.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status != BotStatus.RUNNING:
            logger.warning("Бот {self.name} не запущен, невозможно приостановить" %)
            return False

        logger.info("Приостановка бота {self.name} (id: {self.bot_id})" %)
        self.status = BotStatus.PAUSED

        await send_trading_signal(f"Бот {self.name} приостановлен")

        return True

    async def resume(self) -> bool:
        """
        Возобновляет работу бота после приостановки.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status != BotStatus.PAUSED:
            logger.warning("Бот {self.name} не приостановлен, невозможно возобновить" %)
            return False

        logger.info("Возобновление работы бота {self.name} (id: {self.bot_id})" %)
        self.status = BotStatus.RUNNING

        await send_trading_signal(f"Бот {self.name} возобновил работу")

        return True

    def get_status(self) -> str:
        """
        Получает текущий статус бота.

        Returns:
            Строковое представление статуса
        """
        return self.status.value

    def is_running(self) -> bool:
        """
        Проверяет, запущен ли бот.

        Returns:
            True, если бот запущен, иначе False
        """
        return self.status == BotStatus.RUNNING

    def get_stats(self) -> Dict[str, Any]:
        """
        Получает статистику работы бота.

        Returns:
            Словарь со статистикой
        """
        stats = dict(self.stats)

        # Добавляем дополнительную информацию
        stats["name"] = self.name
        stats["bot_id"] = self.bot_id
        stats["status"] = self.status.value
        stats["exchange"] = self.exchange_id
        stats["symbols"] = self.symbols
        stats["uptime"] = time.time() - self.start_time if self.start_time > 0 else 0
        stats["last_update"] = self.last_update_time

        # Рассчитываем дополнительные метрики
        if stats["trades_count"] > 0:
            stats["win_rate"] = stats["winning_trades"] / stats["trades_count"]

        if stats["winning_trades"] > 0:
            stats["average_profit"] = stats["total_profit"] / stats["winning_trades"]

        if stats["losing_trades"] > 0:
            stats["average_loss"] = stats["total_loss"] / stats["losing_trades"]

        # Рассчитываем общую прибыль
        stats["net_profit"] = stats["total_profit"] - stats["total_loss"]

        # Рассчитываем коэффициент прибыль/риск
        if stats["total_loss"] > 0:
            stats["profit_factor"] = stats["total_profit"] / stats["total_loss"]
        else:
            stats["profit_factor"] = float("inf") if stats["total_profit"] > 0 else 0.0

        return stats

    @async_handle_error
    async def update_symbols(self, symbols: List[str]) -> bool:
        """
        Обновляет список символов для торговли.

        Args:
            symbols: Новый список символов

        Returns:
            True в случае успеха, иначе False
        """
        try:
            logger.info("Обновление списка символов для бота {self.name}: {symbols}" %)
            self.symbols = symbols

            # Если бот запущен, обновляем данные для новых символов
            if self.is_running():
                await self._update_market_data()

            return True

        except Exception as e:
            logger.error(
                f"Ошибка при обновлении символов для бота {self.name}: {str(e)}"
            )
            return False

    @async_handle_error
    async def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Обновляет конфигурацию бота.

        Args:
            config: Словарь с новыми параметрами конфигурации

        Returns:
            True в случае успеха, иначе False
        """
        try:
            logger.info("Обновление конфигурации для бота {self.name}" %)

            # Обновляем базовые параметры
            if "name" in config:
                self.name = config["name"]

            if "exchange_id" in config:
                self.exchange_id = config["exchange_id"]

            if "symbols" in config:
                self.symbols = config["symbols"]

            if "update_interval" in config:
                self.update_interval = float(config["update_interval"])

            # Обновляем дополнительные параметры (в подклассах)
            self._update_config(config)

            return True

        except Exception as e:
            logger.error(
                f"Ошибка при обновлении конфигурации для бота {self.name}: {str(e)}"
            )
            return False

    async def _initialize(self) -> None:
        """
        Инициализирует бота перед запуском.
        Должен быть переопределен в подклассах.
        """
        # Загружаем рыночные данные для всех символов
        await self._update_market_data()

    async def _cleanup(self) -> None:
        """
        Выполняет очистку ресурсов при остановке бота.
        Должен быть переопределен в подклассах.
        """

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет дополнительные параметры конфигурации.
        Должен быть переопределен в подклассах.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """

    @async_handle_error
    async def _update_market_data(self) -> None:
        """
        Обновляет рыночные данные для всех символов.
        """
        for symbol in self.symbols:
            try:
                await self.market_data.get_ticker(self.exchange_id, symbol)
                await self.market_data.get_ohlcv(
                    self.exchange_id, symbol, "1h", limit=100
                )
            except Exception as e:
                logger.warning("Ошибка при обновлении данных для {symbol}: {str(e)}" %)

    async def _run(self) -> None:
        """
        Основной цикл работы бота.
        """
        try:
            logger.info("Основной цикл бота {self.name} запущен" %)

            while True:
                if self.status == BotStatus.RUNNING:
                    try:
                        # Обновляем данные
                        await self._update_market_data()

                        # Выполняем основной шаг бота
                        await self._execute_bot_step()

                        # Обновляем время последнего обновления
                        self.last_update_time = time.time()
                    except Exception as e:
                        logger.error(
                            f"Ошибка в основном цикле бота {self.name}: {str(e)}"
                        )

                # Ждем до следующего обновления
                await asyncio.sleep(self.update_interval)

        except asyncio.CancelledError:
            logger.info("Основной цикл бота {self.name} отменен" %)
            raise
        except Exception as e:
            logger.error("Критическая ошибка в боте {self.name}: {str(e)}" %)
            self.status = BotStatus.ERROR

    async def _execute_bot_step(self) -> None:
        """
        Выполняет один шаг работы бота.
        Должен быть переопределен в подклассах.
        """

    def _update_stats(self, trade_result: float, is_win: bool) -> None:
        """
        Обновляет статистику торговли.

        Args:
            trade_result: Результат сделки (прибыль/убыток)
            is_win: True, если сделка прибыльная, иначе False
        """
        self.stats["trades_count"] += 1

        if is_win:
            self.stats["winning_trades"] += 1
            self.stats["total_profit"] += trade_result
        else:
            self.stats["losing_trades"] += 1
            self.stats["total_loss"] += abs(trade_result)

        # Обновляем максимальную просадку
        current_drawdown = self.stats["total_loss"] / (
            self.stats["total_profit"] + 0.0001
        )
        self.stats["max_drawdown"] = max(self.stats["max_drawdown"], current_drawdown)

        # Обновляем винрейт
        if self.stats["trades_count"] > 0:
            self.stats["win_rate"] = (
                self.stats["winning_trades"] / self.stats["trades_count"]
            )
