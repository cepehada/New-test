"""
Базовый класс для торговых стратегий.
Предоставляет общую функциональность для всех стратегий.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional

from project.config import get_config
from project.data.market_data import MarketData
from project.risk_management.position_sizer import PositionSizer
from project.trade_executor.order_executor import OrderExecutor
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)


class StrategyStatus(Enum):
    """
    Статусы работы стратегии.
    """

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class BaseStrategy(ABC):
    """
    Базовый класс для всех торговых стратегий.
    """

    def __init__(self, config):
        """
        Инициализирует базовую торговую стратегию.

        Args:
            config: Конфигурация стратегии, содержащая:
                - exchange: Экземпляр биржи
                - symbol: Торговый символ
                - timeframe: Таймфрейм для анализа
                - indicators: Список индикаторов
                - stop_loss: Процент стоп-лосса
                - take_profit: Процент тейк-профита
                - position_size: Размер позиции в процентах от капитала
                - max_positions: Максимальное количество позиций
        """
        self.config = get_config()
        self.name = config.name
        self.strategy_id = str(uuid.uuid4())
        self.exchange_id = config.exchange_id
        self.symbols = config.symbols or []
        self.timeframes = config.timeframes or ["1h"]
        self.strategy_config = config.strategy_config or {}
        self.status = StrategyStatus.STOPPED
        self.task = None
        self.start_time = 0
        self.last_update_time = 0
        self.market_data = MarketData.get_instance()
        self.order_executor = OrderExecutor.get_instance()
        self.position_sizer = PositionSizer()
        self.update_interval = 60.0  # секунды

        # Настройки торговли
        self.max_open_positions = self.strategy_config.get("max_open_positions", 5)
        self.position_size_pct = self.strategy_config.get(
            "position_size_pct", 0.02
        )  # 2% от капитала
        self.use_stop_loss = self.strategy_config.get("use_stop_loss", True)
        self.stop_loss_pct = self.strategy_config.get(
            "stop_loss_pct", 0.02
        )  # 2% от цены входа
        self.use_take_profit = self.strategy_config.get("use_take_profit", True)
        self.take_profit_pct = self.strategy_config.get(
            "take_profit_pct", 0.04
        )  # 4% от цены входа
        self.risk_reward_ratio = self.strategy_config.get("risk_reward_ratio", 2.0)

        # Состояние стратегии
        self.open_positions: Dict[str, Dict[str, Any]] = (
            {}
        )  # ключ: "{symbol}" -> данные позиции
        self.active_orders: Dict[str, Dict[str, Any]] = (
            {}
        )  # ключ: "{order_id}" -> данные ордера
        self.signals: Dict[str, Dict[str, Any]] = (
            {}
        )  # ключ: "{symbol}" -> данные сигнала
        self.state: Dict[str, Any] = {
            "last_signals": {},
            "trade_history": [],
            "performance": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0.0,
                "total_loss": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
            },
        }

        logger.debug(
            f"Создана стратегия {self.name} (id: {self.strategy_id}) для {exchange_id}"
        )

    @async_handle_error
    async def start(self) -> bool:
        """
        Запускает стратегию.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status != StrategyStatus.STOPPED:
            logger.warning("Стратегия {self.name} уже запущена или в процессе запуска" %)
            return False

        logger.info("Запуск стратегии {self.name} (id: {self.strategy_id})" %)
        self.status = StrategyStatus.STARTING

        try:
            # Инициализируем стратегию
            await self._initialize()

            # Запускаем основную задачу стратегии
            self.task = asyncio.create_task(self._run())
            self.start_time = time.time()
            self.status = StrategyStatus.RUNNING

            logger.info("Стратегия {self.name} успешно запущена" %)
            await send_trading_signal(f"Стратегия {self.name} запущена")

            return True

        except Exception as e:
            logger.error("Ошибка при запуске стратегии {self.name}: {str(e)}" %)
            self.status = StrategyStatus.ERROR
            return False

    async def stop(self) -> bool:
        """
        Останавливает стратегию.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status == StrategyStatus.STOPPED:
            logger.warning("Стратегия {self.name} уже остановлена" %)
            return False

        logger.info("Остановка стратегии {self.name} (id: {self.strategy_id})" %)
        self.status = StrategyStatus.STOPPING

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
            logger.error("Ошибка при очистке стратегии {self.name}: {str(e)}" %)

        self.status = StrategyStatus.STOPPED
        logger.info("Стратегия {self.name} успешно остановлена" %)
        await send_trading_signal(f"Стратегия {self.name} остановлена")

        return True

    async def pause(self) -> bool:
        """
        Приостанавливает работу стратегии.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status != StrategyStatus.RUNNING:
            logger.warning(
                f"Стратегия {self.name} не запущена, невозможно приостановить"
            )
            return False

        logger.info("Приостановка стратегии {self.name} (id: {self.strategy_id})" %)
        self.status = StrategyStatus.PAUSED

        await send_trading_signal(f"Стратегия {self.name} приостановлена")

        return True

    async def resume(self) -> bool:
        """
        Возобновляет работу стратегии после приостановки.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status != StrategyStatus.PAUSED:
            logger.warning(
                f"Стратегия {self.name} не приостановлена, невозможно возобновить"
            )
            return False

        logger.info(
            f"Возобновление работы стратегии {self.name} (id: {self.strategy_id})"
        )
        self.status = StrategyStatus.RUNNING

        await send_trading_signal(f"Стратегия {self.name} возобновила работу")

        return True

    def get_status(self) -> str:
        """
        Получает текущий статус стратегии.

        Returns:
            Строковое представление статуса
        """
        return self.status.value

    def is_running(self) -> bool:
        """
        Проверяет, запущена ли стратегия.

        Returns:
            True, если стратегия запущена, иначе False
        """
        return self.status == StrategyStatus.RUNNING

    def is_stopped(self) -> bool:
        """
        Проверяет, остановлена ли стратегия.

        Returns:
            True, если стратегия остановлена, иначе False
        """
        return self.status == StrategyStatus.STOPPED

    def get_state(self) -> Dict[str, Any]:
        """
        Получает состояние стратегии.

        Returns:
            Словарь с состоянием стратегии
        """
        state = dict(self.state)

        # Добавляем текущие открытые позиции
        state["open_positions"] = list(self.open_positions.values())

        # Добавляем текущие активные ордера
        state["active_orders"] = list(self.active_orders.values())

        # Добавляем текущие сигналы
        state["signals"] = list(self.signals.values())

        # Добавляем общую информацию о стратегии
        state["name"] = self.name
        state["strategy_id"] = self.strategy_id
        state["status"] = self.status.value
        state["exchange"] = self.exchange_id
        state["symbols"] = self.symbols
        state["timeframes"] = self.timeframes
        state["config"] = self.strategy_config
        state["uptime"] = time.time() - self.start_time if self.start_time > 0 else 0
        state["last_update"] = self.last_update_time

        return state

    @async_handle_error
    async def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Обновляет конфигурацию стратегии.

        Args:
            config: Словарь с новыми параметрами конфигурации

        Returns:
            True в случае успеха, иначе False
        """
        try:
            logger.info("Обновление конфигурации для стратегии {self.name}" %)

            # Обновляем базовые параметры
            if "name" in config:
                self.name = config["name"]

            if "exchange_id" in config:
                self.exchange_id = config["exchange_id"]

            if "symbols" in config:
                self.symbols = config["symbols"]

            if "timeframes" in config:
                self.timeframes = config["timeframes"]

            if "update_interval" in config:
                self.update_interval = float(config["update_interval"])

            # Обновляем настройки торговли
            if "max_open_positions" in config:
                self.max_open_positions = int(config["max_open_positions"])

            if "position_size_pct" in config:
                self.position_size_pct = float(config["position_size_pct"])

            if "use_stop_loss" in config:
                self.use_stop_loss = bool(config["use_stop_loss"])

            if "stop_loss_pct" in config:
                self.stop_loss_pct = float(config["stop_loss_pct"])

            if "use_take_profit" in config:
                self.use_take_profit = bool(config["use_take_profit"])

            if "take_profit_pct" in config:
                self.take_profit_pct = float(config["take_profit_pct"])

            if "risk_reward_ratio" in config:
                self.risk_reward_ratio = float(config["risk_reward_ratio"])

            # Объединяем с текущей конфигурацией
            self.strategy_config.update(config)

            # Обновляем специфические параметры (в подклассах)
            self._update_config(config)

            return True

        except Exception as e:
            logger.error(
                f"Ошибка при обновлении конфигурации для стратегии {self.name}: {str(e)}"
            )
            return False

    @async_handle_error
    async def process_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Обрабатывает внешний сигнал.

        Args:
            signal: Словарь с данными сигнала
                - symbol: символ для торговли
                - action: действие (buy, sell, exit)
                - price: цена (опционально)
                - source: источник сигнала
                - timestamp: время создания сигнала

        Returns:
            True, если сигнал обработан, иначе False
        """
        try:
            if "symbol" not in signal or "action" not in signal:
                logger.warning("Некорректный формат сигнала: {signal}" %)
                return False

            symbol = signal["symbol"]
            action = signal["action"].lower()
            price = signal.get("price")
            source = signal.get("source", "external")

            logger.info(
                f"Получен сигнал для стратегии {self.name}: {action} {symbol} от {source}"
            )

            # Если стратегия не запущена, игнорируем сигнал
            if not self.is_running():
                logger.warning(
                    f"Стратегия {self.name} не запущена, сигнал проигнорирован"
                )
                return False

            # Если символ не в списке отслеживаемых, игнорируем сигнал
            if symbol not in self.symbols:
                logger.warning(
                    f"Символ {symbol} не отслеживается стратегией {self.name}, сигнал проигнорирован"
                )
                return False

            # Обрабатываем сигнал в зависимости от действия
            if action in ["buy", "long"]:
                # Открываем длинную позицию
                return await self._open_position(symbol, "long", price)
            elif action in ["sell", "short"]:
                # Открываем короткую позицию
                return await self._open_position(symbol, "short", price)
            elif action in ["exit", "close"]:
                # Закрываем позицию
                return await self._close_position(symbol)
            else:
                logger.warning("Неизвестное действие в сигнале: {action}" %)
                return False

        except Exception as e:
            logger.error(
                f"Ошибка при обработке сигнала для стратегии {self.name}: {str(e)}"
            )
            return False

    async def _initialize(self) -> None:
        """
        Инициализирует стратегию перед запуском.
        """
        # Загружаем рыночные данные для всех символов и таймфреймов
        await self._update_market_data()

        # Выполняем дополнительную инициализацию в подклассах
        await self._strategy_initialize()

    async def _cleanup(self) -> None:
        """
        Выполняет очистку ресурсов при остановке стратегии.
        """
        # Закрываем все открытые позиции, если настроено
        if self.strategy_config.get("close_positions_on_stop", True):
            for symbol in list(self.open_positions.keys()):
                try:
                    await self._close_position(symbol)
                except Exception as e:
                    logger.error("Ошибка при закрытии позиции {symbol}: {str(e)}" %)

        # Отменяем все активные ордера
        for order_id in list(self.active_orders.keys()):
            try:
                order = self.active_orders[order_id]
                await self.order_executor.cancel_order(
                    order_id=order_id,
                    symbol=order["symbol"],
                    exchange_id=self.exchange_id,
                )
            except Exception as e:
                logger.error("Ошибка при отмене ордера {order_id}: {str(e)}" %)

        # Выполняем дополнительную очистку в подклассах
        await self._strategy_cleanup()

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.
        Должен быть переопределен в подклассах.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """

    @abstractmethod
    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        Должен быть переопределен в подклассах.
        """

    @abstractmethod
    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        Должен быть переопределен в подклассах.
        """

    @abstractmethod
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы на основе текущих рыночных данных.
        Должен быть переопределен в подклассах.

        Returns:
            Словарь с сигналами для каждого символа
        """

    @async_handle_error
    async def _update_market_data(self) -> None:
        """
        Обновляет рыночные данные для всех символов и таймфреймов.
        """
        for symbol in self.symbols:
            try:
                # Получаем текущий тикер
                await self.market_data.get_ticker(self.exchange_id, symbol)

                # Получаем данные OHLCV для всех таймфреймов
                for timeframe in self.timeframes:
                    await self.market_data.get_ohlcv(
                        self.exchange_id, symbol, timeframe, limit=100
                    )
            except Exception as e:
                logger.warning("Ошибка при обновлении данных для {symbol}: {str(e)}" %)

    async def _run(self) -> None:
        """
        Основной цикл работы стратегии.
        """
        try:
            logger.info("Основной цикл стратегии {self.name} запущен" %)

            while True:
                if self.status == StrategyStatus.RUNNING:
                    try:
                        # Обновляем данные
                        await self._update_market_data()

                        # Проверяем состояние открытых позиций
                        await self._check_positions()

                        # Проверяем состояние активных ордеров
                        await self._check_orders()

                        # Генерируем торговые сигналы
                        new_signals = await self._generate_trading_signals()

                        # Обрабатываем новые сигналы
                        await self._process_signals(new_signals)

                        # Обновляем время последнего обновления
                        self.last_update_time = time.time()
                    except Exception as e:
                        logger.error(
                            f"Ошибка в основном цикле стратегии {self.name}: {str(e)}"
                        )

                # Ждем до следующего обновления
                await asyncio.sleep(self.update_interval)

        except asyncio.CancelledError:
            logger.info("Основной цикл стратегии {self.name} отменен" %)
            raise
        except Exception as e:
            logger.error("Критическая ошибка в стратегии {self.name}: {str(e)}" %)
            self.status = StrategyStatus.ERROR

    @async_handle_error
    async def _check_positions(self) -> None:
        """
        Проверяет состояние открытых позиций.
        """
        for symbol, position in list(self.open_positions.items()):
            try:
                # Получаем текущие рыночные данные
                ticker = await self.market_data.get_ticker(self.exchange_id, symbol)
                if not ticker:
                    continue

                current_price = ticker.get("last", 0)
                if current_price <= 0:
                    continue

                # Проверяем, достигнуты ли уровни стоп-лосса или тейк-профита
                entry_price = position["entry_price"]
                side = position["side"]
                stop_loss = position.get("stop_loss")
                take_profit = position.get("take_profit")

                # Рассчитываем текущую прибыль/убыток
                if side == "long":
                    pnl_pct = (current_price / entry_price - 1) * 100
                else:  # short
                    pnl_pct = (entry_price / current_price - 1) * 100

                # Обновляем информацию о позиции
                position["current_price"] = current_price
                position["pnl_pct"] = pnl_pct

                # Проверяем стоп-лосс
                if stop_loss and (
                    (side == "long" and current_price <= stop_loss)
                    or (side == "short" and current_price >= stop_loss)
                ):
                    logger.info(
                        f"Стоп-лосс сработал для {symbol} {side} по цене {current_price}"
                    )
                    await self._close_position(symbol, "stop_loss")
                    continue

                # Проверяем тейк-профит
                if take_profit and (
                    (side == "long" and current_price >= take_profit)
                    or (side == "short" and current_price <= take_profit)
                ):
                    logger.info(
                        f"Тейк-профит сработал для {symbol} {side} по цене {current_price}"
                    )
                    await self._close_position(symbol, "take_profit")
                    continue

            except Exception as e:
                logger.error("Ошибка при проверке позиции {symbol}: {str(e)}" %)

    @async_handle_error
    async def _check_orders(self) -> None:
        """
        Проверяет состояние активных ордеров.
        """
        for order_id, order in list(self.active_orders.items()):
            try:
                # Получаем текущее состояние ордера
                result = await self.order_executor.check_order_status(
                    order_id=order_id,
                    symbol=order["symbol"],
                    exchange_id=self.exchange_id,
                )

                if not result.success:
                    continue

                # Обновляем информацию об ордере
                order["status"] = result.status
                order["filled_quantity"] = result.filled_quantity
                order["average_price"] = result.average_price

                # Если ордер исполнен, обрабатываем его
                if result.status == "closed":
                    # Удаляем ордер из списка активных
                    del self.active_orders[order_id]

                    # Если это ордер открытия позиции, обновляем позицию
                    if order["type"] == "open_position":
                        symbol = order["symbol"]
                        side = order["side"]

                        # Создаем или обновляем позицию
                        if symbol not in self.open_positions:
                            self.open_positions[symbol] = {
                                "symbol": symbol,
                                "side": side,
                                "entry_price": result.average_price or order["price"],
                                "quantity": result.filled_quantity,
                                "entry_time": time.time(),
                                "entry_id": order_id,
                                "strategy_id": self.strategy_id,
                            }

                            # Устанавливаем стоп-лосс и тейк-профит
                            if self.use_stop_loss:
                                stop_loss = self._calculate_stop_loss(
                                    symbol,
                                    side,
                                    self.open_positions[symbol]["entry_price"],
                                )
                                self.open_positions[symbol]["stop_loss"] = stop_loss

                            if self.use_take_profit:
                                take_profit = self._calculate_take_profit(
                                    symbol,
                                    side,
                                    self.open_positions[symbol]["entry_price"],
                                )
                                self.open_positions[symbol]["take_profit"] = take_profit

                            logger.info(
                                f"Открыта позиция {side} по {symbol} по цене {self.open_positions[symbol]['entry_price']}"
                            )

                    # Если это ордер закрытия позиции, фиксируем результат
                    elif order["type"] == "close_position":
                        symbol = order["symbol"]

                        if symbol in self.open_positions:
                            # Рассчитываем прибыль/убыток
                            position = self.open_positions[symbol]
                            entry_price = position["entry_price"]
                            exit_price = result.average_price or order["price"]
                            side = position["side"]
                            quantity = position["quantity"]

                            if side == "long":
                                pnl_pct = (exit_price / entry_price - 1) * 100
                                pnl = (exit_price - entry_price) * quantity
                            else:  # short
                                pnl_pct = (entry_price / exit_price - 1) * 100
                                pnl = (entry_price - exit_price) * quantity

                            # Создаем запись в истории торговли
                            trade_record = {
                                "symbol": symbol,
                                "side": side,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "quantity": quantity,
                                "entry_time": position["entry_time"],
                                "exit_time": time.time(),
                                "pnl": pnl,
                                "pnl_pct": pnl_pct,
                                "exit_reason": order.get("exit_reason", "manual"),
                                "strategy_id": self.strategy_id,
                            }

                            # Добавляем запись в историю
                            self.state["trade_history"].append(trade_record)

                            # Обновляем статистику
                            self._update_performance(trade_record)

                            # Удаляем позицию
                            del self.open_positions[symbol]

                            logger.info(
                                f"Закрыта позиция {side} по {symbol} по цене {exit_price}, PnL: {pnl_pct:.2f}%"
                            )

                # Если ордер отменен, удаляем его из списка активных
                elif result.status in ["canceled", "expired", "rejected"]:
                    del self.active_orders[order_id]
                    logger.info("Ордер {order_id} для {order['symbol']} отменен" %)

            except Exception as e:
                logger.error("Ошибка при проверке ордера {order_id}: {str(e)}" %)

    @async_handle_error
    async def _process_signals(self, signals: Dict[str, Dict[str, Any]]) -> None:
        """
        Обрабатывает торговые сигналы.

        Args:
            signals: Словарь с сигналами для каждого символа
        """
        # Обновляем сигналы
        self.signals = signals

        # Сохраняем последние сигналы
        self.state["last_signals"] = {
            symbol: signal.copy() for symbol, signal in signals.items()
        }

        # Обрабатываем сигналы
        for symbol, signal in signals.items():
            try:
                # Проверяем наличие действия в сигнале
                if "action" not in signal:
                    continue

                action = signal["action"]

                # Проверяем, есть ли уже открытая позиция по этому символу
                has_position = symbol in self.open_positions

                # Обрабатываем действие
                if action == "buy" and not has_position:
                    # Открываем длинную позицию
                    await self._open_position(symbol, "long")

                elif action == "sell" and not has_position:
                    # Открываем короткую позицию
                    await self._open_position(symbol, "short")

                elif action == "exit" and has_position:
                    # Закрываем позицию
                    await self._close_position(symbol, "signal")

                elif action == "hold":
                    # Ничего не делаем
                    pass

            except Exception as e:
                logger.error("Ошибка при обработке сигнала для {symbol}: {str(e)}" %)

    @async_handle_error
    async def _open_position(
        self, symbol: str, side: str, price: Optional[float] = None
    ) -> bool:
        """
        Открывает позицию по указанному символу.

        Args:
            symbol: Символ для торговли
            side: Сторона позиции ('long' или 'short')
            price: Цена входа (None для рыночного входа)

        Returns:
            True, если позиция открыта успешно, иначе False
        """
        try:
            # Проверяем, есть ли уже открытая позиция по этому символу
            if symbol in self.open_positions:
                logger.warning("Уже есть открытая позиция по {symbol}" %)
                return False

            # Проверяем лимит открытых позиций
            if len(self.open_positions) >= self.max_open_positions:
                logger.warning(
                    f"Достигнут лимит открытых позиций ({self.max_open_positions})"
                )
                return False

            # Получаем текущую цену, если не указана
            if price is None:
                ticker = await self.market_data.get_ticker(self.exchange_id, symbol)
                if not ticker:
                    logger.error("Не удалось получить тикер для {symbol}" %)
                    return False

                price = ticker.get("last", 0)
                if price <= 0:
                    logger.error("Некорректная цена для {symbol}: {price}" %)
                    return False

            # Рассчитываем размер позиции
            account_balance = self.strategy_config.get(
                "account_balance", 10000.0
            )  # По умолчанию 10000
            position_size = account_balance * self.position_size_pct

            # Рассчитываем количество в базовой валюте
            quantity = position_size / price

            # Нормализуем количество (округляем)
            quantity = self._normalize_quantity(symbol, quantity)

            if quantity <= 0:
                logger.error("Некорректное количество для {symbol}: {quantity}" %)
                return False

            # Определяем тип и сторону ордера
            order_type = "market" if price is None else "limit"
            order_side = "buy" if side == "long" else "sell"

            # Выполняем ордер
            order_result = await self.order_executor.execute_order(
                symbol=symbol,
                side=order_side,
                amount=quantity,
                order_type=order_type,
                price=price if order_type == "limit" else None,
                exchange_id=self.exchange_id,
            )

            if not order_result.success:
                logger.error(
                    f"Ошибка при выполнении ордера для {symbol}: {order_result.error}"
                )
                return False

            # Добавляем ордер в список активных
            self.active_orders[order_result.order_id] = {
                "id": order_result.order_id,
                "symbol": symbol,
                "side": order_side,
                "type": "open_position",
                "order_type": order_type,
                "price": price,
                "quantity": quantity,
                "status": order_result.status,
                "filled_quantity": order_result.filled_quantity,
                "average_price": order_result.average_price,
                "create_time": time.time(),
                "strategy_id": self.strategy_id,
            }

            logger.info(
                f"Создан ордер на открытие позиции {side} по {symbol}: {order_result.order_id}"
            )

            # Если ордер уже исполнен (market), создаем позицию
            if order_result.status == "closed":
                entry_price = order_result.average_price or price

                self.open_positions[symbol] = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "quantity": order_result.filled_quantity,
                    "entry_time": time.time(),
                    "entry_id": order_result.order_id,
                    "strategy_id": self.strategy_id,
                }

                # Устанавливаем стоп-лосс и тейк-профит
                if self.use_stop_loss:
                    stop_loss = self._calculate_stop_loss(symbol, side, entry_price)
                    self.open_positions[symbol]["stop_loss"] = stop_loss

                if self.use_take_profit:
                    take_profit = self._calculate_take_profit(symbol, side, entry_price)
                    self.open_positions[symbol]["take_profit"] = take_profit

                logger.info("Открыта позиция {side} по {symbol} по цене {entry_price}" %)

            return True

        except Exception as e:
            logger.error("Ошибка при открытии позиции {side} по {symbol}: {str(e)}" %)
            return False

    @async_handle_error
    async def _close_position(self, symbol: str, reason: str = "manual") -> bool:
        """
        Закрывает позицию по указанному символу.

        Args:
            symbol: Символ для торговли
            reason: Причина закрытия (manual, signal, stop_loss, take_profit)

        Returns:
            True, если позиция закрыта успешно, иначе False
        """
        try:
            # Проверяем, есть ли открытая позиция по этому символу
            if symbol not in self.open_positions:
                logger.warning("Нет открытой позиции по {symbol}" %)
                return False

            # Получаем информацию о позиции
            position = self.open_positions[symbol]
            side = position["side"]
            quantity = position["quantity"]

            # Определяем сторону ордера (противоположную стороне позиции)
            order_side = "sell" if side == "long" else "buy"

            # Выполняем ордер
            order_result = await self.order_executor.execute_order(
                symbol=symbol,
                side=order_side,
                amount=quantity,
                order_type="market",
                exchange_id=self.exchange_id,
            )

            if not order_result.success:
                logger.error(
                    f"Ошибка при закрытии позиции для {symbol}: {order_result.error}"
                )
                return False

            # Добавляем ордер в список активных
            self.active_orders[order_result.order_id] = {
                "id": order_result.order_id,
                "symbol": symbol,
                "side": order_side,
                "type": "close_position",
                "order_type": "market",
                "price": None,
                "quantity": quantity,
                "status": order_result.status,
                "filled_quantity": order_result.filled_quantity,
                "average_price": order_result.average_price,
                "create_time": time.time(),
                "exit_reason": reason,
                "strategy_id": self.strategy_id,
            }

            logger.info(
                f"Создан ордер на закрытие позиции {side} по {symbol}: {order_result.order_id}"
            )

            # Если ордер уже исполнен (market), фиксируем результат
            if order_result.status == "closed":
                # Рассчитываем прибыль/убыток
                entry_price = position["entry_price"]
                exit_price = order_result.average_price

                if side == "long":
                    pnl_pct = (exit_price / entry_price - 1) * 100
                    pnl = (exit_price - entry_price) * quantity
                else:  # short
                    pnl_pct = (entry_price / exit_price - 1) * 100
                    pnl = (entry_price - exit_price) * quantity

                # Создаем запись в истории торговли
                trade_record = {
                    "symbol": symbol,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": quantity,
                    "entry_time": position["entry_time"],
                    "exit_time": time.time(),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": reason,
                    "strategy_id": self.strategy_id,
                }

                # Добавляем запись в историю
                self.state["trade_history"].append(trade_record)

                # Обновляем статистику
                self._update_performance(trade_record)

                # Удаляем позицию
                del self.open_positions[symbol]

                logger.info(
                    f"Закрыта позиция {side} по {symbol} по цене {exit_price}, PnL: {pnl_pct:.2f}%"
                )

            return True

        except Exception as e:
            logger.error("Ошибка при закрытии позиции по {symbol}: {str(e)}" %)
            return False

    def _calculate_stop_loss(self, symbol: str, side: str, entry_price: float) -> float:
        """
        Рассчитывает цену стоп-лосса.

        Args:
            symbol: Символ для торговли
            side: Сторона позиции ('long' или 'short')
            entry_price: Цена входа

        Returns:
            Цена стоп-лосса
        """
        if side == "long":
            return entry_price * (1 - self.stop_loss_pct)
        else:  # short
            return entry_price * (1 + self.stop_loss_pct)

    def _calculate_take_profit(
        self, symbol: str, side: str, entry_price: float
    ) -> float:
        """
        Рассчитывает цену тейк-профита.

        Args:
            symbol: Символ для торговли
            side: Сторона позиции ('long' или 'short')
            entry_price: Цена входа

        Returns:
            Цена тейк-профита
        """
        if side == "long":
            return entry_price * (1 + self.take_profit_pct)
        else:  # short
            return entry_price * (1 - self.take_profit_pct)

    def _normalize_quantity(self, symbol: str, quantity: float) -> float:
        """
        Нормализует количество для указанного символа.

        Args:
            symbol: Символ для торговли
            quantity: Исходное количество

        Returns:
            Нормализованное количество
        """
        # Простое округление до 5 знаков после запятой
        # В реальной стратегии здесь должна быть логика с учетом правил биржи
        return round(quantity, 5)

    def _update_performance(self, trade_record: Dict[str, Any]) -> None:
        """
        Обновляет статистику производительности стратегии.

        Args:
            trade_record: Запись о торговой операции
        """
        perf = self.state["performance"]

        # Обновляем счетчики
        perf["total_trades"] += 1

        if trade_record["pnl"] > 0:
            perf["winning_trades"] += 1
            perf["total_profit"] += trade_record["pnl"]
        else:
            perf["losing_trades"] += 1
            perf["total_loss"] += abs(trade_record["pnl"])

        # Рассчитываем винрейт
        if perf["total_trades"] > 0:
            perf["win_rate"] = perf["winning_trades"] / perf["total_trades"]

        # Рассчитываем профит-фактор
        if perf["total_loss"] > 0:
            perf["profit_factor"] = perf["total_profit"] / perf["total_loss"]
        else:
            perf["profit_factor"] = float("inf") if perf["total_profit"] > 0 else 0.0

    def _calculate_profit_and_loss(self, entry_price, current_price, position_type, position_size):
        """Рассчитывает прибыль/убыток для позиции"""
        # Убираем лишние 'elif' после return
        if position_type == 'long':
            return (current_price - entry_price) * position_size
        if position_type == 'short':
            return (entry_price - current_price) * position_size
        return 0
