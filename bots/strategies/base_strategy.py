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
            logger.warning(f"Стратегия {self.name} уже запущена или в процессе запуска")
            return False

        logger.info(f"Запуск стратегии {self.name} (id: {self.strategy_id})")
        self.status = StrategyStatus.STARTING

        try:
            # Инициализируем стратегию
            await self._initialize()

            # Запускаем основную задачу стратегии
            self.task = asyncio.create_task(self._run())
            self.start_time = time.time()
            self.status = StrategyStatus.RUNNING

            logger.info(f"Стратегия {self.name} успешно запущена")
            await send_trading_signal(f"Стратегия {self.name} запущена")

            return True

        except Exception as e:
            logger.error(f"Ошибка при запуске стратегии {self.name}: {str(e)}")
            self.status = StrategyStatus.ERROR
            return False

    async def stop(self) -> bool:
        """
        Останавливает стратегию.

        Returns:
            True в случае успеха, иначе False
        """
        if self.status == StrategyStatus.STOPPED:
            logger.warning(f"Стратегия {self.name} уже остановлена")
            return False

        logger.info(f"Остановка стратегии {self.name} (id: {self.strategy_id})")
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
            logger.error(f"Ошибка при очистке стратегии {self.name}: {str(e)}")

        self.status = StrategyStatus.STOPPED
        logger.info(f"Стратегия {self.name} успешно остановлена")
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

        logger.info(f"Приостановка стратегии {self.name} (id: {self.strategy_id})")
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
            logger.info(f"Обновление конфигурации для стратегии {self.name}")

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
                logger.warning(f"Некорректный формат сигнала: {signal}")
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
                    f"Символ {symbol} не отслеживается стратегией {
                        self.name}, сигнал проигнорирован"
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
                logger.warning(f"Неизвестное действие в сигнале: {action}")
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
                    logger.error(f"Ошибка при закрытии позиции {symbol}: {str(e)}")

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
                logger.error(f"Ошибка при отмене ордера {order_id}: {str(e)}")

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
                logger.warning(f"Ошибка при обновлении данных для {symbol}: {str(e)}")

    async def _run(self) -> None:
        """
        Основной цикл работы стратегии.
        """
        try:
            logger.info(f"Основной цикл стратегии {self.name} запущен")

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
            logger.info(f"Основной цикл стратегии {self.name} отменен")
            raise
        except Exception as e:
            logger.error(f"Критическая ошибка в стратегии {self.name}: {str(e)}")
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
                logger.error(f"Ошибка при проверке позиции {symbol}: {str(e)}")

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
                                f"Открыта позиция {side} по {symbol} по цене {
                                    self.open_positions[symbol]['entry_price']}"
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
                                f"Закрыта позиция {side} по {symbol} по цене {exit_price}, PnL: {
                                    pnl_pct:.2f}%"
                            )

                # Если ордер отменен, удаляем его из списка активных
                elif result.status in ["canceled", "expired", "rejected"]:
                    del self.active_orders[order_id]
                    logger.info(f"Ордер {order_id} для {order['symbol']} отменен")

            except Exception as e:
                logger.error(f"Ошибка при проверке ордера {order_id}: {str(e)}")

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
                logger.error(f"Ошибка при обработке сигнала для {symbol}: {str(e)}")

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
                logger.warning(f"Уже есть открытая позиция по {symbol}")
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
                    logger.error(f"Не удалось получить тикер для {symbol}")
                    return False

                price = ticker.get("last", 0)
                if price <= 0:
                    logger.error(f"Некорректная цена для {symbol}: {price}")
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
                logger.error(f"Некорректное количество для {symbol}: {quantity}")
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

                logger.info(f"Открыта позиция {side} по {symbol} по цене {entry_price}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при открытии позиции {side} по {symbol}: {str(e)}")
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
                logger.warning(f"Нет открытой позиции по {symbol}")
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
            logger.error(f"Ошибка при закрытии позиции по {symbol}: {str(e)}")
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

    def _calculate_profit_and_loss(
        self, entry_price, current_price, position_type, position_size
    ):
        """Рассчитывает прибыль/убыток для позиции"""
        # Убираем лишние 'elif' после return
        if position_type == "long":
            return (current_price - entry_price) * position_size
        if position_type == "short":
            return (entry_price - current_price) * position_size
        return 0


"""
Базовая стратегия для торговых ботов, использующая традиционные индикаторы
вместо машинного обучения для экономии ресурсов.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

from project.utils.logging_utils import setup_logger
from project.technical_analysis.indicators import (
    calculate_ema, calculate_sma, calculate_rsi, 
    calculate_macd, calculate_bollinger_bands
)

logger = setup_logger("base_strategy")


class BaseStrategy:
    """Базовый класс для торговых стратегий"""
    
    def __init__(
        self,
        symbol: str = "",
        timeframe: str = "1h",
        parameters: Dict = None,
        **kwargs
    ):
        """
        Инициализирует базовую стратегию
        
        Args:
            symbol: Торговый символ
            timeframe: Таймфрейм
            parameters: Параметры стратегии
            **kwargs: Дополнительные параметры
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Устанавливаем параметры по умолчанию и из аргументов
        self.parameters = {
            # Общие параметры
            'risk_per_trade': 0.01,  # 1% риска на сделку
            'max_positions': 1,      # Максимум 1 позиция
            
            # Параметры индикаторов
            'fast_ema': 12,          # Период быстрой EMA для MACD
            'slow_ema': 26,          # Период медленной EMA для MACD
            'signal_ema': 9,         # Период сигнальной линии MACD
            'rsi_period': 14,        # Период RSI
            'rsi_overbought': 70,    # Уровень перекупленности RSI
            'rsi_oversold': 30,      # Уровень перепроданности RSI
            'bb_period': 20,         # Период для полос Боллинджера
            'bb_std': 2.0            # Стандартное отклонение для полос Боллинджера
        }
        
        # Обновляем параметры из аргументов
        if parameters:
            self.parameters.update(parameters)
            
        # Данные
        self.data: Optional[pd.DataFrame] = None
        
        # Состояние
        self.position = {
            'is_open': False,
            'direction': None,     # 'long' или 'short'
            'size': 0.0,           # размер позиции
            'entry_price': 0.0,    # цена входа
            'entry_time': None,    # время входа
            'stop_loss': None,     # уровень стоп-лосса
            'take_profit': None    # уровень тейк-профита
        }
        
        # Результаты
        self.results = {
            'trades': [],
            'equity_curve': [],
            'stats': {}
        }
        
        logger.info(f"Base strategy initialized for {symbol} on {timeframe}")
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготавливает данные для стратегии, рассчитывая необходимые индикаторы
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            pd.DataFrame: Обработанные данные с индикаторами
        """
        # Копируем данные
        df = data.copy()
        
        # Проверяем необходимые столбцы
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Рассчитываем индикаторы
        
        # Простые скользящие средние
        df['sma_50'] = calculate_sma(df['close'], 50)
        df['sma_200'] = calculate_sma(df['close'], 200)
        
        # Экспоненциальные скользящие средние
        df['ema_20'] = calculate_ema(df['close'], 20)
        
        # MACD
        macd_result = calculate_macd(
            df['close'], 
            self.parameters['fast_ema'], 
            self.parameters['slow_ema'],
            self.parameters['signal_ema']
        )
        df['macd'] = macd_result['macd']
        df['macd_signal'] = macd_result['signal']
        df['macd_histogram'] = macd_result['histogram']
        
        # RSI
        df['rsi'] = calculate_rsi(df['close'], self.parameters['rsi_period'])
        
        # Bollinger Bands
        bb_result = calculate_bollinger_bands(
            df['close'], 
            self.parameters['bb_period'], 
            self.parameters['bb_std']
        )
        df['bb_upper'] = bb_result['upper']
        df['bb_middle'] = bb_result['middle']
        df['bb_lower'] = bb_result['lower']
        
        # Удаляем NaN значения
        df.dropna(inplace=True)
        
        return df
    
    def update_data(self, data: pd.DataFrame) -> None:
        """
        Обновляет данные стратегии
        
        Args:
            data: DataFrame с новыми данными OHLCV
        """
        # Подготавливаем данные и сохраняем
        self.data = self.prepare_data(data)
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Генерирует торговые сигналы на основе данных
        
        Returns:
            pd.DataFrame: DataFrame с сигналами
        """
        if self.data is None:
            logger.warning("No data available to generate signals")
            return pd.DataFrame()
        
        # Копируем данные
        df = self.data.copy()
        
        # Инициализируем столбцы для сигналов
        df['signal'] = 0    # 1 для сигнала покупки, -1 для сигнала продажи
        df['entry'] = False # True для точек входа в позицию
        df['exit'] = False  # True для точек выхода из позиции
        
        # Здесь реализуется логика генерации сигналов на основе индикаторов
        # Эта базовая реализация должна быть переопределена в подклассах
        
        # Пример простой стратегии пересечения MACD:
        # Сигнал покупки: MACD пересекает сигнальную линию снизу вверх
        # Сигнал продажи: MACD пересекает сигнальную линию сверху вниз
        for i in range(1, len(df)):
            # Проверяем пересечение MACD
            if (df['macd'].iloc[i-1] < df['macd_signal'].iloc[i-1] and 
                df['macd'].iloc[i] > df['macd_signal'].iloc[i]):
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'entry'] = True
            elif (df['macd'].iloc[i-1] > df['macd_signal'].iloc[i-1] and 
                  df['macd'].iloc[i] < df['macd_signal'].iloc[i]):
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'exit'] = True
        
        return df
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Выполняет бэктестирование стратегии на исторических данных
        
        Args:
            data: DataFrame с историческими данными OHLCV
            
        Returns:
            Dict: Результаты бэктестирования
        """
        # Обновляем данные
        self.update_data(data)
        
        # Генерируем сигналы
        signals = self.generate_signals()
        
        # Начальные значения
        initial_balance = 10000.0  # Начальный капитал
        balance = initial_balance
        equity = initial_balance
        max_equity = initial_balance
        drawdown = 0.0
        max_drawdown = 0.0
        trades = []
        equity_curve = []
        
        # Состояние позиции
        position = self.position.copy()
        
        # Проходим по сигналам
        for i, row in signals.iterrows():
            # Расчет equity для текущего бара
            if position['is_open']:
                # Если позиция открыта, рассчитываем нереализованную P&L
                price = row['close']
                if position['direction'] == 'long':
                    unrealized_pnl = position['size'] * (price - position['entry_price'])
                else:  # short
                    unrealized_pnl = position['size'] * (position['entry_price'] - price)
                
                equity = balance + unrealized_pnl
            else:
                equity = balance
            
            # Обновляем максимум equity и drawdown
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
            
            # Записываем в equity curve
            equity_curve.append({
                'timestamp': i,
                'balance': balance,
                'equity': equity,
                'drawdown': drawdown
            })
            
            # Проверяем сигналы на вход/выход
            if not position['is_open']:
                # Если нет открытой позиции, проверяем сигнал на вход
                if row['entry'] and row['signal'] != 0:
                    # Открываем позицию
                    direction = 'long' if row['signal'] > 0 else 'short'
                    position_size = (balance * self.parameters['risk_per_trade'])
                    entry_price = row['close']
                    
                    position = {
                        'is_open': True,
                        'direction': direction,
                        'size': position_size,
                        'entry_price': entry_price,
                        'entry_time': i,
                        'stop_loss': None,
                        'take_profit': None
                    }
                    
                    logger.debug(f"Opened {direction} position at {entry_price}")
            else:
                # Если есть открытая позиция, проверяем сигнал на выход
                if (row['exit'] or 
                   (position['direction'] == 'long' and row['signal'] < 0) or
                   (position['direction'] == 'short' and row['signal'] > 0)):
                    # Закрываем позицию
                    exit_price = row['close']
                    
                    if position['direction'] == 'long':
                        pnl = position['size'] * (exit_price - position['entry_price']) / position['entry_price']
                    else:  # short
                        pnl = position['size'] * (position['entry_price'] - exit_price) / position['entry_price']
                    
                    # Обновляем баланс
                    balance += pnl
                    equity = balance
                    
                    # Записываем сделку
                    trade = {
                        'direction': position['direction'],
                        'entry_time': position['entry_time'],
                        'entry_price': position['entry_price'],
                        'exit_time': i,
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_percent': (pnl / position['size']) * 100
                    }
                    trades.append(trade)
                    
                    # Сбрасываем позицию
                    position = {
                        'is_open': False,
                        'direction': None,
                        'size': 0.0,
                        'entry_price': 0.0,
                        'entry_time': None,
                        'stop_loss': None,
                        'take_profit': None
                    }
                    
                    logger.debug(f"Closed position at {exit_price}, PnL: {pnl:.2f}")
        
        # Рассчитываем статистику
        total_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
        losing_trades = sum(1 for trade in trades if trade['pnl'] < 0)
        
        # Статистика доходности
        total_pnl = balance - initial_balance
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Средний выигрыш/проигрыш
        avg_win = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(trade['pnl'] for trade in trades if trade['pnl'] < 0) / losing_trades if losing_trades > 0 else 0
        
        # Profit Factor
        gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Ожидаемая прибыль
        expectancy = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) if total_trades > 0 else 0
        
        # Коэффициент Шарпа (при годовой безрисковой ставке 0%)
        if len(equity_curve) > 1:
            returns = [e['equity'] / equity_curve[i-1]['equity'] - 1 for i, e in enumerate(equity_curve) if i > 0]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Собираем статистику в словарь
        stats = {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'net_profit': total_pnl,
            'net_profit_percent': (total_pnl / initial_balance) * 100 if initial_balance > 0 else 0,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio
        }
        
        # Сохраняем результаты
        self.results = {
            'trades': trades,
            'equity_curve': equity_curve,
            'stats': stats
        }
        
        logger.info(f"Backtest completed for {self.symbol} on {self.timeframe}")
        logger.info(f"Total trades: {total_trades}, Win rate: {win_rate:.2%}, Profit factor: {profit_factor:.2f}")
        
        return self.results
    
    def get_results(self) -> Dict:
        """
        Возвращает результаты бэктестирования
        
        Returns:
            Dict: Результаты бэктестирования
        """
        return self.results
    
    def plot_results(self, filename: str = None):
        """
        Строит график результатов бэктестирования
        
        Args:
            filename: Если указан, сохраняет график в файл
        """
        # Здесь может быть реализовано построение графиков
        # Для экономии ресурсов, эта функция может быть опциональной
        pass
    
    def optimize(self, data: pd.DataFrame, param_ranges: Dict, metric: str = 'sharpe_ratio') -> Dict:
        """
        Оптимизирует параметры стратегии
        
        Args:
            data: DataFrame с историческими данными OHLCV
            param_ranges: Словарь с диапазонами параметров для оптимизации
            metric: Метрика для оптимизации
            
        Returns:
            Dict: Результаты оптимизации
        """
        # Здесь может быть реализована оптимизация стратегии
        # Для экономии ресурсов, можно использовать простой Grid Search
        pass
    
    def save(self, filename: str) -> bool:
        """
        Сохраняет стратегию в файл
        
        Args:
            filename: Имя файла
            
        Returns:
            bool: True, если сохранение успешно
        """
        try:
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump({
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'parameters': self.parameters,
                    'results': self.results
                }, f)
            logger.info(f"Strategy saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving strategy to {filename}: {str(e)}")
            return False
    
    @classmethod
    def load(cls, filename: str):
        """
        Загружает стратегию из файла
        
        Args:
            filename: Имя файла
            
        Returns:
            BaseStrategy: Экземпляр стратегии
        """
        try:
            import pickle
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            strategy = cls(
                symbol=data['symbol'],
                timeframe=data['timeframe'],
                parameters=data['parameters']
            )
            strategy.results = data['results']
            
            logger.info(f"Strategy loaded from {filename}")
            return strategy
        except Exception as e:
            logger.error(f"Error loading strategy from {filename}: {str(e)}")
            return None


# Реестр стратегий для динамического создания экземпляров
class StrategyRegistry:
    """Реестр стратегий для динамического создания экземпляров"""
    
    _strategies = {}
    
    @classmethod
    def register(cls, strategy_id: str, strategy_class):
        """
        Регистрирует стратегию в реестре
        
        Args:
            strategy_id: ID стратегии
            strategy_class: Класс стратегии
        """
        cls._strategies[strategy_id] = strategy_class
        
    @classmethod
    def get_strategy_class(cls, strategy_id: str):
        """
        Возвращает класс стратегии по ID
        
        Args:
            strategy_id: ID стратегии
            
        Returns:
            class: Класс стратегии
        """
        if strategy_id not in cls._strategies:
            raise ValueError(f"Strategy {strategy_id} not found in registry")
        
        return cls._strategies[strategy_id]
    
    @classmethod
    def create_strategy(cls, strategy_id: str, **kwargs):
        """
        Создает экземпляр стратегии по ID
        
        Args:
            strategy_id: ID стратегии
            **kwargs: Параметры для конструктора стратегии
            
        Returns:
            BaseStrategy: Экземпляр стратегии
        """
        strategy_class = cls.get_strategy_class(strategy_id)
        return strategy_class(**kwargs)
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Возвращает список доступных стратегий
        
        Returns:
            List[str]: Список ID стратегий
        """
        return list(cls._strategies.keys())


# Регистрируем базовую стратегию
StrategyRegistry.register("base", BaseStrategy)
