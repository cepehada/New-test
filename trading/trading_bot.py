import asyncio
import time
import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

from project.data.database import Database
from project.exchange.exchange_manager import get_exchange_manager
from project.trading.strategy_base import Position, Signal, Strategy
from project.utils.logging_utils import setup_logger
from project.utils.notify import NotificationLevel, NotificationManager

logger = setup_logger("trading_bot")


class BotState(Enum):
    """Состояния торгового бота"""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZING = "initializing"


class TradingBot:
    """Класс торгового бота для автоматической торговли"""

    def __init__(
        self,
        config: Dict,
        strategy: Strategy = None,
        database: Database = None,
        notification_manager: NotificationManager = None,
    ):
        """
        Инициализирует торгового бота

        Args:
            config: Конфигурация бота
            strategy: Стратегия торговли
            database: База данных
            notification_manager: Менеджер уведомлений
        """
        self.config = config
        self.strategy = strategy
        self.database = database

        # Генерируем уникальный ID бота, если не указан
        self.bot_id = config.get("bot_id", f"bot_{uuid.uuid4().hex[:8]}")

        # Получаем параметры из конфигурации
        self.symbol = config.get("symbol", "BTC/USDT")
        self.exchange_id = config.get("exchange_id", "binance")
        self.timeframe = config.get("timeframe", "15m")
        self.strategy_id = config.get("strategy_id")

        # Параметры торговли
        self.leverage = config.get("leverage", 1)
        self.margin_type = config.get("margin_type", "isolated")
        self.position_size = config.get(
            "position_size", 0.01
        )  # % от баланса или фиксированный объем
        self.is_position_size_percentage = config.get(
            "is_position_size_percentage", True
        )
        self.max_positions = config.get("max_positions", 1)
        self.allow_shorts = config.get("allow_shorts", False)
        self.take_profit = config.get("take_profit")
        self.stop_loss = config.get("stop_loss")
        self.trailing_stop = config.get("trailing_stop")

        # Параметры исполнения
        self.order_type = config.get("order_type", "market")
        self.post_only = config.get("post_only", False)
        self.reduce_only = config.get("reduce_only", False)
        self.time_in_force = config.get("time_in_force", "GTC")

        # Параметры режима работы
        self.paper_trading = config.get("paper_trading", True)
        self.backtest_mode = config.get("backtest_mode", False)
        self.live_mode = not self.paper_trading and not self.backtest_mode

        # Параметры управления ботом
        self.update_interval = config.get("update_interval", 60)  # секунды
        self.retry_interval = config.get("retry_interval", 5)  # секунды
        self.max_retries = config.get("max_retries", 3)
        self.warmup_bars = config.get("warmup_bars", 100)

        # Уведомления
        self.notification_manager = notification_manager
        self.notification_levels = config.get(
            "notification_levels",
            {
                "trade": NotificationLevel.INFO,
                "error": NotificationLevel.ERROR,
                "position": NotificationLevel.INFO,
                "performance": NotificationLevel.WARNING,
            },
        )

        # Состояние бота
        self.state = BotState.STOPPED
        self.last_update_time = None
        self.last_signal_time = None
        self.last_trade_time = None
        self.start_time = None
        self.error_count = 0
        self.consecutive_errors = 0

        # Данные
        self.data = None
        self.positions = {}
        self.open_orders = {}
        self.signals = []

        # Задача обновления
        self._update_task = None
        self._force_stop = False

        # Инициализируем соединение с биржей
        self._init_exchange()

        # Регистрируем обработчики событий
        self._register_event_handlers()

        # Инициализируем статистику
        self.stats = {
            "trades_count": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "break_even_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "run_time": 0,
            "errors": 0,
        }

        logger.info(
            f"TradingBot initialized: {self.bot_id} | {self.symbol} | {self.exchange_id}"
        )

    async def _init_exchange(self):
        """Инициализирует соединение с биржей"""
        try:
            # Получаем менеджер бирж
            self.exchange_manager = await get_exchange_manager()

            # Получаем инстанс биржи
            self.exchange = await self.exchange_manager.get_exchange(self.exchange_id)

            logger.info("Connected to exchange: {self.exchange_id}" %)
        except Exception as e:
            logger.error("Error connecting to exchange {self.exchange_id}: {str(e)}" %)
            self.state = BotState.ERROR
            raise

    def _register_event_handlers(self):
        """Регистрирует обработчики событий стратегии"""
        if self.strategy:
            # Регистрируем обработчик сигналов
            self.strategy.on_signal = self._on_signal

            # Регистрируем обработчики позиций
            self.strategy.on_position_opened = self._on_position_opened
            self.strategy.on_position_closed = self._on_position_closed

    async def start(self):
        """Запускает торгового бота"""
        if self.state == BotState.RUNNING:
            logger.warning("Bot {self.bot_id} is already running" %)
            return

        logger.info("Starting bot: {self.bot_id}" %)

        # Инициализируем бота
        self.state = BotState.INITIALIZING
        self.start_time = datetime.now()
        self._force_stop = False

        try:
            # Загружаем историю
            await self._load_historical_data()

            # Инициализируем стратегию
            if not self.strategy and self.strategy_id:
                await self._init_strategy()

            # Инициализируем базу данных, если не указана
            if not self.database:
                self.database = Database()
                await self.database.connect()

            # Загружаем сохраненное состояние, если есть
            await self._load_state()

            # Проверяем существующие позиции
            await self._check_existing_positions()

            # Проверяем существующие ордера
            await self._check_existing_orders()

            # Инициализируем уведомления
            if not self.notification_manager:
                self.notification_manager = NotificationManager()

            # Запускаем задачу обновления
            self._update_task = asyncio.create_task(self._update_loop())

            # Обновляем состояние
            self.state = BotState.RUNNING

            # Сохраняем состояние
            await self._save_state()

            # Отправляем уведомление
            await self._send_notification(
                f"Bot {self.bot_id} started",
                f"Trading bot for {self.symbol} on {self.exchange_id} has been started",
                NotificationLevel.INFO,
            )

            logger.info("Bot {self.bot_id} started successfully" %)
        except Exception as e:
            logger.error("Error starting bot {self.bot_id}: {str(e)}" %)
            logger.error(traceback.format_exc())
            self.state = BotState.ERROR
            self.error_count += 1

            # Отправляем уведомление об ошибке
            await self._send_notification(
                f"Error starting bot {self.bot_id}",
                f"Failed to start bot: {str(e)}",
                NotificationLevel.ERROR,
            )

            raise

    async def stop(self):
        """Останавливает торгового бота"""
        if self.state == BotState.STOPPED:
            logger.warning("Bot {self.bot_id} is already stopped" %)
            return

        logger.info("Stopping bot: {self.bot_id}" %)

        # Устанавливаем флаг для остановки
        self._force_stop = True

        # Отменяем задачу обновления
        if self._update_task:
            try:
                self._update_task.cancel()
                await asyncio.gather(self._update_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            self._update_task = None

        # Обновляем состояние
        self.state = BotState.STOPPED

        # Закрываем соединение с базой данных
        if self.database:
            await self.database.disconnect()

        # Сохраняем состояние
        await self._save_state()

        # Отправляем уведомление
        await self._send_notification(
            f"Bot {self.bot_id} stopped",
            f"Trading bot for {self.symbol} on {self.exchange_id} has been stopped",
            NotificationLevel.INFO,
        )

        logger.info("Bot {self.bot_id} stopped successfully" %)

    async def pause(self):
        """Приостанавливает торгового бота"""
        if self.state != BotState.RUNNING:
            logger.warning("Bot {self.bot_id} is not running, cannot pause" %)
            return

        logger.info("Pausing bot: {self.bot_id}" %)

        # Обновляем состояние
        self.state = BotState.PAUSED

        # Сохраняем состояние
        await self._save_state()

        # Отправляем уведомление
        await self._send_notification(
            f"Bot {self.bot_id} paused",
            f"Trading bot for {self.symbol} on {self.exchange_id} has been paused",
            NotificationLevel.INFO,
        )

        logger.info("Bot {self.bot_id} paused successfully" %)

    async def resume(self):
        """Возобновляет работу торгового бота"""
        if self.state != BotState.PAUSED:
            logger.warning("Bot {self.bot_id} is not paused, cannot resume" %)
            return

        logger.info("Resuming bot: {self.bot_id}" %)

        # Обновляем состояние
        self.state = BotState.RUNNING

        # Сохраняем состояние
        await self._save_state()

        # Отправляем уведомление
        await self._send_notification(
            f"Bot {self.bot_id} resumed",
            f"Trading bot for {self.symbol} on {self.exchange_id} has been resumed",
            NotificationLevel.INFO,
        )

        logger.info("Bot {self.bot_id} resumed successfully" %)

    async def _update_loop(self):
        """Основной цикл обновления бота"""
        while not self._force_stop:
            try:
                # Проверяем, активен ли бот
                if self.state != BotState.RUNNING:
                    # Если бот приостановлен или в состоянии ошибки, ждем
                    await asyncio.sleep(self.retry_interval)
                    continue

                # Загружаем новые данные
                await self._update_market_data()

                # Обновляем стратегию
                if self.strategy and self.data is not None:
                    # Передаем данные в стратегию для генерации сигналов
                    signals = await self.strategy.update(self.data)

                    # Обрабатываем сигналы
                    if signals:
                        for signal in signals:
                            await self._process_signal(signal)

                # Обновляем существующие позиции и ордера
                await self._update_positions()
                await self._update_orders()

                # Обновляем состояние
                self.last_update_time = datetime.now()
                self.consecutive_errors = 0

                # Сохраняем состояние
                await self._save_state()

                # Обновляем статистику
                await self._update_stats()

                # Ждем до следующего обновления
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                logger.info("Update loop cancelled for bot {self.bot_id}" %)
                break
            except Exception as e:
                logger.error("Error in update loop for bot {self.bot_id}: {str(e)}" %)
                logger.error(traceback.format_exc())

                # Увеличиваем счетчики ошибок
                self.error_count += 1
                self.consecutive_errors += 1

                # Если слишком много ошибок подряд, переводим бота в состояние ошибки
                if self.consecutive_errors >= self.max_retries:
                    self.state = BotState.ERROR

                    # Отправляем уведомление об ошибке
                    await self._send_notification(
                        f"Bot {self.bot_id} encountered an error",
                        f"Bot stopped due to consecutive errors: {str(e)}",
                        NotificationLevel.ERROR,
                    )

                    logger.error("Bot {self.bot_id} stopped due to consecutive errors" %)
                    break

                # Ждем перед повторной попыткой
                await asyncio.sleep(self.retry_interval)

    async def _load_historical_data(self):
        """Загружает исторические данные с биржи"""
        try:
            logger.info(
                f"Loading historical data for {self.symbol} on {self.exchange_id}"
            )

            # Определяем период загрузки
            limit = self.warmup_bars + 100  # Дополнительный запас

            # Загружаем OHLCV данные
            ohlcv_data = await self.exchange_manager.fetch_ohlcv(
                symbol=self.symbol,
                exchange_id=self.exchange_id,
                timeframe=self.timeframe,
                limit=limit,
            )

            if ohlcv_data is None or len(ohlcv_data) == 0:
                raise ValueError(
                    f"No historical data available for {self.symbol} on {self.exchange_id}"
                )

            # Преобразуем в pandas DataFrame
            import pandas as pd

            # Создаем DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Преобразуем timestamp в datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Устанавливаем timestamp в качестве индекса
            df.set_index("timestamp", inplace=True)

            # Добавляем атрибуты
            df.attrs["symbol"] = self.symbol
            df.attrs["exchange"] = self.exchange_id
            df.attrs["timeframe"] = self.timeframe

            # Сохраняем данные
            self.data = df

            logger.info("Loaded {len(df)} historical bars for {self.symbol}" %)
        except Exception as e:
            logger.error("Error loading historical data: {str(e)}" %)
            raise

    async def _update_market_data(self):
        """Обновляет рыночные данные"""
        try:
            if self.data is None:
                await self._load_historical_data()
                return

            # Получаем последний timestamp
            last_timestamp = self.data.index[-1]

            # Загружаем новые данные
            since = int(last_timestamp.timestamp() * 1000)

            # Загружаем OHLCV данные
            new_data = await self.exchange_manager.fetch_ohlcv(
                symbol=self.symbol,
                exchange_id=self.exchange_id,
                timeframe=self.timeframe,
                since=since,
            )

            if new_data is None or len(new_data) == 0:
                logger.debug("No new market data available")
                return

            # Преобразуем в pandas DataFrame
            import pandas as pd

            # Создаем DataFrame
            new_df = pd.DataFrame(
                new_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Преобразуем timestamp в datetime
            new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms")

            # Устанавливаем timestamp в качестве индекса
            new_df.set_index("timestamp", inplace=True)

            # Объединяем с существующими данными
            self.data = pd.concat([self.data, new_df])

            # Удаляем дубликаты
            self.data = self.data[~self.data.index.duplicated(keep="last")]

            # Сортируем по индексу
            self.data.sort_index(inplace=True)

            # Добавляем атрибуты
            self.data.attrs["symbol"] = self.symbol
            self.data.attrs["exchange"] = self.exchange_id
            self.data.attrs["timeframe"] = self.timeframe

            logger.debug("Updated market data with {len(new_df)} new bars" %)
        except Exception as e:
            logger.error("Error updating market data: {str(e)}" %)
            raise

    async def _init_strategy(self):
        """Инициализирует стратегию"""
        try:
            # Импортируем реестр стратегий
            from project.trading.strategy_base import StrategyRegistry

            # Загружаем параметры стратегии из базы данных
            if self.database:
                strategy_params = await self.database.get_strategy_parameters(
                    self.strategy_id, active_only=True
                )
                if strategy_params and len(strategy_params) > 0:
                    parameters = strategy_params[0].get("parameters", {})
                else:
                    parameters = {}
            else:
                parameters = {}

            # Создаем экземпляр стратегии
            strategy_class = StrategyRegistry.get_strategy_class(self.strategy_id)
            self.strategy = strategy_class(parameters=parameters)

            # Регистрируем обработчики событий
            self._register_event_handlers()

            logger.info(
                f"Strategy {self.strategy_id} initialized with parameters: {parameters}"
            )
        except Exception as e:
            logger.error("Error initializing strategy: {str(e)}" %)
            raise

    async def _load_state(self):
        """Загружает сохраненное состояние из базы данных"""
        if not self.database:
            logger.warning("Database not available, cannot load state")
            return

        try:
            # Загружаем состояние бота
            bot_states = await self.database.get_bot_states()

            for bot_state in bot_states:
                if bot_state.get("bot_id") == self.bot_id:
                    # Восстанавливаем состояние
                    state_data = bot_state.get("state", {})

                    # Обновляем метрики
                    if "stats" in state_data:
                        self.stats.update(state_data["stats"])

                    # Восстанавливаем позиции
                    if "positions" in state_data:
                        for pos_data in state_data["positions"]:
                            position = Position.from_dict(pos_data)
                            self.positions[position.id] = position

                    # Восстанавливаем сигналы
                    if "signals" in state_data:
                        for signal_data in state_data["signals"]:
                            signal = Signal.from_dict(signal_data)
                            self.signals.append(signal)

                    # Восстанавливаем временные метки
                    if "last_update_time" in state_data:
                        self.last_update_time = datetime.fromisoformat(
                            state_data["last_update_time"]
                        )

                    if "last_signal_time" in state_data:
                        self.last_signal_time = datetime.fromisoformat(
                            state_data["last_signal_time"]
                        )

                    if "last_trade_time" in state_data:
                        self.last_trade_time = datetime.fromisoformat(
                            state_data["last_trade_time"]
                        )

                    if "start_time" in state_data:
                        self.start_time = datetime.fromisoformat(
                            state_data["start_time"]
                        )

                    # Восстанавливаем счетчики ошибок
                    if "error_count" in state_data:
                        self.error_count = state_data["error_count"]

                    logger.info("Loaded saved state for bot {self.bot_id}" %)
                    return

            logger.info("No saved state found for bot {self.bot_id}" %)
        except Exception as e:
            logger.error("Error loading bot state: {str(e)}" %)

    async def _save_state(self):
        """Сохраняет состояние бота в базу данных"""
        if not self.database:
            logger.warning("Database not available, cannot save state")
            return

        try:
            # Создаем словарь состояния
            state = {
                "stats": self.stats,
                "positions": [pos.to_dict() for pos in self.positions.values()],
                "signals": [
                    signal.to_dict() for signal in self.signals[-20:]
                ],  # Сохраняем только последние 20 сигналов
                "last_update_time": (
                    self.last_update_time.isoformat() if self.last_update_time else None
                ),
                "last_signal_time": (
                    self.last_signal_time.isoformat() if self.last_signal_time else None
                ),
                "last_trade_time": (
                    self.last_trade_time.isoformat() if self.last_trade_time else None
                ),
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "error_count": self.error_count,
            }

            # Создаем запись для сохранения
            bot_state = {
                "bot_id": self.bot_id,
                "symbol": self.symbol,
                "exchange": self.exchange_id,
                "strategy_id": self.strategy_id,
                "state": state,
                "config": self.config,
                "is_active": self.state != BotState.STOPPED,
                "last_signal_timestamp": (
                    int(self.last_signal_time.timestamp() * 1000)
                    if self.last_signal_time
                    else None
                ),
                "last_trade_timestamp": (
                    int(self.last_trade_time.timestamp() * 1000)
                    if self.last_trade_time
                    else None
                ),
            }

            # Сохраняем состояние
            await self.database.save_bot_state(bot_state)

            logger.debug("Saved state for bot {self.bot_id}" %)
        except Exception as e:
            logger.error("Error saving bot state: {str(e)}" %)

    async def _check_existing_positions(self):
        """Проверяет существующие позиции на бирже"""
        if self.paper_trading or self.backtest_mode:
            return

        try:
            # Получаем открытые позиции с биржи
            positions = await self.exchange_manager.get_positions(
                self.symbol, self.exchange_id
            )

            if positions:
                for position in positions:
                    # Проверяем, что позиция для нашего символа
                    if position.get("symbol") == self.symbol:
                        # Получаем направление
                        amount = float(
                            position.get("contracts", position.get("amount", 0))
                        )
                        direction = (
                            "long" if amount > 0 else "short" if amount < 0 else None
                        )

                        if direction and abs(amount) > 0:
                            # Создаем объект позиции
                            pos = Position(
                                symbol=self.symbol,
                                direction=direction,
                                entry_price=float(
                                    position.get(
                                        "entryPrice", position.get("entry_price", 0)
                                    )
                                ),
                                amount=abs(amount),
                                open_time=datetime.fromtimestamp(
                                    position.get(
                                        "timestamp",
                                        position.get("datetime", time.time() * 1000),
                                    )
                                    / 1000
                                ),
                            )

                            # Добавляем в словарь позиций
                            self.positions[pos.id] = pos

                            logger.info("Found existing position: {pos}" %)
        except Exception as e:
            logger.error("Error checking existing positions: {str(e)}" %)

    async def _check_existing_orders(self):
        """Проверяет существующие ордера на бирже"""
        if self.paper_trading or self.backtest_mode:
            return

        try:
            # Получаем открытые ордера с биржи
            orders = await self.exchange_manager.fetch_open_orders(
                self.symbol, self.exchange_id
            )

            if orders:
                for order in orders:
                    # Проверяем, что ордер для нашего символа
                    if order.get("symbol") == self.symbol:
                        # Сохраняем ордер
                        self.open_orders[order.get("id")] = order

                        logger.info(
                            f"Found existing order: {order.get('id')} | {order.get('type')} | {order.get('side')} | {order.get('price')}"
                        )
        except Exception as e:
            logger.error("Error checking existing orders: {str(e)}" %)

    async def _update_positions(self):
        """Обновляет информацию о позициях"""
        if self.paper_trading or self.backtest_mode:
            return

        try:
            # Получаем открытые позиции с биржи
            positions = await self.exchange_manager.get_positions(
                self.symbol, self.exchange_id
            )

            # Создаем временный словарь для текущих позиций
            current_positions = {}

            if positions:
                for position in positions:
                    # Проверяем, что позиция для нашего символа
                    if position.get("symbol") == self.symbol:
                        # Получаем направление
                        amount = float(
                            position.get("contracts", position.get("amount", 0))
                        )
                        direction = (
                            "long" if amount > 0 else "short" if amount < 0 else None
                        )

                        if direction and abs(amount) > 0:
                            # Создаем или обновляем объект позиции
                            position_id = position.get("id")

                            # Если ID нет, создаем его на основе символа и направления
                            if not position_id:
                                position_id = f"{self.symbol}_{direction}"

                            # Если позиция уже существует, обновляем её
                            if position_id in self.positions:
                                pos = self.positions[position_id]
                                pos.amount = abs(amount)
                                pos.update_price(
                                    float(
                                        position.get(
                                            "markPrice", position.get("mark_price", 0)
                                        )
                                    )
                                )
                            else:
                                # Создаем новую позицию
                                pos = Position(
                                    symbol=self.symbol,
                                    direction=direction,
                                    entry_price=float(
                                        position.get(
                                            "entryPrice", position.get("entry_price", 0)
                                        )
                                    ),
                                    amount=abs(amount),
                                    open_time=datetime.fromtimestamp(
                                        position.get(
                                            "timestamp",
                                            position.get(
                                                "datetime", time.time() * 1000
                                            ),
                                        )
                                        / 1000
                                    ),
                                )

                                # Добавляем в словарь позиций
                                self.positions[position_id] = pos

                                logger.info("New position detected: {pos}" %)

                            # Добавляем в текущие позиции
                            current_positions[position_id] = pos

            # Проверяем, закрылись ли какие-то позиции
            for pos_id, pos in list(self.positions.items()):
                if pos.is_open() and pos_id not in current_positions:
                    # Позиция закрылась
                    pos.close(self.data.iloc[-1]["close"], datetime.now())

                    logger.info("Position closed: {pos}" %)

                    # Обновляем статистику
                    await self._update_stats_after_trade(pos)
        except Exception as e:
            logger.error("Error updating positions: {str(e)}" %)

    async def _update_orders(self):
        """Обновляет информацию об ордерах"""
        if self.paper_trading or self.backtest_mode:
            return

        try:
            # Получаем открытые ордера с биржи
            orders = await self.exchange_manager.fetch_open_orders(
                self.symbol, self.exchange_id
            )

            # Создаем временный словарь для текущих ордеров
            current_orders = {}

            if orders:
                for order in orders:
                    # Проверяем, что ордер для нашего символа
                    if order.get("symbol") == self.symbol:
                        # Сохраняем ордер
                        order_id = order.get("id")
                        current_orders[order_id] = order

                        # Если это новый ордер, добавляем его
                        if order_id not in self.open_orders:
                            self.open_orders[order_id] = order
                            logger.info(
                                f"New order detected: {order_id} | {order.get('type')} | {order.get('side')} | {order.get('price')}"
                            )

            # Проверяем, исполнились ли какие-то ордера
            for order_id, order in list(self.open_orders.items()):
                if order_id not in current_orders:
                    # Ордер исполнился или был отменен
                    # Получаем обновленную информацию
                    updated_order = await self.exchange_manager.fetch_order(
                        order_id, self.symbol, self.exchange_id
                    )

                    if updated_order:
                        # Обновляем ордер
                        self.open_orders[order_id] = updated_order

                        # Проверяем статус
                        status = updated_order.get("status")

                        if status == "closed":
                            logger.info(
                                f"Order executed: {order_id} | {updated_order.get('type')} | {updated_order.get('side')} | {updated_order.get('price')}"
                            )
                            # Удаляем из открытых ордеров
                            del self.open_orders[order_id]
                        elif status == "canceled":
                            logger.info("Order canceled: {order_id}" %)
                            # Удаляем из открытых ордеров
                            del self.open_orders[order_id]
                    else:
                        # Если не удалось получить информацию, удаляем ордер
                        logger.warning("Order not found: {order_id}" %)
                        del self.open_orders[order_id]
        except Exception as e:
            logger.error("Error updating orders: {str(e)}" %)

    async def _process_signal(self, signal: Signal):
        """
        Обрабатывает торговый сигнал

        Args:
            signal: Торговый сигнал
        """
        if not signal.is_valid():
            logger.debug("Signal is not valid: {signal}" %)
            return

        # Обновляем временную метку последнего сигнала
        self.last_signal_time = datetime.now()

        # Добавляем сигнал в список
        self.signals.append(signal)

        # Сохраняем сигнал в базу данных
        if self.database:
            await self.database.save_signal(signal.to_dict())

        logger.info("Processing signal: {signal}" %)

        # Проверяем, активен ли бот
        if self.state != BotState.RUNNING:
            logger.warning("Bot is not running, ignoring signal: {signal}" %)
            return

        # Проверяем тип сигнала
        if signal.direction == "buy":
            # Покупка - открываем длинную позицию
            await self._open_position("long", signal)
        elif signal.direction == "sell":
            # Продажа - если разрешены короткие позиции, открываем короткую позицию
            if self.allow_shorts:
                await self._open_position("short", signal)
            else:
                logger.debug("Short positions not allowed, ignoring signal: {signal}" %)
        elif signal.direction == "close":
            # Закрываем все позиции
            await self._close_all_positions()
        else:
            logger.warning("Unknown signal direction: {signal.direction}" %)

    async def _open_position(self, direction: str, signal: Signal):
        """
        Открывает новую позицию

        Args:
            direction: Направление (long, short)
            signal: Сигнал
        """
        try:
            # Проверяем, можно ли открыть новую позицию
            if (
                len([p for p in self.positions.values() if p.is_open()])
                >= self.max_positions
            ):
                logger.warning(
                    f"Cannot open {direction} position: maximum positions reached"
                )
                return

            # Получаем текущую цену
            current_price = signal.price
            if current_price is None:
                # Если цена не указана в сигнале, используем текущую цену закрытия
                current_price = self.data.iloc[-1]["close"]

            # Рассчитываем размер позиции
            if self.is_position_size_percentage:
                # Получаем баланс
                balance = await self._get_balance()

                # Рассчитываем размер позиции как процент от баланса
                position_value = balance * self.position_size

                # Рассчитываем количество
                amount = position_value / current_price
            else:
                # Используем фиксированный размер
                amount = self.position_size

            # Для бумажной торговли или бэктестинга используем внутреннюю логику
            if self.paper_trading or self.backtest_mode:
                # Создаем позицию
                position = Position(
                    symbol=self.symbol,
                    direction=direction,
                    entry_price=current_price,
                    amount=amount,
                    open_time=datetime.now(),
                    stop_loss=self._calculate_stop_loss(direction, current_price),
                    take_profit=self._calculate_take_profit(direction, current_price),
                )

                # Сохраняем позицию
                self.positions[position.id] = position

                # Вызываем обработчик открытия позиции
                if self.strategy:
                    await self.strategy.on_position_update(position)

                logger.info("Opened {direction} position: {amount} @ {current_price}" %)

                # Отправляем уведомление
                await self._send_notification(
                    f"New {direction} position opened",
                    f"Opened {direction} position for {self.symbol} on {self.exchange_id}\nAmount: {amount}\nPrice: {current_price}",
                    self.notification_levels.get("position", NotificationLevel.INFO),
                )

                return

            # Для реальной торговли создаем ордер на бирже
            order_side = "buy" if direction == "long" else "sell"

            # Создаем ордер
            order = await self.exchange_manager.create_order(
                symbol=self.symbol,
                order_type=self.order_type,
                side=order_side,
                amount=amount,
                price=current_price if self.order_type != "market" else None,
                exchange_id=self.exchange_id,
                params={
                    "timeInForce": self.time_in_force,
                    "postOnly": self.post_only,
                    "reduceOnly": self.reduce_only,
                },
            )

            if order:
                logger.info(
                    f"Created {direction} order: {order.get('id')} | {amount} @ {current_price}"
                )

                # Сохраняем ордер
                self.open_orders[order.get("id")] = order

                # Сохраняем ордер в базу данных
                if self.database:
                    await self.database.save_order(order)

                # Отправляем уведомление
                await self._send_notification(
                    f"New {direction} order created",
                    f"Created {direction} order for {self.symbol} on {self.exchange_id}\nAmount: {amount}\nPrice: {current_price}",
                    self.notification_levels.get("trade", NotificationLevel.INFO),
                )
            else:
                logger.error("Failed to create {direction} order" %)
        except Exception as e:
            logger.error("Error opening {direction} position: {str(e)}" %)

            # Отправляем уведомление об ошибке
            await self._send_notification(
                f"Error opening {direction} position",
                f"Failed to open {direction} position for {self.symbol} on {self.exchange_id}: {str(e)}",
                NotificationLevel.ERROR,
            )

    async def _close_position(self, position_id: str):
        """
        Закрывает позицию

        Args:
            position_id: ID позиции
        """
        try:
            # Проверяем, существует ли позиция
            if position_id not in self.positions:
                logger.warning("Position not found: {position_id}" %)
                return

            position = self.positions[position_id]

            # Проверяем, открыта ли позиция
            if not position.is_open():
                logger.warning("Position already closed: {position_id}" %)
                return

            # Получаем текущую цену
            current_price = self.data.iloc[-1]["close"]

            # Для бумажной торговли или бэктестинга используем внутреннюю логику
            if self.paper_trading or self.backtest_mode:
                # Закрываем позицию
                position.close(current_price, datetime.now())

                # Вызываем обработчик закрытия позиции
                if self.strategy:
                    await self.strategy.on_position_update(position)

                # Обновляем статистику
                await self._update_stats_after_trade(position)

                logger.info("Closed position: {position}" %)

                # Отправляем уведомление
                await self._send_notification(
                    f"Position closed",
                    f"Closed {position.direction} position for {self.symbol} on {self.exchange_id}\nEntry: {position.entry_price}\nExit: {current_price}\nPnL: {position.realized_pnl:.2f}",
                    self.notification_levels.get("position", NotificationLevel.INFO),
                )

                return

            # Для реальной торговли создаем ордер на бирже
            order_side = "sell" if position.direction == "long" else "buy"

            # Создаем ордер
            order = await self.exchange_manager.create_order(
                symbol=self.symbol,
                order_type=self.order_type,
                side=order_side,
                amount=position.amount,
                price=current_price if self.order_type != "market" else None,
                exchange_id=self.exchange_id,
                params={
                    "timeInForce": self.time_in_force,
                    "postOnly": self.post_only,
                    "reduceOnly": True,  # Всегда уменьшать позицию при закрытии
                },
            )

            if order:
                logger.info(
                    f"Created closing order: {order.get('id')} | {position.amount} @ {current_price}"
                )

                # Сохраняем ордер
                self.open_orders[order.get("id")] = order

                # Сохраняем ордер в базу данных
                if self.database:
                    await self.database.save_order(order)

                # Отправляем уведомление
                await self._send_notification(
                    f"Closing order created",
                    f"Created order to close {position.direction} position for {self.symbol} on {self.exchange_id}\nAmount: {position.amount}\nPrice: {current_price}",
                    self.notification_levels.get("trade", NotificationLevel.INFO),
                )
            else:
                logger.error(
                    f"Failed to create closing order for position {position_id}"
                )
        except Exception as e:
            logger.error("Error closing position {position_id}: {str(e)}" %)

            # Отправляем уведомление об ошибке
            await self._send_notification(
                f"Error closing position",
                f"Failed to close position {position_id} for {self.symbol} on {self.exchange_id}: {str(e)}",
                NotificationLevel.ERROR,
            )

    async def _close_all_positions(self):
        """Закрывает все открытые позиции"""
        # Получаем список открытых позиций
        open_positions = [
            pos_id for pos_id, pos in self.positions.items() if pos.is_open()
        ]

        # Закрываем каждую позицию
        for position_id in open_positions:
            await self._close_position(position_id)

    async def _get_balance(self) -> float:
        """
        Получает доступный баланс

        Returns:
            float: Доступный баланс
        """
        if self.paper_trading or self.backtest_mode:
            # Для бумажной торговли или бэктестинга используем начальный баланс из конфигурации
            initial_balance = self.config.get("initial_balance", 10000.0)

            # Учитываем прибыль/убыток от закрытых позиций
            realized_pnl = sum(
                pos.realized_pnl for pos in self.positions.values() if not pos.is_open()
            )

            return initial_balance + realized_pnl

        try:
            # Для реальной торговли получаем баланс с биржи
            balance = await self.exchange_manager.fetch_balance(self.exchange_id)

            if balance:
                # Получаем валюту учета
                quote_currency = self.symbol.split("/")[-1]

                # Получаем доступный баланс
                free_balance = balance.get("free", {}).get(quote_currency, 0.0)

                return float(free_balance)
            else:
                logger.warning("Failed to fetch balance")
                return 0.0
        except Exception as e:
            logger.error("Error getting balance: {str(e)}" %)
            return 0.0

    def _calculate_stop_loss(
        self, direction: str, entry_price: float
    ) -> Optional[float]:
        """
        Рассчитывает уровень стоп-лосса

        Args:
            direction: Направление позиции (long, short)
            entry_price: Цена входа

        Returns:
            Optional[float]: Уровень стоп-лосса или None
        """
        if not self.stop_loss:
            return None

        # Проверяем тип стоп-лосса
        if isinstance(self.stop_loss, float) or isinstance(self.stop_loss, int):
            # Фиксированный процент
            if direction == "long":
                return entry_price * (1 - self.stop_loss)
            else:  # short
                return entry_price * (1 + self.stop_loss)
        elif isinstance(self.stop_loss, dict):
            # Динамический стоп-лосс
            if "atr_multiplier" in self.stop_loss:
                # Используем ATR для расчета стоп-лосса
                multiplier = self.stop_loss["atr_multiplier"]
                atr_period = self.stop_loss.get("atr_period", 14)

                # Рассчитываем ATR
                atr = self._calculate_atr(atr_period)

                if atr is not None:
                    if direction == "long":
                        return entry_price - (atr * multiplier)
                    else:  # short
                        return entry_price + (atr * multiplier)

        return None

    def _calculate_take_profit(
        self, direction: str, entry_price: float
    ) -> Optional[float]:
        """
        Рассчитывает уровень тейк-профита

        Args:
            direction: Направление позиции (long, short)
            entry_price: Цена входа

        Returns:
            Optional[float]: Уровень тейк-профита или None
        """
        if not self.take_profit:
            return None

        # Проверяем тип тейк-профита
        if isinstance(self.take_profit, float) or isinstance(self.take_profit, int):
            # Фиксированный процент
            if direction == "long":
                return entry_price * (1 + self.take_profit)
            else:  # short
                return entry_price * (1 - self.take_profit)

        return None

    def _calculate_atr(self, period: int = 14) -> Optional[float]:
        """
        Рассчитывает индикатор ATR (Average True Range)

        Args:
            period: Период ATR

        Returns:
            Optional[float]: Значение ATR или None
        """
        if self.data is None or len(self.data) < period + 1:
            return None

        try:
            # Получаем данные
            high = self.data["high"].values
            low = self.data["low"].values
            close = self.data["close"].values

            # Рассчитываем True Range
            tr1 = abs(high[1:] - low[1:])
            tr2 = abs(high[1:] - close[:-1])
            tr3 = abs(low[1:] - close[:-1])

            tr = np.maximum(np.maximum(tr1, tr2), tr3)

            # Рассчитываем ATR
            atr = np.mean(tr[-period:])

            return atr
        except Exception as e:
            logger.error("Error calculating ATR: {str(e)}" %)
            return None

    async def _update_stats_after_trade(self, position: Position):
        """
        Обновляет статистику после закрытия позиции

        Args:
            position: Закрытая позиция
        """
        # Увеличиваем счетчик сделок
        self.stats["trades_count"] += 1

        # Обновляем временную метку последней сделки
        self.last_trade_time = datetime.now()

        # Обновляем прибыль/убыток
        pnl = position.realized_pnl
        self.stats["total_pnl"] += pnl

        # Классифицируем сделку
        if pnl > 0:
            self.stats["winning_trades"] += 1
            self.stats["largest_win"] = max(self.stats["largest_win"], pnl)
        elif pnl < 0:
            self.stats["losing_trades"] += 1
            self.stats["largest_loss"] = min(self.stats["largest_loss"], pnl)
        else:
            self.stats["break_even_trades"] += 1

        # Рассчитываем процент выигрышных сделок
        if self.stats["trades_count"] > 0:
            self.stats["win_rate"] = (
                self.stats["winning_trades"] / self.stats["trades_count"]
            )

        # Рассчитываем средний выигрыш/проигрыш
        if self.stats["winning_trades"] > 0:
            self.stats["average_win"] = (
                sum(
                    pos.realized_pnl
                    for pos in self.positions.values()
                    if not pos.is_open() and pos.realized_pnl > 0
                )
                / self.stats["winning_trades"]
            )

        if self.stats["losing_trades"] > 0:
            self.stats["average_loss"] = (
                sum(
                    pos.realized_pnl
                    for pos in self.positions.values()
                    if not pos.is_open() and pos.realized_pnl < 0
                )
                / self.stats["losing_trades"]
            )

        # Рассчитываем profit factor
        total_gains = sum(
            pos.realized_pnl
            for pos in self.positions.values()
            if not pos.is_open() and pos.realized_pnl > 0
        )
        total_losses = abs(
            sum(
                pos.realized_pnl
                for pos in self.positions.values()
                if not pos.is_open() and pos.realized_pnl < 0
            )
        )

        if total_losses > 0:
            self.stats["profit_factor"] = total_gains / total_losses

        # Сохраняем позицию в базу данных
        if self.database:
            await self.database.save_position(position.to_dict())

        # Отправляем уведомление о закрытой позиции
        direction_text = "Long" if position.direction == "long" else "Short"
        notification_level = self.notification_levels.get(
            "trade", NotificationLevel.INFO
        )

        # Если убыточная сделка, повышаем уровень уведомления
        if pnl < 0:
            notification_level = max(notification_level, NotificationLevel.WARNING)

        await self._send_notification(
            f"{direction_text} position closed with P&L: {pnl:.2f}",
            f"Closed {position.direction} position for {self.symbol} on {self.exchange_id}\n"
            f"Entry: {position.entry_price}\n"
            f"Exit: {position.close_price}\n"
            f"Amount: {position.amount}\n"
            f"P&L: {pnl:.2f}\n"
            f"Open time: {position.open_time}\n"
            f"Close time: {position.close_time}",
            notification_level,
        )

    async def _update_stats(self):
        """Обновляет общую статистику бота"""
        # Обновляем время работы
        if self.start_time:
            self.stats["run_time"] = (datetime.now() - self.start_time).total_seconds()

        # Обновляем максимальную просадку
        if self.data is not None and len(self.data) > 0:
            # Получаем историю изменения капитала
            equity_history = []

            # TODO: Реализовать расчет просадки на основе истории изменения капитала

        # Обновляем ошибки
        self.stats["errors"] = self.error_count

    async def _on_signal(self, signal: Signal):
        """
        Обработчик сигналов от стратегии

        Args:
            signal: Торговый сигнал
        """
        await self._process_signal(signal)

    async def _on_position_opened(self, position: Position):
        """
        Обработчик открытия позиции

        Args:
            position: Открытая позиция
        """
        logger.info("Position opened: {position}" %)

        # Сохраняем позицию
        self.positions[position.id] = position

        # Сохраняем позицию в базу данных
        if self.database:
            await self.database.save_position(position.to_dict())

    async def _on_position_closed(self, position: Position):
        """
        Обработчик закрытия позиции

        Args:
            position: Закрытая позиция
        """
        logger.info("Position closed: {position}" %)

        # Обновляем статистику
        await self._update_stats_after_trade(position)

    async def _send_notification(
        self,
        title: str,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
    ):
        """
        Отправляет уведомление

        Args:
            title: Заголовок уведомления
            message: Текст уведомления
            level: Уровень уведомления
        """
        if self.notification_manager:
            try:
                await self.notification_manager.send_notification(title, message, level)
            except Exception as e:
                logger.error("Error sending notification: {str(e)}" %)

    def get_info(self) -> Dict:
        """
        Возвращает информацию о боте

        Returns:
            Dict: Информация о боте
        """
        return {
            "bot_id": self.bot_id,
            "symbol": self.symbol,
            "exchange_id": self.exchange_id,
            "timeframe": self.timeframe,
            "strategy_id": self.strategy_id,
            "state": self.state.value,
            "stats": self.stats,
            "positions": {
                pos_id: pos.to_dict() for pos_id, pos in self.positions.items()
            },
            "last_update_time": (
                self.last_update_time.isoformat() if self.last_update_time else None
            ),
            "last_signal_time": (
                self.last_signal_time.isoformat() if self.last_signal_time else None
            ),
            "last_trade_time": (
                self.last_trade_time.isoformat() if self.last_trade_time else None
            ),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "config": self.config,
        }
