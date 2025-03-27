"""
Стратегия скальпинга.
Торгует на небольших движениях цены с быстрым входом и выходом.
"""

import time
from typing import Any, Dict, List, Optional

from project.bots.strategies.base_strategy import BaseStrategy
from project.technicals.indicators import Indicators
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ScalpingStrategy(BaseStrategy):
    """
    Стратегия скальпинга для быстрой торговли на небольших движениях цены.
    """

    def __init__(
        self,
        name: str = "ScalpingStrategy",
        exchange_id: str = "binance",
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Инициализирует стратегию скальпинга.

        Args:
            name: Имя стратегии
            exchange_id: Идентификатор биржи
            symbols: Список символов для торговли
            timeframes: Список таймфреймов для анализа
            config: Конфигурация стратегии
        """
        # Устанавливаем значения по умолчанию
        config = config or {}
        default_config = {
            "ema_short_period": 9,
            "ema_medium_period": 21,
            "ema_long_period": 50,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "take_profit_pct": 0.005,  # 0.5%
            "stop_loss_pct": 0.003,  # 0.3%
            "max_trade_duration": 30,  # максимальная длительность сделки в минутах
            "min_volume": 1000000,  # минимальный объем для торговли
            "max_spread_pct": 0.001,  # максимальный спред (0.1%)
            "use_order_book": True,  # использовать данные ордербука
            "order_book_imbalance_threshold": 1.5,  # порог дисбаланса ордербука
            "trade_on_momentum": True,  # торговать на импульсе
            "momentum_threshold": 0.002,  # порог импульса (0.2%)
        }

        # Объединяем с пользовательской конфигурацией
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        # Устанавливаем базовые значения
        symbols = symbols or ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT"]
        timeframes = timeframes or ["1m", "5m", "15m"]

        super().__init__(name, exchange_id, symbols, timeframes, config)

        # Дополнительные параметры
        self.cached_orderbooks: Dict[str, Dict[str, Any]] = {}  # symbol -> ордербук
        self.momentum_data: Dict[str, float] = {}  # symbol -> импульс
        self.trade_start_times: Dict[str, float] = {}  # symbol -> время начала сделки

        logger.debug("Создана стратегия скальпинга {self.name}" %)

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """
        # Обновляем параметры индикаторов
        for param in [
            "ema_short_period",
            "ema_medium_period",
            "ema_long_period",
            "rsi_period",
            "rsi_overbought",
            "rsi_oversold",
            "macd_fast_period",
            "macd_slow_period",
            "macd_signal_period",
        ]:
            if param in config:
                self.strategy_config[param] = int(config[param])

        # Обновляем параметры стратегии
        for param in [
            "take_profit_pct",
            "stop_loss_pct",
            "max_spread_pct",
            "order_book_imbalance_threshold",
            "momentum_threshold",
        ]:
            if param in config:
                self.strategy_config[param] = float(config[param])

        for param in ["max_trade_duration", "min_volume"]:
            if param in config:
                self.strategy_config[param] = int(config[param])

        for param in ["use_order_book", "trade_on_momentum"]:
            if param in config:
                self.strategy_config[param] = bool(config[param])

    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        """
        # Устанавливаем меньший интервал обновления для скальпинга
        self.update_interval = 5.0  # 5 секунд

        # Инициализируем кэш ордербуков
        for symbol in self.symbols:
            self.cached_orderbooks[symbol] = {}
            self.momentum_data[symbol] = 0.0

    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        """
        # Нет специфических ресурсов для очистки

    @async_handle_error
    async def _update_market_data(self) -> None:
        """
        Обновляет рыночные данные для всех символов и таймфреймов.
        """
        # Вызываем базовый метод для обновления OHLCV
        await super()._update_market_data()

        # Дополнительно обновляем ордербуки, если используем их
        if self.strategy_config["use_order_book"]:
            for symbol in self.symbols:
                try:
                    orderbook = await self.market_data.get_orderbook(
                        self.exchange_id, symbol, limit=20
                    )
                    self.cached_orderbooks[symbol] = orderbook
                except Exception as e:
                    logger.warning(
                        f"Ошибка при получении ордербука для {symbol}: {str(e)}"
                    )

    @async_handle_error
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы на основе текущих рыночных данных.

        Returns:
            Словарь с сигналами для каждого символа
        """
        signals = {}

        for symbol in self.symbols:
            try:
                # Получаем тикер
                ticker = await self.market_data.get_ticker(self.exchange_id, symbol)
                if not ticker:
                    continue

                # Проверяем объем
                volume = ticker.get("quoteVolume", 0) or ticker.get("volume", 0)
                if volume < self.strategy_config["min_volume"]:
                    continue

                # Получаем текущую цену, спред и проверяем его размер
                current_price = ticker.get("last", 0)
                bid = ticker.get("bid", 0)
                ask = ticker.get("ask", 0)

                if current_price <= 0 or bid <= 0 or ask <= 0:
                    continue

                spread_pct = (ask - bid) / bid
                if spread_pct > self.strategy_config["max_spread_pct"]:
                    logger.debug(
                        f"Слишком большой спред для {symbol}: {spread_pct:.4f}"
                    )
                    continue

                # Проверяем ордербук, если включено
                order_book_signal = None
                if self.strategy_config["use_order_book"]:
                    order_book_signal = self._analyze_order_book(symbol)

                # Рассчитываем импульс, если включено
                momentum_signal = None
                if self.strategy_config["trade_on_momentum"]:
                    momentum_signal = self._calculate_momentum(symbol)

                # Генерируем сигналы для разных таймфреймов
                timeframe_signals = {}
                for timeframe in self.timeframes:
                    timeframe_signals[timeframe] = (
                        await self._generate_timeframe_signal(symbol, timeframe)
                    )

                # Комбинируем сигналы с разных таймфреймов
                combined_signal = self._combine_signals(
                    symbol, timeframe_signals, order_book_signal, momentum_signal
                )

                if combined_signal and combined_signal["action"] != "hold":
                    signals[symbol] = combined_signal

                # Проверяем длительность открытых позиций
                if symbol in self.open_positions:
                    await self._check_position_duration(symbol)

            except Exception as e:
                logger.error("Ошибка при генерации сигналов для {symbol}: {str(e)}" %)

        return signals

    @async_handle_error
    async def _generate_timeframe_signal(
        self, symbol: str, timeframe: str
    ) -> Dict[str, str]:
        """
        Генерирует сигналы для конкретного таймфрейма.

        Args:
            symbol: Символ для торговли
            timeframe: Таймфрейм для анализа

        Returns:
            Словарь с сигналами
        """
        signals = {}

        try:
            # Получаем данные OHLCV
            ohlcv = await self.market_data.get_ohlcv(
                self.exchange_id, symbol, timeframe, limit=100
            )

            if ohlcv.empty:
                return {}

            # Рассчитываем EMA
            ema_short = Indicators.exponential_moving_average(
                ohlcv, self.strategy_config["ema_short_period"]
            )
            ema_medium = Indicators.exponential_moving_average(
                ohlcv, self.strategy_config["ema_medium_period"]
            )
            ema_long = Indicators.exponential_moving_average(
                ohlcv, self.strategy_config["ema_long_period"]
            )

            # Рассчитываем RSI
            rsi = Indicators.relative_strength_index(
                ohlcv, self.strategy_config["rsi_period"]
            )

            # Рассчитываем MACD
            macd_data = Indicators.macd(
                ohlcv,
                self.strategy_config["macd_fast_period"],
                self.strategy_config["macd_slow_period"],
                self.strategy_config["macd_signal_period"],
            )

            # Проверяем наличие данных
            if (
                ema_short.empty
                or ema_medium.empty
                or ema_long.empty
                or rsi.empty
                or macd_data["macd"].empty
                or macd_data["signal"].empty
            ):
                return {}

            # Получаем последние значения
            last_ema_short = ema_short.iloc[-1]
            last_ema_medium = ema_medium.iloc[-1]
            last_ema_long = ema_long.iloc[-1]
            last_rsi = rsi.iloc[-1]
            last_macd = macd_data["macd"].iloc[-1]
            last_signal = macd_data["signal"].iloc[-1]

            # Получаем предыдущие значения
            prev_macd = (
                macd_data["macd"].iloc[-2] if len(macd_data["macd"]) > 1 else last_macd
            )
            prev_signal = (
                macd_data["signal"].iloc[-2]
                if len(macd_data["signal"]) > 1
                else last_signal
            )

            # Сигналы от EMA
            if last_ema_short > last_ema_medium > last_ema_long:
                signals["ema"] = "buy"
            elif last_ema_short < last_ema_medium < last_ema_long:
                signals["ema"] = "sell"
            else:
                signals["ema"] = "hold"

            # Сигналы от RSI
            if last_rsi < self.strategy_config["rsi_oversold"]:
                signals["rsi"] = "buy"
            elif last_rsi > self.strategy_config["rsi_overbought"]:
                signals["rsi"] = "sell"
            else:
                signals["rsi"] = "hold"

            # Сигналы от MACD
            if last_macd > last_signal and prev_macd <= prev_signal:
                signals["macd"] = "buy"
            elif last_macd < last_signal and prev_macd >= prev_signal:
                signals["macd"] = "sell"
            else:
                signals["macd"] = "hold"

            return signals

        except Exception as e:
            logger.error(
                f"Ошибка при генерации сигналов для {symbol} на {timeframe}: {str(e)}"
            )
            return {}

    def _analyze_order_book(self, symbol: str) -> Optional[str]:
        """
        Анализирует ордербук и определяет давление покупателей/продавцов.

        Args:
            symbol: Символ для торговли

        Returns:
            Сигнал (buy/sell/None)
        """
        orderbook = self.cached_orderbooks.get(symbol)
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return None

        bids = orderbook["bids"]
        asks = orderbook["asks"]

        if not bids or not asks:
            return None

        # Суммируем объемы первых 10 уровней
        bid_volume = sum(bid[1] for bid in bids[:10] if len(bid) > 1)
        ask_volume = sum(ask[1] for ask in asks[:10] if len(ask) > 1)

        if bid_volume <= 0 or ask_volume <= 0:
            return None

        # Рассчитываем отношение объемов
        volume_ratio = bid_volume / ask_volume

        # Определяем сигнал
        threshold = self.strategy_config["order_book_imbalance_threshold"]

        if volume_ratio > threshold:
            return "buy"  # Больше покупок
        elif volume_ratio < 1 / threshold:
            return "sell"  # Больше продаж

        return None

    def _calculate_momentum(self, symbol: str) -> Optional[str]:
        """
        Рассчитывает импульс и определяет направление движения.

        Args:
            symbol: Символ для торговли

        Returns:
            Сигнал (buy/sell/None)
        """
        try:
            # Получаем данные OHLCV для минутного таймфрейма
            ohlcv = self.market_data.get_ohlcv(self.exchange_id, symbol, "1m")

            if ohlcv.empty or len(ohlcv) < 5:
                return None

            # Рассчитываем изменение цены за последние 5 минут
            price_5min_ago = ohlcv["close"].iloc[-5]
            current_price = ohlcv["close"].iloc[-1]

            if price_5min_ago <= 0:
                return None

            # Рассчитываем импульс (изменение в процентах)
            momentum = current_price / price_5min_ago - 1

            # Сохраняем значение импульса
            self.momentum_data[symbol] = momentum

            # Определяем сигнал на основе импульса
            threshold = self.strategy_config["momentum_threshold"]

            if momentum > threshold:
                return "buy"
            elif momentum < -threshold:
                return "sell"

            return None

        except Exception as e:
            logger.error("Ошибка при расчете импульса для {symbol}: {str(e)}" %)
            return None

    def _combine_signals(
        self,
        symbol: str,
        timeframe_signals: Dict[str, Dict[str, str]],
        order_book_signal: Optional[str],
        momentum_signal: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Комбинирует сигналы из разных источников.

        Args:
            symbol: Символ для торговли
            timeframe_signals: Сигналы с разных таймфреймов
            order_book_signal: Сигнал от ордербука
            momentum_signal: Сигнал от импульса

        Returns:
            Комбинированный сигнал или None
        """
        if not timeframe_signals:
            return None

        # Подсчитываем количество сигналов каждого типа
        buy_count = 0
        sell_count = 0
        hold_count = 0

        # Учитываем веса таймфреймов (меньшие таймфреймы важнее для скальпинга)
        timeframe_weights = {}
        for i, tf in enumerate(self.timeframes):
            timeframe_weights[tf] = (len(self.timeframes) - i) / len(self.timeframes)

        # Подсчитываем сигналы с разных таймфреймов
        for tf, signals in timeframe_signals.items():
            weight = timeframe_weights[tf]

            for indicator, signal in signals.items():
                if signal == "buy":
                    buy_count += weight
                elif signal == "sell":
                    sell_count += weight
                else:
                    hold_count += weight

        # Учитываем сигнал ордербука
        if order_book_signal:
            if order_book_signal == "buy":
                buy_count += 1.5  # Сигнал ордербука имеет больший вес
            elif order_book_signal == "sell":
                sell_count += 1.5

        # Учитываем сигнал импульса
        if momentum_signal:
            if momentum_signal == "buy":
                buy_count += 1.0
            elif momentum_signal == "sell":
                sell_count += 1.0

        # Определяем итоговый сигнал
        action = "hold"
        if buy_count >= 2 * sell_count and buy_count > hold_count:
            action = "buy"
        elif sell_count >= 2 * buy_count and sell_count > hold_count:
            action = "sell"

        # Если у нас уже есть открытая позиция по этому символу
        if symbol in self.open_positions:
            position = self.open_positions[symbol]
            position_side = position["side"]

            # Если сигнал противоположен стороне позиции, закрываем позицию
            if (position_side == "long" and action == "sell") or (
                position_side == "short" and action == "buy"
            ):
                action = "exit"
            # Если сигнал совпадает со стороной позиции или "hold", ничего не делаем
            else:
                action = "hold"

        if action == "hold":
            return None

        # Получаем текущую цену
        ticker = self.market_data.get_ticker(self.exchange_id, symbol)
        current_price = ticker.get("last", 0) if ticker else 0

        # Формируем итоговый сигнал
        return {
            "symbol": symbol,
            "action": action,
            "price": current_price,
            "buy_score": buy_count,
            "sell_score": sell_count,
            "hold_score": hold_count,
            "order_book_signal": order_book_signal,
            "momentum_signal": momentum_signal,
            "momentum_value": self.momentum_data.get(symbol, 0),
            "timestamp": time.time(),
        }

    @async_handle_error
    async def _check_position_duration(self, symbol: str) -> None:
        """
        Проверяет длительность открытой позиции и закрывает ее, если превышен лимит.

        Args:
            symbol: Символ для торговли
        """
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]

        # Проверяем время открытия позиции
        entry_time = position.get("entry_time", 0)
        current_time = time.time()

        # Рассчитываем длительность в минутах
        duration_minutes = (current_time - entry_time) / 60

        # Если превышен лимит, закрываем позицию
        if duration_minutes > self.strategy_config["max_trade_duration"]:
            logger.info(
                f"Закрываем позицию по {symbol} из-за превышения максимального времени: "
                f"{duration_minutes:.1f} минут > {self.strategy_config['max_trade_duration']} минут"
            )

            await self._close_position(symbol, "time_limit")
