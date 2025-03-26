"""
Основная торговая стратегия.
Реализует комбинированную стратегию на основе нескольких индикаторов.
"""

import asyncio
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set

from project.config import get_config
from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error
from project.data.market_data import MarketData
from project.technicals.indicators import Indicators
from project.technicals.patterns import Patterns
from project.bots.strategies.base_strategy import BaseStrategy, StrategyStatus

logger = get_logger(__name__)


class MainStrategy(BaseStrategy):
    """
    Основная торговая стратегия, использующая несколько индикаторов.
    """

    def __init__(
        self,
        name: str = "MainStrategy",
        exchange_id: str = "binance",
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Инициализирует основную стратегию.

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
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "ema_short_period": 9,
            "ema_medium_period": 21,
            "ema_long_period": 50,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "atr_period": 14,
            "min_volume": 1000000,  # Минимальный объем для торговли
            "trend_confirmation": True,  # Требовать подтверждения тренда
            "use_patterns": True,  # Использовать свечные паттерны
            "signal_confirmation_count": 2,  # Минимальное количество подтверждающих сигналов
        }

        # Объединяем с пользовательской конфигурацией
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        # Устанавливаем базовые значения
        symbols = symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        timeframes = timeframes or ["1h", "4h", "1d"]

        super().__init__(name, exchange_id, symbols, timeframes, config)

        # Дополнительные параметры
        self.indicators_data: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = (
            {}
        )  # symbol -> timeframe -> indicator -> DataFrame
        self.last_analyzed_candle: Dict[str, Dict[str, int]] = (
            {}
        )  # symbol -> timeframe -> timestamp
        self.trend_direction: Dict[str, str] = {}  # symbol -> trend

        logger.debug(f"Создана основная стратегия {self.name}")

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """
        # Обновляем параметры индикаторов
        if "rsi_period" in config:
            self.strategy_config["rsi_period"] = int(config["rsi_period"])

        if "rsi_overbought" in config:
            self.strategy_config["rsi_overbought"] = float(config["rsi_overbought"])

        if "rsi_oversold" in config:
            self.strategy_config["rsi_oversold"] = float(config["rsi_oversold"])

        if "ema_short_period" in config:
            self.strategy_config["ema_short_period"] = int(config["ema_short_period"])

        if "ema_medium_period" in config:
            self.strategy_config["ema_medium_period"] = int(config["ema_medium_period"])

        if "ema_long_period" in config:
            self.strategy_config["ema_long_period"] = int(config["ema_long_period"])

        if "macd_fast_period" in config:
            self.strategy_config["macd_fast_period"] = int(config["macd_fast_period"])

        if "macd_slow_period" in config:
            self.strategy_config["macd_slow_period"] = int(config["macd_slow_period"])

        if "macd_signal_period" in config:
            self.strategy_config["macd_signal_period"] = int(
                config["macd_signal_period"]
            )

        if "bollinger_period" in config:
            self.strategy_config["bollinger_period"] = int(config["bollinger_period"])

        if "bollinger_std" in config:
            self.strategy_config["bollinger_std"] = float(config["bollinger_std"])

        if "atr_period" in config:
            self.strategy_config["atr_period"] = int(config["atr_period"])

        if "min_volume" in config:
            self.strategy_config["min_volume"] = float(config["min_volume"])

        if "trend_confirmation" in config:
            self.strategy_config["trend_confirmation"] = bool(
                config["trend_confirmation"]
            )

        if "use_patterns" in config:
            self.strategy_config["use_patterns"] = bool(config["use_patterns"])

        if "signal_confirmation_count" in config:
            self.strategy_config["signal_confirmation_count"] = int(
                config["signal_confirmation_count"]
            )

    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        """
        # Инициализируем структуры данных
        for symbol in self.symbols:
            # Инициализируем данные индикаторов
            self.indicators_data[symbol] = {}

            # Инициализируем последние проанализированные свечи
            self.last_analyzed_candle[symbol] = {}

            # Инициализируем направление тренда
            self.trend_direction[symbol] = "neutral"

            for timeframe in self.timeframes:
                self.indicators_data[symbol][timeframe] = {}
                self.last_analyzed_candle[symbol][timeframe] = 0

        # Рассчитываем начальные индикаторы
        await self._calculate_indicators()

    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        """
        # Нет специфических ресурсов для очистки
        pass

    @async_handle_error
    async def _calculate_indicators(self) -> None:
        """
        Рассчитывает технические индикаторы для всех символов и таймфреймов.
        """
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    # Получаем данные OHLCV
                    ohlcv = await self.market_data.get_ohlcv(
                        self.exchange_id, symbol, timeframe, limit=100
                    )

                    if ohlcv.empty:
                        logger.warning(
                            f"Нет данных OHLCV для {symbol} на таймфрейме {timeframe}"
                        )
                        continue

                    # Получаем временную метку последней свечи
                    last_timestamp = ohlcv.index[-1].timestamp()

                    # Если эта свеча уже была проанализирована, пропускаем
                    if last_timestamp <= self.last_analyzed_candle[symbol][timeframe]:
                        continue

                    # Обновляем временную метку последней проанализированной свечи
                    self.last_analyzed_candle[symbol][timeframe] = last_timestamp

                    # Рассчитываем RSI
                    rsi = Indicators.relative_strength_index(
                        ohlcv, self.strategy_config["rsi_period"]
                    )
                    self.indicators_data[symbol][timeframe]["rsi"] = rsi

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

                    self.indicators_data[symbol][timeframe]["ema_short"] = ema_short
                    self.indicators_data[symbol][timeframe]["ema_medium"] = ema_medium
                    self.indicators_data[symbol][timeframe]["ema_long"] = ema_long

                    # Рассчитываем MACD
                    macd_data = Indicators.macd(
                        ohlcv,
                        self.strategy_config["macd_fast_period"],
                        self.strategy_config["macd_slow_period"],
                        self.strategy_config["macd_signal_period"],
                    )

                    self.indicators_data[symbol][timeframe]["macd"] = macd_data["macd"]
                    self.indicators_data[symbol][timeframe]["macd_signal"] = macd_data[
                        "signal"
                    ]
                    self.indicators_data[symbol][timeframe]["macd_histogram"] = (
                        macd_data["histogram"]
                    )

                    # Рассчитываем полосы Боллинджера
                    bb_data = Indicators.bollinger_bands(
                        ohlcv,
                        self.strategy_config["bollinger_period"],
                        self.strategy_config["bollinger_std"],
                    )

                    self.indicators_data[symbol][timeframe]["bb_upper"] = bb_data[
                        "upper"
                    ]
                    self.indicators_data[symbol][timeframe]["bb_middle"] = bb_data[
                        "middle"
                    ]
                    self.indicators_data[symbol][timeframe]["bb_lower"] = bb_data[
                        "lower"
                    ]

                    # Рассчитываем ATR
                    atr = Indicators.average_true_range(
                        ohlcv, self.strategy_config["atr_period"]
                    )
                    self.indicators_data[symbol][timeframe]["atr"] = atr

                    # Если включено использование паттернов, рассчитываем их
                    if self.strategy_config["use_patterns"]:
                        # Бычьи паттерны
                        bullish_engulfing = Patterns.engulfing(ohlcv, bullish=True)
                        hammer = Patterns.hammer(ohlcv)
                        morning_star = Patterns.morning_star(ohlcv)

                        # Медвежьи паттерны
                        bearish_engulfing = Patterns.engulfing(ohlcv, bullish=False)
                        shooting_star = Patterns.shooting_star(ohlcv)
                        evening_star = Patterns.evening_star(ohlcv)

                        self.indicators_data[symbol][timeframe][
                            "bullish_engulfing"
                        ] = bullish_engulfing
                        self.indicators_data[symbol][timeframe]["hammer"] = hammer
                        self.indicators_data[symbol][timeframe][
                            "morning_star"
                        ] = morning_star
                        self.indicators_data[symbol][timeframe][
                            "bearish_engulfing"
                        ] = bearish_engulfing
                        self.indicators_data[symbol][timeframe][
                            "shooting_star"
                        ] = shooting_star
                        self.indicators_data[symbol][timeframe][
                            "evening_star"
                        ] = evening_star

                    # Определяем направление тренда на основе EMA
                    self._determine_trend(symbol, timeframe)

                except Exception as e:
                    logger.error(
                        f"Ошибка при расчете индикаторов для {symbol} на {timeframe}: {str(e)}"
                    )

    def _determine_trend(self, symbol: str, timeframe: str) -> None:
        """
        Определяет направление тренда на основе EMA.

        Args:
            symbol: Символ для торговли
            timeframe: Таймфрейм для анализа
        """
        try:
            if (
                timeframe != self.timeframes[-1]
            ):  # Используем только самый старший таймфрейм
                return

            # Получаем данные EMA
            ema_short = self.indicators_data[symbol][timeframe]["ema_short"]
            ema_medium = self.indicators_data[symbol][timeframe]["ema_medium"]
            ema_long = self.indicators_data[symbol][timeframe]["ema_long"]

            # Проверяем наличие данных
            if ema_short.empty or ema_medium.empty or ema_long.empty:
                return

            # Получаем последние значения
            last_short = ema_short.iloc[-1]
            last_medium = ema_medium.iloc[-1]
            last_long = ema_long.iloc[-1]

            # Определяем направление тренда
            if last_short > last_medium > last_long:
                self.trend_direction[symbol] = "bullish"
            elif last_short < last_medium < last_long:
                self.trend_direction[symbol] = "bearish"
            else:
                self.trend_direction[symbol] = "neutral"

        except Exception as e:
            logger.error(
                f"Ошибка при определении тренда для {symbol} на {timeframe}: {str(e)}"
            )

    @async_handle_error
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы на основе текущих рыночных данных.

        Returns:
            Словарь с сигналами для каждого символа
        """
        # Рассчитываем индикаторы
        await self._calculate_indicators()

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

                # Получаем текущую цену
                current_price = ticker.get("last", 0)
                if current_price <= 0:
                    continue

                # Генерируем сигналы для всех таймфреймов
                timeframe_signals = {}
                for timeframe in self.timeframes:
                    timeframe_signals[timeframe] = self._generate_timeframe_signals(
                        symbol, timeframe, current_price
                    )

                # Комбинируем сигналы с разных таймфреймов
                combined_signal = self._combine_timeframe_signals(
                    symbol, timeframe_signals
                )

                if combined_signal:
                    signals[symbol] = combined_signal

            except Exception as e:
                logger.error(f"Ошибка при генерации сигналов для {symbol}: {str(e)}")

        return signals

    def _generate_timeframe_signals(
        self, symbol: str, timeframe: str, current_price: float
    ) -> Dict[str, Any]:
        """
        Генерирует торговые сигналы для конкретного таймфрейма.

        Args:
            symbol: Символ для торговли
            timeframe: Таймфрейм для анализа
            current_price: Текущая цена

        Returns:
            Словарь с сигналами
        """
        signals = {}

        try:
            # Проверяем наличие рассчитанных индикаторов
            if (
                symbol not in self.indicators_data
                or timeframe not in self.indicators_data[symbol]
            ):
                return {}

            indicators = self.indicators_data[symbol][timeframe]

            # Сигналы от RSI
            if "rsi" in indicators and not indicators["rsi"].empty:
                last_rsi = indicators["rsi"].iloc[-1]

                # Проверяем условия перекупленности/перепроданности
                if last_rsi is not None:
                    if last_rsi < self.strategy_config["rsi_oversold"]:
                        signals["rsi"] = "buy"
                    elif last_rsi > self.strategy_config["rsi_overbought"]:
                        signals["rsi"] = "sell"
                    else:
                        signals["rsi"] = "hold"

            # Сигналы от MACD
            if all(k in indicators for k in ["macd", "macd_signal", "macd_histogram"]):
                if not indicators["macd"].empty and not indicators["macd_signal"].empty:
                    last_macd = indicators["macd"].iloc[-1]
                    last_signal = indicators["macd_signal"].iloc[-1]
                    prev_macd = (
                        indicators["macd"].iloc[-2]
                        if len(indicators["macd"]) > 1
                        else None
                    )
                    prev_signal = (
                        indicators["macd_signal"].iloc[-2]
                        if len(indicators["macd_signal"]) > 1
                        else None
                    )

                    # Проверяем пересечение MACD и сигнальной линии
                    if prev_macd is not None and prev_signal is not None:
                        if last_macd > last_signal and prev_macd <= prev_signal:
                            signals["macd"] = "buy"
                        elif last_macd < last_signal and prev_macd >= prev_signal:
                            signals["macd"] = "sell"
                        else:
                            signals["macd"] = "hold"

            # Сигналы от полос Боллинджера
            if all(k in indicators for k in ["bb_upper", "bb_middle", "bb_lower"]):
                if (
                    not indicators["bb_upper"].empty
                    and not indicators["bb_lower"].empty
                ):
                    last_upper = indicators["bb_upper"].iloc[-1]
                    last_lower = indicators["bb_lower"].iloc[-1]

                    # Проверяем пробой полос Боллинджера
                    if current_price < last_lower:
                        signals["bollinger"] = "buy"
                    elif current_price > last_upper:
                        signals["bollinger"] = "sell"
                    else:
                        signals["bollinger"] = "hold"

            # Сигналы от EMA
            if all(k in indicators for k in ["ema_short", "ema_medium", "ema_long"]):
                if (
                    not indicators["ema_short"].empty
                    and not indicators["ema_medium"].empty
                    and not indicators["ema_long"].empty
                ):
                    last_short = indicators["ema_short"].iloc[-1]
                    last_medium = indicators["ema_medium"].iloc[-1]
                    last_long = indicators["ema_long"].iloc[-1]

                    # Проверяем пересечение EMA
                    if last_short > last_medium > last_long:
                        signals["ema"] = "buy"
                    elif last_short < last_medium < last_long:
                        signals["ema"] = "sell"
                    else:
                        signals["ema"] = "hold"

            # Сигналы от свечных паттернов (если включено)
            if self.strategy_config["use_patterns"]:
                pattern_signals = []

                # Бычьи паттерны
                if (
                    "bullish_engulfing" in indicators
                    and not indicators["bullish_engulfing"].empty
                ):
                    if indicators["bullish_engulfing"].iloc[-1]:
                        pattern_signals.append("buy")

                if "hammer" in indicators and not indicators["hammer"].empty:
                    if indicators["hammer"].iloc[-1]:
                        pattern_signals.append("buy")

                if (
                    "morning_star" in indicators
                    and not indicators["morning_star"].empty
                ):
                    if indicators["morning_star"].iloc[-1]:
                        pattern_signals.append("buy")

                # Медвежьи паттерны
                if (
                    "bearish_engulfing" in indicators
                    and not indicators["bearish_engulfing"].empty
                ):
                    if indicators["bearish_engulfing"].iloc[-1]:
                        pattern_signals.append("sell")

                if (
                    "shooting_star" in indicators
                    and not indicators["shooting_star"].empty
                ):
                    if indicators["shooting_star"].iloc[-1]:
                        pattern_signals.append("sell")

                if (
                    "evening_star" in indicators
                    and not indicators["evening_star"].empty
                ):
                    if indicators["evening_star"].iloc[-1]:
                        pattern_signals.append("sell")

                # Определяем общий сигнал от паттернов
                if pattern_signals:
                    # Считаем количество бычьих и медвежьих сигналов
                    buy_count = pattern_signals.count("buy")
                    sell_count = pattern_signals.count("sell")

                    if buy_count > sell_count:
                        signals["patterns"] = "buy"
                    elif sell_count > buy_count:
                        signals["patterns"] = "sell"

            return signals

        except Exception as e:
            logger.error(
                f"Ошибка при генерации сигналов для {symbol} на {timeframe}: {str(e)}"
            )
            return {}

    def _combine_timeframe_signals(
        self, symbol: str, timeframe_signals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Комбинирует сигналы с разных таймфреймов.

        Args:
            symbol: Символ для торговли
            timeframe_signals: Словарь с сигналами для каждого таймфрейма

        Returns:
            Комбинированный сигнал
        """
        if not timeframe_signals:
            return {}

        try:
            # Подсчитываем количество бычьих и медвежьих сигналов для каждого индикатора
            indicator_signals = {
                "rsi": {"buy": 0, "sell": 0, "hold": 0},
                "macd": {"buy": 0, "sell": 0, "hold": 0},
                "bollinger": {"buy": 0, "sell": 0, "hold": 0},
                "ema": {"buy": 0, "sell": 0, "hold": 0},
                "patterns": {"buy": 0, "sell": 0, "hold": 0},
            }

            # Учитываем веса таймфреймов (старшие таймфреймы имеют больший вес)
            timeframe_weights = {}
            for i, tf in enumerate(self.timeframes):
                timeframe_weights[tf] = (i + 1) / len(self.timeframes)

            # Подсчитываем сигналы с учетом весов таймфреймов
            for tf, signals in timeframe_signals.items():
                weight = timeframe_weights[tf]

                for indicator, signal in signals.items():
                    if indicator in indicator_signals:
                        indicator_signals[indicator][signal] += weight

            # Определяем финальный сигнал для каждого индикатора
            final_signals = {}
            for indicator, counts in indicator_signals.items():
                max_signal = max(counts, key=counts.get)
                if counts[max_signal] > 0:
                    final_signals[indicator] = max_signal

            # Если нет сигналов, возвращаем пустой словарь
            if not final_signals:
                return {}

            # Подсчитываем общее количество сигналов
            buy_count = sum(1 for signal in final_signals.values() if signal == "buy")
            sell_count = sum(1 for signal in final_signals.values() if signal == "sell")
            hold_count = sum(1 for signal in final_signals.values() if signal == "hold")

            # Требуемое количество подтверждающих сигналов
            required_confirmations = self.strategy_config["signal_confirmation_count"]

            # Формируем финальный сигнал
            if buy_count >= required_confirmations:
                action = "buy"
            elif sell_count >= required_confirmations:
                action = "sell"
            else:
                action = "hold"

            # Проверяем подтверждение тренда, если требуется
            if self.strategy_config["trend_confirmation"]:
                trend = self.trend_direction[symbol]

                # Если тренд не совпадает с сигналом, держим позицию
                if (action == "buy" and trend != "bullish") or (
                    action == "sell" and trend != "bearish"
                ):
                    action = "hold"

            # Если у нас уже есть открытая позиция по этому символу, проверяем сигнал
            if symbol in self.open_positions:
                position = self.open_positions[symbol]
                position_side = position["side"]

                # Если сигнал противоположен стороне позиции, закрываем позицию
                if (position_side == "long" and action == "sell") or (
                    position_side == "short" and action == "buy"
                ):
                    action = "exit"
                # Если сигнал совпадает со стороной позиции или "hold", продолжаем держать
                else:
                    action = "hold"
            # Если позиции нет, и сигнал "hold", не делаем ничего
            elif action == "hold":
                return {}

            # Получаем текущую цену
            ticker = self.market_data.get_ticker(self.exchange_id, symbol)
            current_price = ticker.get("last", 0) if ticker else 0

            # Формируем сигнал
            combined_signal = {
                "symbol": symbol,
                "action": action,
                "price": current_price,
                "indicators": final_signals,
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "hold_signals": hold_count,
                "trend": self.trend_direction[symbol],
                "timestamp": time.time(),
            }

            return combined_signal

        except Exception as e:
            logger.error(f"Ошибка при комбинировании сигналов для {symbol}: {str(e)}")
            return {}

    def analyze_market(self, data):
        """
        Analyzes market data to generate trading signals.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Dictionary with analysis results
        """
        trend = self._determine_trend(data)
        volatility = self._calculate_volatility(data)
        signals = self._generate_signals(trend, volatility, data)
        return signals
    
    def _determine_trend(self, data):
        """Determines market trend based on data"""
        # Fewer branches...
    
    def _calculate_volatility(self, data):
        """Calculates market volatility"""
        # Fewer branches...
    
    def _generate_signals(self, trend, volatility, data):
        """Generates signals based on trend and volatility"""
        # Fewer branches...
