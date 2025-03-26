"""
Модуль для генерации торговых сигналов.
Предоставляет функции для создания сигналов на основе технического анализа.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from project.utils.logging_utils import get_logger
from project.utils.error_handler import handle_error
from project.technical_analysis.indicators import Indicators
from project.technical_analysis.patterns import Patterns

logger = get_logger(__name__)


class SignalGenerator:
    """
    Класс для генерации торговых сигналов на основе технического анализа.
    """

    @staticmethod
    @handle_error
    def macd_crossover(
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе пересечения линий MACD.

        Args:
            data: DataFrame с данными OHLCV
            fast_period: Период быстрой EMA
            slow_period: Период медленной EMA
            signal_period: Период сигнальной линии

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < slow_period + signal_period:
            logger.warning(
                "Недостаточно данных для генерации сигналов MACD с периодами %s, %s, %s",
                fast_period,
                slow_period,
                signal_period,
            )
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные MACD
        macd_data = Indicators.macd(data, fast_period, slow_period, signal_period)
        macd_line = macd_data["macd"]
        signal_line = macd_data["signal"]

        # Генерируем сигналы на основе пересечения
        buy_signal = (macd_line > signal_line) & (
            macd_line.shift(1) <= signal_line.shift(1)
        )
        sell_signal = (macd_line < signal_line) & (
            macd_line.shift(1) >= signal_line.shift(1)
        )

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def rsi_overbought_oversold(
        data: pd.DataFrame,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе перекупленности/перепроданности RSI.

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета RSI
            overbought: Уровень перекупленности
            oversold: Уровень перепроданности

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < period + 1:
            logger.warning(
                "Недостаточно данных для генерации сигналов RSI с периодом %s",
                period,
            )
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные RSI
        rsi = Indicators.relative_strength_index(data, period)

        # Генерируем сигналы на основе уровней
        buy_signal = (rsi < oversold) & (rsi.shift(1) >= oversold)
        sell_signal = (rsi > overbought) & (rsi.shift(1) <= overbought)

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def moving_average_crossover(
        data: pd.DataFrame,
        fast_period: int = 20,
        slow_period: int = 50,
        column: str = "close",
    ) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе пересечения скользящих средних.

        Args:
            data: DataFrame с данными OHLCV
            fast_period: Период быстрой скользящей средней
            slow_period: Период медленной скользящей средней
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < slow_period:
            logger.warning(
                "Недостаточно данных для генерации сигналов MA с периодами %s, %s",
                fast_period,
                slow_period,
            )
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные скользящих средних
        fast_ma = Indicators.moving_average(data, fast_period, column)
        slow_ma = Indicators.moving_average(data, slow_period, column)

        # Генерируем сигналы на основе пересечения
        buy_signal = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        sell_signal = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def bollinger_bands_breakout(
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе пробоя полос Боллинджера.

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета
            std_dev: Множитель стандартного отклонения
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < period:
            logger.warning(
                "Недостаточно данных для генерации сигналов BB с периодом %s",
                period,
            )
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные полос Боллинджера
        bb = Indicators.bollinger_bands(data, period, std_dev, column)
        upper = bb["upper"]
        lower = bb["lower"]

        # Генерируем сигналы на основе пробоя
        buy_signal = (data[column] < lower) & (data[column].shift(1) >= lower.shift(1))
        sell_signal = (data[column] > upper) & (data[column].shift(1) <= upper.shift(1))

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def stochastic_crossover(
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        overbought: float = 80.0,
        oversold: float = 20.0,
    ) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе пересечения линий стохастического осциллятора.

        Args:
            data: DataFrame с данными OHLCV
            k_period: Период для %K
            d_period: Период для %D
            overbought: Уровень перекупленности
            oversold: Уровень перепроданности

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < k_period + d_period:
            logger.warning(
                "Недостаточно данных для генерации сигналов Stochastic с периодами %s, %s",
                k_period,
                d_period,
            )
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные стохастического осциллятора
        stochastic = Indicators.stochastic_oscillator(data, k_period, d_period)
        k = stochastic["k"]
        d = stochastic["d"]

        # Генерируем сигналы на основе пересечения в зонах перекупленности/перепроданности
        buy_signal = (k > d) & (k.shift(1) <= d.shift(1)) & (k < oversold)
        sell_signal = (k < d) & (k.shift(1) >= d.shift(1)) & (k > overbought)

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def pattern_based_signals(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе свечных паттернов.

        Args:
            data: DataFrame с данными OHLCV

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < 3:
            logger.warning(
                "Недостаточно данных для генерации сигналов на основе паттернов"
            )
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные о паттернах
        bullish_engulfing = Patterns.engulfing(data, bullish=True)
        bearish_engulfing = Patterns.engulfing(data, bullish=False)
        hammer = Patterns.hammer(data)
        shooting_star = Patterns.shooting_star(data)
        morning_star = Patterns.morning_star(data)
        evening_star = Patterns.evening_star(data)

        # Генерируем сигналы на основе паттернов
        buy_signal = bullish_engulfing | hammer | morning_star
        sell_signal = bearish_engulfing | shooting_star | evening_star

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def adx_directional_movement(
        data: pd.DataFrame, period: int = 14, threshold: float = 25.0
    ) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе индекса среднего направления движения (ADX).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета
            threshold: Пороговое значение для определения силы тренда

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < period + 1:
            logger.warning(
                "Недостаточно данных для генерации сигналов ADX с периодом %s",
                period,
            )
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные ADX
        adx_data = Indicators.average_directional_index(data, period)
        adx = adx_data["adx"]
        di_plus = adx_data["di_plus"]
        di_minus = adx_data["di_minus"]

        # Генерируем сигналы на основе ADX и индикаторов направления
        strong_trend = adx > threshold
        buy_signal = (
            strong_trend
            & (di_plus > di_minus)
            & (di_plus.shift(1) <= di_minus.shift(1))
        )
        sell_signal = (
            strong_trend
            & (di_plus < di_minus)
            & (di_plus.shift(1) >= di_minus.shift(1))
        )

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def ichimoku_signals(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Генерирует сигналы на основе индикатора Ichimoku Cloud.

        Args:
            data: DataFrame с данными OHLCV

        Returns:
            Словарь с сигналами (buy, sell)
        """
        if len(data) < 52 + 26:  # максимальный период + смещение
            logger.warning("Недостаточно данных для генерации сигналов Ichimoku")
            return {
                "buy": pd.Series(False, index=data.index),
                "sell": pd.Series(False, index=data.index),
            }

        # Получаем данные Ichimoku
        ichimoku = Indicators.ichimoku_cloud(data)
        tenkan_sen = ichimoku["tenkan_sen"]
        kijun_sen = ichimoku["kijun_sen"]
        senkou_span_a = ichimoku["senkou_span_a"]
        senkou_span_b = ichimoku["senkou_span_b"]

        # Генерируем сигналы на основе Ichimoku
        price_above_cloud = (data["close"] > senkou_span_a) & (
            data["close"] > senkou_span_b
        )
        price_below_cloud = (data["close"] < senkou_span_a) & (
            data["close"] < senkou_span_b
        )

        # Сигналы покупки:
        # 1. Tenkan-sen пересекает Kijun-sen снизу вверх (TK-крест)
        # 2. Цена выше облака (сильный тренд)
        tk_cross_bullish = (tenkan_sen > kijun_sen) & (
            tenkan_sen.shift(1) <= kijun_sen.shift(1)
        )
        buy_signal = tk_cross_bullish & price_above_cloud

        # Сигналы продажи:
        # 1. Tenkan-sen пересекает Kijun-sen сверху вниз (TK-крест)
        # 2. Цена ниже облака (сильный тренд)
        tk_cross_bearish = (tenkan_sen < kijun_sen) & (
            tenkan_sen.shift(1) >= kijun_sen.shift(1)
        )
        sell_signal = tk_cross_bearish & price_below_cloud

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def combine_signals(
        signals_list: List[Dict[str, pd.Series]], weights: Optional[List[float]] = None
    ) -> Dict[str, pd.Series]:
        """
        Объединяет несколько сигналов с весами.

        Args:
            signals_list: Список словарей с сигналами
            weights: Список весов для каждого набора сигналов (None для равных весов)

        Returns:
            Словарь с объединенными сигналами (buy, sell)
        """
        if not signals_list:
            logger.warning("Пустой список сигналов для объединения")
            return {"buy": pd.Series(dtype=bool), "sell": pd.Series(dtype=bool)}

        # Проверяем, что все сигналы имеют одинаковый индекс
        index = signals_list[0]["buy"].index
        for signals in signals_list:
            if not signals["buy"].index.equals(index):
                logger.warning("Индексы сигналов не совпадают")
                return {
                    "buy": pd.Series(False, index=index),
                    "sell": pd.Series(False, index=index),
                }

        # Устанавливаем равные веса, если не указаны
        if weights is None:
            weights = [1.0 / len(signals_list)] * len(signals_list)
        elif len(weights) != len(signals_list):
            logger.warning("Количество весов не соответствует количеству сигналов")
            weights = [1.0 / len(signals_list)] * len(signals_list)

        # Инициализируем результаты
        buy_score = pd.Series(0.0, index=index)
        sell_score = pd.Series(0.0, index=index)

        # Объединяем сигналы с весами
        for i, signals in enumerate(signals_list):
            buy_score += signals["buy"].astype(float) * weights[i]
            sell_score += signals["sell"].astype(float) * weights[i]

        # Определяем окончательные сигналы (простой порог 0.5)
        buy_signal = buy_score > 0.5
        sell_signal = sell_score > 0.5

        return {"buy": buy_signal, "sell": sell_signal}

    @staticmethod
    @handle_error
    def filter_signals(
        signals: Dict[str, pd.Series], lookback_period: int = 5
    ) -> Dict[str, pd.Series]:
        """
        Фильтрует сигналы для предотвращения частых переключений.

        Args:
            signals: Словарь с сигналами (buy, sell)
            lookback_period: Период для проверки предыдущих сигналов

        Returns:
            Словарь с отфильтрованными сигналами (buy, sell)
        """
        if "buy" not in signals or "sell" not in signals:
            logger.warning("Некорректный формат сигналов для фильтрации")
            return signals

        buy_signal = signals["buy"].copy()
        sell_signal = signals["sell"].copy()

        # Создаем маски для проверки недавних сигналов
        recent_buy = (
            buy_signal.rolling(window=lookback_period).sum().shift(1).fillna(0) > 0
        )
        recent_sell = (
            sell_signal.rolling(window=lookback_period).sum().shift(1).fillna(0) > 0
        )

        # Фильтруем сигналы
        filtered_buy = buy_signal & (~recent_buy)
        filtered_sell = sell_signal & (~recent_sell)

        return {"buy": filtered_buy, "sell": filtered_sell}

    @staticmethod
    @handle_error
    def generate_all_signals(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Генерирует все типы сигналов и объединяет их.

        Args:
            data: DataFrame с данными OHLCV

        Returns:
            Словарь с объединенными сигналами (buy, sell)
        """
        # Генерируем сигналы разных типов
        macd_signals = SignalGenerator.macd_crossover(data)
        rsi_signals = SignalGenerator.rsi_overbought_oversold(data)
        ma_signals = SignalGenerator.moving_average_crossover(data)
        bb_signals = SignalGenerator.bollinger_bands_breakout(data)
        stoch_signals = SignalGenerator.stochastic_crossover(data)
        pattern_signals = SignalGenerator.pattern_based_signals(data)

        # Создаем список сигналов
        signals_list = [
            macd_signals,
            rsi_signals,
            ma_signals,
            bb_signals,
            stoch_signals,
            pattern_signals,
        ]

        # Определяем веса для разных типов сигналов
        weights = [0.2, 0.15, 0.2, 0.15, 0.15, 0.15]

        # Объединяем сигналы
        combined_signals = SignalGenerator.combine_signals(signals_list, weights)

        # Фильтруем сигналы
        filtered_signals = SignalGenerator.filter_signals(combined_signals)

        return filtered_signals
