"""
Модуль для расчета технических индикаторов.
Предоставляет функции для анализа ценовых данных и создания индикаторов.
"""

# Standard imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
try:
    import numpy as np
    import pandas as pd
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas"])
    import numpy as np
    import pandas as pd

# Local imports
from project.utils.cache_utils import cache
from project.utils.error_handler import handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Indicators:
    """
    Класс для расчета технических индикаторов.
    """

    @staticmethod
    @handle_error
    def moving_average(
        data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.Series:
        """
        Рассчитывает простую скользящую среднюю (SMA).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Series со значениями SMA
        """
        if len(data) < period:
            logger.warning("Недостаточно данных для расчета SMA с периодом %s", period)
            return pd.Series(index=data.index)

        return data[column].rolling(window=period).mean()

    @staticmethod
    @handle_error
    def exponential_moving_average(
        data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.Series:
        """
        Рассчитывает экспоненциальную скользящую среднюю (EMA).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Series со значениями EMA
        """
        if len(data) < period:
            logger.warning("Недостаточно данных для расчета EMA с периодом %s", period)
            return pd.Series(index=data.index)

        return data[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    @handle_error
    def bollinger_bands(
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> Dict[str, pd.Series]:
        """
        Рассчитывает полосы Боллинджера.

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета
            std_dev: Множитель стандартного отклонения
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Словарь с верхней, средней и нижней полосами
        """
        if len(data) < period:
            logger.warning(
                "Недостаточно данных для расчета полос Боллинджера с периодом %s", period
            )
            return {
                "upper": pd.Series(index=data.index),
                "middle": pd.Series(index=data.index),
                "lower": pd.Series(index=data.index),
            }

        middle = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std(ddof=0)

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {"upper": upper, "middle": middle, "lower": lower}

    @staticmethod
    @handle_error
    def relative_strength_index(
        data: pd.DataFrame, period: int = 14, column: str = "close"
    ) -> pd.Series:
        """
        Рассчитывает индекс относительной силы (RSI).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Series со значениями RSI
        """
        if len(data) < period + 1:
            logger.warning("Недостаточно данных для расчета RSI с периодом %s", period)
            return pd.Series(index=data.index)

        # Рассчитываем изменения цены
        delta = data[column].diff()

        # Разделяем положительные и отрицательные изменения
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Рассчитываем среднее значение за период
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Рассчитываем относительную силу
        rs = avg_gain / avg_loss

        # Рассчитываем RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    @handle_error
    def macd(
        data: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ) -> Dict[str, pd.Series]:
        """
        Рассчитывает индикатор MACD (Moving Average Convergence Divergence).

        Args:
            data: DataFrame с данными OHLCV
            fast_period: Период быстрой EMA
            slow_period: Период медленной EMA
            signal_period: Период сигнальной линии
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Словарь с линией MACD, сигнальной линией и гистограммой
        """
        if len(data) < slow_period + signal_period:
            logger.warning(
                "Недостаточно данных для расчета MACD с периодами %s, %s, %s",
                fast_period,
                slow_period,
                signal_period,
            )
            return {
                "macd": pd.Series(index=data.index),
                "signal": pd.Series(index=data.index),
                "histogram": pd.Series(index=data.index),
            }

        # Рассчитываем быструю и медленную EMA
        fast_ema = data[column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data[column].ewm(span=slow_period, adjust=False).mean()

        # Рассчитываем линию MACD
        macd_line = fast_ema - slow_ema

        # Рассчитываем сигнальную линию
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Рассчитываем гистограмму
        histogram = macd_line - signal_line

        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    @handle_error
    def stochastic_oscillator(
        data: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Рассчитывает стохастический осциллятор.

        Args:
            data: DataFrame с данными OHLCV
            k_period: Период для %K
            d_period: Период для %D

        Returns:
            Словарь с линиями %K и %D
        """
        if len(data) < k_period:
            logger.warning(
                "Недостаточно данных для расчета стохастического осциллятора с периодом %s",
                k_period,
            )
            return {"k": pd.Series(index=data.index), "d": pd.Series(index=data.index)}

        # Находим минимумы и максимумы за период
        low_min = data["low"].rolling(window=k_period).min()
        high_max = data["high"].rolling(window=k_period).max()

        # Рассчитываем %K
        k = 100 * ((data["close"] - low_min) / (high_max - low_min))

        # Рассчитываем %D (SMA от %K)
        d = k.rolling(window=d_period).mean()

        return {"k": k, "d": d}

    @staticmethod
    @handle_error
    def average_true_range(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Рассчитывает средний истинный диапазон (ATR).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета

        Returns:
            Series со значениями ATR
        """
        if len(data) < 2:
            logger.warning("Недостаточно данных для расчета ATR")
            return pd.Series(index=data.index)

        # Рассчитываем истинный диапазон
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Рассчитываем ATR
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    @handle_error
    def ichimoku_cloud(
        data: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_span_b_period: int = 52,
        displacement: int = 26,
    ) -> Dict[str, pd.Series]:
        """
        Рассчитывает индикатор Ichimoku Cloud (Облако Ишимоку).

        Args:
            data: DataFrame с данными OHLCV
            tenkan_period: Период для линии Tenkan-sen
            kijun_period: Период для линии Kijun-sen
            senkou_span_b_period: Период для линии Senkou Span B
            displacement: Смещение облака вперед

        Returns:
            Словарь с линиями индикатора
        """
        if len(data) < max(tenkan_period, kijun_period, senkou_span_b_period):
            logger.warning(
                "Недостаточно данных для расчета Ichimoku Cloud с периодами %s, %s, %s",
                tenkan_period,
                kijun_period,
                senkou_span_b_period,
            )
            return {
                "tenkan_sen": pd.Series(index=data.index),
                "kijun_sen": pd.Series(index=data.index),
                "senkou_span_a": pd.Series(index=data.index),
                "senkou_span_b": pd.Series(index=data.index),
                "chikou_span": pd.Series(index=data.index),
            }

        # Tenkan-sen (преобразующая линия): (highest high + lowest low) / 2 for the past tenkan_period
        tenkan_sen = (
            data["high"].rolling(window=tenkan_period).max()
            + data["low"].rolling(window=tenkan_period).min()
        ) / 2

        # Kijun-sen (базовая линия): (highest high + lowest low) / 2 for the past kijun_period
        kijun_sen = (
            data["high"].rolling(window=kijun_period).max()
            + data["low"].rolling(window=kijun_period).min()
        ) / 2

        # Senkou Span A (первая опережающая линия): (Tenkan-sen + Kijun-sen) / 2 displaced forward by displacement periods
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

        # Senkou Span B (вторая опережающая линия): (highest high + lowest low) / 2 for the past senkou_span_b_period displaced forward by displacement periods
        senkou_span_b = (
            (
                data["high"].rolling(window=senkou_span_b_period).max()
                + data["low"].rolling(window=senkou_span_b_period).min()
            )
            / 2
        ).shift(displacement)

        # Chikou Span (запаздывающая линия): current closing price displaced backwards by displacement periods
        chikou_span = data["close"].shift(-displacement)

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
        }

    @staticmethod
    @handle_error
    def volume_weighted_average_price(
        data: pd.DataFrame, period: int = None
    ) -> pd.Series:
        """
        Рассчитывает средневзвешенную цену по объему (VWAP).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета (None для расчета за весь период)

        Returns:
            Series со значениями VWAP
        """
        if "volume" not in data.columns:
            logger.warning("Данные не содержат информацию об объеме для расчета VWAP")
            return pd.Series(index=data.index)

        # Рассчитываем типичную цену
        typical_price = (data["high"] + data["low"] + data["close"]) / 3

        # Рассчитываем произведение цены на объем
        price_volume = typical_price * data["volume"]

        if period is None:
            # Рассчитываем VWAP за весь период
            cumulative_price_volume = price_volume.cumsum()
            cumulative_volume = data["volume"].cumsum()
            vwap = cumulative_price_volume / cumulative_volume
        else:
            # Рассчитываем VWAP за указанный период
            cumulative_price_volume = price_volume.rolling(window=period).sum()
            cumulative_volume = data["volume"].rolling(window=period).sum()
            vwap = cumulative_price_volume / cumulative_volume

        return vwap

    @staticmethod
    @handle_error
    def fibonacci_retracement(
        data: pd.DataFrame, high_period: int = None, low_period: int = None
    ) -> Dict[str, float]:
        """
        Рассчитывает уровни Фибоначчи.

        Args:
            data: DataFrame с данными OHLCV
            high_period: Период для поиска максимума (None для использования всего периода)
            low_period: Период для поиска минимума (None для использования всего периода)

        Returns:
            Словарь с уровнями Фибоначчи
        """
        if len(data) < 2:
            logger.warning("Недостаточно данных для расчета уровней Фибоначчи")
            return {}

        # Находим максимум и минимум
        if high_period:
            high_val = data["high"].rolling(window=high_period).max().iloc[-1]
        else:
            high_val = data["high"].max()

        if low_period:
            low_val = data["low"].rolling(window=low_period).min().iloc[-1]
        else:
            low_val = data["low"].min()

        # Рассчитываем разницу
        diff = high_val - low_val

        # Рассчитываем уровни Фибоначчи
        levels = {
            "0.0": low_val,
            "0.236": low_val + 0.236 * diff,
            "0.382": low_val + 0.382 * diff,
            "0.5": low_val + 0.5 * diff,
            "0.618": low_val + 0.618 * diff,
            "0.786": low_val + 0.786 * diff,
            "1.0": high_val,
        }

        return levels

    @staticmethod
    @handle_error
    def average_directional_index(
        data: pd.DataFrame, period: int = 14
    ) -> Dict[str, pd.Series]:
        """
        Рассчитывает индекс среднего направления движения (ADX).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета

        Returns:
            Словарь с индексом ADX и индикаторами направления
        """
        if len(data) < period + 1:
            logger.warning("Недостаточно данных для расчета ADX с периодом %s", period)
            return {
                "adx": pd.Series(index=data.index),
                "di_plus": pd.Series(index=data.index),
                "di_minus": pd.Series(index=data.index),
            }

        # Рассчитываем изменения цены
        high_diff = data["high"].diff()
        low_diff = -data["low"].diff()

        # Рассчитываем направленное движение
        plus_dm = ((high_diff > 0) & (high_diff > low_diff)) * high_diff
        minus_dm = ((low_diff > 0) & (low_diff > high_diff)) * low_diff

        # Заменяем NaN на 0
        plus_dm = plus_dm.fillna(0)
        minus_dm = minus_dm.fillna(0)

        # Рассчитываем ATR
        atr = Indicators.average_true_range(data, period)

        # Рассчитываем показатели направления
        di_plus = 100 * (plus_dm.rolling(window=period).sum() / atr)
        di_minus = 100 * (minus_dm.rolling(window=period).sum() / atr)

        # Рассчитываем разницу и сумму показателей направления
        di_diff = abs(di_plus - di_minus)
        di_sum = di_plus + di_minus

        # Рассчитываем индекс направления
        dx = 100 * (di_diff / di_sum.replace(0, float("nan")))

        # Рассчитываем ADX
        adx = dx.rolling(window=period).mean()

        return {"adx": adx, "di_plus": di_plus, "di_minus": di_minus}

    @staticmethod
    @handle_error
    def on_balance_volume(data: pd.DataFrame) -> pd.Series:
        """
        Рассчитывает индикатор On Balance Volume (OBV).

        Args:
            data: DataFrame с данными OHLCV

        Returns:
            Series со значениями OBV
        """
        if len(data) < 2 или "volume" not in data.columns:
            logger.warning("Недостаточно данных для расчета OBV")
            return pd.Series(index=data.index)

        # Рассчитываем изменение цены
        price_change = data["close"].diff()

        # Определяем объемы для суммирования
        volume = data["volume"].copy()
        volume.loc[price_change < 0] = -volume.loc[price_change < 0]
        volume.loc[price_change == 0] = 0

        # Рассчитываем OBV
        obv = volume.cumsum()

        return obv

    @staticmethod
    @handle_error
    def accumulation_distribution_line(data: pd.DataFrame) -> pd.Series:
        """
        Рассчитывает линию накопления/распределения (A/D Line).

        Args:
            data: DataFrame с данными OHLCV

        Returns:
            Series со значениями A/D Line
        """
        if len(data) < 1 или "volume" not in data.columns:
            logger.warning("Недостаточно данных для расчета A/D Line")
            return pd.Series(index=data.index)

        # Рассчитываем множитель объема
        high_low = data["high"] - data["low"]
        close_low = data["close"] - data["low"]
        high_close = data["high"] - data["close"]

        # Обрабатываем случай, когда high == low
        high_low = high_low.replace(0, float("nan"))

        money_flow_multiplier = ((close_low - high_close) / high_low).fillna(0)

        # Рассчитываем объем потока денег
        money_flow_volume = money_flow_multiplier * data["volume"]

        # Рассчитываем A/D Line
        ad_line = money_flow_volume.cumsum()

        return ad_line

    @staticmethod
    @handle_error
    def rate_of_change(
        data: pd.DataFrame, period: int = 14, column: str = "close"
    ) -> pd.Series:
        """
        Рассчитывает индикатор скорости изменения (ROC).

        Args:
            data: DataFrame с данными OHLCV
            period: Период для расчета
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Series со значениями ROC
        """
        if len(data) < period:
            logger.warning("Недостаточно данных для расчета ROC с периодом %s", period)
            return pd.Series(index=data.index)

        # Рассчитываем ROC
        roc = ((data[column] / data[column].shift(period)) - 1) * 100

        return roc
