"""
Модуль для расчета технических индикаторов.
Предоставляет функции для анализа ценовых данных и создания индикаторов.
"""

# Standard imports
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
    def moving_average(data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Рассчитывает простое скользящее среднее (SMA).

        Args:
            data: DataFrame с данными
            period: Период для расчета
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            pandas.Series: Значения индикатора
        """
        if len(data) < period:
            logger.warning("Недостаточно данных для расчета MA с периодом %d", period)
            return pd.Series(index=data.index)

        return data[column].rolling(window=period).mean()

    @staticmethod
    @handle_error
    def exponential_moving_average(data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Рассчитывает экспоненциальное скользящее среднее (EMA).

        Args:
            data: DataFrame с данными
            period: Период для расчета
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            pandas.Series: Значения индикатора
        """
        if len(data) < period:
            logger.warning("Недостаточно данных для расчета EMA с периодом %d", period)
            return pd.Series(index=data.index)
            
        return data[column].ewm(span=period, adjust=False).mean()

    @staticmethod
    @handle_error
    def relative_strength_index(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Рассчитывает индекс относительной силы (RSI).

        Args:
            data: DataFrame с данными
            period: Период для расчета
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            pandas.Series: Значения индикатора
        """
        if len(data) < period + 1:
            logger.warning("Недостаточно данных для расчета RSI с периодом %d", period)
            return pd.Series(index=data.index)
            
        delta = data[column].diff()
        
        # Получаем положительные и отрицательные изменения
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Рассчитываем среднее значение за период
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Рассчитываем RSI
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)  # Избегаем деления на ноль
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    @staticmethod
    @handle_error
    def macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, 
            signal_period: int = 9, column: str = 'close') -> Dict[str, pd.Series]:
        """
        Рассчитывает индикатор MACD (Moving Average Convergence Divergence).

        Args:
            data: DataFrame с данными
            fast_period: Период быстрой EMA
            slow_period: Период медленной EMA
            signal_period: Период сигнальной линии
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Dict[str, pd.Series]: Словарь с компонентами MACD
        """
        if len(data) < slow_period + signal_period:
            logger.warning(
                "Недостаточно данных для расчета MACD с периодами %d, %d, %d",
                fast_period, slow_period, signal_period
            )
            empty_series = pd.Series(index=data.index)
            return {"macd": empty_series, "signal": empty_series, "histogram": empty_series}
            
        # Рассчитываем быструю и медленную EMA
        fast_ema = Indicators.exponential_moving_average(data, fast_period, column)
        slow_ema = Indicators.exponential_moving_average(data, slow_period, column)
        
        # Рассчитываем линию MACD
        macd_line = fast_ema - slow_ema
        
        # Рассчитываем сигнальную линию
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Рассчитываем гистограмму
        histogram = macd_line - signal_line
        
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    @staticmethod
    @handle_error
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, 
                       column: str = 'close') -> Dict[str, pd.Series]:
        """
        Рассчитывает полосы Боллинджера.

        Args:
            data: DataFrame с данными
            period: Период для расчета
            std_dev: Множитель стандартного отклонения
            column: Столбец для расчета (по умолчанию 'close')

        Returns:
            Dict[str, pd.Series]: Словарь с компонентами полос Боллинджера
        """
        if len(data) < period:
            logger.warning("Недостаточно данных для расчета BB с периодом %d", period)
            empty_series = pd.Series(index=data.index)
            return {"upper": empty_series, "middle": empty_series, "lower": empty_series}
        
        # Рассчитываем среднюю линию (SMA)
        middle = Indicators.moving_average(data, period, column)
        
        # Рассчитываем стандартное отклонение
        rolling_std = data[column].rolling(window=period).std()
        
        # Рассчитываем верхнюю и нижнюю полосы
        upper = middle + (rolling_std * std_dev)
        lower = middle - (rolling_std * std_dev)
        
        return {"upper": upper, "middle": middle, "lower": lower}

    @staticmethod
    @handle_error
    def stochastic_oscillator(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Рассчитывает стохастический осциллятор.

        Args:
            data: DataFrame с данными
            k_period: Период для %K
            d_period: Период для %D

        Returns:
            Dict[str, pd.Series]: Словарь с компонентами стохастика
        """
        if len(data) < k_period + d_period:
            logger.warning(
                "Недостаточно данных для расчета стохастика с периодами %d, %d",
                k_period, d_period
            )
            empty_series = pd.Series(index=data.index)
            return {"k": empty_series, "d": empty_series}
        
        # Находим минимум и максимум за период k_period
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        # Рассчитываем %K
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        
        # Рассчитываем %D (SMA от %K)
        d = k.rolling(window=d_period).mean()
        
        return {"k": k, "d": d}

    @staticmethod
    @handle_error
    def average_directional_index(data: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Рассчитывает индекс среднего направления движения (ADX).

        Args:
            data: DataFrame с данными
            period: Период для расчета

        Returns:
            Dict[str, pd.Series]: Словарь с компонентами ADX
        """
        if len(data) < period * 2:
            logger.warning("Недостаточно данных для расчета ADX с периодом %d", period)
            empty_series = pd.Series(index=data.index)
            return {"adx": empty_series, "di_plus": empty_series, "di_minus": empty_series}
        
        # Рассчитываем True Range
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        # Рассчитываем Directional Movement
        plus_dm = data['high'].diff()
        minus_dm = -data['low'].diff()
        
        # Оставляем только положительные значения
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        # Сглаживаем DirectionalMovement и TrueRange
        smoothed_plus_dm = plus_dm.ewm(span=period, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(span=period, adjust=False).mean()
        smoothed_tr = true_range.ewm(span=period, adjust=False).mean()
        
        # Рассчитываем индексы направления
        di_plus = 100 * (smoothed_plus_dm / smoothed_tr)
        di_minus = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # Рассчитываем DX и ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return {"adx": adx, "di_plus": di_plus, "di_minus": di_minus}

    @staticmethod
    @handle_error
    def ichimoku_cloud(data: pd.DataFrame, conversion_period: int = 9, base_period: int = 26,
                    lagging_span2_period: int = 52, displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Рассчитывает облако Ишимоку.

        Args:
            data: DataFrame с данными
            conversion_period: Период для Tenkan-sen (Conversion Line)
            base_period: Период для Kijun-sen (Base Line)
            lagging_span2_period: Период для Senkou Span B
            displacement: Смещение для Senkou Span A и B

        Returns:
            Dict[str, pd.Series]: Словарь с компонентами облака Ишимоку
        """
        if len(data) < max(conversion_period, base_period, lagging_span2_period) + displacement:
            logger.warning("Недостаточно данных для расчета Ichimoku")
            empty_series = pd.Series(index=data.index)
            return {
                "tenkan_sen": empty_series,
                "kijun_sen": empty_series,
                "senkou_span_a": empty_series,
                "senkou_span_b": empty_series,
                "chikou_span": empty_series
            }
        
        # Рассчитываем Tenkan-sen (Conversion Line)
        period_high = data['high'].rolling(window=conversion_period).max()
        period_low = data['low'].rolling(window=conversion_period).min()
        tenkan_sen = (period_high + period_low) / 2
        
        # Рассчитываем Kijun-sen (Base Line)
        period_high = data['high'].rolling(window=base_period).max()
        period_low = data['low'].rolling(window=base_period).min()
        kijun_sen = (period_high + period_low) / 2
        
        # Рассчитываем Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Рассчитываем Senkou Span B (Leading Span B)
        period_high = data['high'].rolling(window=lagging_span2_period).max()
        period_low = data['low'].rolling(window=lagging_span2_period).min()
        senkou_span_b = ((period_high + period_low) / 2).shift(displacement)
        
        # Рассчитываем Chikou Span (Lagging Span)
        chikou_span = data['close'].shift(-displacement)
        
        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span
        }
    
    @staticmethod
    @handle_error
    def fibonacci_retracement(data: pd.DataFrame, high_period: int = 20, low_period: int = 20) -> Dict[str, float]:
        """
        Рассчитывает уровни коррекции Фибоначчи.

        Args:
            data: DataFrame с данными
            high_period: Период для определения максимума
            low_period: Период для определения минимума

        Returns:
            Dict[str, float]: Словарь с уровнями Фибоначчи
        """
        if len(data) < max(high_period, low_period):
            logger.warning("Недостаточно данных для расчета уровней Фибоначчи")
            return {}
        
        # Находим максимум и минимум за период
        high = data['high'].rolling(window=high_period).max().iloc[-1]
        low = data['low'].rolling(window=low_period).min().iloc[-1]
        
        # Проверяем, что максимум и минимум найдены
        if pd.isna(high) or pd.isna(low) or high == low:
            logger.warning("Не удалось определить максимум и минимум для уровней Фибоначчи")
            return {}
            
        # Рассчитываем уровни коррекции
        diff = high - low
        
        # Стандартные уровни коррекции Фибоначчи
        levels = {
            "0.0": low,
            "0.236": low + diff * 0.236,
            "0.382": low + diff * 0.382,
            "0.5": low + diff * 0.5,
            "0.618": low + diff * 0.618,
            "0.786": low + diff * 0.786,
            "1.0": high,
            # Расширения Фибоначчи
            "1.272": high + diff * 0.272,
            "1.618": high + diff * 0.618,
            "2.0": high + diff,
        }
        
        return levels
