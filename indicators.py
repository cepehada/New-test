"""
Indicators Module.
Вычисляет технические индикаторы: RSI, VWAP и 
Bollinger Bands на основе реальных данных.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("Indicators")

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """
    Вычисляет RSI для списка цен.
    
    Args:
        prices (List[float]): Список цен закрытия.
        period (int): Период расчета RSI.
    
    Returns:
        Optional[float]: Значение RSI или None, если данных
        недостаточно.
    """
    if len(prices) < period + 1:
        logger.error("Недостаточно данных для расчета RSI")
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_vwap(prices: List[float],
                   volumes: List[float]) -> Optional[float]:
    """
    Вычисляет VWAP на основе цен и объемов.
    
    Args:
        prices (List[float]): Список цен.
        volumes (List[float]): Список объемов.
    
    Returns:
        Optional[float]: Значение VWAP или None при ошибке.
    """
    if not prices or not volumes or len(prices) != len(volumes):
        logger.error("Неверные данные для VWAP")
        return None
    total_volume = sum(volumes)
    if total_volume == 0:
        logger.error("Общий объем равен 0")
        return None
    cumulative = sum(p * v for p, v in zip(prices, volumes))
    return cumulative / total_volume

def calculate_bollinger_bands(prices: List[float],
                              period: int = 20,
                              std_dev: float = 2.0) -> Tuple[Optional[float],
                                                           Optional[float],
                                                           Optional[float]]:
    """
    Вычисляет линии Боллинджера: нижнюю, SMA и верхнюю.
    
    Args:
        prices (List[float]): Список цен закрытия.
        period (int): Период для SMA.
        std_dev (float): Множитель стандартного отклонения.
    
    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]:
        (нижняя, центральная, верхняя линии) или (None, None, None)
        при недостатке данных.
    """
    if len(prices) < period:
        logger.error("Недостаточно данных для Боллинджера")
        return None, None, None
    series = pd.Series(prices)
    sma = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    if std is None:
        logger.error("Ошибка расчета стандартного отклонения")
        return None, None, None
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return lower, sma, upper

if __name__ == "__main__":
    sample_prices = [100, 102, 101, 103, 105, 104, 106, 107, 108, 110,
                     109, 111, 112, 113, 115]
    sample_volumes = [10, 12, 11, 13, 15, 14, 16, 17, 18, 20,
                      19, 21, 22, 23, 25]
    rsi_val = calculate_rsi(sample_prices, period=14)
    vwap_val = calculate_vwap(sample_prices, sample_volumes)
    bb_lower, bb_center, bb_upper = calculate_bollinger_bands(
        sample_prices, period=14, std_dev=2
    )
    print(f"RSI: {rsi_val}")
    print(f"VWAP: {vwap_val}")
    print(f"Bollinger Bands: Lower={bb_lower}, Center={bb_center}, Upper={bb_upper}")
