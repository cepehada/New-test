"""
Enhanced Indicators.
Реализует расширенный расчет индикаторов, включая ATR (Average
True Range) на основе реальных свечных данных.
"""

import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("EnhancedIndicators")

def calculate_atr(candles: List[Dict[str, float]],
                  period: int = 14) -> Optional[float]:
    """
    Рассчитывает ATR (Average True Range) по списку свечей.
    
    Каждая свеча задается словарем с ключами: "high", "low", "close".
    ATR вычисляется как скользящая средняя True Range за указанный период.
    
    Args:
        candles (List[Dict[str, float]]): Список свечей.
        period (int): Период для расчета ATR.
    
    Returns:
        Optional[float]: Значение ATR или None, если данных недостаточно.
    """
    if len(candles) < period + 1:
        logger.error("Недостаточно свечей для расчета ATR")
        return None

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close),
                 abs(low - prev_close))
        true_ranges.append(tr)
    atr = np.mean(true_ranges[-period:])
    logger.info(f"ATR рассчитан: {atr}")
    return atr

if __name__ == "__main__":
    sample_candles = [
        {"high": 110, "low": 100, "close": 105},
        {"high": 112, "low": 102, "close": 108},
        {"high": 115, "low": 107, "close": 110},
        {"high": 116, "low": 108, "close": 112},
        {"high": 118, "low": 110, "close": 115},
        {"high": 120, "low": 112, "close": 118},
        {"high": 121, "low": 113, "close": 119},
        {"high": 123, "low": 115, "close": 121},
        {"high": 125, "low": 117, "close": 123},
        {"high": 126, "low": 118, "close": 124},
        {"high": 128, "low": 120, "close": 126},
        {"high": 130, "low": 122, "close": 128},
        {"high": 132, "low": 124, "close": 130},
        {"high": 133, "low": 125, "close": 131},
        {"high": 135, "low": 127, "close": 133},
    ]
    atr_val = calculate_atr(sample_candles, period=14)
    print(f"Calculated ATR: {atr_val}")
