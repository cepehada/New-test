"""
Technical Analysis Module.
Реализует функции для определения пробития линий
Боллинджера и анализа нескольких таймфреймов.
"""

import logging
from typing import List, Dict, Any
from project.technicals.indicators import calculate_bollinger_bands

logger = logging.getLogger("TechnicalAnalysis")

def bollinger_breakout_signal(prices: List[float],
                              period: int = 20,
                              std_dev: float = 2.0) -> Dict[str, Any]:
    """
    Определяет, пробита ли цена верхняя или нижняя линия Боллинджера.
    
    Args:
        prices (List[float]): Список цен закрытия.
        period (int): Период расчёта SMA.
        std_dev (float): Множитель стандартного отклонения.
    
    Returns:
        Dict[str, Any]: Сигнал пробития с линиями:
            {
              "breakout": bool,
              "direction": "up" или "down",
              "lower": float,
              "center": float,
              "upper": float
            }
    """
    lower, center, upper = calculate_bollinger_bands(prices,
                                                     period,
                                                     std_dev)
    if lower is None or center is None or upper is None:
        logger.error("Ошибка расчёта линий Боллинджера")
        return {"breakout": False}
    last_price = prices[-1]
    if last_price > upper:
        signal = {"breakout": True, "direction": "up",
                  "lower": lower, "center": center, "upper": upper}
    elif last_price < lower:
        signal = {"breakout": True, "direction": "down",
                  "lower": lower, "center": center, "upper": upper}
    else:
        signal = {"breakout": False,
                  "lower": lower, "center": center, "upper": upper}
    logger.info(f"Bollinger breakout signal: {signal}")
    return signal

def multi_timeframe_analysis(prices_dict: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    Выполняет анализ нескольких таймфреймов для подтверждения сигнала.
    
    Args:
        prices_dict (Dict[str, List[float]]): Словарь, где ключи – таймфреймы,
            например, "1m", "5m", "1h", а значения – списки цен закрытия.
    
    Returns:
        Dict[str, Any]: Сводный анализ с сигналами для каждого таймфрейма и
            консенсусом:
            {
              "1m": signal,
              "5m": signal,
              "1h": signal,
              "consensus": bool
            }
    """
    results = {}
    consensus = True
    breakout_direction = None
    for timeframe, prices in prices_dict.items():
        signal = bollinger_breakout_signal(prices)
        results[timeframe] = signal
        if signal.get("breakout"):
            direction = signal.get("direction")
            if breakout_direction is None:
                breakout_direction = direction
            elif breakout_direction != direction:
                consensus = False
        else:
            consensus = False
    results["consensus"] = consensus
    logger.info(f"Multi-timeframe analysis: {results}")
    return results

if __name__ == "__main__":
    # Пример использования для нескольких таймфреймов
    prices_1m = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
    prices_5m = [100, 102, 101, 104, 107, 105, 108, 110, 109, 112,
                 115, 113, 116, 118, 117, 120, 122, 121, 123, 125]
    prices_1h = [100, 105, 102, 108, 110, 115, 113, 117, 120, 118,
                 122, 125, 123, 128, 130, 127, 132, 135, 133, 137]
    prices_dict = {
        "1m": prices_1m,
        "5m": prices_5m,
        "1h": prices_1h
    }
    analysis = multi_timeframe_analysis(prices_dict)
    print(analysis)
