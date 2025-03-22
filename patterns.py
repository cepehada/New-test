"""
Patterns Module.
Рассчитывает уровни коррекции Фибоначчи на основе исторических
цен, используя реальные данные (максимум, минимум).
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger("Patterns")

def fibonacci_retracement(prices: List[float]) -> Optional[Dict[str, float]]:
    """
    Рассчитывает уровни коррекции Фибоначчи для заданного списка цен.
    
    Находит максимум и минимум, затем вычисляет уровни:
      23.6%, 38.2%, 50%, 61.8%, 78.6%.
      
    Args:
        prices (List[float]): Список цен.
        
    Returns:
        Optional[Dict[str, float]]: Словарь уровней, либо None,
                                    если расчет невозможен.
    """
    if not prices:
        logger.error("Список цен пуст, расчет невозможен")
        return None
    high = max(prices)
    low = min(prices)
    diff = high - low
    if diff == 0:
        logger.error("Разница между максимумом и минимумом равна 0")
        return None
    levels = {
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff
    }
    logger.info(f"Fibonacci levels рассчитаны: {levels}")
    return levels

if __name__ == "__main__":
    sample_prices = [100, 102, 105, 103, 107, 110, 108, 112, 115, 111]
    levels = fibonacci_retracement(sample_prices)
    print(f"Fibonacci Retracement Levels: {levels}")
