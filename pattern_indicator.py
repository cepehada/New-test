"""
Pattern Indicator.
Реализует обнаружение паттерна "Голова и плечи" в ряду цен.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("PatternIndicator")

def detect_head_and_shoulders(
    prices: List[float],
    shoulder_tolerance: float = 0.03,
    min_pattern_length: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Обнаруживает паттерн "Голова и плечи" в списке цен.

    Паттерн определяется по трем локальным пикам, где центральный
    является самой высокой точкой, а плечи примерно равны.

    Args:
        prices (List[float]): Список цен закрытия.
        shoulder_tolerance (float): Допустимая относительная разница
            между плечами (например, 0.03 = 3%).
        min_pattern_length (int): Минимальное число точек для паттерна.

    Returns:
        Optional[Dict[str, Any]]: Если паттерн обнаружен, возвращает
            словарь с индексами и значениями: {
                "left_shoulder": int,
                "head": int,
                "right_shoulder": int,
                "left_peak": float,
                "head_peak": float,
                "right_peak": float,
                "shoulder_diff": float
            },
            иначе None.
    """
    n = len(prices)
    if n < min_pattern_length:
        logger.error("Недостаточно данных для паттерна")
        return None

    # Находим локальные максимумы
    peaks = []
    for i in range(1, n - 1):
        if prices[i] > prices[i - 1] and prices[i] > prices[i + 1]:
            peaks.append(i)
    if len(peaks) < 3:
        logger.info("Пиков недостаточно для паттерна")
        return None

    # Ищем три последовательных пика
    for i in range(len(peaks) - 2):
        left = peaks[i]
        head = peaks[i + 1]
        right = peaks[i + 2]
        if prices[head] <= prices[left] or prices[head] <= prices[right]:
            continue
        # Проверяем равенство плеч: относительная разница
        shoulder_diff = abs(prices[left] - prices[right])
        avg_shoulder = (prices[left] + prices[right]) / 2
        if avg_shoulder == 0:
            continue
        rel_diff = shoulder_diff / avg_shoulder
        if rel_diff > shoulder_tolerance:
            continue
        # Проверяем наличие "спадов" между пиками
        trough_left = min(prices[left:head])
        trough_right = min(prices[head:right + 1])
        if (prices[left] - trough_left) < 0.01 * prices[left] or \
           (prices[right] - trough_right) < 0.01 * prices[right]:
            continue
        pattern = {
            "left_shoulder": left,
            "head": head,
            "right_shoulder": right,
            "left_peak": prices[left],
            "head_peak": prices[head],
            "right_peak": prices[right],
            "shoulder_diff": rel_diff
        }
        logger.info("Паттерн 'Голова и плечи' обнаружен")
        return pattern
    logger.info("Паттерн 'Голова и плечи' не обнаружен")
    return None

if __name__ == "__main__":
    # Пример: ряд цен для проверки алгоритма
    sample_prices = [100, 105, 103, 110, 108, 107, 112, 109, 107, 105,
                     104, 106, 103, 107, 105]
    pattern = detect_head_and_shoulders(sample_prices)
    if pattern:
        print("Паттерн найден:")
        print(pattern)
    else:
        print("Паттерн не обнаружен")
