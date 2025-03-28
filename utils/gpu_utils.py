"""
Облегченная версия утилит для работы с вычислениями.
Использует только CPU для экономии ресурсов.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, Optional

from project.utils.logging_utils import setup_logger

logger = setup_logger("computation_utils")

# Флаг доступности ускорения
ACCELERATION_AVAILABLE = False

def get_device_info() -> Dict:
    """
    Возвращает информацию о вычислительных устройствах
    
    Returns:
        Dict: Информация об устройствах
    """
    return {
        "device": "cpu",
        "acceleration_available": False,
        "reason": "GPU acceleration disabled to save resources"
    }

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Рассчитывает простое скользящее среднее
    
    Args:
        data: Исходные данные
        window: Размер окна
        
    Returns:
        np.ndarray: Значения скользящего среднего
    """
    return np.array(pd.Series(data).rolling(window=window, min_periods=1).mean())

def crossover(data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
    """
    Определяет пересечения двух линий
    
    Args:
        data1: Первый набор данных
        data2: Второй набор данных
        
    Returns:
        np.ndarray: Значения 1 (пересечение вверх), -1 (пересечение вниз) или 0
    """
    result = np.zeros(len(data1))
    
    for i in range(1, len(data1)):
        if data1[i-1] < data2[i-1] and data1[i] >= data2[i]:
            result[i] = 1
        elif data1[i-1] > data2[i-1] and data1[i] <= data2[i]:
            result[i] = -1
    
    return result

def is_acceleration_available() -> bool:
    """
    Проверяет доступность ускорения вычислений
    
    Returns:
        bool: False, т.к. ускорение отключено
    """
    return ACCELERATION_AVAILABLE

logger.info("Using CPU-only computation utils (GPU acceleration disabled)")
