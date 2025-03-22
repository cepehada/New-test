"""
Var Calculator.
Рассчитывает Value at Risk (VaR) на основе исторических
доходностей с использованием исторической симуляции и
Monte Carlo моделирования.
"""

import numpy as np
import logging
from typing import List

logger = logging.getLogger("VarCalculator")

def calculate_var_historical(returns: List[float],
                             confidence: float = 0.95) -> float:
    """
    Рассчитывает VaR методом исторической симуляции.
    
    Функция принимает реальные исторические доходности и
    вычисляет процентиль, отражающий риск потерь.
    
    Args:
        returns (List[float]): Исторические доходности.
        confidence (float): Уровень доверия, напр. 0.95.
        
    Returns:
        float: Значение VaR (отрицательное число), которое
               отражает риск потерь.
               
    Raises:
        ValueError: Если список доходностей пуст.
    """
    if not returns:
        logger.error("Список доходностей пуст")
        raise ValueError("Список доходностей пуст")
    arr = np.array(returns)
    try:
        percentile = (1 - confidence) * 100
        var_value = np.percentile(arr, percentile)
        logger.info(
            f"Historical VaR для {confidence*100}% доверия: {var_value}"
        )
        return var_value
    except Exception as e:
        logger.error(f"Ошибка расчета Historical VaR: {e}")
        raise

def calculate_var_monte_carlo(returns: List[float],
                              num_simulations: int = 10000,
                              confidence: float = 0.95) -> float:
    """
    Рассчитывает VaR методом Monte Carlo моделирования.
    
    Генерируются случайные доходности на основе среднего и стандартного
    отклонения исторических данных, после чего вычисляется процентиль.
    
    Args:
        returns (List[float]): Исторические доходности.
        num_simulations (int): Количество симуляций (по умолчанию 10000).
        confidence (float): Уровень доверия, напр. 0.95.
        
    Returns:
        float: Значение VaR (отрицательное число).
        
    Raises:
        ValueError: Если список доходностей пуст.
    """
    if not returns:
        logger.error("Список доходностей пуст")
        raise ValueError("Список доходностей пуст")
    try:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        simulated = np.random.normal(mean_return, std_return,
                                      num_simulations)
        percentile = (1 - confidence) * 100
        var_value = np.percentile(simulated, percentile)
        logger.info(
            f"Monte Carlo VaR для {confidence*100}% доверия: {var_value}"
        )
        return var_value
    except Exception as e:
        logger.error(f"Ошибка расчета Monte Carlo VaR: {e}")
        raise

if __name__ == "__main__":
    # Пример расчета VaR
    sample_returns = [0.01, -0.02, 0.005, -0.015, 0.02, -0.01,
                      0.003, -0.005, 0.007, -0.008, 0.012, -0.009]
    try:
        var_hist = calculate_var_historical(sample_returns,
                                            confidence=0.95)
        print(f"Historical VaR: {var_hist:.4f}")
        var_mc = calculate_var_monte_carlo(sample_returns,
                                           num_simulations=10000,
                                           confidence=0.95)
        print(f"Monte Carlo VaR: {var_mc:.4f}")
    except Exception as err:
        print(f"Ошибка: {err}")
