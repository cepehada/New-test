"""
Metrics Module.
Рассчитывает дополнительные торговые метрики: Sortino Ratio,
Calmar Ratio, Omega Ratio и другие показатели эффективности торговли.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Union, Tuple

logger = logging.getLogger("Metrics")


def calculate_sortino_ratio(
    returns: List[float], risk_free: float = 0.02 / 252, periods_per_year: int = 252, 
    mar: Optional[float] = None
) -> Optional[float]:
    """
    Рассчитывает коэффициент Sortino Ratio.

    Формула:
      (mean(returns) - risk_free) / std(negative returns)

    Args:
        returns (List[float]): Список дневных доходностей.
        risk_free (float): Безрисковая ставка (дневная).
        periods_per_year (int): Количество периодов в году (252 для торговых дней).
        mar (Optional[float]): Минимальная приемлемая доходность. Если None, используется risk_free.

    Returns:
        Optional[float]: Sortino Ratio или None, если данных недостаточно.
    """
    if not returns or len(returns) < 2:
        logger.error("Недостаточно данных для Sortino Ratio")
        return None
    
    # Очистка от NaN значений
    returns_arr = np.array(returns)
    returns_arr = returns_arr[~np.isnan(returns_arr)]
    
    if len(returns_arr) < 2:
        logger.error("Недостаточно валидных данных для Sortino Ratio после очистки от NaN")
        return None
    
    # Если MAR не указан, используем безрисковую ставку
    target_return = mar if mar is not None else risk_free
    
    excess = returns_arr - target_return
    mean_excess = np.mean(excess)
    
    # Используем только отрицательные избыточные доходности для downside deviation
    negative_excess = excess[excess < 0]
    
    if negative_excess.size == 0:
        logger.info("Нет отрицательных избыточных доходностей, Sortino = inf")
        return float("inf")
    
    # Корректный расчет downside deviation с использованием среднеквадратичного отклонения отрицательных избытков
    downside_deviation = np.sqrt(np.mean(negative_excess**2))
    
    if downside_deviation == 0:
        logger.error("Downside deviation равен 0, расчет невозможен")
        return None
    
    sortino = mean_excess / downside_deviation
    return sortino * np.sqrt(periods_per_year)


def calculate_calmar_ratio(
    roi: float, max_drawdown: float, years: float = 1.0
) -> Optional[float]:
    """
    Рассчитывает коэффициент Calmar Ratio.

    Формула:
      Calmar = (Annualized ROI) / max_drawdown

    Args:
        roi (float): ROI в процентах.
        max_drawdown (float): Максимальная просадка (%) портфеля.
        years (float): Количество лет, за которые достигнут ROI.

    Returns:
        Optional[float]: Calmar Ratio или None, если max_drawdown равен 0.
    """
    if max_drawdown <= 0:
        logger.error(f"Max Drawdown должен быть положительным, получено: {max_drawdown}")
        return None
    
    # Аннуализируем ROI, если период не равен одному году
    annualized_roi = roi if years == 1.0 else ((1 + roi/100)**(1/years) - 1) * 100
    
    return annualized_roi / max_drawdown


def calculate_omega_ratio(
    returns: List[float], threshold: float = 0.0, periods_per_year: int = 252
) -> Optional[float]:
    """
    Рассчитывает коэффициент Omega.
    
    Omega ratio - отношение вероятностно-взвешенных прибылей к вероятностно-взвешенным убыткам.
    
    Формула:
      Omega = sum(max(returns - threshold, 0)) / sum(max(threshold - returns, 0))
    
    Args:
        returns (List[float]): Список дневных доходностей.
        threshold (float): Пороговое значение для определения прибыли/убытка.
        periods_per_year (int): Количество периодов в году (252 для торговых дней).
        
    Returns:
        Optional[float]: Omega Ratio или None, если данных недостаточно.
    """
    if not returns or len(returns) < 2:
        logger.error("Недостаточно данных для Omega Ratio")
        return None
    
    # Очистка от NaN значений
    returns_arr = np.array(returns)
    returns_arr = returns_arr[~np.isnan(returns_arr)]
    
    if len(returns_arr) < 2:
        logger.error("Недостаточно валидных данных для Omega Ratio после очистки от NaN")
        return None
    
    excess = returns_arr - threshold
    
    # Сумма положительных избыточных доходностей
    sum_positive = np.sum(np.maximum(excess, 0))
    
    # Сумма отрицательных избыточных доходностей (преобразованных в положительные)
    sum_negative = np.sum(np.maximum(-excess, 0))
    
    if sum_negative == 0:
        logger.info("Нет отрицательных избыточных доходностей относительно порога, Omega = inf")
        return float("inf")
    
    omega = sum_positive / sum_negative
    return omega


def calculate_metrics_pack(
    returns: List[float], 
    roi: float, 
    max_drawdown: float, 
    years: float = 1.0,
    risk_free: float = 0.02 / 252, 
    periods_per_year: int = 252
) -> Dict[str, Union[float, str]]:
    """
    Рассчитывает комплексный набор метрик.
    
    Args:
        returns (List[float]): Список дневных доходностей.
        roi (float): ROI в процентах.
        max_drawdown (float): Максимальная просадка (%) портфеля.
        years (float): Количество лет, за которые достигнут ROI.
        risk_free (float): Безрисковая ставка (дневная).
        periods_per_year (int): Количество периодов в году.
        
    Returns:
        Dict[str, Union[float, str]]: Словарь со всеми метриками.
    """
    metrics = {}
    
    # Базовые показатели
    metrics["roi"] = roi
    metrics["max_drawdown"] = max_drawdown
    
    # Sharpe Ratio
    try:
        # Очистка от NaN значений
        returns_arr = np.array(returns)
        returns_arr = returns_arr[~np.isnan(returns_arr)]
        
        if len(returns_arr) >= 2:
            excess = returns_arr - risk_free
            sharpe = (np.mean(excess) / np.std(excess, ddof=1)) * np.sqrt(periods_per_year)
            metrics["sharpe_ratio"] = sharpe
        else:
            metrics["sharpe_ratio"] = "N/A"
    except Exception as e:
        logger.error(f"Ошибка при расчете Sharpe Ratio: {str(e)}")
        metrics["sharpe_ratio"] = "Error"
    
    # Sortino Ratio
    try:
        sortino = calculate_sortino_ratio(returns, risk_free, periods_per_year)
        metrics["sortino_ratio"] = sortino if sortino is not None else "N/A"
    except Exception as e:
        logger.error(f"Ошибка при расчете Sortino Ratio: {str(e)}")
        metrics["sortino_ratio"] = "Error"
    
    # Calmar Ratio
    try:
        calmar = calculate_calmar_ratio(roi, max_drawdown, years)
        metrics["calmar_ratio"] = calmar if calmar is not None else "N/A"
    except Exception as e:
        logger.error(f"Ошибка при расчете Calmar Ratio: {str(e)}")
        metrics["calmar_ratio"] = "Error"
    
    # Omega Ratio
    try:
        omega = calculate_omega_ratio(returns, risk_free, periods_per_year)
        metrics["omega_ratio"] = omega if omega is not None else "N/A"
    except Exception as e:
        logger.error(f"Ошибка при расчете Omega Ratio: {str(e)}")
        metrics["omega_ratio"] = "Error"
    
    # Annual Volatility
    try:
        returns_arr = np.array(returns)
        returns_arr = returns_arr[~np.isnan(returns_arr)]
        
        if len(returns_arr) >= 2:
            annual_vol = np.std(returns_arr, ddof=1) * np.sqrt(periods_per_year)
            metrics["annual_volatility"] = annual_vol
        else:
            metrics["annual_volatility"] = "N/A"
    except Exception as e:
        logger.error(f"Ошибка при расчете Annual Volatility: {str(e)}")
        metrics["annual_volatility"] = "Error"
    
    return metrics


if __name__ == "__main__":
    # Пример тестовых данных
    sample_returns = [0.01, -0.005, 0.015, -0.02, 0.005, 0.01]
    sortino = calculate_sortino_ratio(sample_returns)
    roi = 20.0  # ROI в процентах
    max_dd = 15.0  # Максимальная просадка в процентах
    calmar = calculate_calmar_ratio(roi, max_dd)
    omega = calculate_omega_ratio(sample_returns)
    
    print(f"Sortino Ratio: {sortino}")
    print(f"Calmar Ratio: {calmar}")
    print(f"Omega Ratio: {omega}")
    
    # Расчет комплексного набора метрик
    metrics_pack = calculate_metrics_pack(sample_returns, roi, max_dd)
    print("\nКомплексный набор метрик:")
    for name, value in metrics_pack.items():
        print(f"{name}: {value}")