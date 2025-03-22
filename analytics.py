"""
Analytics Module.
Реализует расчет основных торговых метрик:
- Sharpe Ratio (годовой)
- Сортино Ratio (годовой)
- Максимальная просадка (Max Drawdown)
- ROI (Return on Investment)
- Коэффициент выигрыша (Win Rate)
- Среднее соотношение прибыль/убыток (Profit/Loss Ratio)
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger("Analytics")


def calculate_sharpe_ratio(
    returns: List[float], risk_free: float = 0.02 / 252, periods_per_year: int = 252
) -> float:
    """
    Рассчитывает годовой коэффициент Шарпа.

    Args:
        returns (List[float]): Список дневных доходностей.
        risk_free (float): Безрисковая ставка (дневная).
        periods_per_year (int): Количество периодов в году (252 для торговых дней).

    Returns:
        float: Годовой коэффициент Шарпа.
    """
    if len(returns) < 2:
        logger.error("Недостаточно данных для Sharpe Ratio")
        return 0.0
    
    # Преобразование в numpy массив и удаление NaN
    returns_arr = np.array(returns)
    returns_arr = returns_arr[~np.isnan(returns_arr)]
    
    if len(returns_arr) < 2:
        logger.error("Недостаточно валидных данных для Sharpe Ratio после удаления NaN")
        return 0.0
    
    excess = returns_arr - risk_free
    mean_ex = np.mean(excess)
    std_ex = np.std(excess, ddof=1)  # ddof=1 для несмещенной оценки
    
    if std_ex == 0:
        logger.warning("Стандартное отклонение равно 0, возвращаем 0 для Sharpe Ratio")
        return 0.0
    
    daily_sharpe = mean_ex / std_ex
    return daily_sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: List[float], risk_free: float = 0.02 / 252, periods_per_year: int = 252
) -> float:
    """
    Рассчитывает коэффициент Сортино (Sharpe ratio, но только с учетом отрицательной волатильности).

    Args:
        returns (List[float]): Список дневных доходностей.
        risk_free (float): Безрисковая ставка (дневная).
        periods_per_year (int): Количество периодов в году (252 для торговых дней).

    Returns:
        float: Годовой коэффициент Сортино.
    """
    if len(returns) < 2:
        logger.error("Недостаточно данных для Sortino Ratio")
        return 0.0
    
    # Преобразование в numpy массив и удаление NaN
    returns_arr = np.array(returns)
    returns_arr = returns_arr[~np.isnan(returns_arr)]
    
    if len(returns_arr) < 2:
        logger.error("Недостаточно валидных данных для Sortino Ratio после удаления NaN")
        return 0.0
    
    excess = returns_arr - risk_free
    mean_ex = np.mean(excess)
    
    # Вычисляем только отрицательные отклонения (downside deviation)
    negative_returns = excess[excess < 0]
    
    if len(negative_returns) == 0:
        logger.warning("Нет отрицательных доходностей, возвращаем большое положительное значение")
        return 100.0  # Условно большое положительное значение
    
    downside_deviation = np.sqrt(np.mean(negative_returns**2))
    
    if downside_deviation == 0:
        logger.warning("Downside deviation равно 0, возвращаем 0 для Sortino Ratio")
        return 0.0
    
    daily_sortino = mean_ex / downside_deviation
    return daily_sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(values: List[float]) -> Tuple[float, Dict]:
    """
    Рассчитывает максимальную просадку портфеля и возвращает информацию о ней.

    Args:
        values (List[float]): История значений портфеля.

    Returns:
        Tuple[float, Dict]: Максимальная просадка (%) и информация о периоде просадки.
    """
    if not values or len(values) < 2:
        logger.error("История портфеля пуста или недостаточно данных")
        return 0.0, {"start_idx": 0, "end_idx": 0, "peak_idx": 0}
    
    # Преобразование в numpy массив и удаление NaN
    values_arr = np.array(values)
    valid_indices = ~np.isnan(values_arr)
    values_arr = values_arr[valid_indices]
    
    if len(values_arr) < 2:
        logger.error("Недостаточно валидных данных для Max Drawdown после удаления NaN")
        return 0.0, {"start_idx": 0, "end_idx": 0, "peak_idx": 0}
    
    # Вычисление просадки
    peak_idx = 0
    max_dd = 0.0
    max_dd_info = {"start_idx": 0, "end_idx": 0, "peak_idx": 0}
    
    for i in range(len(values_arr)):
        if values_arr[i] > values_arr[peak_idx]:
            peak_idx = i
        
        dd = (values_arr[peak_idx] - values_arr[i]) / values_arr[peak_idx]
        
        if dd > max_dd:
            max_dd = dd
            max_dd_info = {
                "start_idx": peak_idx,
                "end_idx": i,
                "peak_idx": peak_idx,
                "peak_value": values_arr[peak_idx],
                "trough_value": values_arr[i]
            }
    
    return max_dd * 100, max_dd_info


def calculate_roi(initial_equity: float, final_equity: float, days: Optional[int] = None) -> Dict[str, float]:
    """
    Рассчитывает ROI (Return on Investment) в процентах.
    Опционально рассчитывает аннуализированный ROI, если указаны дни.

    Args:
        initial_equity (float): Начальный капитал.
        final_equity (float): Конечный капитал.
        days (Optional[int]): Количество дней инвестирования для аннуализации.

    Returns:
        Dict[str, float]: Словарь с ROI и аннуализированным ROI (если указаны дни).
    """
    result = {}
    
    if initial_equity <= 0:
        logger.error(f"Начальный капитал должен быть положительным, получено: {initial_equity}")
        return {"roi": 0.0}
    
    roi = ((final_equity - initial_equity) / initial_equity) * 100
    result["roi"] = roi
    
    if days and days > 0:
        # Аннуализированный ROI
        ann_roi = ((1 + roi/100) ** (365/days) - 1) * 100
        result["annualized_roi"] = ann_roi
    
    return result


def calculate_win_rate(trades: List[float]) -> Dict[str, Union[float, int]]:
    """
    Рассчитывает статистику выигрышей/проигрышей.

    Args:
        trades (List[float]): Список результатов сделок (прибыль/убыток).

    Returns:
        Dict[str, Union[float, int]]: Словарь со статистикой.
    """
    if not trades:
        logger.error("Список сделок пуст")
        return {"win_rate": 0.0, "win_count": 0, "loss_count": 0, "breakeven_count": 0}
    
    # Преобразование в numpy массив и удаление NaN
    trades_arr = np.array(trades)
    trades_arr = trades_arr[~np.isnan(trades_arr)]
    
    if len(trades_arr) == 0:
        logger.error("Нет валидных сделок после удаления NaN")
        return {"win_rate": 0.0, "win_count": 0, "loss_count": 0, "breakeven_count": 0}
    
    win_count = np.sum(trades_arr > 0)
    loss_count = np.sum(trades_arr < 0)
    breakeven_count = np.sum(trades_arr == 0)
    
    total_trades = len(trades_arr)
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0
    
    # Расчет соотношения средней прибыли к среднему убытку (profit/loss ratio)
    avg_profit = np.mean(trades_arr[trades_arr > 0]) if win_count > 0 else 0
    avg_loss = abs(np.mean(trades_arr[trades_arr < 0])) if loss_count > 0 else 0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
    
    return {
        "win_rate": win_rate,
        "win_count": int(win_count),
        "loss_count": int(loss_count),
        "breakeven_count": int(breakeven_count),
        "total_trades": total_trades,
        "avg_profit": float(avg_profit),
        "avg_loss": float(avg_loss),
        "profit_loss_ratio": float(profit_loss_ratio)
    }


if __name__ == "__main__":
    # Пример тестовых данных для Sharpe Ratio
    sample_returns = [0.01, -0.005, 0.015, -0.02, 0.005, 0.01]
    sharpe = calculate_sharpe_ratio(sample_returns)
    print(f"Sharpe Ratio: {sharpe:.4f}")
    
    # Рассчитываем также Sortino Ratio
    sortino = calculate_sortino_ratio(sample_returns)
    print(f"Sortino Ratio: {sortino:.4f}")

    # Пример тестовой истории портфеля для Max Drawdown
    sample_values = [100, 105, 102, 110, 108, 107, 112, 109]
    max_dd, dd_info = calculate_max_drawdown(sample_values)
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Drawdown info: {dd_info}")

    # Пример расчёта ROI
    roi_data = calculate_roi(1000, 1200, days=30)
    print(f"ROI: {roi_data['roi']:.2f}%")
    if 'annualized_roi' in roi_data:
        print(f"Annualized ROI: {roi_data['annualized_roi']:.2f}%")
    
    # Пример расчета статистики сделок
    sample_trades = [100, -50, 70, -30, 20, 80, -25]
    trade_stats = calculate_win_rate(sample_trades)
    print(f"Win Rate: {trade_stats['win_rate']:.2f}%")
    print(f"Profit/Loss Ratio: {trade_stats['profit_loss_ratio']:.2f}")