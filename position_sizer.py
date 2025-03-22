"""
Position Sizer.
Рассчитывает оптимальный размер позиции на основе капитала, риска,
цены входа, стоп‑лосса и ATR (волатильности).
"""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger("PositionSizer")

def calculate_dynamic_position_size(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_price: float,
    atr: float
) -> Optional[float]:
    """
    Рассчитывает оптимальный размер позиции с учетом реальных данных.
    
    Формула:
      position_size = (account_equity * (risk_percent / 100)) /
                      max(|entry_price - stop_loss_price|, atr)
    
    Если расстояние между ценой входа и стоп‑лоссом равно 0, 
    функция возвращает None.
    
    Args:
        account_equity (float): Общий капитал на счете.
        risk_percent (float): Процент капитала для риска в сделке.
        entry_price (float): Цена входа.
        stop_loss_price (float): Цена стоп‑лосса.
        atr (float): Значение ATR (Average True Range).
    
    Returns:
        Optional[float]: Оптимальный размер позиции или None, если
        расчет невозможен.
    """
    base_risk = abs(entry_price - stop_loss_price)
    risk_per_unit = base_risk if base_risk > atr else atr
    if risk_per_unit == 0:
        logger.error("Риск за единицу равен 0, расчет невозможен.")
        return None
    risk_amount = account_equity * (risk_percent / 100)
    position_size = risk_amount / risk_per_unit
    logger.info(
        f"Капитал: {account_equity}, риск%: {risk_percent}, вход: {entry_price}, "
        f"стоп: {stop_loss_price}, ATR: {atr}, размер позиции: {position_size:.4f}"
    )
    return position_size

if __name__ == "__main__":
    equity = 10000.0
    risk = 1.0          # 1% риска
    entry = 50000.0
    stop = 49500.0
    atr_value = 100.0   # Примерное значение ATR
    size = calculate_dynamic_position_size(equity, risk, entry, stop, atr_value)
    if size is not None:
        print(f"Динамический размер позиции: {size:.4f}")
    else:
        print("Расчет размера позиции невозможен.")
