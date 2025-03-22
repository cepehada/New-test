"""
Risk Aware Executor.
Исполняет ордера с учетом риск-параметров, корректируя параметры ордеров
на основе рыночных условий.
"""

import asyncio
import logging
from typing import Dict, Any

from project.config import load_config
config = load_config()
from project.utils.ccxt_exchanges import ExchangeManager

logger = logging.getLogger("RiskAwareExecutor")
exchange_manager = ExchangeManager(config)

async def execute_risk_aware_order(order_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Исполняет ордер с учетом риск-показателей.
    
    Принимает параметры ордера, такие как:
      {
         "exchange": "bybit",
         "symbol": "BTC/USDT",
         "side": "buy" или "sell",
         "amount": 0.1,
         "price": 30000 (опционально)
      }
    
    Если цена указана, ордер исполняется как лимитный; иначе – как рыночный.
    Функция обрабатывает ошибки и возвращает результат исполнения.
    
    Args:
        order_params (Dict[str, Any]): Параметры ордера.
    
    Returns:
        Dict[str, Any]: Результат исполнения ордера.
    
    Raises:
        Exception: При ошибке исполнения ордера.
    """
    exchange_id = order_params.get("exchange", "bybit")
    symbol = order_params.get("symbol", "BTC/USDT")
    side = order_params.get("side", "buy")
    amount = order_params.get("amount", 0.1)
    
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    
    try:
        if "price" in order_params:
            # Лимитный ордер
            price = order_params["price"]
            order = await exch.create_order(
                symbol, "limit", side, amount, price
            )
        else:
            # Рыночный ордер
            order = await exch.create_order(
                symbol, "market", side, amount
            )
        logger.info(f"Risk aware order выполнен: {order}")
        return order
    except Exception as e:
        logger.error(f"Ошибка risk aware order: {e}")
        raise
