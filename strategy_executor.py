"""
Strategy Executor.
Реализует выполнение торговых стратегий на основе входящего сигнала.
Выбирает стратегию и вызывает соответствующие функции ордер-экзекуции,
используя параметр exchange из сигнала или значение по умолчанию.
"""

import asyncio
import logging
from typing import Dict, Any

from project.config import load_config
config = load_config()
from project.trade_executor.order_executor import (
    dynamic_twap_order,
    dynamic_iceberg_order,
    dynamic_stop_loss_by_atr,
    partial_close_trade,
    limit_if_touched_order,
    exit_on_time_condition,
    safe_exit_on_huge_spike
)
from project.trade_executor.risk_aware_executor import (
    execute_risk_aware_order
)

logger = logging.getLogger("StrategyExecutor")

async def execute_strategy(signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выполняет торговую стратегию на основе входящего сигнала.
    
    Args:
        signal (Dict[str, Any]): Сигнал, содержащий параметры для стратегии,
            например:
            {
                "type": "twap",
                "exchange": "bybit",
                "symbol": "BTC/USDT",
                "side": "buy",
                "amount": 0.1,
                "duration": 60,
                "slices": 6,
                ... (дополнительные параметры)
            }
    
    Returns:
        Dict[str, Any]: Результат исполнения стратегии.
    
    Raises:
        Exception: При ошибке исполнения стратегии.
    """
    strategy_type = signal.get("type", "default")
    exchange_id = signal.get("exchange", "bybit")
    symbol = signal.get("symbol", "BTC/USDT")
    side = signal.get("side", "buy")
    amount = signal.get("amount", 0.1)
    
    result = {}
    try:
        if strategy_type == "twap":
            duration = signal.get("duration", 60)
            slices = signal.get("slices", 6)
            result = await dynamic_twap_order(
                exchange_id, symbol, side, amount, duration, slices
            )
        elif strategy_type == "iceberg":
            chunk_size = signal.get("chunk_size", 0.05)
            result = await dynamic_iceberg_order(
                exchange_id, symbol, side, amount, chunk_size
            )
        elif strategy_type == "stoploss":
            atr_multiplier = signal.get("atr_multiplier", 1.5)
            result = await dynamic_stop_loss_by_atr(
                exchange_id, symbol, side, amount, atr_multiplier
            )
        elif strategy_type == "partial_close":
            percentage = signal.get("percentage", 50)
            result = await partial_close_trade(
                exchange_id, symbol, side, amount, percentage
            )
        elif strategy_type == "limit_if_touched":
            trigger_price = signal.get("trigger_price")
            limit_price = signal.get("limit_price")
            if trigger_price is None or limit_price is None:
                raise ValueError("Не указаны trigger_price или limit_price")
            result = await limit_if_touched_order(
                exchange_id, symbol, side, amount, trigger_price, limit_price
            )
        elif strategy_type == "exit_time":
            duration = signal.get("duration", 60)
            result = await exit_on_time_condition(
                exchange_id, symbol, side, amount, duration
            )
        elif strategy_type == "safe_exit":
            spike_threshold = signal.get("spike_threshold", 5.0)
            result = await safe_exit_on_huge_spike(
                exchange_id, symbol, side, amount, spike_threshold
            )
        else:
            # Если тип неизвестен, используем риск-ориентированное исполнение
            result = await execute_risk_aware_order({
                "exchange": exchange_id,
                "symbol": symbol,
                "side": side,
                "amount": amount
            })
        logger.info(f"Стратегия {strategy_type} выполнена: {result}")
    except Exception as e:
        logger.error(f"Ошибка стратегии {strategy_type}: {e}")
        raise
    return result
