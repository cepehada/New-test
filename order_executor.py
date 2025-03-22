"""
Модуль исполнения ордеров.
Создание различных типов ордеров на биржах (TWAP, Iceberg и другие).
"""

import asyncio
import logging
from typing import Dict, Any, List
from project.utils.ccxt_exchanges import ExchangeManager
from project.config import load_config

logger = logging.getLogger("OrderExecutor")
config = load_config()
exchange_manager = ExchangeManager(config)


async def dynamic_twap_order(exchange: str, symbol: str, side: str,
                             total_amount: float, duration: int
                             ) -> Dict[str, Any]:
    """
    Создает динамический TWAP-ордер.

    Args:
        exchange (str): Идентификатор биржи.
        symbol (str): Торговая пара (например, "BTC/USDT").
        side (str): Направление ("buy" или "sell").
        total_amount (float): Общий объем.
        duration (int): Время исполнения ордера в секундах.

    Returns:
        Dict[str, Any]: Результат исполнения ордера.
    """
    exch = exchange_manager.get_exchange(exchange)
    chunk_time = config["trade"]["twap_chunk_time"]
    num_chunks = max(1, duration // chunk_time)
    chunk_amount = total_amount / num_chunks
    orders = []

    for i in range(num_chunks):
        try:
            order = await exch.create_order(
                symbol, "market", side, chunk_amount
            )
            orders.append(order)
            logger.info(f"TWAP-ордер {i+1}/{num_chunks} создан: {order}")
        except Exception as error:
            logger.error(f"Ошибка создания TWAP-ордера: {error}")
        await asyncio.sleep(chunk_time)

    return {"status": "filled", "orders": orders}


async def dynamic_iceberg_order(exchange: str, symbol: str, side: str,
                                total_amount: float, visible_amount: float
                                ) -> Dict[str, Any]:
    """
    Создает динамический Iceberg-ордер.

    Args:
        exchange (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): Направление.
        total_amount (float): Общий объем.
        visible_amount (float): Видимый объем.

    Returns:
        Dict[str, Any]: Результат исполнения ордера.
    """
    exch = exchange_manager.get_exchange(exchange)
    remaining_amount = total_amount
    orders = []

    while remaining_amount > 0:
        current_amount = min(visible_amount, remaining_amount)
        try:
            order = await exch.create_order(
                symbol, "limit", side, current_amount
            )
            orders.append(order)
            remaining_amount -= current_amount
            logger.info(f"Iceberg-ордер создан: {order}")
        except Exception as error:
            logger.error(f"Ошибка создания Iceberg-ордера: {error}")

        await asyncio.sleep(config["trade"]["iceberg_refresh_interval"])

    return {"status": "filled", "orders": orders}


async def dynamic_stop_loss_by_atr(exchange: str, symbol: str, side: str,
                                   amount: float, atr: float,
                                   multiplier: float) -> None:
    """
    Устанавливает стоп-лосс на основе ATR.

    Args:
        exchange (str): Биржа.
        symbol (str): Торговая пара.
        side (str): Направление сделки.
        amount (float): Объем.
        atr (float): Текущее значение ATR.
        multiplier (float): Множитель ATR для стоп-лосса.
    """
    exch = exchange_manager.get_exchange(exchange)
    ticker = await exch.fetch_ticker(symbol)
    last_price = ticker["last"]

    stop_loss_price = (last_price - atr * multiplier if side == "buy"
                       else last_price + atr * multiplier)

    try:
        await exch.create_order(
            symbol, "stop_loss_limit", "sell" if side == "buy" else "buy",
            amount, stop_loss_price, {"stopPrice": stop_loss_price}
        )
        logger.info(f"Стоп-лосс по ATR установлен на цене {stop_loss_price}")
    except Exception as error:
        logger.error(f"Ошибка установки стоп-лосса по ATR: {error}")


async def batch_create_orders(exchange: str, symbol: str, side: str,
                              orders_params: List[Dict[str, Any]]
                              ) -> List[Dict[str, Any]]:
    """
    Создает пакет лимитных ордеров.

    Args:
        exchange (str): Биржа.
        symbol (str): Торговая пара.
        side (str): Направление.
        orders_params (List[Dict[str, Any]]): Параметры ордеров.

    Returns:
        List[Dict[str, Any]]: Список результатов создания ордеров.
    """
    exch = exchange_manager.get_exchange(exchange)
    created_orders = []

    for params in orders_params:
        try:
            order = await exch.create_order(
                symbol, "limit", side, params["amount"], params["price"]
            )
            created_orders.append(order)
            logger.info(f"Лимитный ордер создан: {order}")
        except Exception as error:
            logger.error(f"Ошибка создания лимитного ордера: {error}")

    return created_orders
