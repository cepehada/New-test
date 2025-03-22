"""
Модуль продвинутых ордеров.
Реализует функционал трейлинг-стоп ордеров для бирж.
"""

import asyncio
import logging
from typing import Any
from project.utils.ccxt_exchanges import ExchangeManager
from project.config import load_config

logger = logging.getLogger("AdvancedOrders")
config = load_config()
exchange_manager = ExchangeManager(config)


async def trailing_stop(exchange: str, symbol: str, side: str,
                        amount: float, trigger_price: float,
                        trail_percent: float) -> None:
    """
    Выполняет трейлинг-стоп ордер до его срабатывания.

    Args:
        exchange (str): Идентификатор биржи.
        symbol (str): Торговая пара (например, "BTC/USDT").
        side (str): Направление ("buy" или "sell").
        amount (float): Количество актива.
        trigger_price (float): Цена активации трейлинга.
        trail_percent (float): Процент для трейлинга от текущей цены.
    """
    try:
        exch = exchange_manager.get_exchange(exchange)
        triggered = False
        initial_price = (await exch.fetch_ticker(symbol))["last"]
        logger.info(f"Начальная цена {symbol}: {initial_price}")

        while not triggered:
            current_price = (await exch.fetch_ticker(symbol))["last"]
            price_change = abs((current_price - trigger_price) /
                               trigger_price) * 100

            if side.lower() == "sell":
                if current_price <= trigger_price * (
                        1 - trail_percent / 100):
                    triggered = True
            else:  # side = "buy"
                if current_price >= trigger_price * (
                        1 + trail_percent / 100):
                    triggered = True

            logger.info(f"Трейлинг-стоп {symbol}: текущая цена "
                        f"{current_price}, изменение {price_change:.2f}%")

            if triggered:
                order = await exch.create_order(symbol, "market", side, amount)
                logger.info(f"Трейлинг-стоп сработал, ордер исполнен: {order}")
                break

            await asyncio.sleep(config["trade"]["trailing_check_interval"])

    except Exception as error:
        logger.error(f"Ошибка выполнения трейлинг-стоп ордера: {error}")
