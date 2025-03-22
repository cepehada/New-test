"""
Order Executor.
Реализует исполнение ордеров:
- TWAP: разбивка крупного ордера на равномерные части.
- Iceberg: скрытое исполнение ордеров малыми партиями.
- Dynamic SL: стоп‑лосс по ATR.
- Partial Close: частичное закрытие позиции.
- Limit if Touched: лимитный ордер при достижении триггера.
- Exit on Time: выход по истечении заданного времени.
- Safe Exit: безопасный выход при резком движении цены.
"""

import asyncio
import logging
import time
from typing import Dict, Any

from project.config import load_config
config = load_config()
from project.utils.ccxt_exchanges import ExchangeManager

logger = logging.getLogger("OrderExecutor")
exchange_manager = ExchangeManager(config)


async def dynamic_twap_order(exchange_id: str, symbol: str, side: str,
                              total_amount: float, duration: int,
                              slices: int) -> Dict[str, Any]:
    """
    Разбивает ордер на равномерные части (TWAP) для минимизации влияния.
    
    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): "buy" или "sell".
        total_amount (float): Общий объем ордера.
        duration (int): Время исполнения ордера (сек).
        slices (int): Количество частей.
    
    Returns:
        Dict[str, Any]: Сводный результат ордеров.
    """
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    slice_amount = total_amount / slices
    delay = duration / slices
    orders = []
    for i in range(slices):
        try:
            order = await exch.create_order(symbol, "market", side,
                                            slice_amount)
            orders.append(order)
            logger.info(f"TWAP ордер {i+1}/{slices} выполнен")
        except Exception as e:
            logger.error(f"TWAP ордер {i+1} ошибка: {e}")
        await asyncio.sleep(delay)
    return {"orders": orders}


async def dynamic_iceberg_order(exchange_id: str, symbol: str, side: str,
                                  total_amount: float, chunk_size: float) -> Dict[str, Any]:
    """
    Исполняет Iceberg-ордер, скрывая реальный объем сделки.
    
    Делит общий объем на партии (chunk_size) и отправляет лимитные ордера.
    
    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): "buy" или "sell".
        total_amount (float): Общий объем ордера.
        chunk_size (float): Размер одной партии.
    
    Returns:
        Dict[str, Any]: Результаты исполнения ордеров.
    """
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    executed = []
    remaining = total_amount
    while remaining > 0:
        curr_chunk = min(chunk_size, remaining)
        try:
            order = await exch.create_order(symbol, "limit", side,
                                            curr_chunk)
            executed.append(order)
            logger.info(f"Iceberg ордер на {curr_chunk} выполнен")
        except Exception as e:
            logger.error(f"Ошибка Iceberg ордера: {e}")
        remaining -= curr_chunk
        await asyncio.sleep(1)
    return {"orders": executed}


async def dynamic_stop_loss_by_atr(exchange_id: str, symbol: str, side: str,
                                     amount: float, atr_multiplier: float) -> Dict[str, Any]:
    """
    Устанавливает динамический стоп‑лосс на основе ATR.
    
    Вычисляет ATR, получает текущую цену и рассчитывает рекомендуемый SL,
    затем отправляет ордер типа "stop".
    
    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): "buy" или "sell".
        amount (float): Объем сделки.
        atr_multiplier (float): Множитель ATR.
    
    Returns:
        Dict[str, Any]: Результат установки стоп‑лосса.
    """
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    from project.technicals.enhanced_indicators import calculate_atr
    try:
        atr_val = await calculate_atr(exchange_id, symbol)
        ticker = await exch.fetch_ticker(symbol)
        current_price = ticker.get("last", 0)
        if side.lower() == "buy":
            sl_price = current_price - atr_multiplier * atr_val
        else:
            sl_price = current_price + atr_multiplier * atr_val
        order = await exch.create_order(symbol, "stop", side, amount, sl_price)
        logger.info(
            f"Dynamic SL order выполнен: {order} (ATR: {atr_val}, SL: {sl_price})"
        )
        return order
    except Exception as e:
        logger.error(f"Ошибка dynamic_stop_loss_by_atr: {e}")
        raise


async def partial_close_trade(exchange_id: str, symbol: str, side: str,
                                amount: float, percentage: float) -> Dict[str, Any]:
    """
    Частично закрывает позицию по заданному проценту.
    
    Вычисляет объем для закрытия и отправляет рыночный ордер.
    
    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): "buy" или "sell".
        amount (float): Общий объем позиции.
        percentage (float): Процент закрытия (например, 50).
    
    Returns:
        Dict[str, Any]: Результат частичного закрытия.
    """
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    close_amt = amount * (percentage / 100)
    try:
        order = await exch.create_order(symbol, "market", side, close_amt)
        logger.info(
            f"Partial close: {percentage}% от {amount} для {symbol} выполнен"
        )
        return order
    except Exception as e:
        logger.error(f"Partial close trade error: {e}")
        raise


async def limit_if_touched_order(exchange_id: str, symbol: str, side: str,
                                 amount: float, trigger_price: float,
                                 limit_price: float) -> Dict[str, Any]:
    """
    Размещает лимитный ордер типа "If Touched".
    
    Если текущая цена достигает trigger_price, выставляется лимитный ордер
    по limit_price.
    
    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): "buy" или "sell".
        amount (float): Объем ордера.
        trigger_price (float): Цена срабатывания.
        limit_price (float): Цена лимитного ордера.
    
    Returns:
        Dict[str, Any]: Результат исполнения.
    """
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    try:
        ticker = await exch.fetch_ticker(symbol)
        current_price = ticker.get("last", 0)
        condition = (side.lower() == "buy" and current_price <= trigger_price) or \
                    (side.lower() == "sell" and current_price >= trigger_price)
        if condition:
            order = await exch.create_order(symbol, "limit", side, amount, limit_price)
            logger.info(
                f"Limit if touched order выполнен для {symbol}: {order}"
            )
            return order
        else:
            logger.info(
                f"Trigger price не достигнута для {symbol}: текущая цена {current_price}"
            )
            return {"status": "trigger_not_met", "current_price": current_price}
    except Exception as e:
        logger.error(f"Ошибка лимитного ордера: {e}")
        raise


async def exit_on_time_condition(exchange_id: str, symbol: str, side: str,
                                 amount: float, duration: int) -> Dict[str, Any]:
    """
    Выходит из сделки по истечении заданного времени.
    
    Ожидает duration секунд, затем отправляет рыночный ордер.
    
    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): "buy" или "sell".
        amount (float): Объем для выхода.
        duration (int): Время ожидания (сек).
    
    Returns:
        Dict[str, Any]: Результат исполнения ордера.
    """
    await asyncio.sleep(duration)
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    try:
        order = await exch.create_order(symbol, "market", side, amount)
        logger.info(f"Exit on time order выполнен для {symbol}: {order}")
        return order
    except Exception as e:
        logger.error(f"Ошибка выхода по времени: {e}")
        raise


async def safe_exit_on_huge_spike(exchange_id: str, symbol: str, side: str,
                                  amount: float, spike_threshold: float) -> Dict[str, Any]:
    """
    Выполняет безопасный выход при резком движении цены.
    
    Если за короткий промежуток времени цена изменилась более чем
    на spike_threshold (%), отправляется рыночный ордер.
    
    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        side (str): "buy" или "sell".
        amount (float): Объем позиции.
        spike_threshold (float): Порог изменения цены (%).
    
    Returns:
        Dict[str, Any]: Результат исполнения ордера.
    """
    exch = exchange_manager.get_exchange(exchange_id)
    if not exch:
        raise ValueError(f"Биржа {exchange_id} не найдена")
    try:
        ticker_before = await exch.fetch_ticker(symbol)
        price_before = ticker_before.get("last", 0)
        await asyncio.sleep(1)
        ticker_after = await exch.fetch_ticker(symbol)
        price_after = ticker_after.get("last", 0)
        spike = abs(price_after - price_before) / price_before * 100
        if spike >= spike_threshold:
            order = await exch.create_order(symbol, "market", side, amount)
            logger.info(
                f"Safe exit order выполнен для {symbol} при spike {spike:.2f}%: {order}"
            )
            return order
        else:
            logger.info(
                f"Spike {spike:.2f}% для {symbol} ниже порога, выход не требуется"
            )
            return {"status": "no_spike", "spike": spike}
    except Exception as e:
        logger.error(f"Ошибка safe_exit_on_huge_spike: {e}")
        raise
