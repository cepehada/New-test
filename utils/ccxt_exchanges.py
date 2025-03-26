"""
Модуль для работы с биржевыми API через библиотеку CCXT.
Предоставляет общий интерфейс для взаимодействия с различными криптовалютными биржами.
"""

import ccxt.async_support as ccxt
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from project.config import get_config
from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error, async_with_retry

logger = get_logger(__name__)

# Кэш для хранения экземпляров бирж
_exchange_instances: Dict[str, ccxt.Exchange] = {}


@async_with_retry(
    max_retries=3,
    retry_delay=2.0,
    exceptions=(ccxt.NetworkError, ccxt.ExchangeNotAvailable),
)
async def connect_exchange(exchange_id: str) -> ccxt.Exchange:
    """
    Создает и настраивает соединение с указанной биржей.

    Args:
        exchange_id: Идентификатор биржи (binance, bybit, ...)

    Returns:
        Настроенный экземпляр биржи
    """
    global _exchange_instances

    # Проверяем наличие экземпляра в кэше
    if (
        exchange_id in _exchange_instances
        and not _exchange_instances[exchange_id].closed
    ):
        logger.debug(f"Использование существующего соединения с {exchange_id}")
        return _exchange_instances[exchange_id]

    # Получаем настройки для указанной биржи
    config = get_config()
    try:
        exchange_settings = config.get_exchange_settings(exchange_id)
    except ValueError as e:
        logger.error(f"Ошибка при получении настроек биржи {exchange_id}: {str(e)}")
        raise

    # Проверяем наличие класса биржи в CCXT
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        logger.error(f"Биржа {exchange_id} не поддерживается CCXT")
        raise ValueError(f"Биржа {exchange_id} не поддерживается")

    # Создаем экземпляр биржи с настройками
    exchange_options = {
        "apiKey": exchange_settings.API_KEY,
        "secret": exchange_settings.API_SECRET,
        "timeout": 30000,  # 30 секунд
        "enableRateLimit": True,
        "options": {},
    }

    # Настройка тестовой сети, если включен режим paper trading
    if exchange_settings.TESTNET:
        if exchange_id == "binance":
            exchange_options["options"]["defaultType"] = "future"
            exchange_options["urls"] = {
                "api": {
                    "public": "https://testnet.binancefuture.com/fapi/v1",
                    "private": "https://testnet.binancefuture.com/fapi/v1",
                }
            }
        elif exchange_id == "bybit":
            exchange_options["urls"] = {"api": "https://api-testnet.bybit.com"}

    try:
        exchange = exchange_class(exchange_options)
        logger.info(f"Соединение с биржей {exchange_id} создано")

        # Загружаем рынки для валидации символов и получения лимитов
        await exchange.load_markets()
        logger.debug(f"Рынки для {exchange_id} загружены")

        # Сохраняем в кэш
        _exchange_instances[exchange_id] = exchange
        return exchange

    except ccxt.AuthenticationError as e:
        logger.error(f"Ошибка аутентификации на бирже {exchange_id}: {str(e)}")
        raise
    except ccxt.ExchangeNotAvailable as e:
        logger.error(f"Биржа {exchange_id} недоступна: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при подключении к бирже {exchange_id}: {str(e)}")
        raise


async def close_exchange(exchange_id: str) -> None:
    """
    Закрывает соединение с указанной биржей.

    Args:
        exchange_id: Идентификатор биржи
    """
    global _exchange_instances

    if exchange_id in _exchange_instances:
        try:
            await _exchange_instances[exchange_id].close()
            logger.info(f"Соединение с биржей {exchange_id} закрыто")
            del _exchange_instances[exchange_id]
        except Exception as e:
            logger.error(
                f"Ошибка при закрытии соединения с биржей {exchange_id}: {str(e)}"
            )


async def close_all_exchanges() -> None:
    """
    Закрывает все открытые соединения с биржами.
    """
    global _exchange_instances

    for exchange_id, exchange in list(_exchange_instances.items()):
        try:
            await exchange.close()
            logger.debug(f"Соединение с биржей {exchange_id} закрыто")
        except Exception as e:
            logger.error(
                f"Ошибка при закрытии соединения с биржей {exchange_id}: {str(e)}"
            )

    _exchange_instances.clear()
    logger.info("Все соединения с биржами закрыты")


@async_handle_error
async def fetch_ticker(exchange_id: str, symbol: str) -> Dict[str, Any]:
    """
    Получает текущий тикер для указанного символа.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары (например, BTC/USDT)

    Returns:
        Данные тикера
    """
    exchange = await connect_exchange(exchange_id)

    try:
        ticker = await exchange.fetch_ticker(symbol)
        logger.debug(f"Получен тикер для {symbol} на {exchange_id}")
        return ticker
    except ccxt.BadSymbol as e:
        logger.error(f"Неверный символ {symbol} для биржи {exchange_id}: {str(e)}")
        raise
    except Exception as e:
        logger.error(
            f"Ошибка при получении тикера для {symbol} на {exchange_id}: {str(e)}"
        )
        raise


@async_handle_error
async def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    since: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[List[float]]:
    """
    Получает исторические свечи OHLCV для указанного символа.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        timeframe: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d, ...)
        since: Временная метка начала в миллисекундах
        limit: Ограничение на количество свечей

    Returns:
        Список свечей OHLCV
    """
    exchange = await connect_exchange(exchange_id)

    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        logger.debug(
            f"Получено {len(ohlcv)} свечей {timeframe} для {symbol} на {exchange_id}"
        )
        return ohlcv
    except Exception as e:
        logger.error(
            f"Ошибка при получении OHLCV для {symbol} на {exchange_id}: {str(e)}"
        )
        raise


@async_handle_error
async def fetch_order_book(
    exchange_id: str, symbol: str, limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Получает книгу ордеров для указанного символа.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        limit: Ограничение на количество уровней

    Returns:
        Данные книги ордеров
    """
    exchange = await connect_exchange(exchange_id)

    try:
        order_book = await exchange.fetch_order_book(symbol, limit)
        logger.debug(f"Получена книга ордеров для {symbol} на {exchange_id}")
        return order_book
    except Exception as e:
        logger.error(
            f"Ошибка при получении книги ордеров для {symbol} на {exchange_id}: {str(e)}"
        )
        raise


@async_handle_error
async def fetch_balance(exchange_id: str) -> Dict[str, Any]:
    """
    Получает текущий баланс аккаунта.

    Args:
        exchange_id: Идентификатор биржи

    Returns:
        Данные баланса
    """
    exchange = await connect_exchange(exchange_id)

    try:
        balance = await exchange.fetch_balance()
        # Не логируем весь баланс из соображений безопасности
        logger.debug(f"Получен баланс на {exchange_id}")
        return balance
    except Exception as e:
        logger.error(f"Ошибка при получении баланса на {exchange_id}: {str(e)}")
        raise


@async_handle_error
async def create_order(
    exchange_id: str,
    symbol: str,
    type: str,
    side: str,
    amount: float,
    price: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Создает ордер на указанной бирже.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        type: Тип ордера (market, limit, ...)
        side: Сторона ордера (buy, sell)
        amount: Количество
        price: Цена (для лимитных ордеров)
        params: Дополнительные параметры

    Returns:
        Данные созданного ордера
    """
    exchange = await connect_exchange(exchange_id)
    params = params or {}

    try:
        order = await exchange.create_order(symbol, type, side, amount, price, params)
        logger.info(
            f"Создан ордер {side} {type} на {exchange_id} для {symbol}: "
            f"количество={amount}, цена={price if price else 'рыночная'}"
        )
        return order
    except Exception as e:
        logger.error(f"Ошибка при создании ордера на {exchange_id}: {str(e)}")
        raise


@async_handle_error
async def cancel_order(
    exchange_id: str,
    order_id: str,
    symbol: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Отменяет ордер на указанной бирже.

    Args:
        exchange_id: Идентификатор биржи
        order_id: Идентификатор ордера
        symbol: Символ торговой пары
        params: Дополнительные параметры

    Returns:
        Данные отмененного ордера
    """
    exchange = await connect_exchange(exchange_id)
    params = params or {}

    try:
        result = await exchange.cancel_order(order_id, symbol, params)
        logger.info(f"Отменен ордер {order_id} на {exchange_id} для {symbol}")
        return result
    except Exception as e:
        logger.error(f"Ошибка при отмене ордера {order_id} на {exchange_id}: {str(e)}")
        raise


@async_handle_error
async def fetch_order(
    exchange_id: str,
    order_id: str,
    symbol: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Получает информацию о ордере по его идентификатору.

    Args:
        exchange_id: Идентификатор биржи
        order_id: Идентификатор ордера
        symbol: Символ торговой пары
        params: Дополнительные параметры

    Returns:
        Данные ордера
    """
    exchange = await connect_exchange(exchange_id)
    params = params or {}

    try:
        order = await exchange.fetch_order(order_id, symbol, params)
        logger.debug(f"Получена информация о ордере {order_id} на {exchange_id}")
        return order
    except Exception as e:
        logger.error(
            f"Ошибка при получении информации о ордере {order_id} на {exchange_id}: {str(e)}"
        )
        raise


@async_handle_error
async def fetch_orders(
    exchange_id: str,
    symbol: str,
    since: Optional[int] = None,
    limit: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Получает список ордеров для указанного символа.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        since: Временная метка начала в миллисекундах
        limit: Ограничение на количество ордеров
        params: Дополнительные параметры

    Returns:
        Список ордеров
    """
    exchange = await connect_exchange(exchange_id)
    params = params or {}

    try:
        orders = await exchange.fetch_orders(symbol, since, limit, params)
        logger.debug(f"Получено {len(orders)} ордеров для {symbol} на {exchange_id}")
        return orders
    except Exception as e:
        logger.error(
            f"Ошибка при получении списка ордеров для {symbol} на {exchange_id}: {str(e)}"
        )
        raise


@async_handle_error
async def fetch_open_orders(
    exchange_id: str,
    symbol: Optional[str] = None,
    since: Optional[int] = None,
    limit: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Получает список открытых ордеров.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары (None для всех символов)
        since: Временная метка начала в миллисекундах
        limit: Ограничение на количество ордеров
        params: Дополнительные параметры

    Returns:
        Список открытых ордеров
    """
    exchange = await connect_exchange(exchange_id)
    params = params or {}

    try:
        orders = await exchange.fetch_open_orders(symbol, since, limit, params)
        logger.debug(f"Получено {len(orders)} открытых ордеров на {exchange_id}")
        return orders
    except Exception as e:
        logger.error(
            f"Ошибка при получении списка открытых ордеров на {exchange_id}: {str(e)}"
        )
        raise


@async_handle_error
async def fetch_closed_orders(
    exchange_id: str,
    symbol: Optional[str] = None,
    since: Optional[int] = None,
    limit: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Получает список закрытых ордеров.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары (None для всех символов)
        since: Временная метка начала в миллисекундах
        limit: Ограничение на количество ордеров
        params: Дополнительные параметры

    Returns:
        Список закрытых ордеров
    """
    exchange = await connect_exchange(exchange_id)
    params = params or {}

    try:
        orders = await exchange.fetch_closed_orders(symbol, since, limit, params)
        logger.debug(f"Получено {len(orders)} закрытых ордеров на {exchange_id}")
        return orders
    except Exception as e:
        logger.error(
            f"Ошибка при получении списка закрытых ордеров на {exchange_id}: {str(e)}"
        )
        raise


@async_handle_error
async def fetch_my_trades(
    exchange_id: str,
    symbol: Optional[str] = None,
    since: Optional[int] = None,
    limit: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Получает список сделок пользователя.

    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары (None для всех символов)
        since: Временная метка начала в миллисекундах
        limit: Ограничение на количество сделок
        params: Дополнительные параметры

    Returns:
        Список сделок
    """
    exchange = await connect_exchange(exchange_id)
    params = params or {}

    try:
        trades = await exchange.fetch_my_trades(symbol, since, limit, params)
        logger.debug(f"Получено {len(trades)} сделок на {exchange_id}")
        return trades
    except Exception as e:
        logger.error(f"Ошибка при получении списка сделок на {exchange_id}: {str(e)}")
        raise
