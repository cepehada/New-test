"""
Модуль для работы с биржами через библиотеку CCXT.
Предоставляет унифицированный интерфейс для работы с различными биржами.
"""

# Стандартные импорты
import asyncio
import time
from typing import Dict, List, Any, Optional

# Сторонние импорты
import ccxt.async_support as ccxt_async

# Внутренние импорты
from project.config import get_config
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Кэш для экземпляров бирж
_exchange_instances = {}


async def create_exchange(exchange_id: str, config=None) -> ccxt_async.Exchange:
    """
    Создает экземпляр биржи CCXT.
    
    Args:
        exchange_id: Идентификатор биржи
        config: Конфигурация (если None, будет использована глобальная)
        
    Returns:
        Экземпляр биржи
    """
    # Получаем конфигурацию, если не передана
    if config is None:
        config = get_config()
    
    # Создаем экземпляр биржи
    try:
        exchange_class = getattr(ccxt_async, exchange_id)
        
        # Получаем настройки API для биржи
        api_key = config.get(f"{exchange_id.upper()}_API_KEY", "")
        api_secret = config.get(f"{exchange_id.upper()}_API_SECRET", "")
        
        # Создаем настройки
        exchange_options = {
            'apiKey': api_key,
            'secret': api_secret,
            'timeout': 30000,  # 30 секунд таймаут
            'enableRateLimit': True,  # Включаем ограничение скорости запросов
        }
        
        # Добавляем дополнительные опции для некоторых бирж
        if exchange_id == 'binance':
            exchange_options['options'] = {
                'adjustForTimeDifference': True,
                'recvWindow': 60000,  # 60 секунд окно для запросов
            }
        elif exchange_id == 'kucoin':
            exchange_options['options'] = {
                'versions': {
                    'public': {'GET': {'orderBook/level2_100': 'v1'}},
                }
            }
            
        # Настройки тестовой сети (если используется)
        if config.get('ENABLE_PAPER_TRADING', False):
            logger.info("Используется тестовая сеть для биржи %s", exchange_id)
            if exchange_id == 'binance':
                exchange_options['options']['defaultType'] = 'future'
                exchange_options['options']['test'] = True
            elif exchange_id == 'bybit':
                exchange_options['testnet'] = True
                
        # Создаем экземпляр биржи
        exchange = exchange_class(exchange_options)
        
        # Устанавливаем параметры
        exchange.verbose = False  # Отключаем подробное логирование CCXT
        
        # Возвращаем экземпляр
        logger.info("Создан экземпляр биржи %s", exchange_id)
        return exchange
        
    except Exception as e:
        logger.error("Не удалось создать экземпляр биржи %s: %s", exchange_id, str(e))
        raise


async def get_exchange(exchange_id: str, config=None) -> ccxt_async.Exchange:
    """
    Получает экземпляр биржи из кэша или создает новый.
    
    Args:
        exchange_id: Идентификатор биржи
        config: Конфигурация (если None, будет использована глобальная)
        
    Returns:
        Экземпляр биржи
    """
    global _exchange_instances
    
    # Создаем экземпляр биржи, если его нет в кэше
    if exchange_id not in _exchange_instances:
        try:
            _exchange_instances[exchange_id] = await create_exchange(exchange_id, config)
            logger.info("Добавлен экземпляр биржи %s в кэш", exchange_id)
        except Exception as e:
            logger.error("Не удалось получить экземпляр биржи %s: %s", exchange_id, str(e))
            return None
    
    return _exchange_instances.get(exchange_id)


async def close_exchange(exchange_id: str) -> bool:
    """
    Закрывает соединение с биржей.
    
    Args:
        exchange_id: Идентификатор биржи
        
    Returns:
        True, если соединение успешно закрыто
    """
    global _exchange_instances
    
    # Закрываем соединение, если биржа есть в кэше
    if exchange_id in _exchange_instances:
        try:
            await _exchange_instances[exchange_id].close()
            logger.info("Закрыто соединение с биржей %s", exchange_id)
            del _exchange_instances[exchange_id]
            return True
        except Exception as e:
            logger.error("Не удалось закрыть соединение с биржей %s: %s", exchange_id, str(e))
    
    return False


async def close_all_exchanges() -> None:
    """
    Закрывает все соединения с биржами.
    """
    global _exchange_instances
    
    # Закрываем все соединения
    for exchange_id in list(_exchange_instances.keys()):
        await close_exchange(exchange_id)
        
    _exchange_instances = {}
    logger.info("Закрыты все соединения с биржами")


async def fetch_ticker(exchange_id: str, symbol: str) -> Dict:
    """
    Получает тикер для указанной пары на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        
    Returns:
        Данные тикера
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return {}
    
    try:
        # Получаем тикер
        ticker = await exchange.fetch_ticker(symbol)
        logger.debug(
            "Получен тикер для %s на %s: last=%.8f, bid=%.8f, ask=%.8f", 
            symbol, exchange_id, 
            ticker.get('last', 0), 
            ticker.get('bid', 0), 
            ticker.get('ask', 0)
        )
        return ticker
    except Exception as e:
        logger.error(
            "Ошибка при получении тикера для %s на %s: %s", 
            symbol, exchange_id, str(e)
        )
        return {}


async def fetch_order_book(
    exchange_id: str, symbol: str, limit: int = 20
) -> Dict:
    """
    Получает стакан заказов для указанной пары на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        limit: Количество уровней стакана
        
    Returns:
        Данные стакана заказов
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return {}
    
    try:
        # Получаем стакан заказов
        orderbook = await exchange.fetch_order_book(symbol, limit)
        logger.debug(
            "Получен стакан заказов для %s на %s: %d bids, %d asks", 
            symbol, exchange_id, 
            len(orderbook.get('bids', [])), 
            len(orderbook.get('asks', []))
        )
        return orderbook
    except Exception as e:
        logger.error(
            "Ошибка при получении стакана заказов для %s на %s: %s", 
            symbol, exchange_id, str(e)
        )
        return {}


async def fetch_ohlcv(
    exchange_id: str, 
    symbol: str, 
    timeframe: str = '1h', 
    limit: int = 100,
    since: Optional[int] = None
) -> List:
    """
    Получает OHLCV данные для указанной пары на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        timeframe: Временной интервал
        limit: Количество свечей
        since: Начальное время (timestamp в мс)
        
    Returns:
        Список OHLCV данных
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return []
    
    try:
        # Получаем OHLCV данные
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        logger.debug(
            "Получены OHLCV данные для %s на %s (%s): %d свечей", 
            symbol, exchange_id, timeframe, len(ohlcv)
        )
        return ohlcv
    except Exception as e:
        logger.error(
            "Ошибка при получении OHLCV данных для %s на %s: %s", 
            symbol, exchange_id, str(e)
        )
        return []


async def fetch_balance(exchange_id: str) -> Dict:
    """
    Получает баланс аккаунта на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        
    Returns:
        Данные баланса
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return {}
    
    try:
        # Получаем баланс
        balance = await exchange.fetch_balance()
        logger.debug("Получен баланс на %s", exchange_id)
        return balance
    except Exception as e:
        logger.error("Ошибка при получении баланса на %s: %s", exchange_id, str(e))
        return {}


async def create_order(
    exchange_id: str,
    symbol: str,
    type_order: str,
    side: str,
    amount: float,
    price: Optional[float] = None,
    params: Dict = None
) -> Dict:
    """
    Создает заказ на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары
        type_order: Тип заказа (market, limit и т.д.)
        side: Сторона (buy, sell)
        amount: Количество
        price: Цена (для лимитных заказов)
        params: Дополнительные параметры
        
    Returns:
        Данные созданного заказа
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return {}
    
    try:
        # Создаем заказ
        params = params or {}
        order = await exchange.create_order(symbol, type_order, side, amount, price, params)
        logger.info(
            "Создан %s заказ на %s для %s: %s %f по цене %s", 
            type_order, exchange_id, symbol, side, 
            amount, price if price else "рыночная"
        )
        return order
    except Exception as e:
        logger.error(
            "Ошибка при создании заказа на %s для %s: %s", 
            exchange_id, symbol, str(e)
        )
        return {}


async def cancel_order(
    exchange_id: str, 
    order_id: str, 
    symbol: str, 
    params: Dict = None
) -> Dict:
    """
    Отменяет заказ на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        order_id: Идентификатор заказа
        symbol: Символ торговой пары
        params: Дополнительные параметры
        
    Returns:
        Данные отмененного заказа
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return {}
    
    try:
        # Отменяем заказ
        params = params or {}
        result = await exchange.cancel_order(order_id, symbol, params)
        logger.info("Отменен заказ %s на %s для %s", order_id, exchange_id, symbol)
        return result
    except Exception as e:
        logger.error(
            "Ошибка при отмене заказа %s на %s: %s", 
            order_id, exchange_id, str(e)
        )
        return {}


async def fetch_order(
    exchange_id: str, 
    order_id: str, 
    symbol: str,
    params: Dict = None
) -> Dict:
    """
    Получает информацию о заказе на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        order_id: Идентификатор заказа
        symbol: Символ торговой пары
        params: Дополнительные параметры
        
    Returns:
        Данные заказа
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return {}
    
    try:
        # Получаем информацию о заказе
        params = params or {}
        order = await exchange.fetch_order(order_id, symbol, params)
        logger.debug(
            "Получена информация о заказе %s на %s для %s: статус=%s", 
            order_id, exchange_id, symbol, order.get('status', 'unknown')
        )
        return order
    except Exception as e:
        logger.error(
            "Ошибка при получении информации о заказе %s на %s: %s", 
            order_id, exchange_id, str(e)
        )
        return {}


async def fetch_open_orders(
    exchange_id: str, 
    symbol: Optional[str] = None,
    params: Dict = None
) -> List:
    """
    Получает список открытых заказов на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары (None для всех пар)
        params: Дополнительные параметры
        
    Returns:
        Список открытых заказов
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return []
    
    try:
        # Получаем открытые заказы
        params = params or {}
        orders = await exchange.fetch_open_orders(symbol, None, None, params)
        logger.debug(
            "Получены открытые заказы на %s%s: найдено %d", 
            exchange_id, f" для {symbol}" if symbol else "", len(orders)
        )
        return orders
    except Exception as e:
        logger.error(
            "Ошибка при получении открытых заказов на %s: %s", 
            exchange_id, str(e)
        )
        return []


async def fetch_closed_orders(
    exchange_id: str, 
    symbol: Optional[str] = None,
    params: Dict = None
) -> List:
    """
    Получает список закрытых заказов на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары (None для всех пар)
        params: Дополнительные параметры
        
    Returns:
        Список закрытых заказов
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return []
    
    try:
        # Получаем закрытые заказы
        params = params or {}
        orders = await exchange.fetch_closed_orders(symbol, None, None, params)
        logger.debug(
            "Получены закрытые заказы на %s%s: найдено %d", 
            exchange_id, f" для {symbol}" if symbol else "", len(orders)
        )
        return orders
    except Exception as e:
        logger.error(
            "Ошибка при получении закрытых заказов на %s: %s", 
            exchange_id, str(e)
        )
        return []


async def fetch_my_trades(
    exchange_id: str, 
    symbol: Optional[str] = None,
    since: Optional[int] = None,
    limit: int = 50,
    params: Dict = None
) -> List:
    """
    Получает список сделок пользователя на бирже.
    
    Args:
        exchange_id: Идентификатор биржи
        symbol: Символ торговой пары (None для всех пар)
        since: Начальное время (timestamp в мс)
        limit: Максимальное количество сделок
        params: Дополнительные параметры
        
    Returns:
        Список сделок
    """
    # Получаем экземпляр биржи
    exchange = await get_exchange(exchange_id)
    
    if not exchange:
        logger.error("Не удалось получить экземпляр биржи %s", exchange_id)
        return []
    
    try:
        # Получаем сделки пользователя
        params = params or {}
        trades = await exchange.fetch_my_trades(symbol, since, limit, params)
        logger.debug(
            "Получены сделки пользователя на %s%s: найдено %d", 
            exchange_id, f" для {symbol}" if symbol else "", len(trades)
        )
        return trades
    except Exception as e:
        logger.error(
            "Ошибка при получении сделок пользователя на %s: %s", 
            exchange_id, str(e)
        )
        return []
