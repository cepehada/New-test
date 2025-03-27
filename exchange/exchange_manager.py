import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import ccxt.async_support as ccxt
from project.utils.error_handler import ExchangeErrorHandler, handle_exchange_errors
from project.utils.logging_utils import setup_logger
from project.utils.websocket_client import (
    WebSocketClient,
    get_message_processor,
    get_ws_pool,
)

logger = setup_logger("exchange_manager")

# Словарь для хранения экземпляров бирж
exchange_instances = {}

# Список поддерживаемых бирж
SUPPORTED_EXCHANGES = [
    'binance', 'binanceus', 'bybit', 'kucoin', 'okx', 'huobi',
    'mexc', 'bitget', 'gateio', 'kraken', 'bitfinex', 'coinbase'
]

# Словарь соответствия названий бирж между различными библиотеками
EXCHANGE_NAME_MAP = {
    'binance': {'ccxt': 'binance', 'cryptofeed': 'BINANCE'},
    'binanceus': {'ccxt': 'binanceus', 'cryptofeed': 'BINANCE_US'},
    'bybit': {'ccxt': 'bybit', 'cryptofeed': 'BYBIT'},
    'kucoin': {'ccxt': 'kucoin', 'cryptofeed': 'KUCOIN'},
    'okx': {'ccxt': 'okx', 'cryptofeed': 'OKX'},
    'huobi': {'ccxt': 'huobi', 'cryptofeed': 'HUOBI'},
    'mexc': {'ccxt': 'mexc', 'cryptofeed': 'MEXC'},
    'bitget': {'ccxt': 'bitget', 'cryptofeed': 'BITGET'},
    'gateio': {'ccxt': 'gateio', 'cryptofeed': 'GATEIO'},
    'kraken': {'ccxt': 'kraken', 'cryptofeed': 'KRAKEN'},
    'bitfinex': {'ccxt': 'bitfinex', 'cryptofeed': 'BITFINEX'},
    'coinbase': {'ccxt': 'coinbase', 'cryptofeed': 'COINBASE'}
}


class ExchangeConfig:
    """Класс для хранения конфигурации биржи"""

    def __init__(self, exchange_id: str, config: Dict = None):
        """
        Инициализирует конфигурацию биржи

        Args:
            exchange_id: ID биржи
            config: Словарь с конфигурацией
        """
        self.exchange_id = exchange_id
        self.config = config or {}

        # API ключи
        self.api_key = self.config.get('api_key', '')
        self.api_secret = self.config.get('api_secret', '')
        self.api_password = self.config.get('api_password', '')

        # Настройки подключения
        self.timeout = self.config.get('timeout', 30000)
        self.enable_rate_limit = self.config.get('enable_rate_limit', True)
        self.verbose = self.config.get('verbose', False)

        # Настройки торговли
        self.sandbox = self.config.get('sandbox', False)
        self.default_type = self.config.get('default_type', 'spot')

        # WebSocket настройки
        self.use_websocket = self.config.get('use_websocket', True)
        self.ws_type = self.config.get('ws_type', 'public')

        # Прокси настройки
        self.proxy = self.config.get('proxy', '')
        self.proxy_username = self.config.get('proxy_username', '')
        self.proxy_password = self.config.get('proxy_password', '')

    def to_ccxt_config(self) -> Dict:
        """
        Преобразует конфигурацию в формат CCXT

        Returns:
            Dict: Конфигурация для CCXT
        """
        ccxt_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'timeout': self.timeout,
            'enableRateLimit': self.enable_rate_limit,
            'verbose': self.verbose,
            'options': {
                'defaultType': self.default_type
            }
        }

        # Добавляем пароль, если он есть
        if self.api_password:
            ccxt_config['password'] = self.api_password

        # Добавляем прокси, если он есть
        if self.proxy:
            ccxt_config['proxy'] = self.proxy
            if self.proxy_username and self.proxy_password:
                ccxt_config['proxy_username'] = self.proxy_username
                ccxt_config['proxy_password'] = self.proxy_password

        # Дополнительные специфичные настройки для отдельных бирж
        if self.exchange_id == 'binance':
            if self.sandbox:
                ccxt_config['options']['defaultType'] = 'future'
                ccxt_config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binancefuture.com/fapi/v1',
                        'private': 'https://testnet.binancefuture.com/fapi/v1',
                    }
                }
        elif self.exchange_id == 'bybit':
            if self.sandbox:
                ccxt_config['urls'] = {
                    'api': 'https://api-testnet.bybit.com'
                }

        # Настройки для margin/futures
        if self.default_type in ['margin', 'future', 'swap', 'delivery']:
            ccxt_config['options']['defaultType'] = self.default_type

        return ccxt_config


class ExchangeRateLimit:
    """
    Class to manage rate limits for exchange API requests.
    
    Handles limiting the number of requests according to exchange requirements
    and implements backoff strategies when limits are reached.
    """

    def __init__(self, rate_limit: Dict = None):
        """
        Инициализирует объект с ограничениями

        Args:
            rate_limit: Словарь с настройками ограничений
        """
        self.rate_limit = rate_limit or {}

        # Максимальное количество запросов
        self.max_requests = self.rate_limit.get('max_requests', 1200)

        # Период в секундах
        self.period = self.rate_limit.get('period', 60)

        # Штраф за превышение лимита
        self.penalty = self.rate_limit.get('penalty', 2.0)

        # Буфер запасных запросов (%)
        self.buffer = self.rate_limit.get('buffer', 5)

        # Счетчик запросов
        self.request_count = 0

        # Время начала отсчета
        self.start_time = time.time()

        # Время последнего запроса
        self.last_request_time = 0

        # Блокировка для синхронизации
        self.lock = asyncio.Lock()

    async def acquire(self):
        """
        Acquire permission to make a request, waiting if necessary to respect rate limits.
        
        Blocks until a request can be made according to the configured rate limits.
        Updates internal counters when a request is permitted.
        """
        async with self.lock:
            # Calculate remaining time in the current period
            current_time = time.time()
            elapsed = current_time - self.start_time

            # Apply buffer to limit for safety margin
            effective_limit = max(1, self.max_requests - self.buffer)

            # Check if we've reached the rate limit
            if self.request_count >= effective_limit:
                # Calculate time to wait if we've hit the limit
                remaining_time = self.period - elapsed

                if remaining_time > 0:
                    logger.warning(
                        f"Rate limit ({effective_limit} per {self.period}s) reached, waiting {remaining_time:.2f}s")
                    await asyncio.sleep(remaining_time)

                    # Reset counter
                    self.request_count = 0
                    self.start_time = time.time()

            # Increment request counter
            self.request_count += 1

            # Update last request time
            self.last_request_time = time.time()

    async def update_limit(self, rate_limit: Dict):
        """
        Updates rate limit settings.

        Args:
            rate_limit: Dictionary containing rate limit settings.
                Supported keys: 'max_requests', 'period', 'penalty', 'buffer'
        """
        async with self.lock:
            # Update settings
            self.rate_limit = rate_limit
            self.max_requests = rate_limit.get('max_requests', self.max_requests)
            self.period = rate_limit.get('period', self.period)
            self.penalty = rate_limit.get('penalty', self.penalty)
            self.buffer = rate_limit.get('buffer', self.buffer)

    def get_usage(self) -> Dict:
        """
        Returns information about rate limit usage.

        Returns:
            Dict: Rate limit usage information including:
                - used: Number of requests used in current period
                - limit: Maximum number of requests allowed
                - remaining: Number of requests remaining
                - reset_in: Seconds until the limit resets
        """
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Check if period has expired
        if elapsed >= self.period:
            return {
                'used': 0,
                'limit': self.max_requests,
                'remaining': self.max_requests,
                'reset_in': 0
            }

        return {
            'used': self.request_count,
            'limit': self.max_requests,
            'remaining': self.max_requests - self.request_count,
            'reset_in': self.period - elapsed
        }


class ExchangeCache:
    """
    Class for caching data from exchanges.
    
    Provides methods to store and retrieve exchange data with configurable TTL (time-to-live)
    to minimize redundant API calls.
    """

    def __init__(self, cache_ttl: Dict = None):
        """
        Initializes the cache.

        Args:
            cache_ttl: Dictionary with TTL values for different data types (in seconds)
        """
        self.cache_ttl = cache_ttl or {
            'markets': 3600,    # 1 hour
            'tickers': 10,      # 10 seconds
            'orderbooks': 5,    # 5 seconds
            'ohlcv': 60,        # 1 minute
            'trades': 10,       # 10 seconds
            'balance': 10,      # 10 seconds
            'orders': 10,       # 10 seconds
            'my_trades': 60     # 1 minute
        }

        # Данные кеша
        self.cache = {}

        # Время последнего обновления
        self.cache_time = {}

        # Блокировка для синхронизации
        self.lock = asyncio.Lock()

    async def get(self, key: str, subkey: str = None) -> Optional[Any]:
        """
        Получает данные из кеша

        Args:
            key: Ключ кеша
            subkey: Подключ кеша

        Returns:
            Optional[Any]: Данные из кеша или None, если кеш устарел
        """
        async with self.lock:
            # Генерируем полный ключ
            full_key = key
            if subkey:
                full_key = f"{key}_{subkey}"

            # Проверяем наличие данных
            if full_key not in self.cache:
                return None

            # Проверяем время жизни кеша
            ttl = self.cache_ttl.get(key, 60)
            if time.time() - self.cache_time.get(full_key, 0) > ttl:
                return None

            return self.cache[full_key]

    async def set(self, key: str, data: Any, subkey: str = None):
        """
        Сохраняет данные в кеш

        Args:
            key: Ключ кеша
            data: Данные для сохранения
            subkey: Подключ кеша
        """
        async with self.lock:
            # Генерируем полный ключ
            full_key = key
            if subkey:
                full_key = f"{key}_{subkey}"

            # Сохраняем данные
            self.cache[full_key] = data
            self.cache_time[full_key] = time.time()

    async def invalidate(self, key: str = None, subkey: str = None):
        """
        Инвалидирует кеш

        Args:
            key: Ключ кеша
            subkey: Подключ кеша
        """
        async with self.lock:
            if key and subkey:
                # Инвалидируем конкретный ключ
                full_key = f"{key}_{subkey}"
                if full_key in self.cache:
                    del self.cache[full_key]
                if full_key in self.cache_time:
                    del self.cache_time[full_key]
            elif key:
                # Инвалидируем все данные для ключа
                keys_to_delete = []
                for full_key in self.cache:
                    if full_key.startswith(f"{key}_"):
                        keys_to_delete.append(full_key)

                for full_key in keys_to_delete:
                    del self.cache[full_key]
                    if full_key in self.cache_time:
                        del self.cache_time[full_key]
            else:
                # Инвалидируем весь кеш
                self.cache = {}
                self.cache_time = {}

    async def update_ttl(self, cache_ttl: Dict):
        """
        Обновляет время жизни кеша

        Args:
            cache_ttl: Словарь с временем жизни кеша
        """
        async with self.lock:
            self.cache_ttl.update(cache_ttl)

    def get_stats(self) -> Dict:
        """
        Возвращает статистику кеша

        Returns:
            Dict: Статистика кеша
        """
        stats = {
            'size': len(self.cache),
            'types': {},
            'hit_ratio': 0.0
        }

        # Собираем статистику по типам данных
        for full_key in self.cache:
            key_parts = full_key.split('_', 1)
            key = key_parts[0]

            if key not in stats['types']:
                stats['types'][key] = 0

            stats['types'][key] += 1

        return stats


class ExchangeManager:
    """Класс для управления соединениями с биржами"""

    def __init__(self, config: Dict = None):
        """
        Инициализирует менеджер бирж

        Args:
            config: Словарь с конфигурацией
        """
        self.config = config or {}

        # Конфигурации бирж
        self.exchange_configs = {}

        # Ограничения частоты запросов
        self.rate_limits = {}

        # Кеш данных
        self.cache = ExchangeCache(self.config.get('cache_ttl', {}))

        # WebSocket соединения
        self.websocket_clients = {}

        # Словарь соответствия символов между биржами
        self.symbol_mapping = {}

        # Словарь соответствия временных интервалов между биржами
        self.timeframe_mapping = {}

        # Настройки для автоматического выбора биржи
        self.exchange_priorities = self.config.get('exchange_priorities', {})
        self.exchange_scores = {}

        # Счетчики ошибок
        self.error_counters = {}

        # Признак инициализации
        self.initialized = False

        # Блокировка для синхронизации
        self.lock = asyncio.Lock()

        # Загружаем конфигурации бирж
        for exchange_id, exchange_config in self.config.get('exchanges', {}).items():
            if exchange_id in SUPPORTED_EXCHANGES:
                self.exchange_configs[exchange_id] = ExchangeConfig(exchange_id, exchange_config)

        # Инициализируем соответствия символов
        self._init_symbol_mappings()

        # Инициализируем соответствия временных интервалов
        self._init_timeframe_mappings()

        logger.info("Exchange manager initialized with {len(self.exchange_configs)} exchanges" %)

    async def start(self):
        """Запускает менеджер бирж"""
        if self.initialized:
            logger.warning("Exchange manager is already initialized")
            return

        async with self.lock:
            # Инициализируем экземпляры бирж
            for exchange_id, exchange_config in self.exchange_configs.items():
                # Инициализируем ограничения частоты запросов
                rate_limit_config = self.config.get('rate_limits', {}).get(exchange_id, {})
                self.rate_limits[exchange_id] = ExchangeRateLimit(rate_limit_config)

                # Инициализируем счетчики ошибок
                self.error_counters[exchange_id] = {
                    'total': 0,
                    'connection': 0,
                    'auth': 0,
                    'rate_limit': 0,
                    'insufficient_funds': 0,
                    'order_not_found': 0,
                    'other': 0
                }

                # Инициализируем WebSocket соединения, если настроены
                if exchange_config.use_websocket:
                    await self._init_websocket(exchange_id, exchange_config)

            # Обновляем рейтинг бирж
            await self._update_exchange_scores()

            self.initialized = True
            logger.info("Exchange manager started")

    async def stop(self):
        """Останавливает менеджер бирж"""
        if not self.initialized:
            logger.warning("Exchange manager is not initialized")
            return

        async with self.lock:
            # Закрываем все экземпляры бирж
            await self._close_all_exchanges()

            # Закрываем все WebSocket соединения
            for exchange_id, ws_client in self.websocket_clients.items():
                try:
                    await ws_client.disconnect()
                except Exception as e:
                    logger.error("Error disconnecting WebSocket for {exchange_id}: {str(e)}" %)

            self.websocket_clients = {}
            self.initialized = False

            logger.info("Exchange manager stopped")

    async def _close_all_exchanges(self):
        """Закрывает все экземпляры бирж"""
        for exchange_id, exchange in list(exchange_instances.items()):
            try:
                await exchange.close()
                logger.debug("Closed connection to exchange: {exchange_id}" %)
            except Exception as e:
                logger.error("Error closing exchange {exchange_id}: {str(e)}" %)

        # Очищаем словарь экземпляров
        exchange_instances.clear()

    async def _init_websocket(self, exchange_id: str, exchange_config: ExchangeConfig):
        """
        Инициализирует WebSocket соединение для биржи

        Args:
            exchange_id: ID биржи
            exchange_config: Конфигурация биржи
        """
        try:
            # Получаем URL для WebSocket
            ws_url = self._get_websocket_url(exchange_id, exchange_config)
            if not ws_url:
                logger.warning("WebSocket URL not found for exchange: {exchange_id}" %)
                return

            # Получаем пул WebSocket соединений
            ws_pool = get_ws_pool()

            # Создаем обработчик сообщений
            async def on_ws_message(message):
                try:
                    # Обработка сообщения
                    processor = get_message_processor()
                    await processor.process_message(message)
                except Exception as e:
                    logger.error("Error processing WebSocket message from {exchange_id}: {str(e)}" %)

            # Создаем обработчик подключения
            async def on_ws_connect():
                logger.info("Connected to {exchange_id} WebSocket" %)

                # Отправляем сообщения авторизации, если необходимо
                if exchange_config.ws_type == 'private':
                    auth_message = self._get_websocket_auth(exchange_id, exchange_config)
                    if auth_message:
                        await self.websocket_clients[exchange_id].send(auth_message)

            # Создаем обработчик отключения
            async def on_ws_disconnect():
                logger.info("Disconnected from {exchange_id} WebSocket" %)

            # Создаем обработчик ошибок
            async def on_ws_error(error):
                logger.error("WebSocket error for {exchange_id}: {error}" %)

                # Увеличиваем счетчик ошибок
                self.error_counters[exchange_id]['connection'] += 1
                self.error_counters[exchange_id]['total'] += 1

            # Создаем WebSocket клиент
            conn_id, client = await ws_pool.create_connection(
                url=ws_url,
                on_message=on_ws_message,
                on_connect=on_ws_connect,
                on_disconnect=on_ws_disconnect,
                on_error=on_ws_error,
                auto_reconnect=True,
                max_reconnect_attempts=5,
                reconnect_delay=5.0,
                ping_interval=30.0
            )

            # Добавляем провайдер авторизации, если необходимо
            if exchange_config.ws_type == 'private':
                auth_provider = self._get_websocket_auth_provider(exchange_id)
                auth_params = {
                    'api_key': exchange_config.api_key,
                    'api_secret': exchange_config.api_secret,
                    'api_password': exchange_config.api_password
                }
                client.set_auth_provider(auth_provider, auth_params)

            # Сохраняем клиент
            self.websocket_clients[exchange_id] = client

            # Подключаемся
            await client.connect()

            logger.info("Initialized WebSocket for exchange: {exchange_id}" %)

        except Exception as e:
            logger.error("Error initializing WebSocket for {exchange_id}: {str(e)}" %)

    def _get_websocket_url(self, exchange_id: str, exchange_config: ExchangeConfig) -> Optional[str]:
        """
        Возвращает URL для WebSocket соединения

        Args:
            exchange_id: ID биржи
            exchange_config: Конфигурация биржи

        Returns:
            Optional[str]: URL для WebSocket или None
        """
        # URL WebSocket для разных бирж
        ws_urls = {
            'binance': {
                'public': 'wss://stream.binance.com:9443/ws',
                'private': 'wss://stream.binance.com:9443/ws'
            },
            'binanceus': {
                'public': 'wss://stream.binance.us:9443/ws',
                'private': 'wss://stream.binance.us:9443/ws'
            },
            'bybit': {
                'public': 'wss://stream.bybit.com/v5/public',
                'private': 'wss://stream.bybit.com/v5/private'
            },
            'kucoin': {
                'public': 'wss://ws-api.kucoin.com/endpoint',
                'private': 'wss://ws-api.kucoin.com/endpoint'
            },
            'okx': {
                'public': 'wss://ws.okx.com:8443/ws/v5/public',
                'private': 'wss://ws.okx.com:8443/ws/v5/private'
            },
            'huobi': {
                'public': 'wss://api.huobi.pro/ws',
                'private': 'wss://api.huobi.pro/ws/v2'
            },
            'mexc': {
                'public': 'wss://wbs.mexc.com/ws',
                'private': 'wss://wbs.mexc.com/ws'
            },
            'bitget': {
                'public': 'wss://ws.bitget.com/spot/v1/stream',
                'private': 'wss://ws.bitget.com/spot/v1/stream'
            },
            'gateio': {
                'public': 'wss://api.gateio.ws/ws/v4/',
                'private': 'wss://api.gateio.ws/ws/v4/'
            },
            'kraken': {
                'public': 'wss://ws.kraken.com',
                'private': 'wss://ws-auth.kraken.com'
            },
            'bitfinex': {
                'public': 'wss://api-pub.bitfinex.com/ws/2',
                'private': 'wss://api.bitfinex.com/ws/2'
            },
            'coinbase': {
                'public': 'wss://ws-feed.exchange.coinbase.com',
                'private': 'wss://ws-feed.exchange.coinbase.com'
            }
        }

        # URL для тестнета, если включен sandbox режим
        sandbox_ws_urls = {
            'binance': {
                'public': 'wss://stream.binancefuture.com/ws',
                'private': 'wss://stream.binancefuture.com/ws'
            },
            'bybit': {
                'public': 'wss://stream-testnet.bybit.com/v5/public',
                'private': 'wss://stream-testnet.bybit.com/v5/private'
            }
        }

        # Выбираем URL
        urls = sandbox_ws_urls.get(exchange_id, {}) if exchange_config.sandbox else ws_urls.get(exchange_id, {})
        ws_type = exchange_config.ws_type

        return urls.get(ws_type)

    def _get_websocket_auth(self, exchange_id: str, exchange_config: ExchangeConfig) -> Optional[Dict]:
        """
        Возвращает сообщение авторизации для WebSocket

        Args:
            exchange_id: ID биржи
            exchange_config: Конфигурация биржи

        Returns:
            Optional[Dict]: Сообщение авторизации или None
        """
        # Проверяем наличие ключей
        if not exchange_config.api_key or not exchange_config.api_secret:
            logger.warning("Missing API keys for {exchange_id} WebSocket authentication" %)
            return None

        # Генерируем сообщение авторизации для разных бирж
        if exchange_id == 'binance':
            # Для Binance
            timestamp = int(time.time() * 1000)
            signature = hmac.new(
                exchange_config.api_secret.encode('utf-8'),
                f"timestamp={timestamp}".encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            return {
                'method': 'AUTHENTICATE',
                'params': {
                    'apiKey': exchange_config.api_key,
                    'timestamp': timestamp,
                    'signature': signature
                },
                'id': int(time.time())
            }

        elif exchange_id == 'bybit':
            # Для Bybit
            timestamp = int(time.time() * 1000)
            signature = hmac.new(
                exchange_config.api_secret.encode('utf-8'),
                f"GET/realtime{timestamp}".encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            return {
                'op': 'auth',
                'args': [exchange_config.api_key, timestamp, signature]
            }

        elif exchange_id == 'okx':
            # Для OKX
            timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
            message = timestamp + 'GET' + '/users/self/verify'
            signature = base64.b64encode(
                hmac.new(
                    exchange_config.api_secret.encode('utf-8'),
                    message.encode('utf-8'),
                    hashlib.sha256
                ).digest()
            ).decode('utf-8')

            return {
                'op': 'login',
                'args': [{
                    'apiKey': exchange_config.api_key,
                    'passphrase': exchange_config.api_password,
                    'timestamp': timestamp,
                    'sign': signature
                }]
            }

        # Добавить другие биржи по необходимости

        logger.warning("WebSocket authentication not implemented for {exchange_id}" %)
        return None

    def _get_websocket_auth_provider(self, exchange_id: str) -> Optional[Callable]:
        """
        Возвращает провайдер авторизации для WebSocket

        Args:
            exchange_id: ID биржи

        Returns:
            Optional[Callable]: Провайдер авторизации или None
        """
        # Провайдер авторизации для разных бирж
        if exchange_id == 'binance':
            async def binance_auth_provider(auth_params, url):
                api_key = auth_params.get('api_key')
                api_secret = auth_params.get('api_secret')

                if not api_key or not api_secret:
                    return {}

                return {'X-MBX-APIKEY': api_key}

            return binance_auth_provider

        elif exchange_id == 'bybit':
            async def bybit_auth_provider(auth_params, url):
                api_key = auth_params.get('api_key')
                api_secret = auth_params.get('api_secret')

                if not api_key or not api_secret:
                    return {}

                return {}  # Для Bybit авторизация в сообщении, а не в заголовках

            return bybit_auth_provider

        # Добавить другие биржи по необходимости

        return None

    def _init_symbol_mappings(self):
        """Инициализирует соответствия символов между биржами"""
        # Загружаем соответствия из конфигурации
        self.symbol_mapping = self.config.get('symbol_mapping', {})

        # Добавляем базовые соответствия
        if not self.symbol_mapping:
            # Пример базовых соответствий
            self.symbol_mapping = {
                'BTC/USDT': {
                    'binance': 'BTC/USDT',
                    'bybit': 'BTCUSDT',
                    'okx': 'BTC-USDT',
                    'kucoin': 'BTC-USDT'
                },
                'ETH/USDT': {
                    'binance': 'ETH/USDT',
                    'bybit': 'ETHUSDT',
                    'okx': 'ETH-USDT',
                    'kucoin': 'ETH-USDT'
                }
                # Добавить другие символы по необходимости
            }

    def _init_timeframe_mappings(self):
        """Инициализирует соответствия временных интервалов между биржами"""
        # Загружаем соответствия из конфигурации
        self.timeframe_mapping = self.config.get('timeframe_mapping', {})

        # Добавляем базовые соответствия
        if not self.timeframe_mapping:
            # Пример базовых соответствий
            self.timeframe_mapping = {
                '1m': {
                    'binance': '1m',
                    'bybit': '1',
                    'okx': '1m',
                    'kucoin': '1min'
                },
                '5m': {
                    'binance': '5m',
                    'bybit': '5',
                    'okx': '5m',
                    'kucoin': '5min'
                },
                '15m': {
                    'binance': '15m',
                    'bybit': '15',
                    'okx': '15m',
                    'kucoin': '15min'
                },
                '30m': {
                    'binance': '30m',
                    'bybit': '30',
                    'okx': '30m',
                    'kucoin': '30min'
                },
                '1h': {
                    'binance': '1h',
                    'bybit': '60',
                    'okx': '1H',
                    'kucoin': '1hour'
                },
                '4h': {
                    'binance': '4h',
                    'bybit': '240',
                    'okx': '4H',
                    'kucoin': '4hour'
                },
                '1d': {
                    'binance': '1d',
                    'bybit': 'D',
                    'okx': '1D',
                    'kucoin': '1day'
                },
                '1w': {
                    'binance': '1w',
                    'bybit': 'W',
                    'okx': '1W',
                    'kucoin': '1week'
                }
                # Добавить другие интервалы по необходимости
            }

    async def _update_exchange_scores(self):
        """Обновляет рейтинг бирж для автоматического выбора"""
        for exchange_id in self.exchange_configs.keys():
            try:
                # Базовая оценка из приоритетов
                base_score = self.exchange_priorities.get(exchange_id, 50)

                # Снижаем оценку на основе счетчиков ошибок
                error_penalty = min(50, self.error_counters[exchange_id]['total'])

                # Вычисляем итоговую оценку
                final_score = max(0, base_score - error_penalty)

                self.exchange_scores[exchange_id] = final_score

            except Exception as e:
                logger.error("Error updating score for {exchange_id}: {str(e)}" %)
                self.exchange_scores[exchange_id] = 0

    async def get_exchange(self, exchange_id: str) -> ccxt.Exchange:
        """
        Возвращает экземпляр биржи

        Args:
            exchange_id: ID биржи

        Returns:
            ccxt.Exchange: Экземпляр биржи
        """
        # Проверяем, есть ли конфигурация для этой биржи
        if exchange_id not in self.exchange_configs:
            raise ValueError(f"Unknown exchange: {exchange_id}")

        # Проверяем, есть ли уже экземпляр
        global exchange_instances
        if exchange_id in exchange_instances:
            return exchange_instances[exchange_id]

        try:
            # Получаем конфигурацию
            exchange_config = self.exchange_configs[exchange_id]

            # Преобразуем конфигурацию для CCXT
            ccxt_config = exchange_config.to_ccxt_config()

            # Создаем экземпляр биржи
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class(ccxt_config)

            # Устанавливаем логирование
            exchange.logger = logger

            # Добавляем в словарь экземпляров
            exchange_instances[exchange_id] = exchange

            return exchange

        except Exception as e:
            logger.error("Error creating exchange {exchange_id}: {str(e)}" %)
            raise

    async def select_best_exchange(self, symbol: str, required_features: List[str] = None) -> Tuple[str, float]:
        """
        Выбирает лучшую биржу для работы с указанным символом

        Args:
            symbol: Символ для торговли
            required_features: Список необходимых функций

        Returns:
            Tuple[str, float]: ID биржи и её оценка
        """
        best_exchange = None
        best_score = -1

        # Обновляем рейтинг бирж
        await self._update_exchange_scores()

        for exchange_id, score in sorted(self.exchange_scores.items(), key=lambda x: x[1], reverse=True):
            try:
                # Проверяем, поддерживает ли биржа символ
                mapped_symbol = await self.map_symbol(symbol, exchange_id)
                if not mapped_symbol:
                    continue

                # Проверяем требуемые функции
                if required_features:
                    exchange = await self.get_exchange(exchange_id)
                    has_all_features = True

                    for feature in required_features:
                        if feature not in exchange.has or not exchange.has[feature]:
                            has_all_features = False
                            break

                    if not has_all_features:
                        continue

                # Если оценка выше текущей лучшей, обновляем
                if score > best_score:
                    best_exchange = exchange_id
                    best_score = score

            except Exception as e:
                logger.error("Error checking exchange {exchange_id}: {str(e)}" %)

        return best_exchange, best_score

    async def map_symbol(self, symbol: str, exchange_id: str) -> Optional[str]:
        """
        Преобразует символ для указанной биржи

        Args:
            symbol: Исходный символ
            exchange_id: ID биржи

        Returns:
            Optional[str]: Преобразованный символ или None
        """
        # Проверяем в словаре соответствий
        if symbol in self.symbol_mapping and exchange_id in self.symbol_mapping[symbol]:
            return self.symbol_mapping[symbol][exchange_id]

        try:
            # Пробуем получить маркеты биржи
            exchange = await self.get_exchange(exchange_id)

            # Проверяем, загружены ли маркеты
            if not exchange.markets:
                await exchange.load_markets()

            # Пробуем найти символ в маркетах
            if symbol in exchange.markets:
                return symbol

            # Пробуем нормализовать символ
            normalized = self._normalize_symbol(symbol)

            for market_symbol in exchange.markets:
                if self._normalize_symbol(market_symbol) == normalized:
                    return market_symbol

            # Если не нашли, возвращаем None
            return None

        except Exception as e:
            logger.error("Error mapping symbol {symbol} for {exchange_id}: {str(e)}" %)
            return None

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Нормализует символ для сравнения

        Args:
            symbol: Исходный символ

        Returns:
            str: Нормализованный символ
        """
        # Удаляем все специальные символы
        normalized = symbol.replace('/', '').replace('-', '').replace('_', '')
        return normalized.upper()

    async def map_symbol_across_exchanges(self, symbol: str, source_exchange: str, target_exchange: str) -> Optional[str]:
        """
        Преобразует символ с одной биржи для другой

        Args:
            symbol: Символ на исходной бирже
            source_exchange: Исходная биржа
            target_exchange: Целевая биржа

        Returns:
            Optional[str]: Преобразованный символ или None
        """
        # Если биржи совпадают, возвращаем исходный символ
        if source_exchange == target_exchange:
            return symbol

        try:
            # Получаем базовый и котируемый активы
            parts = None

            # Пробуем разделить символ
            if '/' in symbol:
                parts = symbol.split('/')
            elif '-' in symbol:
                parts = symbol.split('-')
            else:
                # Для символов без разделителя пробуем найти в маркетах исходной биржи
                exchange = await self.get_exchange(source_exchange)

                # Загружаем маркеты, если не загружены
                if not exchange.markets:
                    await exchange.load_markets()

                if symbol in exchange.markets:
                    market = exchange.markets[symbol]
                    parts = [market['base'], market['quote']]
                else:
                    # Не удалось разделить символ
                    return None

            base, quote = parts

            # Получаем стандартный формат символа для целевой биржи
            target_exchange_obj = await self.get_exchange(target_exchange)

            # Загружаем маркеты, если не загружены
            if not target_exchange_obj.markets:
                await target_exchange_obj.load_markets()

            # Нормализуем базу и котировку
            base = base.upper()
            quote = quote.upper()

            # Ищем соответствующий символ на целевой бирже
            for market_symbol, market in target_exchange_obj.markets.items():
                if market['base'].upper() == base and market['quote'].upper() == quote:
                    return market_symbol

            # Если не нашли, возвращаем None
            return None

        except Exception as e:
            logger.error("Error mapping symbol {symbol} from {source_exchange} to {target_exchange}: {str(e)}" %)
            return None

    async def map_timeframe(self, timeframe: str, exchange_id: str) -> Optional[str]:
        """
        Преобразует временной интервал для указанной биржи

        Args:
            timeframe: Исходный временной интервал
            exchange_id: ID биржи

        Returns:
            Optional[str]: Преобразованный временной интервал или None
        """
        # Проверяем в словаре соответствий
        if timeframe in self.timeframe_mapping and exchange_id in self.timeframe_mapping[timeframe]:
            return self.timeframe_mapping[timeframe][exchange_id]

        try:
            # Пробуем получить поддерживаемые интервалы
            exchange = await self.get_exchange(exchange_id)

            # Проверяем, поддерживает ли биржа временной интервал
            if hasattr(exchange, 'timeframes') and exchange.timeframes and timeframe in exchange.timeframes:s:
                return timeframe

            # Если не нашли, возвращаем None
            return None

        except Exception as e:
            logger.error("Error mapping timeframe {timeframe} for {exchange_id}: {str(e)}" %)
            return None

    async def fetch_ticker(self, symbol: str, exchange_id: str) -> Optional[Dict]:
        """
        Получает тикер для указанного символа

        Args:
            symbol: Символ
            exchange_id: ID биржи

        Returns:
            Optional[Dict]: Тикер или None
        """
        # Проверяем в кеше
        cached_ticker = await self.cache.get('tickers', f"{exchange_id}_{symbol}")
        if cached_ticker:
            return cached_ticker

        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()

            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)

            # Получаем тикер
            ticker = await exchange.fetch_ticker(mapped_symbol)

            # Сохраняем в кеш
            await self.cache.set('tickers', ticker, f"{exchange_id}_{symbol}")

            return ticker

        except Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching ticker for {symbol} on {exchange_id}: {str(e)}" %)
            return None

    async def fetch_order_book(self, symbol: str, exchange_id: str, limit: int = 20) -> Optional[Dict]:
        """
        Получает книгу ордеров для указанного символа

        Args:
            symbol: Символ
            exchange_id: ID биржи
            limit: Количество уровней

        Returns:
            Optional[Dict]: Книга ордеров или None
        """
        # Проверяем в кеше
        cached_orderbook = await self.cache.get('orderbooks', f"{exchange_id}_{symbol}_{limit}")
        if cached_orderbook:
            return cached_orderbook

        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()

            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)

            # Получаем книгу ордеров
            orderbook = await exchange.fetch_order_book(mapped_symbol, limit)

            # Сохраняем в кеш
            await self.cache.set('orderbooks', orderbook, f"{exchange_id}_{symbol}_{limit}")

            return orderbook

        except Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching order book for {symbol} on {exchange_id}: {str(e)}" %)
            return None

    async def fetch_ohlcv(self, symbol: str, exchange_id: str, timeframe: str = '1h', limit: int = 100, since: int = None) -> Optional[List]:
        """
        Получает OHLCV данные для указанного символа

        Args:
            symbol: Символ
            exchange_id: ID биржи
            timeframe: Временной интервал
            limit: Количество свечей
            since: Начальная метка времени

        Returns:
            Optional[List]: OHLCV данные или None
        """
        # Проверяем в кеше
        cache_key = f"{exchange_id}_{symbol}_{timeframe}_{limit}_{since}"
        cached_ohlcv = await self.cache.get('ohlcv', cache_key)
        if cached_ohlcv:
            return cached_ohlcv

        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()

            # Преобразуем символ и временной интервал
            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None

            mapped_timeframe = await self.map_timeframe(timeframe, exchange_id)
            if not mapped_timeframe:
                logger.warning("Timeframe {timeframe} not supported on {exchange_id}" %)
                return None

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)

            # Получаем OHLCV данные
            ohlcv = await exchange.fetch_ohlcv(mapped_symbol, mapped_timeframe, since, limit)

            # Сохраняем в кеш
            await self.cache.set('ohlcv', ohlcv, cache_key)

            return ohlcv

        except Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching OHLCV for {symbol} on {exchange_id}: {str(e)}" %)
            return None

    async def fetch_trades(self, symbol: str, exchange_id: str, limit: int = 100, since: int = None) -> Optional[List]:
        """
        Получает историю сделок для указанного символа

        Args:
            symbol: Символ
            exchange_id: ID биржи
            limit: Количество сделок
            since: Начальная метка времени

        Returns:
            Optional[List]: История сделок или None
        """
        # Проверяем в кеше
        cache_key = f"{exchange_id}_{symbol}_{limit}_{since}"
        cached_trades = await self.cache.get('trades', cache_key)
        if cached_trades:
            return cached_trades

        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()

            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)

            # Получаем историю сделок
            trades = await exchange.fetch_trades(mapped_symbol, since, limit)

            # Сохраняем в кеш
            await self.cache.set('trades', trades, cache_key)

            return trades

        except Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching trades for {symbol} on {exchange_id}: {str(e)}" %)
            return None

    async def fetch_balance(self, exchange_id: str) -> Optional[Dict]:
        """
        Получает баланс для указанной биржи

        Args:
            exchange_id: ID биржи

        Returns:
            Optional[Dict]: Баланс или None
        """
        # Проверяем в кеше
        cached_balance = await self.cache.get('balance', exchange_id)
        if cached_balance:
            return cached_balance

        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)

            # Получаем баланс
            balance = await exchange.fetch_balance()

            # Сохраняем в кеш
            await self.cache.set('balance', balance, exchange_id)

            return balance

        except Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching balance on {exchange_id}: {str(e)}" %)
            return None

    async def create_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None, exchange_id: str = None, params: Dict = None) -> Optional[Dict]:
        """
        Создает ордер

        Args:
            symbol: Символ
            order_type: Тип ордера
            side: Сторона (buy/sell)
            amount: Количество
            price: Цена
            exchange_id: ID биржи
            params: Дополнительные параметры

        Returns:
            Optional[Dict]: Информация о созданном ордере или None
        """
        # Если биржа не указана, выбираем лучшуюа, выбираем лучшую
        if not exchange_id:
            exchange_id, _ = await self.select_best_exchange(symbol, ['createOrder'])elect_best_exchange(symbol, ['createOrder'])
            if not exchange_id:
                logger.error("No suitable exchange found for {symbol}" %)logger.error("No suitable exchange found for {symbol}" %)
                return None

        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()f.rate_limits[exchange_id].acquire()

            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:ot mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)= await self.get_exchange(exchange_id)

            # Создаем ордер
            params = params or {} or {}
            order = await exchange.create_order(mapped_symbol, order_type, side, amount, price, params)nt, price, params)

            # Инвалидируем кеш баланса
            await self.cache.invalidate('balance', exchange_id)e.invalidate('balance', exchange_id)

            return orderreturn order

        except Exception as e:
            # Увеличиваем счетчик ошибок            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)r_counter(exchange_id, e)

            logger.error("Error creating order for {symbol} on {exchange_id}: {str(e)}" %)creating order for {symbol} on {exchange_id}: {str(e)}")
            return None

    async def cancel_order(self, order_id: str, symbol: str, exchange_id: str) -> Optional[Dict]:    async def cancel_order(self, order_id: str, symbol: str, exchange_id: str) -> Optional[Dict]:
        """
        Отменяет ордер

        Args:
            order_id: ID ордера
            symbol: Символ
            exchange_id: ID биржи            exchange_id: ID биржи

        Returns:
            Optional[Dict]: Информация об отмененном ордере или None            Optional[Dict]: Информация об отмененном ордере или None
        """
        try:        try:
            # Отслеживаем ограничения частоты запросовничения частоты запросов
            await self.rate_limits[exchange_id].acquire()nge_id].acquire()

            # Преобразуем символ            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:ed_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None

            # Получаем экземпляр биржи экземпляр биржи
            exchange = await self.get_exchange(exchange_id)            exchange = await self.get_exchange(exchange_id)

            # Отменяем ордер
            order = await exchange.cancel_order(order_id, mapped_symbol)exchange.cancel_order(order_id, mapped_symbol)

            # Инвалидируем кеш баланса            # Инвалидируем кеш баланса
            await self.cache.invalidate('balance', exchange_id)t self.cache.invalidate('balance', exchange_id)

            return order return order

        except Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)            self._increment_error_counter(exchange_id, e)

            logger.error("Error canceling order {order_id} for {symbol} on {exchange_id}: {str(e)}" %)l} on {exchange_id}: {str(e)}")
            return None

    async def fetch_order(self, order_id: str, symbol: str, exchange_id: str) -> Optional[Dict]:elf, order_id: str, symbol: str, exchange_id: str) -> Optional[Dict]:
        """        """
        Получает информацию об ордере

        Args:        Args:
            order_id: ID ордераера
            symbol: Символ
            exchange_id: ID биржи            exchange_id: ID биржи

        Returns:
            Optional[Dict]: Информация об ордере или None            Optional[Dict]: Информация об ордере или None
        """
        try:        try:
            # Отслеживаем ограничения частоты запросовничения частоты запросов
            await self.rate_limits[exchange_id].acquire()nge_id].acquire()

            # Преобразуем символ            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:ed_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None

            # Получаем экземпляр биржии
            exchange = await self.get_exchange(exchange_id)            exchange = await self.get_exchange(exchange_id)

            # Получаем информацию об ордереию об ордере
            order = await exchange.fetch_order(order_id, mapped_symbol)exchange.fetch_order(order_id, mapped_symbol)

            return order            return order

        except Exception as e:
            # Увеличиваем счетчик ошибок # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching order {order_id} for {symbol} on {exchange_id}: {str(e)}" %)} for {symbol} on {exchange_id}: {str(e)}")
            return None            return None

    async def fetch_orders(self, symbol: str, exchange_id: str, since: int = None, limit: int = None) -> Optional[List]: int = None, limit: int = None) -> Optional[List]:
        """
        Получает список ордеров

        Args:        Args:
            symbol: Символ
            exchange_id: ID биржи
            since: Начальная метка времени            since: Начальная метка времени
            limit: Максимальное количество ордеровордеров

        Returns:        Returns:
            Optional[List]: Список ордеров или Nonet]: Список ордеров или None
        """        """
        try:
            # Отслеживаем ограничения частоты запросовстоты запросов
            await self.rate_limits[exchange_id].acquire()

            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)ol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:            if not mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return None     return None

            # Получаем экземпляр биржи            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)xchange = await self.get_exchange(exchange_id)

            # Получаем список ордероверов
            orders = await exchange.fetch_orders(mapped_symbol, since, limit)orders(mapped_symbol, since, limit)

            return orders            return orders

        except Exception as e:
            # Увеличиваем счетчик ошибок # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching orders for {symbol} on {exchange_id}: {str(e)}" %)bol} on {exchange_id}: {str(e)}")
            return None            return None

    async def fetch_open_orders(self, symbol: str = None, exchange_id: str = None) -> Optional[List]: str = None) -> Optional[List]:
        """
        Получает список открытых ордеров

        Args:        Args:
            symbol: Символ
            exchange_id: ID биржи

        Returns:
            Optional[List]: Список открытых ордеров или None
        """        """
        # Если биржа не указана, выбираем лучшуюказана, выбираем лучшую
        if not exchange_id:        if not exchange_id:
            if symbol:
                exchange_id, _ = await self.select_best_exchange(symbol, ['fetchOpenOrders'])elf.select_best_exchange(symbol, ['fetchOpenOrders'])
            else:
                # Если символ не указан, выбираем первую доступную биржу                # Если символ не указан, выбираем первую доступную биржу
                exchange_id = next(iter(self.exchange_configs.keys()), None)

            if not exchange_id:            if not exchange_id:
                logger.error("No suitable exchange found")
                return None     return None

        try:        try:
            # Отслеживаем ограничения частоты запросов Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()e_limits[exchange_id].acquire()

            # Преобразуем символ, если указан            # Преобразуем символ, если указан
            mapped_symbol = Noneed_symbol = None
            if symbol:
                mapped_symbol = await self.map_symbol(symbol, exchange_id)     mapped_symbol = await self.map_symbol(symbol, exchange_id)
                if not mapped_symbol:
                    logger.warning("Symbol {symbol} not found on {exchange_id}" %)warning(f"Symbol {symbol} not found on {exchange_id}")
                    return Noneturn None

            # Получаем экземпляр биржиучаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)

            # Получаем список открытых ордеров            # Получаем список открытых ордеров
            orders = await exchange.fetch_open_orders(mapped_symbol)ange.fetch_open_orders(mapped_symbol)

            return orders

        except Exception as e:pt Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching open orders on {exchange_id}: {str(e)}" %)n orders on {exchange_id}: {str(e)}")
            return None

    async def fetch_my_trades(self, symbol: str, exchange_id: str, since: int = None, limit: int = None) -> Optional[List]:int = None, limit: int = None) -> Optional[List]:
        """
        Получает список сделок пользователя

        Args:        Args:
            symbol: Символ
            exchange_id: ID биржи
            since: Начальная метка времени            since: Начальная метка времени
            limit: Максимальное количество сделоклок

        Returns:        Returns:
            Optional[List]: Список сделок или None]: Список сделок или None
        """        """
        # Проверяем в кеше
        cache_key = f"{exchange_id}_{symbol}_{limit}_{since}"bol}_{limit}_{since}"
        cached_trades = await self.cache.get('my_trades', cache_key) cache_key)
        if cached_trades:        if cached_trades:
            return cached_trades

        try:        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire() await self.rate_limits[exchange_id].acquire()

            # Преобразуем символ            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)apped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)Symbol {symbol} not found on {exchange_id}")
                return None

            # Получаем экземпляр биржи            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)ange = await self.get_exchange(exchange_id)

            # Получаем список сделок # Получаем список сделок
            trades = await exchange.fetch_my_trades(mapped_symbol, since, limit) exchange.fetch_my_trades(mapped_symbol, since, limit)

            # Сохраняем в кеш
            await self.cache.set('my_trades', trades, cache_key)che.set('my_trades', trades, cache_key)

            return trades            return trades

        except Exception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)            self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching my trades for {symbol} on {exchange_id}: {str(e)}" %)xchange_id}: {str(e)}")
            return None

    def _increment_error_counter(self, exchange_id: str, error: Exception):unter(self, exchange_id: str, error: Exception):
        """        """
        Увеличивает счетчики ошибок

        Args:        Args:
            exchange_id: ID биржи
            error: Исключение
        """        """
        if exchange_id not in self.error_counters: self.error_counters:
            return

        # Увеличиваем общий счетчикий счетчик
        self.error_counters[exchange_id]['total'] += 1        self.error_counters[exchange_id]['total'] += 1

        # Определяем тип ошибки
        error_str = str(error).lower()

        if 'connection' in error_str or 'timeout' in error_str or 'network' in error_str:
            self.error_counters[exchange_id]['connection'] += 1counters[exchange_id]['connection'] += 1
        elif 'auth' in error_str or 'api key' in error_str or 'signature' in error_str:        elif 'auth' in error_str or 'api key' in error_str or 'signature' in error_str:
            self.error_counters[exchange_id]['auth'] += 1
        elif 'rate limit' in error_str or 'too many requests' in error_str:f 'rate limit' in error_str or 'too many requests' in error_str:
            self.error_counters[exchange_id]['rate_limit'] += 1hange_id]['rate_limit'] += 1
        elif 'insufficient' in error_str or 'balance' in error_str:        elif 'insufficient' in error_str or 'balance' in error_str:
            self.error_counters[exchange_id]['insufficient_funds'] += 1elf.error_counters[exchange_id]['insufficient_funds'] += 1
        elif 'order not found' in error_str or 'no such order' in error_str: error_str or 'no such order' in error_str:
            self.error_counters[exchange_id]['order_not_found'] += 1rs[exchange_id]['order_not_found'] += 1
        else:e:
            self.error_counters[exchange_id]['other'] += 1r'] += 1

    async def get_error_stats(self, exchange_id: str = None) -> Dict:    async def get_error_stats(self, exchange_id: str = None) -> Dict:
        """
        Возвращает статистику ошибок

        Args:
            exchange_id: ID биржи

        Returns:
            Dict: Статистика ошибок
        """
        if exchange_id:
            return self.error_counters.get(exchange_id, {})
        else:
            return self.error_counters

    async def get_rate_limit_usage(self, exchange_id: str = None) -> Dict:
        """
        Возвращает информацию об использовании ограничений частоты запросоващает информацию об использовании ограничений частоты запросов

        Args:        Args:
            exchange_id: ID биржи

        Returns:
            Dict: Информация об использовании            Dict: Информация об использовании
        """
        if exchange_id:
            return self.rate_limits.get(exchange_id, ExchangeRateLimit()).get_usage()            return self.rate_limits.get(exchange_id, ExchangeRateLimit()).get_usage()
        else:
            return {ex_id: limiter.get_usage() for ex_id, limiter in self.rate_limits.items()}get_usage() for ex_id, limiter in self.rate_limits.items()}

    async def get_cache_stats(self) -> Dict:_stats(self) -> Dict:
        """
        Возвращает статистику кешаащает статистику кеша

        Returns:        Returns:
            Dict: Статистика кеша
        """
        return self.cache.get_stats()

    async def clear_cache(self, key: str = None, subkey: str = None): clear_cache(self, key: str = None, subkey: str = None):
        """
        Очищает кеш        Очищает кеш

        Args:
            key: Ключ кеша key: Ключ кеша
            subkey: Подключ кешаключ кеша
        """
        await self.cache.invalidate(key, subkey) self.cache.invalidate(key, subkey)

    async def subscribe_to_ticker(self, symbol: str, exchange_id: str, callback: Callable = None) -> bool:    async def subscribe_to_ticker(self, symbol: str, exchange_id: str, callback: Callable = None) -> bool:
        """
        Подписывается на обновления тикераписывается на обновления тикера

        Args:        Args:
            symbol: Символol: Символ
            exchange_id: ID биржи
            callback: Функция для обработки обновлений callback: Функция для обработки обновлений

        Returns:        Returns:
            bool: True, если подписка успешна, иначе False
        """
        if exchange_id not in self.websocket_clients:_id not in self.websocket_clients:
            logger.warning("WebSocket not initialized for {exchange_id}" %)            logger.warning("WebSocket not initialized for {exchange_id}" %)
            return Falseeturn False

        try:
            # Преобразуем символ # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)l(symbol, exchange_id)
            if not mapped_symbol:            if not mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return False     return False

            # Формируем сообщение подписки            # Формируем сообщение подписки
            subscription = self._get_ticker_subscription(exchange_id, mapped_symbol)ubscription = self._get_ticker_subscription(exchange_id, mapped_symbol)
            if not subscription:ption:
                logger.warning("Ticker subscription not supported for {exchange_id}" %)Ticker subscription not supported for {exchange_id}")
                return False

            # Добавляем подпискубавляем подписку
            self.websocket_clients[exchange_id].add_subscription(subscription)iption(subscription)

            # Регистрируем обработчик, если указан
            if callback:
                message_type = self._get_ticker_message_type(exchange_id)type = self._get_ticker_message_type(exchange_id)
                if message_type:                if message_type:
                    processor = get_message_processor()        processor = get_message_processor()
                    processor.register_handler(message_type, callback)gister_handler(message_type, callback)

            return True

        except Exception as e:e:
            logger.error("Error subscribing to ticker for {symbol} on {exchange_id}: {str(e)}" %)            logger.error("Error subscribing to ticker for {symbol} on {exchange_id}: {str(e)}" %)
            return False

    def _get_ticker_subscription(self, exchange_id: str, symbol: str) -> Optional[Dict]:(self, exchange_id: str, symbol: str) -> Optional[Dict]:
        """
        Возвращает сообщение подписки на тикер подписки на тикер

        Args:
            exchange_id: ID биржи
            symbol: Символ            symbol: Символ

        Returns:
            Optional[Dict]: Сообщение подписки или None
        """
        # Сообщения подписки для разных бирж
        if exchange_id == 'binance':
            return {            return {
                'method': 'SUBSCRIBE',': 'SUBSCRIBE',
                'params': [f"{symbol.lower()}@ticker"],                'params': [f"{symbol.lower()}@ticker"],
                'id': int(time.time()).time())
            }
        elif exchange_id == 'bybit': == 'bybit':
            return {            return {
                'op': 'subscribe',
                'args': [f"tickers.{symbol}"]     'args': [f"tickers.{symbol}"]
            }
        elif exchange_id == 'okx':        elif exchange_id == 'okx':
            instrument_type = 'SPOT'  # или 'SWAP', 'FUTURES'nstrument_type = 'SPOT'  # или 'SWAP', 'FUTURES'
            return {
                'op': 'subscribe',scribe',
                'args': [{                'args': [{
                    'channel': 'tickers',    'channel': 'tickers',
                    'instId': symbol
                }]     }]
            }

        # Добавить другие биржи по необходимостиругие биржи по необходимости

        return None

    def _get_ticker_message_type(self, exchange_id: str) -> Optional[str]:ticker_message_type(self, exchange_id: str) -> Optional[str]:
        """
        Возвращает тип сообщения для тикераип сообщения для тикера

        Args:
            exchange_id: ID биржиxchange_id: ID биржи

        Returns:
            Optional[str]: Тип сообщения или None[str]: Тип сообщения или None
        """
        # Типы сообщений для разных биржля разных бирж
        if exchange_id == 'binance':
            return '24hrTicker'
        elif exchange_id == 'bybit':nge_id == 'bybit':
            return 'tickers'eturn 'tickers'
        elif exchange_id == 'okx':        elif exchange_id == 'okx':
            return 'tickers'

        # Добавить другие биржи по необходимостидругие биржи по необходимости

        return None

    async def unsubscribe_from_ticker(self, symbol: str, exchange_id: str) -> bool: symbol: str, exchange_id: str) -> bool:
        """        """
        Отписывается от обновлений тикераывается от обновлений тикера

        Args:        Args:
            symbol: Символol: Символ
            exchange_id: ID биржи

        Returns:
            bool: True, если отписка успешна, иначе False успешна, иначе False
        """
        if exchange_id not in self.websocket_clients:ebsocket_clients:
            logger.warning("WebSocket not initialized for {exchange_id}" %)"WebSocket not initialized for {exchange_id}")
            return False

        try:        try:
            # Преобразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)                logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return False

            # Формируем сообщение отписки
            unsubscription = self._get_ticker_unsubscription(exchange_id, mapped_symbol)            unsubscription = self._get_ticker_unsubscription(exchange_id, mapped_symbol)
            if not unsubscription:f not unsubscription:
                logger.warning("Ticker unsubscription not supported for {exchange_id}" %)ning(f"Ticker unsubscription not supported for {exchange_id}")
                return False

            # Отправляем сообщение отпискиправляем сообщение отписки
            await self.websocket_clients[exchange_id].send(unsubscription)d(unsubscription)

            # Удаляем из списка подписок
            subscription = self._get_ticker_subscription(exchange_id, mapped_symbol)ed_symbol)
            if subscription:ion:
                self.websocket_clients[exchange_id].remove_subscription(subscription)                self.websocket_clients[exchange_id].remove_subscription(subscription)

            return True

        except Exception as e:
            logger.error("Error unsubscribing from ticker for {symbol} on {exchange_id}: {str(e)}" %)xchange_id}: {str(e)}")
            return False

    def _get_ticker_unsubscription(self, exchange_id: str, symbol: str) -> Optional[Dict]:exchange_id: str, symbol: str) -> Optional[Dict]:
        """
        Возвращает сообщение отписки от тикераки от тикера

        Args:
            exchange_id: ID биржи            exchange_id: ID биржи
            symbol: Символ

        Returns:        Returns:
            Optional[Dict]: Сообщение отписки или Noneписки или None
        """
        # Сообщения отписки для разных бирждля разных бирж
        if exchange_id == 'binance':
            return {            return {
                'method': 'UNSUBSCRIBE',': 'UNSUBSCRIBE',
                'params': [f"{symbol.lower()}@ticker"],                'params': [f"{symbol.lower()}@ticker"],
                'id': int(time.time()).time())
            }
        elif exchange_id == 'bybit': == 'bybit':
            return {            return {
                'op': 'unsubscribe',
                'args': [f"tickers.{symbol}"]     'args': [f"tickers.{symbol}"]
            }
        elif exchange_id == 'okx':        elif exchange_id == 'okx':
            return {eturn {
                'op': 'unsubscribe',e',
                'args': [{
                    'channel': 'tickers',                    'channel': 'tickers',
                    'instId': symbol    'instId': symbol
                }]
            } }

        # Добавить другие биржи по необходимостиеобходимости

        return None

    async def subscribe_to_order_book(self, symbol: str, exchange_id: str, depth: int = 20, callback: Callable = None) -> bool:self, symbol: str, exchange_id: str, depth: int = 20, callback: Callable = None) -> bool:
        """
        Подписывается на обновления книги ордеровкниги ордеров

        Args:
            symbol: Символ
            exchange_id: ID биржиxchange_id: ID биржи
            depth: Глубина книги ордероврдеров
            callback: Функция для обработки обновлений: Функция для обработки обновлений

        Returns:
            bool: True, если подписка успешна, иначе Falseешна, иначе False
        """
        if exchange_id not in self.websocket_clients:e_id not in self.websocket_clients:
            logger.warning("WebSocket not initialized for {exchange_id}" %)ogger.warning(f"WebSocket not initialized for {exchange_id}")
            return False            return False

        try:        try:
            # Преобразуем символразуем символ
            mapped_symbol = await self.map_symbol(symbol, exchange_id)            mapped_symbol = await self.map_symbol(symbol, exchange_id)
            if not mapped_symbol:
                logger.warning("Symbol {symbol} not found on {exchange_id}" %)     logger.warning("Symbol {symbol} not found on {exchange_id}" %)
                return False

            # Формируем сообщение подписки Формируем сообщение подписки
            subscription = self._get_orderbook_subscription(exchange_id, mapped_symbol, depth) self._get_orderbook_subscription(exchange_id, mapped_symbol, depth)
            if not subscription:
                logger.warning("Order book subscription not supported for {exchange_id}" %)ook subscription not supported for {exchange_id}")
                return False

            # Добавляем подпискубавляем подписку
            self.websocket_clients[exchange_id].add_subscription(subscription)iption(subscription)

            # Регистрируем обработчик, если указан
            if callback:
                message_type = self._get_orderbook_message_type(exchange_id)type = self._get_orderbook_message_type(exchange_id)
                if message_type:                if message_type:
                    processor = get_message_processor()        processor = get_message_processor()
                    processor.register_handler(message_type, callback)gister_handler(message_type, callback)

            return True

        except Exception as e:e:
            logger.error("Error subscribing to order book for {symbol} on {exchange_id}: {str(e)}" %)            logger.error("Error subscribing to order book for {symbol} on {exchange_id}: {str(e)}" %)
            return False

    def _get_orderbook_subscription(self, exchange_id: str, symbol: str, depth: int = 20) -> Optional[Dict]:ion(self, exchange_id: str, symbol: str, depth: int = 20) -> Optional[Dict]:
        """
        Возвращает сообщение подписки на книгу ордеров подписки на книгу ордеров

        Args:
            exchange_id: ID биржи
            symbol: Символ            symbol: Символ
            depth: Глубина книги ордеров

        Returns:
            Optional[Dict]: Сообщение подписки или Noneщение подписки или None
        """
        # Сообщения подписки для разных бирж
        if exchange_id == 'binance':        if exchange_id == 'binance':
            return {
                'method': 'SUBSCRIBE',                'method': 'SUBSCRIBE',
                'params': [f"{symbol.lower()}@depth{depth}"],symbol.lower()}@depth{depth}"],
                'id': int(time.time())
            }
        elif exchange_id == 'bybit':        elif exchange_id == 'bybit':
            return {
                'op': 'subscribe',     'op': 'subscribe',
                'args': [f"orderbook.{depth}.{symbol}"]]
            }            }
        elif exchange_id == 'okx':exchange_id == 'okx':
            return {
                'op': 'subscribe',scribe',
                'args': [{
                    'channel': 'books',                    'channel': 'books',
                    'instId': symbol,    'instId': symbol,
                    'depth': depth
                }]     }]
            }

        # Добавить другие биржи по необходимостиругие биржи по необходимости

        return None

    def _get_orderbook_message_type(self, exchange_id: str) -> Optional[str]:orderbook_message_type(self, exchange_id: str) -> Optional[str]:
        """
        Возвращает тип сообщения для книги ордеровип сообщения для книги ордеров

        Args:
            exchange_id: ID биржиxchange_id: ID биржи

        Returns:
            Optional[str]: Тип сообщения или Noneбщения или None
        """
        # Типы сообщений для разных биржж
        if exchange_id == 'binance':
            return 'depthUpdate'
        elif exchange_id == 'bybit':nge_id == 'bybit':
            return 'orderbook'eturn 'orderbook'
        elif exchange_id == 'okx':        elif exchange_id == 'okx':
            return 'books'

        # Добавить другие биржи по необходимостидругие биржи по необходимости

        return None

    async def get_deposit_address(self, currency: str, exchange_id: str) -> Optional[Dict]:str, exchange_id: str) -> Optional[Dict]:
        """        """
        Получает адрес для пополнения балансаает адрес для пополнения баланса

        Args:        Args:
            currency: Валютаency: Валюта
            exchange_id: ID биржи

        Returns:
            Optional[Dict]: Информация об адресе или Noneия об адресе или None
        """
        try:
            # Отслеживаем ограничения частоты запросовничения частоты запросов
            await self.rate_limits[exchange_id].acquire()[exchange_id].acquire()

            # Получаем экземпляр биржи            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)xchange_id)

            # Проверяем, поддерживает ли биржа эту функциюряем, поддерживает ли биржа эту функцию
            if not exchange.has.get('fetchDepositAddress', False):            if not exchange.has.get('fetchDepositAddress', False):
                logger.warning("Exchange {exchange_id} does not support deposit addresses" %))
                return None     return None

            # Получаем адрес            # Получаем адрес
            address = await exchange.fetch_deposit_address(currency)ddress = await exchange.fetch_deposit_address(currency)

            return address

        except Exception as e:xception as e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e) self._increment_error_counter(exchange_id, e)

            logger.error("Error fetching deposit address for {currency} on {exchange_id}: {str(e)}" %)ess for {currency} on {exchange_id}: {str(e)}")
            return None

    async def withdraw(self, currency: str, amount: float, address: str, exchange_id: str, tag: str = None, params: Dict = None) -> Optional[Dict]: str, amount: float, address: str, exchange_id: str, tag: str = None, params: Dict = None) -> Optional[Dict]:
        """
        Выводит средства        Выводит средства

        Args:
            currency: Валюта
            amount: Сумма
            address: Адрес            address: Адрес
            exchange_id: ID биржибиржи
            tag: Тег или memo
            params: Дополнительные параметры            params: Дополнительные параметры

        Returns:        Returns:
            Optional[Dict]: Информация о выводе или Noneформация о выводе или None
        """
        try:
            # Отслеживаем ограничения частоты запросов            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()

            # Получаем экземпляр биржи            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)

            # Проверяем, поддерживает ли биржа эту функцию поддерживает ли биржа эту функцию
            if not exchange.has.get('withdraw', False):            if not exchange.has.get('withdraw', False):
                logger.warning("Exchange {exchange_id} does not support withdrawals" %)   logger.warning("Exchange {exchange_id} does not support withdrawals" %)
                return None

            # Формируем параметрыраметры
            withdraw_params = params or {}ams or {}
            if tag:
                withdraw_params['tag'] = tag

            # Выполняем выводполняем вывод
            withdrawal = await exchange.withdraw(currency, amount, address, tag, withdraw_params)y, amount, address, tag, withdraw_params)

            # Инвалидируем кеш баланса# Инвалидируем кеш баланса
            await self.cache.invalidate('balance', exchange_id)hange_id)

            return withdrawal            return withdrawal

        except Exception as e:
            # Увеличиваем счетчик ошибок            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)

            logger.error("Error withdrawing {amount} {currency} to {address} on {exchange_id}: {str(e)}" %)ange_id}: {str(e)}")
            return None

    async def get_transfer_fee(self, currency: str, exchange_id: str) -> Optional[Dict]:lf, currency: str, exchange_id: str) -> Optional[Dict]:
        """
        Получает комиссию за выводмиссию за вывод

        Args:        Args:
            currency: Валюта
            exchange_id: ID биржи

        Returns:
            Optional[Dict]: Информация о комиссии или None
        """        """
        try:
            # Отслеживаем ограничения частоты запросов            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()mits[exchange_id].acquire()

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)            exchange = await self.get_exchange(exchange_id)

            # Получаем информацию о валютеинформацию о валюте
            currencies = await exchange.fetch_currencies()            currencies = await exchange.fetch_currencies()

            if not currencies or currency not in currencies: if not currencies or currency not in currencies:
                logger.warning("Currency {currency} not found on {exchange_id}" %)urrency {currency} not found on {exchange_id}")
                return None                return None

            # Возвращаем информацию о комиссииормацию о комиссии
            currency_info = currencies[currency]ncies[currency]

            return {rn {
                'currency': currency,
                'withdraw_fee': currency_info.get('fee'),     'withdraw_fee': currency_info.get('fee'),
                'withdraw_min': currency_info.get('limits', {}).get('withdraw', {}).get('min'),    'withdraw_min': currency_info.get('limits', {}).get('withdraw', {}).get('min'),
                'withdraw_max': currency_info.get('limits', {}).get('withdraw', {}).get('max'),its', {}).get('withdraw', {}).get('max'),
                'networks': currency_info.get('networks', {}) {})
            }            }

        except Exception as e:
            # Увеличиваем счетчик ошибок            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)exchange_id, e)

            logger.error("Error fetching transfer fee for {currency} on {exchange_id}: {str(e)}" %)            logger.error("Error fetching transfer fee for {currency} on {exchange_id}: {str(e)}" %)
            return None

    async def get_exchange_info(self, exchange_id: str) -> Optional[Dict]:info(self, exchange_id: str) -> Optional[Dict]:
        """        """
        Получает общую информацию о бирже

        Args:        Args:
            exchange_id: ID биржи_id: ID биржи

        Returns:
            Optional[Dict]: Информация о бирже или None
        """
        try:
            # Получаем экземпляр биржи Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id)            exchange = await self.get_exchange(exchange_id)

            # Формируем информацию
            info = {
                'id': exchange_id,                'id': exchange_id,
                'name': exchange.name,
                'countries': exchange.countries,ies': exchange.countries,
                'urls': exchange.urls,                'urls': exchange.urls,
                'version': exchange.version,
                'api_docs': exchange.urls.get('api') or exchange.urls.get('apiDocs'),     'api_docs': exchange.urls.get('api') or exchange.urls.get('apiDocs'),
                'has': exchange.has,
                'timeframes': exchange.timeframes,                'timeframes': exchange.timeframes,
                'timeout': exchange.timeout,   'timeout': exchange.timeout,
                'rate_limit': exchange.rate_limit,hange.rate_limit,
                'user_agent': exchange.userAgent,                'user_agent': exchange.userAgent,
                'rate_limit_usage': self.rate_limits[exchange_id].get_usage() if exchange_id in self.rate_limits else None,'rate_limit_usage': self.rate_limits[exchange_id].get_usage() if exchange_id in self.rate_limits else None,
                'error_stats': self.error_counters.get(exchange_id, {}),exchange_id, {}),
                'score': self.exchange_scores.get(exchange_id, 0)     'score': self.exchange_scores.get(exchange_id, 0)
            }}

            return info

        except Exception as e:
            logger.error("Error fetching exchange info for {exchange_id}: {str(e)}" %)rror(f"Error fetching exchange info for {exchange_id}: {str(e)}")
            return None

    async def get_markets(self, exchange_id: str, reload: bool = False) -> Optional[Dict]:, reload: bool = False) -> Optional[Dict]:
        """
        Получает список рынков

        Args:
            exchange_id: ID биржи
            reload: Перезагрузить маркеты

        Returns:
            Optional[Dict]: Список рынков или None
        """
        # Проверяем в кеше
        cached_markets = await self.cache.get('markets', exchange_id)d_markets = await self.cache.get('markets', exchange_id)
        if cached_markets and not reload:        if cached_markets and not reload:
            return cached_marketsed_markets

        try:
            # Отслеживаем ограничения частоты запросов
            await self.rate_limits[exchange_id].acquire()rate_limits[exchange_id].acquire()

            # Получаем экземпляр биржи
            exchange = await self.get_exchange(exchange_id) exchange = await self.get_exchange(exchange_id)

            # Загружаем рынки            # Загружаем рынки
            markets = await exchange.load_markets()arkets = await exchange.load_markets()

            # Сохраняем в кеш
            await self.cache.set('markets', markets, exchange_id)            await self.cache.set('markets', markets, exchange_id)

            return markets

        except Exception as e:s e:
            # Увеличиваем счетчик ошибок
            self._increment_error_counter(exchange_id, e)ter(exchange_id, e)

            logger.error("Error fetching markets on {exchange_id}: {str(e)}" %)            logger.error("Error fetching markets on {exchange_id}: {str(e)}" %)
            return Nonereturn None

    def send_order(self, exchange_id, order_params):
        """Отправляет ордер на биржу"""        """Отправляет ордер на биржу"""
        if exchange_id in self.exchanges:es:
            # ...existing code...
            pass            pass
        elif exchange_id in self.paper_trading_exchanges:elf.paper_trading_exchanges:
            # Добавляем пропущенный блок после elif
            return self.paper_trading_exchanges[exchange_id].send_order(order_params)            return self.paper_trading_exchanges[exchange_id].send_order(order_params)
        else:
            raise ValueError(f"Exchange {exchange_id} not found")


# Глобальный экземпляр менеджера бирж# Глобальный экземпляр менеджера бирж
_exchange_manager = None


async def get_exchange_manager() -> ExchangeManager:async def get_exchange_manager() -> ExchangeManager:
    """
    Возвращает глобальный экземпляр менеджера биржый экземпляр менеджера бирж

    Returns:
        ExchangeManager: Менеджер бирж
    """
    global _exchange_manager

    if _exchange_manager is None:
        _exchange_manager = ExchangeManager()
        await _exchange_manager.start()

    return _exchange_manager


async def get_exchange_instance(exchange_id: str) -> ccxt.Exchange:nge_id: str) -> ccxt.Exchange:
    """
    Возвращает экземпляр биржи    Возвращает экземпляр биржи

    Args:
        exchange_id: ID биржи exchange_id: ID биржи

    Returns:    Returns:
        ccxt.Exchange: Экземпляр биржи.Exchange: Экземпляр биржи
    """
    exchange_manager = await get_exchange_manager()hange_manager = await get_exchange_manager()
    return await exchange_manager.get_exchange(exchange_id)nager.get_exchange(exchange_id)
