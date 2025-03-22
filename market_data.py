"""
Market Data.
Предоставляет методы для получения тикеров и ордербуков
в реальном времени с кэшированием и таймаутами.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Tuple
from project.config import load_config
config = load_config()
from project.utils.ccxt_exchanges import ExchangeManager

logger = logging.getLogger("MarketData")

class MarketData:
    """
    Класс для получения рыночных данных с бирж.
    
    Методы get_ticker() и get_order_book() возвращают данные,
    используя кэш с TTL.
    """

    def __init__(self) -> None:
        self.ex_manager = ExchangeManager(config)
        # cache: key -> (timestamp, data)
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.cache_ttl: int = config.system.CACHE_TTL

    def _is_cache_valid(self, key: str) -> bool:
        """
        Проверяет, действительны ли кэшированные данные.
        
        Args:
            key (str): Ключ кэша.
        
        Returns:
            bool: True, если данные действительны.
        """
        if key in self.cache:
            timestamp, _ = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return True
        return False

    async def get_ticker(self, exch_id: str, symbol: str) -> Dict[str, Any]:
        """
        Получает тикер для заданной биржи и пары.
        
        Args:
            exch_id (str): Идентификатор биржи.
            symbol (str): Торговая пара.
        
        Returns:
            Dict[str, Any]: Тикер.
        """
        key = f"ticker_{exch_id}_{symbol}"
        if self._is_cache_valid(key):
            _, data = self.cache[key]
            return data
        exch = self.ex_manager.get_exchange(exch_id)
        try:
            ticker = await asyncio.wait_for(
                exch.fetch_ticker(symbol), timeout=5
            )
            self.cache[key] = (time.time(), ticker)
            return ticker
        except Exception as e:
            logger.error(
                f"Ошибка получения тикера {exch_id} {symbol}: {e}"
            )
            return {}

    async def get_order_book(self, exch_id: str, symbol: str) -> Dict[str, Any]:
        """
        Получает ордербук для заданной биржи и пары.
        
        Args:
            exch_id (str): Идентификатор биржи.
            symbol (str): Торговая пара.
        
        Returns:
            Dict[str, Any]: Ордербук.
        """
        key = f"orderbook_{exch_id}_{symbol}"
        if self._is_cache_valid(key):
            _, data = self.cache[key]
            return data
        exch = self.ex_manager.get_exchange(exch_id)
        try:
            order_book = await asyncio.wait_for(
                exch.fetch_order_book(symbol), timeout=5
            )
            self.cache[key] = (time.time(), order_book)
            return order_book
        except Exception as e:
            logger.error(
                f"Ошибка получения ордербука {exch_id} {symbol}: {e}"
            )
            return {}
