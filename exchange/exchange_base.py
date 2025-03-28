import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from project.utils.error_handler import async_with_retry
from project.utils.logging_utils import setup_logger

logger = setup_logger("exchange_base")

class BaseExchangeAdapter(ABC):
    """Базовый класс для адаптеров бирж"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализирует адаптер биржи
        
        Args:
            config: Конфигурация (необязательно)
        """
        self.config = config or {}
        self.exchange_id = "base"  # Должно быть переопределено в подклассах
        self.name = "Base Exchange"  # Должно быть переопределено в подклассах
        self.ccxt_instance = None
        self.rate_limiter = None
        self.websocket = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Инициализирует адаптер биржи
        
        Returns:
            bool: True в случае успешной инициализации
        """
        pass
        
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Останавливает адаптер биржи
        
        Returns:
            bool: True в случае успешной остановки
        """
        pass
        
    @async_with_retry(retries=3)
    async def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """
        Получает тикер для указанного символа
        
        Args:
            symbol: Символ (пара)
            
        Returns:
            Optional[Dict]: Данные тикера или None при ошибке
        """
        # Реализация по умолчанию использует CCXT
        # Подклассы могут переопределить для оптимизации
        if not self.ccxt_instance:
            logger.error(f"{self.name}: CCXT-экземпляр не инициализирован")
            return None
            
        try:
            return await self.ccxt_instance.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"{self.name}: Ошибка получения тикера для {symbol}: {str(e)}")
            return None
    
    # Добавьте другие общие методы для всех бирж
