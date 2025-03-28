from typing import Dict, List, Optional

from project.exchange.exchange_base import BaseExchangeAdapter
from project.exchange.adapters.binance_adapter import BinanceAdapter
from project.exchange.adapters.bybit_adapter import BybitAdapter
from project.exchange.adapters.htx_adapter import HTXAdapter
from project.exchange.adapters.phemex_adapter import PhemexAdapter
from project.exchange.adapters.mex_adapter import MEXAdapter
from project.utils.logging_utils import setup_logger

logger = setup_logger("exchange_factory")

class ExchangeFactory:
    """Фабрика для создания адаптеров бирж"""
    
    _adapters: Dict[str, type] = {
        "binance": BinanceAdapter,
        "bybit": BybitAdapter,
        "htx": HTXAdapter,
        "phemex": PhemexAdapter,
        "mex": MEXAdapter,
        # Можно добавить другие биржи
    }
    
    @classmethod
    def create(cls, exchange_id: str, config: Optional[Dict] = None) -> Optional[BaseExchangeAdapter]:
        """
        Создает адаптер для указанной биржи
        
        Args:
            exchange_id: Идентификатор биржи
            config: Конфигурация (необязательно)
            
        Returns:
            BaseExchangeAdapter: Экземпляр адаптера или None при ошибке
        """
        adapter_class = cls._adapters.get(exchange_id.lower())
        if not adapter_class:
            logger.error(f"Неподдерживаемая биржа: {exchange_id}")
            return None
            
        try:
            return adapter_class(config)
        except Exception as e:
            logger.error(f"Ошибка создания адаптера для {exchange_id}: {str(e)}")
            return None
    
    @classmethod
    def get_supported_exchanges(cls) -> List[str]:
        """
        Возвращает список поддерживаемых бирж
        
        Returns:
            List[str]: Список идентификаторов поддерживаемых бирж
        """
        return list(cls._adapters.keys())
