import asyncio
import time
from typing import Dict, List, Any, Optional

import ccxt.async_support as ccxt

from project.exchange.exchange_base import BaseExchangeAdapter
from project.exchange.exchange_errors import ExchangeErrorHandler
from project.utils.logging_utils import setup_logger

logger = setup_logger("binance_adapter")


class BinanceAdapter(BaseExchangeAdapter):
    """Адаптер для работы с биржей Binance"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Инициализирует адаптер Binance

        Args:
            config: Конфигурация (необязательно)
        """
        super().__init__(config)
        self.exchange_id = "binance"
        self.name = "Binance"

    async def initialize(self) -> bool:
        """
        Инициализирует адаптер Binance

        Returns:
            bool: True в случае успешной инициализации
        """
        try:
            # Получаем настройки API
            api_key = self.config.get("BINANCE_API_KEY", "")
            api_secret = self.config.get("BINANCE_API_SECRET", "")

            # Создаем экземпляр CCXT
            self.ccxt_instance = ccxt.binance(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    "timeout": 30000,
                    "options": {"adjustForTimeDifference": True, "recvWindow": 60000},
                }
            )

            # Настройка тестовой сети (если используется)
            if self.config.get("ENABLE_PAPER_TRADING", False):
                self.ccxt_instance.options["defaultType"] = "future"
                self.ccxt_instance.options["test"] = True
                logger.info(f"{self.name}: Включен режим тестовой сети")

            # Проверка соединения
            await self.ccxt_instance.load_markets()
            logger.info(f"{self.name}: Адаптер инициализирован успешно")
            return True

        except Exception as e:
            error_code, error_msg, _ = ExchangeErrorHandler.handle_error(
                e, self.exchange_id
            )
            logger.error(f"{self.name}: Ошибка инициализации адаптера: {error_msg}")
            return False

    async def shutdown(self) -> bool:
        """
        Останавливает адаптер Binance

        Returns:
            bool: True в случае успешной остановки
        """
        try:
            if self.ccxt_instance:
                await self.ccxt_instance.close()
                self.ccxt_instance = None

            logger.info(f"{self.name}: Адаптер остановлен")
            return True

        except Exception as e:
            logger.error(f"{self.name}: Ошибка при остановке адаптера: {str(e)}")
            return False

    # Специфичные для Binance методы могут быть добавлены здесь
