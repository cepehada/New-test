"""
Главный модуль приложения.
Запускает бота и все фоновые сервисы.
"""

import asyncio
import logging
from aiogram import Bot, Dispatcher
from project.config import load_config
from project.utils.logging_utils import configure_logging
from project.bots.arbitrage.core import ArbitrageBot
from project.bots.news.news_bot import NewsBot
from project.integrations.binance_liquidations import (
    BinanceLiquidationMonitor
)
from project.integrations.telegram_integration import router as tg_router
from project.risk_management.portfolio_manager import (
    PortfolioManager
)

config = load_config()

class TradingApplication:
    """
    Класс приложения. Инициализирует и запускает сервисы.
    """

    def __init__(self) -> None:
        configure_logging(config["logging"])
        self.bot = Bot(token=config["telegram"]["main_token"])
        self.dp = Dispatcher()
        self.dp.include_router(tg_router)
        self.services = {
            "arbitrage": ArbitrageBot(config),
            "news": NewsBot(config),
            "liquidations": BinanceLiquidationMonitor(),
            "portfolio": PortfolioManager()
        }

    async def _start_service(self, name: str) -> None:
        """
        Запускает сервис с автоматическим перезапуском.
        """
        service = self.services[name]
        while True:
            try:
                await service.run()
            except Exception as e:
                logging.error(f"Service {name} failed: {e}")
                await asyncio.sleep(10)

    async def run(self) -> None:
        """
        Запускает поллинг бота и фоновые сервисы.
        """
        tasks = [
            self.dp.start_polling(self.bot),
            self._start_service("arbitrage"),
            self._start_service("news"),
            self._start_service("liquidations")
        ]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    app = TradingApplication()
    asyncio.run(app.run())
