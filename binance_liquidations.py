"""
Модуль мониторинга ликвидаций на Binance.
Использует WebSocket для получения данных о ликвидациях.
"""

import asyncio
import json
import logging
import websockets
from typing import Any
from project.config import load_config
from project.utils.notify import NotificationManager

config = load_config()
logger = logging.getLogger("BinanceLiquidation")
notifier = NotificationManager()

class BinanceLiquidationMonitor:
    """
    Класс для мониторинга ликвидаций на Binance.
    Подключается к WebSocket и обрабатывает сообщения.
    """

    def __init__(self) -> None:
        self.ws_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        self.active = True
        self.threshold = 500000  # Порог в долларах

    async def process_message(self, message: dict) -> None:
        """
        Обрабатывает сообщение WebSocket.
        
        Args:
            message (dict): Данные ликвидации.
        """
        try:
            data = message.get("o", {})
            symbol = data.get("s", "UNKNOWN")
            side = "LONG" if data.get("S", "") == "BUY" else "SHORT"
            quantity = float(data.get("q", 0))
            price = float(data.get("ap", 0))
            notional = quantity * price
            if notional >= self.threshold:
                alert = (f"🚨 Ликвидация на {symbol}: {side} " 
                         f"Объем: ${notional:,.2f}")
                logger.info(alert)
                await notifier.send_alert(alert)
        except Exception as e:
            logger.error(f"Process message error: {e}")

    async def run(self) -> None:
        """
        Запускает подключение к WebSocket Binance и обрабатывает сообщения.
        """
        while self.active:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Подключено к Binance WS")
                    async for msg in ws:
                        data = json.loads(msg)
                        await self.process_message(data)
            except Exception as e:
                logger.error(f"Binance WS error: {e}")
                await asyncio.sleep(5)

