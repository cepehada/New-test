"""
Модуль уведомлений.
Отправляет сообщения в Telegram через API.
"""

import logging
import aiohttp
from project.config import load_config

config = load_config()

class NotificationManager:
    """
    Класс для отправки уведомлений в Telegram.
    """

    def __init__(self) -> None:
        self.telegram_token = config["telegram"]["main_token"]
        self.chat_id = config["telegram"]["main_chat_id"]
        self.api_url = (f"https://api.telegram.org/bot"
                        f"{self.telegram_token}/sendMessage")
        self.logger = logging.getLogger("NotificationManager")

    async def send_alert(self, message: str) -> None:
        """
        Отправляет уведомление в Telegram.

        Args:
            message (str): Текст уведомления.
        """
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(self.api_url,
                                     json=payload) as resp:
                    if resp.status != 200:
                        self.logger.error(
                            f"Telegram alert failed: {await resp.text()}")
        except Exception as e:
            self.logger.error(f"Send alert exception: {e}")

    async def multi_chat_notify_admin(self,
                                      message: str,
                                      chat_ids: list) -> None:
        """
        Отправляет уведомление в несколько чатов.

        Args:
            message (str): Текст сообщения.
            chat_ids (list): Список идентификаторов чатов.
        """
        for cid in chat_ids:
            payload = {"chat_id": cid, "text": message,
                       "parse_mode": "Markdown"}
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(self.api_url,
                                         json=payload) as resp:
                        if resp.status != 200:
                            self.logger.error(
                                f"Alert to {cid} failed: "
                                f"{await resp.text()}")
            except Exception as e:
                self.logger.error(
                    f"Multi chat notify error for {cid}: {e}")
