"""
Модуль для интеграции со Slack.
Предоставляет функции для отправки сообщений и уведомлений в Slack.
"""

import asyncio
from typing import Any, Dict, List, Optional

from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

logger = get_logger(__name__)


class SlackIntegration:
    """
    Класс для взаимодействия со Slack API.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "SlackIntegration":
        """
        Получает экземпляр интеграции со Slack (Singleton).

        Returns:
            Экземпляр класса SlackIntegration
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, token: Optional[str] = None):
        """
        Инициализирует интеграцию со Slack.

        Args:
            token: Токен Slack API (None для использования из конфигурации)
        """
        self.config = get_config()
        self.token = token

        # В реальном приложении токен должен быть в конфигурации
        # self.token = token or self.config.SLACK_API_TOKEN

        if not self.token:
            logger.warning("Slack-интеграция не будет работать - токен не указан")
            self.client = None
            return

        self.client = AsyncWebClient(token=self.token)
        self.default_channel = "#general"  # Канал по умолчанию

        logger.info("Интеграция со Slack инициализирована")

    @async_handle_error
    async def send_message(
        self,
        text: str,
        channel: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        thread_ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Отправляет сообщение в Slack.

        Args:
            text: Текст сообщения
            channel: ID канала или его имя (None для использования канала по умолчанию)
            attachments: Прикрепления к сообщению
            blocks: Блоки сообщения (для форматирования)
            thread_ts: Timestamp для ответа в треде

        Returns:
            Результат отправки сообщения
        """
        if not self.client:
            logger.warning("Slack-клиент не настроен - сообщение не отправлено")
            return {"ok": False, "error": "Client not configured"}

        channel = channel or self.default_channel

        try:
            response = await self.client.chat_postMessage(
                channel=channel,
                text=text,
                attachments=attachments,
                blocks=blocks,
                thread_ts=thread_ts,
            )

            logger.debug("Сообщение отправлено в Slack канал {channel}" %)
            return response
        except SlackApiError as e:
            logger.error("Ошибка Slack API при отправке сообщения: {str(e)}" %)
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.error("Ошибка при отправке сообщения в Slack: {str(e)}" %)
            return {"ok": False, "error": str(e)}

    @async_handle_error
    async def send_file(
        self,
        file_path: str,
        title: Optional[str] = None,
        comment: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Отправляет файл в Slack.

        Args:
            file_path: Путь к файлу
            title: Заголовок файла
            comment: Комментарий к файлу
            channel: ID канала или его имя (None для использования канала по умолчанию)

        Returns:
            Результат отправки файла
        """
        if not self.client:
            logger.warning("Slack-клиент не настроен - файл не отправлен")
            return {"ok": False, "error": "Client not configured"}

        channel = channel or self.default_channel

        try:
            response = await self.client.files_upload(
                file=file_path, title=title, initial_comment=comment, channels=channel
            )

            logger.debug("Файл {file_path} отправлен в Slack канал {channel}" %)
            return response
        except SlackApiError as e:
            logger.error("Ошибка Slack API при отправке файла: {str(e)}" %)
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.error("Ошибка при отправке файла в Slack: {str(e)}" %)
            return {"ok": False, "error": str(e)}

    @async_handle_error
    async def send_chart(
        self,
        chart_path: str,
        title: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Отправляет график в Slack.

        Args:
            chart_path: Путь к файлу графика
            title: Заголовок графика
            channel: ID канала или его имя (None для использования канала по умолчанию)

        Returns:
            Результат отправки графика
        """
        comment = "Trading Chart" if not title else f"Trading Chart: {title}"
        return await self.send_file(chart_path, title, comment, channel)

    @async_handle_error
    async def create_channel(self, name: str) -> Dict[str, Any]:
        """
        Создает новый канал в Slack.

        Args:
            name: Имя канала

        Returns:
            Результат создания канала
        """
        if not self.client:
            logger.warning("Slack-клиент не настроен - канал не создан")
            return {"ok": False, "error": "Client not configured"}

        try:
            response = await self.client.conversations_create(name=name)

            logger.info("Создан Slack канал {name}" %)
            return response
        except SlackApiError as e:
            logger.error("Ошибка Slack API при создании канала: {str(e)}" %)
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.error("Ошибка при создании канала в Slack: {str(e)}" %)
            return {"ok": False, "error": str(e)}

    @async_handle_error
    async def get_channel_history(
        self, channel: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Получает историю сообщений в канале.

        Args:
            channel: ID канала или его имя (None для использования канала по умолчанию)
            limit: Максимальное количество сообщений

        Returns:
            Список сообщений
        """
        if not self.client:
            logger.warning("Slack-клиент не настроен - история не получена")
            return []

        channel = channel or self.default_channel

        try:
            response = await self.client.conversations_history(
                channel=channel, limit=limit
            )

            logger.debug("Получена история Slack канала {channel}" %)
            return response.get("messages", [])
        except SlackApiError as e:
            logger.error("Ошибка Slack API при получении истории: {str(e)}" %)
            return []
        except Exception as e:
            logger.error("Ошибка при получении истории Slack: {str(e)}" %)
            return []

    @async_handle_error
    async def send_alert(
        self,
        title: str,
        message: str,
        color: str = "danger",
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Отправляет предупреждающее сообщение в Slack.

        Args:
            title: Заголовок сообщения
            message: Текст сообщения
            color: Цвет сообщения (good, warning, danger)
            channel: ID канала или его имя (None для использования канала по умолчанию)

        Returns:
            Результат отправки сообщения
        """
        attachment = {
            "color": color,
            "title": title,
            "text": message,
            "ts": int(asyncio.get_event_loop().time()),
        }

        return await self.send_message(
            text=f"Alert: {title}", channel=channel, attachments=[attachment]
        )

    @async_handle_error
    async def send_trade_notification(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        order_type: str,
        status: str,
        channel: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Отправляет уведомление о сделке в Slack.

        Args:
            symbol: Символ торговой пары
            side: Сторона сделки (buy/sell)
            amount: Количество
            price: Цена
            order_type: Тип ордера
            status: Статус сделки
            channel: ID канала или его имя (None для использования канала по умолчанию)

        Returns:
            Результат отправки уведомления
        """
        # Определяем цвет и эмодзи в зависимости от стороны сделки
        color = "#36a64f" if side.lower() == "buy" else "#d72b3f"
        emoji = "📈" if side.lower() == "buy" else "📉"

        # Создаем блоки сообщения
        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Trade Notification* {emoji}"},
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Symbol:*\n{symbol}"},
                    {"type": "mrkdwn", "text": f"*Side:*\n{side.upper()}"},
                    {"type": "mrkdwn", "text": f"*Amount:*\n{amount}"},
                    {"type": "mrkdwn", "text": f"*Price:*\n{price}"},
                    {"type": "mrkdwn", "text": f"*Type:*\n{order_type}"},
                    {"type": "mrkdwn", "text": f"*Status:*\n{status}"},
                ],
            },
        ]

        # Создаем вложение для цвета
        attachments = [{"color": color, "blocks": []}]

        return await self.send_message(
            text=f"Trade: {side.upper()} {amount} {symbol} at {price}",
            channel=channel,
            blocks=blocks,
            attachments=attachments,
        )
