import asyncio
import json
import os
import smtplib
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Union

import httpx
from project.utils.logging_utils import setup_logger

logger = setup_logger("notification")


class NotificationConfig:
    """Конфигурация системы уведомлений"""

    def __init__(self, config_dict: Dict = None):
        config = config_dict or {}

        # Общие настройки
        self.enabled = config.get("enabled", True)
        self.notification_levels = config.get(
            "notification_levels", ["error", "warning", "info"]
        )
        self.max_notifications_per_minute = config.get(
            "max_notifications_per_minute", 10
        )
        self.throttle_time = config.get("throttle_time", 60)  # в секундах
        self.queue_size = config.get("queue_size", 100)

        # Настройки Telegram
        self.telegram_enabled = config.get("telegram_enabled", False)
        self.telegram_bot_token = config.get("telegram_bot_token", "")
        self.telegram_chat_ids = config.get("telegram_chat_ids", [])
        self.telegram_parse_mode = config.get("telegram_parse_mode", "HTML")

        # Настройки Email
        self.email_enabled = config.get("email_enabled", False)
        self.email_server = config.get("email_server", "smtp.gmail.com")
        self.email_port = config.get("email_port", 587)
        self.email_use_tls = config.get("email_use_tls", True)
        self.email_username = config.get("email_username", "")
        self.email_password = config.get("email_password", "")
        self.email_from = config.get("email_from", "")
        self.email_to = config.get("email_to", [])
        self.email_subject_prefix = config.get("email_subject_prefix", "[Trading Bot]")

        # Настройки Discord webhook
        self.discord_enabled = config.get("discord_enabled", False)
        self.discord_webhook_url = config.get("discord_webhook_url", "")
        self.discord_username = config.get("discord_username", "Trading Bot")
        self.discord_avatar_url = config.get("discord_avatar_url", "")

        # Настройки Push-уведомлений
        self.push_enabled = config.get("push_enabled", False)
        self.push_service = config.get(
            "push_service", "pushover"
        )  # pushover, pushbullet, etc.
        self.push_api_key = config.get("push_api_key", "")
        self.push_user_key = config.get("push_user_key", "")

        # Настройки Slack
        self.slack_enabled = config.get("slack_enabled", False)
        self.slack_webhook_url = config.get("slack_webhook_url", "")
        self.slack_channel = config.get("slack_channel", "#alerts")
        self.slack_username = config.get("slack_username", "Trading Bot")
        self.slack_icon_emoji = config.get(
            "slack_icon_emoji", ":chart_with_upwards_trend:"
        )

        # Фильтры уведомлений
        self.categories_filter = config.get(
            "categories_filter", []
        )  # Пустой список означает "все категории"
        self.include_system_info = config.get("include_system_info", True)

        # Шаблоны сообщений
        self.templates_dir = config.get("templates_dir", "templates/notifications")
        self.load_templates()

    def load_templates(self):
        """Загружает шаблоны сообщений из файлов"""
        self.templates = {
            "default": {
                "title": "{level}: {title}",
                "body": "{message}\n\nTime: {timestamp}",
            }
        }

        # Проверяем наличие директории с шаблонами
        templates_path = Path(self.templates_dir)
        if not templates_path.exists():
            logger.warning(f"Templates directory {self.templates_dir} not found")
            return

        # Загружаем шаблоны из файлов
        for template_file in templates_path.glob("*.json"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    template_data = json.load(f)
                    self.templates[template_file.stem] = template_data
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {str(e)}")


class NotificationService:
    """Сервис для отправки уведомлений через различные каналы"""

    _instance = None
    _initialized = False

    def __new__(cls, config: Dict = None):
        if cls._instance is None:
            cls._instance = super(NotificationService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Dict = None):
        if self._initialized:
            return

        self.config = NotificationConfig(config)
        self.queue = asyncio.Queue(maxsize=self.config.queue_size)
        self.history = []
        self.notification_count = 0
        self.last_reset_time = time.time()
        self.processing_task = None
        self._initialized = True
        logger.info("Сервис уведомлений инициализирован")

    async def start(self):
        """Запускает обработчик очереди уведомлений"""
        if self.processing_task is not None:
            logger.warning("Обработчик уведомлений уже запущен")
            return

        self.processing_task = asyncio.create_task(self._process_notification_queue())
        logger.info("Обработчик уведомлений запущен")

    async def stop(self):
        """Останавливает обработчик уведомлений"""
        if self.processing_task is None:
            logger.warning("Обработчик уведомлений не запущен")
            return

        self.processing_task.cancel()
        try:
            await self.processing_task
        except asyncio.CancelledError:
            pass
        self.processing_task = None
        logger.info("Обработчик уведомлений остановлен")

    async def _process_notification_queue(self):
        """Обрабатывает очередь уведомлений"""
        while True:
            try:
                # Получаем уведомление из очереди
                notification = await self.queue.get()

                # Отправляем уведомление
                await self._send_notification(notification)

                # Отмечаем задачу как выполненную
                self.queue.task_done()

                # Небольшая пауза для предотвращения спама
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("Обработчик очереди уведомлений отменен")
                break
            except Exception as e:
                logger.error(f"Ошибка при обработке уведомления: {str(e)}")

    async def _send_notification(self, notification: Dict):
        """Отправляет уведомление через все активные каналы"""
        # Применяем форматирование
        formatted = self._format_notification(notification, notification.get('template', 'default'))

        # Добавляем системную информацию
        formatted = self._add_system_info(formatted)

        # Отправляем через разные каналы
        sent = False

        # Telegram
        if self.config.telegram_enabled:
            telegram_sent = await self._send_telegram(formatted)
            sent = sent or telegram_sent

        # Другие каналы - здесь можно добавить

        # Добавляем в историю
        self._add_to_history(formatted)

        # Увеличиваем счетчик уведомлений
        self.notification_count += 1

        # Сбрасываем счетчик, если прошел период троттлинга
        current_time = time.time()
        if current_time - self.last_reset_time > self.config.throttle_time:
            self.notification_count = 0
            self.last_reset_time = current_time

    def _format_notification(self, notification: Dict, template_name: str) -> Dict:
        """
        Форматирует уведомление с использованием шаблона

        Args:
            notification: Словарь с данными уведомления
            template_name: Имя шаблона

        Returns:
            Dict: Отформатированное уведомление
        """
        # Получаем шаблон или используем шаблон по умолчанию
        template = self.config.templates.get(
            template_name, self.config.templates["default"]
        )

        # Подготавливаем данные для подстановки в шаблон
        data = {
            "title": notification.get("title", "Notification"),
            "message": notification.get("message", ""),
            "level": notification.get("level", "info"),
            "timestamp": notification.get("timestamp", datetime.now().isoformat()),
            "category": notification.get("category", "general"),
        }

        # Добавляем все остальные данные из уведомления
        for key, value in notification.items():
            if key not in data:
                data[key] = value

        # Форматируем заголовок и текст
        title_template = template.get(
            "title", self.config.templates["default"]["title"]
        )
        body_template = template.get("body", self.config.templates["default"]["body"])

        try:
            title = title_template.format(**data)
            body = body_template.format(**data)
        except KeyError as e:
            logger.warning(f"Missing key in notification template: {str(e)}")
            title = notification.get("title", "Notification")
            body = notification.get("message", "")

        # Возвращаем отформатированное уведомление
        return {
            "title": title,
            "body": body,
            "level": notification.get("level", "info"),
            "original": notification,
        }

    def _add_system_info(self, notification: Dict) -> Dict:
        """
        Добавляет системную информацию к уведомлению

        Args:
            notification: Отформатированное уведомление

        Returns:
            Dict: Уведомление с добавленной системной информацией
        """
        # Добавляем информацию о времени, хосте, и т.д.
        body = notification["body"]
        body += f"\n\nSystem: {os.uname().nodename}"
        body += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        notification["body"] = body
        return notification

    async def _send_telegram(self, notification: Dict) -> bool:
        """
        Отправляет уведомление через Telegram

        Args:
            notification: Отформатированное уведомление

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        if not self.http_client:
            return False

        for chat_id in self.config.telegram_chat_ids:
            try:
                # Формируем сообщение
                message = f"<b>{notification['title']}</b>\n\n{notification['body']}"

                # Готовим данные для запроса
                data = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": self.config.telegram_parse_mode,
                    "disable_web_page_preview": True,
                }

                # Отправляем запрос
                url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
                response = await self.http_client.post(url, json=data)

                # Проверяем результат
                if response.status_code != 200:
                    logger.error(f"Telegram API error: {response.text}")
                    return False

            except Exception as e:
                logger.error(f"Error sending Telegram notification: {str(e)}")
                return False

        return True

    def _add_to_history(self, notification: Dict):
        """
        Добавляет уведомление в историю

        Args:
            notification: Уведомление для добавления в историю
        """
        # Добавляем отметку времени, если ее нет
        if "timestamp" not in notification:
            notification["timestamp"] = datetime.now().isoformat()

        # Добавляем уведомление в начало списка
        self.history.insert(0, notification)

        # Ограничиваем размер истории
        if len(self.history) > self.config.queue_size:
            self.history = self.history[: self.config.queue_size]


# Глобальный экземпляр сервиса уведомлений
_notification_service = None


def init_notification_service(config: Dict = None):
    """
    Инициализирует глобальный сервис уведомлений

    Args:
        config: Конфигурация сервиса
    """
    global _notification_service
    _notification_service = NotificationService(config)
    return _notification_service


async def start_notification_service():
    """Запускает глобальный сервис уведомлений"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    await _notification_service.start()


async def stop_notification_service():
    """Останавливает глобальный сервис уведомлений"""
    global _notification_service
    if _notification_service is not None:
        await _notification_service.stop()
