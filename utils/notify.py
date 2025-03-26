import asyncio
import logging
import json
import httpx
import smtplib
import os
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

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
        return cls._instance

    def __init__(self, config: Dict = None):
        if self._initialized:
            return

        self.config = NotificationConfig(config)

        # Очередь уведомлений
        self.notification_queue = asyncio.Queue(maxsize=self.config.queue_size)

        # Счетчик уведомлений и метка времени для ограничения частоты
        self.notification_count = 0
        self.last_reset_time = time.time()

        # HTTP-клиент для отправки запросов
        self.http_client = None

        # Флаг для остановки фоновой задачи
        self._stop_requested = False

        # Задача обработки очереди уведомлений
        self._processor_task = None

        # История отправленных уведомлений
        self.notification_history: List[Dict] = []
        self.max_history_size = 100

        self._initialized = True
        logger.info("Notification service initialized")

    async def start(self):
        """Запускает сервис уведомлений"""
        if not self.config.enabled:
            logger.info("Notification service is disabled")
            return

        if self._processor_task is not None:
            logger.warning("Notification service is already running")
            return

        # Создаем HTTP-клиент
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Запускаем обработчик очереди
        self._stop_requested = False
        self._processor_task = asyncio.create_task(self._process_notification_queue())

        logger.info("Notification service started")

    async def stop(self):
        """Останавливает сервис уведомлений"""
        if self._processor_task is None:
            logger.warning("Notification service is not running")
            return

        self._stop_requested = True

        # Отменяем задачу обработки очереди
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None

        # Закрываем HTTP-клиент
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        logger.info("Notification service stopped")

    async def _process_notification_queue(self):
        """Обрабатывает очередь уведомлений"""
        while not self._stop_requested:
            try:
                # Получаем уведомление из очереди
                notification = await self.notification_queue.get()

                # Обрабатываем ограничение частоты
                current_time = time.time()
                if current_time - self.last_reset_time > self.config.throttle_time:
                    # Сбрасываем счетчик, если прошло достаточно времени
                    self.notification_count = 0
                    self.last_reset_time = current_time

                # Проверяем, не превышен ли лимит уведомлений
                if self.notification_count >= self.config.max_notifications_per_minute:
                    logger.warning(
                        f"Rate limit exceeded: {self.config.max_notifications_per_minute} notifications per {self.config.throttle_time} seconds"
                    )

                    # Если это важное уведомление (error), все равно отправляем
                    if notification.get("level") != "error":
                        self.notification_queue.task_done()
                        continue

                # Отправляем уведомление через все настроенные каналы
                await self._send_notification(notification)

                # Увеличиваем счетчик
                self.notification_count += 1

                # Добавляем в историю
                self._add_to_history(notification)

                # Отмечаем задачу как выполненную
                self.notification_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing notification: {str(e)}")
                await asyncio.sleep(1)

    async def _send_notification(self, notification: Dict):
        """
        Отправляет уведомление через все настроенные каналы

        Args:
            notification: Словарь с данными уведомления
        """
        level = notification.get("level", "info")

        # Проверяем, нужно ли отправлять уведомление этого уровня
        if level not in self.config.notification_levels:
            return

        # Проверяем фильтры категорий
        category = notification.get("category", "general")
        if (
            self.config.categories_filter
            and category not in self.config.categories_filter
        ):
            return

        # Форматируем сообщение с использованием шаблона
        template_name = notification.get("template", "default")
        notification_formatted = self._format_notification(notification, template_name)

        # Добавляем системную информацию, если требуется
        if self.config.include_system_info:
            notification_formatted = self._add_system_info(notification_formatted)

        # Отправляем через разные каналы
        tasks = []

        # Telegram
        if (
            self.config.telegram_enabled
            and self.config.telegram_bot_token
            and self.config.telegram_chat_ids
        ):
            tasks.append(self._send_telegram(notification_formatted))

        # Email
        if (
            self.config.email_enabled
            and self.config.email_username
            and self.config.email_password
            and self.config.email_to
        ):
            tasks.append(self._send_email(notification_formatted))

        # Discord
        if self.config.discord_enabled and self.config.discord_webhook_url:
            tasks.append(self._send_discord(notification_formatted))

        # Push-уведомления
        if self.config.push_enabled and self.config.push_api_key:
            tasks.append(self._send_push(notification_formatted))

        # Slack
        if self.config.slack_enabled and self.config.slack_webhook_url:
            tasks.append(self._send_slack(notification_formatted))

        # Ждем завершения всех задач
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Логируем результаты
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Error sending notification via channel {i}: {str(result)}"
                    )

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

    async def _send_email(self, notification: Dict) -> bool:
        """
        Отправляет уведомление по электронной почте

        Args:
            notification: Отформатированное уведомление

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        try:
            # Создаем сообщение
            msg = MIMEMultipart()
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(self.config.email_to)
            msg["Subject"] = (
                f"{self.config.email_subject_prefix} {notification['title']}"
            )

            # Добавляем текст сообщения
            msg.attach(MIMEText(notification["body"], "plain"))

            # Отправляем сообщение
            with smtplib.SMTP(
                self.config.email_server, self.config.email_port
            ) as server:
                if self.config.email_use_tls:
                    server.starttls()
                server.login(self.config.email_username, self.config.email_password)
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            return False

    async def _send_discord(self, notification: Dict) -> bool:
        """
        Отправляет уведомление через Discord webhook

        Args:
            notification: Отформатированное уведомление

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        if not self.http_client:
            return False

        try:
            # Определяем цвет в зависимости от уровня
            color_map = {
                "error": 0xFF0000,  # красный
                "warning": 0xFFAA00,  # оранжевый
                "info": 0x00AAFF,  # синий
                "success": 0x00FF00,  # зеленый
                "debug": 0xAAAAAA,  # серый
            }

            level = notification["original"].get("level", "info")
            color = color_map.get(level, 0x00AAFF)

            # Готовим данные для запроса
            data = {
                "username": self.config.discord_username,
                "avatar_url": self.config.discord_avatar_url,
                "embeds": [
                    {
                        "title": notification["title"],
                        "description": notification["body"],
                        "color": color,
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
            }

            # Отправляем запрос
            response = await self.http_client.post(
                self.config.discord_webhook_url, json=data
            )

            # Проверяем результат
            if response.status_code not in [200, 204]:
                logger.error(f"Discord API error: {response.text}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error sending Discord notification: {str(e)}")
            return False

    async def _send_push(self, notification: Dict) -> bool:
        """
        Отправляет Push-уведомление

        Args:
            notification: Отформатированное уведомление

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        if not self.http_client:
            return False

        try:
            if self.config.push_service == "pushover":
                # Готовим данные для запроса Pushover
                data = {
                    "token": self.config.push_api_key,
                    "user": self.config.push_user_key,
                    "title": notification["title"],
                    "message": notification["body"],
                    "priority": self._get_pushover_priority(
                        notification["original"].get("level", "info")
                    ),
                }

                # Отправляем запрос
                response = await self.http_client.post(
                    "https://api.pushover.net/1/messages.json", data=data
                )

                # Проверяем результат
                if response.status_code != 200:
                    logger.error(f"Pushover API error: {response.text}")
                    return False

                return True

            elif self.config.push_service == "pushbullet":
                # Готовим данные для запроса Pushbullet
                data = {
                    "type": "note",
                    "title": notification["title"],
                    "body": notification["body"],
                }

                headers = {
                    "Access-Token": self.config.push_api_key,
                    "Content-Type": "application/json",
                }

                # Отправляем запрос
                response = await self.http_client.post(
                    "https://api.pushbullet.com/v2/pushes", json=data, headers=headers
                )

                # Проверяем результат
                if response.status_code != 200:
                    logger.error(f"Pushbullet API error: {response.text}")
                    return False

                return True

            else:
                logger.error(f"Unsupported push service: {self.config.push_service}")
                return False

        except Exception as e:
            logger.error(f"Error sending push notification: {str(e)}")
            return False

    def _get_pushover_priority(self, level: str) -> int:
        """
        Возвращает приоритет Pushover в зависимости от уровня уведомления

        Args:
            level: Уровень уведомления

        Returns:
            int: Приоритет Pushover
        """
        priority_map = {
            "error": 1,  # High priority
            "warning": 0,  # Normal priority
            "info": -1,  # Low priority
            "success": -1,  # Low priority
            "debug": -2,  # Lowest priority
        }

        return priority_map.get(level, 0)

    async def _send_slack(self, notification: Dict) -> bool:
        """
        Отправляет уведомление через Slack webhook

        Args:
            notification: Отформатированное уведомление

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        if not self.http_client:
            return False

        try:
            # Определяем цвет в зависимости от уровня
            color_map = {
                "error": "#FF0000",  # красный
                "warning": "#FFAA00",  # оранжевый
                "info": "#00AAFF",  # синий
                "success": "#00FF00",  # зеленый
                "debug": "#AAAAAA",  # серый
            }

            level = notification["original"].get("level", "info")
            color = color_map.get(level, "#00AAFF")

            # Готовим данные для запроса
            data = {
                "channel": self.config.slack_channel,
                "username": self.config.slack_username,
                "icon_emoji": self.config.slack_icon_emoji,
                "attachments": [
                    {
                        "title": notification["title"],
                        "text": notification["body"],
                        "color": color,
                        "ts": int(time.time()),
                    }
                ],
            }

            # Отправляем запрос
            response = await self.http_client.post(
                self.config.slack_webhook_url, json=data
            )

            # Проверяем результат
            if response.status_code != 200:
                logger.error(f"Slack API error: {response.text}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error sending Slack notification: {str(e)}")
            return False

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
        self.notification_history.insert(0, notification)

        # Ограничиваем размер истории
        if len(self.notification_history) > self.max_history_size:
            self.notification_history = self.notification_history[
                : self.max_history_size
            ]

    async def get_notification_history(
        self, limit: int = None, level: str = None, category: str = None
    ) -> List[Dict]:
        """
        Возвращает историю отправленных уведомлений с опциональной фильтрацией

        Args:
            limit: Максимальное количество уведомлений
            level: Фильтр по уровню уведомления
            category: Фильтр по категории уведомления

        Returns:
            List[Dict]: Список уведомлений
        """
        filtered_history = self.notification_history

        # Применяем фильтр по уровню
        if level:
            filtered_history = [n for n in filtered_history if n.get("level") == level]

        # Применяем фильтр по категории
        if category:
            filtered_history = [
                n for n in filtered_history if n.get("category") == category
            ]

        # Применяем ограничение количества
        if limit:
            filtered_history = filtered_history[:limit]

        return filtered_history

    async def clear_notification_history(self):
        """Очищает историю отправленных уведомлений"""
        self.notification_history = []
        logger.info("Notification history cleared")

    async def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info",
        category: str = "general",
        template: str = "default",
        **kwargs,
    ) -> bool:
        """
        Отправляет уведомление

        Args:
            title: Заголовок уведомления
            message: Текст уведомления
            level: Уровень важности (error, warning, info, success, debug)
            category: Категория уведомления
            template: Название шаблона для форматирования
            **kwargs: Дополнительные параметры

        Returns:
            bool: True, если уведомление добавлено в очередь, иначе False
        """
        if not self.config.enabled:
            return False

        # Проверяем уровень уведомления
        if level not in self.config.notification_levels:
            return False

        # Проверяем фильтры категорий
        if (
            self.config.categories_filter
            and category not in self.config.categories_filter
        ):
            return False

        # Создаем уведомление
        notification = {
            "title": title,
            "message": message,
            "level": level,
            "category": category,
            "template": template,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }

        # Пытаемся добавить уведомление в очередь
        try:
            # Используем asyncio.wait_for для ограничения времени ожидания
            await asyncio.wait_for(
                self.notification_queue.put(notification), timeout=1.0
            )
            return True
        except asyncio.TimeoutError:
            logger.warning("Notification queue is full, dropping notification")
            return False
        except Exception as e:
            logger.error(f"Error adding notification to queue: {str(e)}")
            return False


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


async def send_notification(
    title: str,
    message: str,
    level: str = "info",
    category: str = "general",
    template: str = "default",
    **kwargs,
) -> bool:
    """
    Отправляет уведомление через глобальный сервис

    Args:
        title: Заголовок уведомления
        message: Текст уведомления
        level: Уровень важности (error, warning, info, success, debug)
        category: Категория уведомления
        template: Название шаблона для форматирования
        **kwargs: Дополнительные параметры

    Returns:
        bool: True, если уведомление добавлено в очередь, иначе False
    """
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
        await _notification_service.start()
    return await _notification_service.send_notification(
        title, message, level, category, template, **kwargs
    )


async def get_notification_history(
    limit: int = None, level: str = None, category: str = None
) -> List[Dict]:
    """
    Возвращает историю отправленных уведомлений с опциональной фильтрацией

    Args:
        limit: Максимальное количество уведомлений
        level: Фильтр по уровню уведомления
        category: Фильтр по категории уведомления

    Returns:
        List[Dict]: Список уведомлений
    """
    global _notification_service
    if _notification_service is None:
        return []
    return await _notification_service.get_notification_history(limit, level, category)


async def clear_notification_history():
    """Очищает историю отправленных уведомлений"""
    global _notification_service
    if _notification_service is not None:
        await _notification_service.clear_notification_history()


class TelegramBot:
    """Класс для работы с Telegram-ботом"""

    def __init__(self, token: str, allowed_chat_ids: List[str] = None):
        """
        Инициализирует Telegram-бота

        Args:
            token: Токен бота
            allowed_chat_ids: Список разрешенных ID чатов
        """
        self.token = token
        self.allowed_chat_ids = set(allowed_chat_ids or [])
        self.http_client = None
        self.webhook_url = None
        self.webhook_secret = None
        self.update_offset = 0
        self.handlers = {}
        self._stop_requested = False
        self._polling_task = None
        self._webhook_task = None
        self._last_update_time = 0

        logger.info("Telegram bot initialized")

    async def start_polling(self):
        """Запускает опрос обновлений от Telegram API"""
        if self._polling_task is not None:
            logger.warning("Telegram bot polling is already running")
            return

        # Создаем HTTP-клиент
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Запускаем задачу опроса
        self._stop_requested = False
        self._polling_task = asyncio.create_task(self._polling_loop())

        logger.info("Telegram bot polling started")

    async def stop_polling(self):
        """Останавливает опрос обновлений"""
        if self._polling_task is None:
            logger.warning("Telegram bot polling is not running")
            return

        self._stop_requested = True

        # Отменяем задачу опроса
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None

        # Закрываем HTTP-клиент
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        logger.info("Telegram bot polling stopped")

    async def set_webhook(self, url: str, secret: str = None):
        """
        Устанавливает webhook для получения обновлений

        Args:
            url: URL для webhook
            secret: Секретный токен для проверки запросов
        """
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            # Устанавливаем webhook
            webhook_params = {
                "url": url,
                "allowed_updates": ["message", "callback_query", "inline_query"],
            }

            if secret:
                webhook_params["secret_token"] = secret

            response = await self.http_client.post(
                f"https://api.telegram.org/bot{self.token}/setWebhook",
                json=webhook_params,
            )

            if response.status_code != 200:
                logger.error(f"Failed to set webhook: {response.text}")
                return False

            # Сохраняем данные webhook
            self.webhook_url = url
            self.webhook_secret = secret

            logger.info(f"Webhook set to {url}")
            return True

        except Exception as e:
            logger.error(f"Error setting webhook: {str(e)}")
            return False

    async def remove_webhook(self):
        """Удаляет webhook"""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            response = await self.http_client.post(
                f"https://api.telegram.org/bot{self.token}/deleteWebhook"
            )

            if response.status_code != 200:
                logger.error(f"Failed to remove webhook: {response.text}")
                return False

            # Сбрасываем данные webhook
            self.webhook_url = None
            self.webhook_secret = None

            logger.info("Webhook removed")
            return True

        except Exception as e:
            logger.error(f"Error removing webhook: {str(e)}")
            return False

    async def _polling_loop(self):
        """Цикл опроса обновлений от Telegram API"""
        while not self._stop_requested:
            try:
                # Получаем обновления
                updates = await self._get_updates()

                # Обрабатываем обновления
                if updates:
                    for update in updates:
                        await self._process_update(update)

                        # Обновляем offset для следующего запроса
                        if update.get("update_id", 0) >= self.update_offset:
                            self.update_offset = update["update_id"] + 1

                    # Обновляем время последнего обновления
                    self._last_update_time = time.time()

                # Пауза между опросами
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Telegram polling loop: {str(e)}")
                await asyncio.sleep(5)

    async def _get_updates(self) -> List[Dict]:
        """
        Получает обновления от Telegram API

        Returns:
            List[Dict]: Список обновлений
        """
        if not self.http_client:
            return []

        try:
            # Параметры запроса
            params = {
                "offset": self.update_offset,
                "timeout": 10,
                "allowed_updates": ["message", "callback_query", "inline_query"],
            }

            # Отправляем запрос
            response = await self.http_client.post(
                f"https://api.telegram.org/bot{self.token}/getUpdates", json=params
            )

            # Проверяем результат
            if response.status_code != 200:
                logger.error(f"Telegram API error: {response.text}")
                return []

            # Парсим ответ
            data = response.json()
            if data.get("ok"):
                return data.get("result", [])

            return []

        except Exception as e:
            logger.error(f"Error getting Telegram updates: {str(e)}")
            return []

    async def _process_update(self, update: Dict):
        """
        Обрабатывает обновление от Telegram

        Args:
            update: Обновление от Telegram API
        """
        # Проверяем тип обновления
        if "message" in update:
            # Обрабатываем сообщение
            message = update["message"]
            chat_id = str(message.get("chat", {}).get("id"))

            # Проверяем, разрешен ли чат
            if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
                logger.warning(f"Message from unauthorized chat {chat_id}")
                return

            # Проверяем наличие текста
            text = message.get("text", "")
            if not text:
                return

            # Обрабатываем команду
            if text.startswith("/"):
                command = text.split()[0][1:]  # Убираем "/" и берем первое слово
                await self._handle_command(command, message)
            else:
                # Обрабатываем обычное сообщение
                await self._handle_message(message)

        elif "callback_query" in update:
            # Обрабатываем callback-запрос
            callback_query = update["callback_query"]
            chat_id = str(callback_query.get("message", {}).get("chat", {}).get("id"))

            # Проверяем, разрешен ли чат
            if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
                logger.warning(f"Callback from unauthorized chat {chat_id}")
                return

            await self._handle_callback(callback_query)

    async def _handle_command(self, command: str, message: Dict):
        """
        Обрабатывает команду

        Args:
            command: Команда (без /)
            message: Сообщение от Telegram API
        """
        # Проверяем наличие обработчика для команды
        handler = self.handlers.get(f"command_{command}")
        if handler:
            await handler(message)
        else:
            # Обработчик по умолчанию
            handler = self.handlers.get("command_default")
            if handler:
                await handler(message)

    async def _handle_message(self, message: Dict):
        """
        Обрабатывает обычное сообщение

        Args:
            message: Сообщение от Telegram API
        """
        # Обработчик обычных сообщений
        handler = self.handlers.get("message")
        if handler:
            await handler(message)

    async def _handle_callback(self, callback_query: Dict):
        """
        Обрабатывает callback-запрос

        Args:
            callback_query: Callback-запрос от Telegram API
        """
        # Получаем данные callback
        data = callback_query.get("data", "")

        # Проверяем наличие обработчика для callback
        handler = self.handlers.get(f"callback_{data}")
        if handler:
            await handler(callback_query)
        else:
            # Обработчик по умолчанию
            handler = self.handlers.get("callback_default")
            if handler:
                await handler(callback_query)

    def add_handler(self, event_type: str, handler: callable):
        """
        Добавляет обработчик события

        Args:
            event_type: Тип события (command_start, message, callback_default и т.д.)
            handler: Функция-обработчик
        """
        self.handlers[event_type] = handler
        logger.debug(f"Added handler for {event_type}")

    async def send_message(
        self, chat_id: str, text: str, parse_mode: str = "HTML", **kwargs
    ) -> bool:
        """
        Отправляет сообщение в чат

        Args:
            chat_id: ID чата
            text: Текст сообщения
            parse_mode: Режим форматирования текста (HTML, Markdown)
            **kwargs: Дополнительные параметры

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            # Готовим данные для запроса
            data = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                **kwargs,
            }

            # Отправляем запрос
            response = await self.http_client.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage", json=data
            )

            # Проверяем результат
            if response.status_code != 200:
                logger.error(f"Telegram API error: {response.text}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")
            return False

    async def send_photo(
        self, chat_id: str, photo: Union[str, bytes], caption: str = None, **kwargs
    ) -> bool:
        """
        Отправляет фото в чат

        Args:
            chat_id: ID чата
            photo: URL или данные фотографии
            caption: Подпись к фото
            **kwargs: Дополнительные параметры

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            if isinstance(photo, str) and (
                photo.startswith("http://") or photo.startswith("https://")
            ):
                # Фото по URL
                data = {"chat_id": chat_id, "photo": photo}

                if caption:
                    data["caption"] = caption

                data.update(kwargs)

                # Отправляем запрос
                response = await self.http_client.post(
                    f"https://api.telegram.org/bot{self.token}/sendPhoto", json=data
                )
            else:
                # Фото как файл
                files = {"photo": photo}

                data = {"chat_id": chat_id}

                if caption:
                    data["caption"] = caption

                data.update(kwargs)

                # Отправляем запрос
                response = await self.http_client.post(
                    f"https://api.telegram.org/bot{self.token}/sendPhoto",
                    data=data,
                    files=files,
                )

            # Проверяем результат
            if response.status_code != 200:
                logger.error(f"Telegram API error: {response.text}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error sending Telegram photo: {str(e)}")
            return False

    async def edit_message_text(
        self, chat_id: str, message_id: int, text: str, **kwargs
    ) -> bool:
        """
        Редактирует текст сообщения

        Args:
            chat_id: ID чата
            message_id: ID сообщения
            text: Новый текст
            **kwargs: Дополнительные параметры

        Returns:
            bool: True, если редактирование успешно, иначе False
        """
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            # Готовим данные для запроса
            data = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                **kwargs,
            }

            # Отправляем запрос
            response = await self.http_client.post(
                f"https://api.telegram.org/bot{self.token}/editMessageText", json=data
            )

            # Проверяем результат
            if response.status_code != 200:
                logger.error(f"Telegram API error: {response.text}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error editing Telegram message: {str(e)}")
            return False

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = None,
        show_alert: bool = False,
        **kwargs,
    ) -> bool:
        """
        Отвечает на callback-запрос

        Args:
            callback_query_id: ID callback-запроса
            text: Текст ответа
            show_alert: Показать ответ как предупреждение
            **kwargs: Дополнительные параметры

        Returns:
            bool: True, если ответ успешен, иначе False
        """
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            # Готовим данные для запроса
            data = {
                "callback_query_id": callback_query_id,
                "show_alert": show_alert,
                **kwargs,
            }

            if text:
                data["text"] = text

            # Отправляем запрос
            response = await self.http_client.post(
                f"https://api.telegram.org/bot{self.token}/answerCallbackQuery",
                json=data,
            )

            # Проверяем результат
            if response.status_code != 200:
                logger.error(f"Telegram API error: {response.text}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error answering callback query: {str(e)}")
            return False

    async def send_invoice(
        self,
        chat_id: str,
        title: str,
        description: str,
        payload: str,
        provider_token: str,
        currency: str,
        prices: List[Dict[str, Any]],
        **kwargs,
    ) -> bool:
        """
        Отправляет инвойс для оплаты

        Args:
            chat_id: ID чата
            title: Заголовок инвойса
            description: Описание инвойса
            payload: Данные инвойса
            provider_token: Токен платежной системы
            currency: Валюта (USD, EUR, RUB и т.д.)
            prices: Список цен [{label: str, amount: int}]
            **kwargs: Дополнительные параметры

        Returns:
            bool: True, если инвойс успешно отправлен, иначе False
        """
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=30.0)

        try:
            # Готовим данные для запроса
            data = {
                "chat_id": chat_id,
                "title": title,
                "description": description,
                "payload": payload,
                "provider_token": provider_token,
                "currency": currency,
                "prices": prices,
                **kwargs,
            }

            # Отправляем запрос
            response = await self.http_client.post(
                f"https://api.telegram.org/bot{self.token}/sendInvoice", json=data
            )

            # Проверяем результат
            if response.status_code != 200:
                logger.error(f"Telegram API error: {response.text}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error sending invoice: {str(e)}")
            return False


class EmailService:
    """Класс для работы с электронной почтой"""

    def __init__(self, config: Dict = None):
        """
        Инициализирует сервис электронной почты

        Args:
            config: Конфигурация сервиса
        """
        config = config or {}

        self.server = config.get("server", "smtp.gmail.com")
        self.port = config.get("port", 587)
        self.use_tls = config.get("use_tls", True)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.from_address = config.get("from_address", "")
        self.signature = config.get("signature", "")

        logger.info("Email service initialized")

    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        html: bool = False,
        attachments: List[str] = None,
    ) -> bool:
        """
        Отправляет электронное письмо

        Args:
            to: Адрес или список адресов получателей
            subject: Тема письма
            body: Текст письма
            html: Текст в формате HTML
            attachments: Список путей к файлам для прикрепления

        Returns:
            bool: True, если отправка успешна, иначе False
        """
        try:
            # Создаем сообщение
            msg = MIMEMultipart()
            msg["From"] = self.from_address

            # Получатели
            if isinstance(to, str):
                to = [to]
            msg["To"] = ", ".join(to)

            msg["Subject"] = subject

            # Добавляем текст сообщения
            if html:
                # HTML-версия
                html_content = body
                if self.signature:
                    html_content += f"<br><br>{self.signature}"
                msg.attach(MIMEText(html_content, "html"))
            else:
                # Текстовая версия
                text_content = body
                if self.signature:
                    text_content += f"\n\n{self.signature}"
                msg.attach(MIMEText(text_content, "plain"))

            # Добавляем вложения
            if attachments:
                for attachment_path in attachments:
                    try:
                        with open(attachment_path, "rb") as f:
                            attachment_data = f.read()

                        # Определяем тип файла
                        import mimetypes

                        content_type, encoding = mimetypes.guess_type(attachment_path)
                        if content_type is None:
                            content_type = "application/octet-stream"

                        main_type, sub_type = content_type.split("/", 1)

                        # Создаем вложение
                        attachment = MIMEText(
                            attachment_data, _subtype=sub_type, _charset="utf-8"
                        )
                        attachment.add_header(
                            "Content-Disposition",
                            "attachment",
                            filename=os.path.basename(attachment_path),
                        )
                        msg.attach(attachment)
                    except Exception as e:
                        logger.error(
                            f"Error adding attachment {attachment_path}: {str(e)}"
                        )

            # Отправляем сообщение
            loop = asyncio.get_event_loop()

            await loop.run_in_executor(None, self._send_email_sync, msg.as_string(), to)

            logger.info(f"Email sent to {', '.join(to)}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False

    def _send_email_sync(self, message_string: str, recipients: List[str]):
        """
        Синхронная функция для отправки сообщения

        Args:
            message_string: Строка сообщения
            recipients: Список получателей
        """
        with smtplib.SMTP(self.server, self.port) as server:
            if self.use_tls:
                server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            server.sendmail(self.from_address, recipients, message_string)
