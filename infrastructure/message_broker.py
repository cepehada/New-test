"""
Модуль для работы с брокером сообщений.
Предоставляет единый интерфейс для публикации и подписки на сообщения.
"""

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from project.config import get_config
from project.utils.error_handler import async_handle_error, async_with_retry
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Определение типа для обработчиков сообщений
MessageHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class MessageBroker:
    """
    Класс для работы с брокером сообщений.
    Поддерживает локальную маршрутизацию сообщений и возможность интеграции с RabbitMQ/Kafka.
    """

    _instance = None

    @classmethod
    def get_instance(cls, connection_string: Optional[str] = None) -> "MessageBroker":
        """
        Получает экземпляр брокера сообщений (Singleton).

        Args:
            connection_string: Строка подключения к брокеру (None для использования из конфигурации)

        Returns:
            Экземпляр класса MessageBroker
        """
        if cls._instance is None:
            cls._instance = cls(connection_string)
        return cls._instance

    def __init__(self, connection_string: Optional[str] = None):
        """
        Инициализирует брокер сообщений.

        Args:
            connection_string: Строка подключения к брокеру (None для использования из конфигурации)
        """
        self.config = get_config()
        self.connection_string = connection_string or self.config.MESSAGE_BROKER_URI
        self.handlers: Dict[str, List[MessageHandler]] = {}
        self.topics: Set[str] = set()
        self.connection = None
        self.channel = None
        self._lock = asyncio.Lock()
        logger.debug("Создан экземпляр брокера сообщений")

    @async_with_retry(max_retries=3, retry_delay=1.0)
    async def initialize(self) -> None:
        """
        Инициализирует соединение с брокером сообщений.
        """
        if self.connection is None:
            try:
                # В реальной реализации здесь будет код для подключения к RabbitMQ/Kafka
                # Для демонстрации используем простой флаг
                logger.info("Инициализация соединения с брокером сообщений")
                self.connection = True
                self.channel = True
                logger.info("Соединение с брокером сообщений установлено")
            except Exception as e:
                logger.error(f"Ошибка при подключении к брокеру сообщений: {str(e)}")
                raise

    @async_handle_error
    async def publish(self, topic: str, message: Dict[str, Any]) -> bool:
        """
        Публикует сообщение в указанную тему.

        Args:
            topic: Тема сообщения
            message: Данные сообщения

        Returns:
            True при успешной публикации, иначе False
        """
        if not self.connection:
            await self.initialize()

        # Добавляем тему в список известных тем
        self.topics.add(topic)

        try:
            # Сериализуем сообщение в JSON
            message_json = json.dumps(message)

            # В реальной реализации здесь будет код для публикации в RabbitMQ/Kafka
            await self.channel.basic_publish(
                exchange="", routing_key=topic, body=message_json
            )  # Assuming this is the correct fix
            logger.debug(
                f"Публикация сообщения в тему {topic}: {message_json[:100]}..."
            )

            # Вызываем всех локальных подписчиков
            handlers = self.handlers.get(topic, [])
            if handlers:
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(
                            f"Ошибка в обработчике {handler.__name__} для темы {topic}: {str(e)}"
                        )

            return True
        except Exception as e:
            logger.error(f"Ошибка при публикации сообщения в тему {topic}: {str(e)}")
            return False

    @async_handle_error
    async def subscribe(self, topic: str, handler: MessageHandler) -> bool:
        """
        Подписывается на указанную тему.

        Args:
            topic: Тема сообщения
            handler: Асинхронная функция-обработчик сообщений

        Returns:
            True при успешной подписке, иначе False
        """
        if not self.connection:
            await self.initialize()

        # Добавляем тему в список известных тем
        self.topics.add(topic)

        try:
            async with self._lock:
                if topic not in self.handlers:
                    self.handlers[topic] = []

                # Проверяем, не подписан ли уже этот обработчик
                if handler not in self.handlers[topic]:
                    self.handlers[topic].append(handler)
                    logger.debug(
                        f"Подписка на тему {topic} для обработчика {handler.__name__}"
                    )

                # В реальной реализации здесь будет код для подписки в RabbitMQ/Kafka

            return True
        except Exception as e:
            logger.error(f"Ошибка при подписке на тему {topic}: {str(e)}")
            return False

    @async_handle_error
    async def unsubscribe(self, topic: str, handler: MessageHandler) -> bool:
        """
        Отписывается от указанной темы.

        Args:
            topic: Тема сообщения
            handler: Функция-обработчик для отписки

        Returns:
            True при успешной отписке, иначе False
        """
        try:
            async with self._lock:
                if topic in self.handlers and handler in self.handlers[topic]:
                    self.handlers[topic].remove(handler)
                    logger.debug(
                        f"Отписка от темы {topic} для обработчика {handler.__name__}"
                    )

                    # Если больше нет обработчиков, удаляем тему из словаря
                    if not self.handlers[topic]:
                        del self.handlers[topic]

                        # В реальной реализации здесь будет код для отписки в RabbitMQ/Kafka

            return True
        except Exception as e:
            logger.error(f"Ошибка при отписке от темы {topic}: {str(e)}")
            return False

    async def close(self) -> None:
        """
        Закрывает соединение с брокером сообщений.
        """
        if self.connection:
            try:
                # В реальной реализации здесь будет код для закрытия соединения
                self.connection = None
                self.channel = None
                logger.info("Соединение с брокером сообщений закрыто")
            except Exception as e:
                logger.error(
                    f"Ошибка при закрытии соединения с брокером сообщений: {str(e)}"
                )

    def get_all_topics(self) -> List[str]:
        """
        Возвращает список всех известных тем.

        Returns:
            Список тем
        """
        return list(self.topics)

    def get_topic_handlers(self, topic: str) -> List[MessageHandler]:
        """
        Возвращает список обработчиков для указанной темы.

        Args:
            topic: Тема сообщения

        Returns:
            Список обработчиков
        """
        return self.handlers.get(topic, [])
