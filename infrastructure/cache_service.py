"""
Модуль для централизованного управления кэшированием.
Предоставляет интерфейс для кэширования данных с возможностью использования разных бэкендов.
"""

import asyncio
import time
from typing import Any, Dict, Generic, Optional, Set, TypeVar

from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Типовая переменная для типизации
T = TypeVar("T")


class CacheEntry(Generic[T]):
    """
    Класс для хранения записи в кэше с временем жизни.
    """

    def __init__(self, value: T, ttl: float, created_at: Optional[float] = None):
        """
        Инициализирует запись кэша.

        Args:
            value: Значение для хранения
            ttl: Время жизни в секундах
            created_at: Время создания записи (по умолчанию текущее время)
        """
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()

    def is_expired(self) -> bool:
        """
        Проверяет, истекло ли время жизни записи.

        Returns:
            True, если запись истекла, иначе False
        """
        return time.time() > self.created_at + self.ttl

    def time_left(self) -> float:
        """
        Возвращает оставшееся время жизни записи в секундах.

        Returns:
            Оставшееся время в секундах (отрицательное, если запись истекла)
        """
        return (self.created_at + self.ttl) - time.time()


class CacheService:
    """
    Сервис для централизованного управления кэшированием.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "CacheService":
        """
        Получает экземпляр сервиса кэширования (Singleton).

        Returns:
            Экземпляр класса CacheService
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Инициализирует сервис кэширования.
        """
        self.config = get_config()
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.tags: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task = None
        logger.debug("Создан экземпляр сервиса кэширования")

    async def start_cleanup_task(self, interval: float = 300.0) -> None:
        """
        Запускает периодическую задачу по очистке истекших записей.

        Args:
            interval: Интервал выполнения в секундах
        """
        if self._cleanup_task is not None:
            logger.warning("Задача очистки кэша уже запущена")
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(interval)
                    await self.cleanup_expired()
                except asyncio.CancelledError:
                    logger.info("Задача очистки кэша отменена")
                    break
                except Exception as e:
                    logger.error(f"Ошибка в задаче очистки кэша: {str(e)}")
                    await asyncio.sleep(interval)

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Задача очистки кэша запущена с интервалом {interval} секунд")

    async def stop_cleanup_task(self) -> None:
        """
        Останавливает задачу очистки кэша.
        """
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Задача очистки кэша остановлена")

    @async_handle_error
    async def set(
        self, key: str, value: T, ttl: float = 3600.0, tags: Optional[list] = None
    ) -> None:
        """
        Устанавливает значение в кэш.

        Args:
            key: Ключ кэша
            value: Значение для хранения
            ttl: Время жизни в секундах
            tags: Список тегов для группировки ключей
        """
        async with self._lock:
            entry = CacheEntry(value, ttl)
            self.memory_cache[key] = entry

            # Добавляем теги
            tags = tags or []
            for tag in tags:
                if tag not in self.tags:
                    self.tags[tag] = set()
                self.tags[tag].add(key)

            await self.redis.set(
                key, value, expire=ttl
            )  # Assuming this is the correct fix

            logger.debug(
                f"Установлено значение в кэш для ключа {key}, TTL={ttl}с, теги={tags}"
            )

    @async_handle_error
    async def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Получает значение из кэша.

        Args:
            key: Ключ кэша
            default: Значение по умолчанию, если ключ не найден или истек

        Returns:
            Значение из кэша или default, если ключ не найден или истек
        """
        entry = self.memory_cache.get(key)

        if entry is None:
            logger.debug(f"Ключ {key} не найден в кэше")
            return default

        if entry.is_expired():
            logger.debug(f"Ключ {key} истек в кэше")
            # Удаляем истекшую запись
            async with self._lock:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    # Удаляем ключ из всех тегов
                    for tag_keys in self.tags.values():
                        tag_keys.discard(key)
            return default

        logger.debug(f"Получено значение из кэша для ключа {key}")
        return entry.value

    @async_handle_error
    async def delete(self, key: str) -> bool:
        """
        Удаляет ключ из кэша.

        Args:
            key: Ключ кэша

        Returns:
            True, если ключ был удален, иначе False
        """
        async with self._lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
                # Удаляем ключ из всех тегов
                for tag_keys in self.tags.values():
                    tag_keys.discard(key)
                logger.debug(f"Ключ {key} удален из кэша")
                return True

            logger.debug(f"Ключ {key} не найден в кэше для удаления")
            return False

    @async_handle_error
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Инвалидирует все ключи с указанным тегом.

        Args:
            tag: Тег для инвалидации

        Returns:
            Количество инвалидированных ключей
        """
        async with self._lock:
            keys_to_invalidate = self.tags.get(tag, set())
            count = len(keys_to_invalidate)

            for key in list(keys_to_invalidate):
                if key in self.memory_cache:
                    del self.memory_cache[key]

            # Очищаем тег
            if tag in self.tags:
                self.tags[tag] = set()

            logger.debug(f"Инвалидировано {count} ключей с тегом {tag}")
            return count

    @async_handle_error
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Инвалидирует все ключи, соответствующие шаблону.

        Args:
            pattern: Подстрока для поиска в ключах

        Returns:
            Количество инвалидированных ключей
        """
        async with self._lock:
            keys_to_invalidate = [k for k in self.memory_cache.keys() if pattern in k]
            count = len(keys_to_invalidate)

            for key in keys_to_invalidate:
                del self.memory_cache[key]
                # Удаляем ключ из всех тегов
                for tag_keys in self.tags.values():
                    tag_keys.discard(key)

            logger.debug(f"Инвалидировано {count} ключей по шаблону '{pattern}'")
            return count

    @async_handle_error
    async def clear(self) -> int:
        """
        Очищает весь кэш.

        Returns:
            Количество удаленных элементов
        """
        async with self._lock:
            count = len(self.memory_cache)
            self.memory_cache.clear()
            self.tags.clear()
            logger.info(f"Кэш полностью очищен ({count} элементов)")
            return count

    @async_handle_error
    async def cleanup_expired(self) -> int:
        """
        Удаляет все истекшие записи из кэша.

        Returns:
            Количество удаленных элементов
        """
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, entry in self.memory_cache.items()
                if current_time > entry.created_at + entry.ttl
            ]

            count = len(expired_keys)

            for key in expired_keys:
                del self.memory_cache[key]
                # Удаляем ключ из всех тегов
                for tag_keys in self.tags.values():
                    tag_keys.discard(key)

            # Удаляем пустые теги
            empty_tags = [tag for tag, keys in self.tags.items() if not keys]
            for tag in empty_tags:
                del self.tags[tag]

            logger.debug(f"Удалено {count} истекших записей из кэша")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику использования кэша.

        Returns:
            Словарь со статистикой
        """
        current_time = time.time()
        total_entries = len(self.memory_cache)
        expired_entries = sum(
            1
            for entry in self.memory_cache.values()
            if current_time > entry.created_at + entry.ttl
        )
        active_entries = total_entries - expired_entries

        # Группировка по тегам
        tag_counts = {tag: len(keys) for tag, keys in self.tags.items()}

        return {
            "total_entries": total_entries,
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "total_tags": len(self.tags),
            "tag_counts": tag_counts,
        }
