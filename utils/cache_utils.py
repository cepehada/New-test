"""
Модуль кэширования данных.
Предоставляет функции для временного хранения часто используемых данных.
"""

import time
import asyncio
import functools
from typing import Dict, Any, Optional, Callable, TypeVar, Tuple, cast
import logging

from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Типовые переменные для типизации
T = TypeVar("T")
K = TypeVar("K")

# Глобальный кэш
_cache: Dict[str, Tuple[Any, float, float]] = {}


def cache(ttl: float = 60.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Декоратор для кэширования результатов функции.

    Args:
        ttl: Время жизни кэша в секундах

    Returns:
        Декорированная функция
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Создаем ключ кэша на основе функции и аргументов
            cache_key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(kwargs)}"

            # Проверяем наличие и актуальность кэша
            if cache_key in _cache:
                value, timestamp, _ttl = _cache[cache_key]
                if time.time() - timestamp < _ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return value

            # Выполняем функцию и кэшируем результат
            result = func(*args, **kwargs)
            _cache[cache_key] = (result, time.time(), ttl)
            logger.debug(f"Cache miss for {func.__name__}, stored result")

            return result

        return wrapper

    return decorator


def async_cache(ttl: float = 60.0) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Декоратор для кэширования результатов асинхронной функции.

    Args:
        ttl: Время жизни кэша в секундах

    Returns:
        Декорированная асинхронная функция
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Создаем ключ кэша на основе функции и аргументов
            cache_key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(kwargs)}"

            # Проверяем наличие и актуальность кэша
            if cache_key in _cache:
                value, timestamp, _ttl = _cache[cache_key]
                if time.time() - timestamp < _ttl:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return value

            # Выполняем функцию и кэшируем результат
            result = await func(*args, **kwargs)
            _cache[cache_key] = (result, time.time(), ttl)
            logger.debug(f"Cache miss for {func.__name__}, stored result")

            return result

        return cast(Callable[..., T], wrapper)

    return decorator


def invalidate_cache(key_pattern: Optional[str] = None) -> None:
    """
    Инвалидирует кэш по шаблону ключа.

    Args:
        key_pattern: Шаблон ключа для инвалидации (None для всего кэша)
    """
    global _cache

    if key_pattern is None:
        # Инвалидируем весь кэш
        count = len(_cache)
        _cache.clear()
        logger.info(f"Invalidated entire cache ({count} entries)")
    else:
        # Инвалидируем кэш по шаблону
        keys_to_remove = [k for k in _cache.keys() if key_pattern in k]
        for k in keys_to_remove:
            del _cache[k]
        logger.info(
            f"Invalidated {len(keys_to_remove)} cache entries matching pattern '{key_pattern}'"
        )


def get_cache_size() -> int:
    """
    Возвращает количество записей в кэше.

    Returns:
        Количество записей в кэше
    """
    return len(_cache)


def get_cache_stats() -> Dict[str, Any]:
    """
    Возвращает статистику кэша.

    Returns:
        Словарь со статистикой кэша
    """
    current_time = time.time()
    active_entries = sum(1 for _, ts, ttl in _cache.values() if current_time - ts < ttl)
    expired_entries = len(_cache) - active_entries

    # Группировка по префиксам функций
    function_counts: Dict[str, int] = {}
    for key in _cache:
        prefix = key.split(":")[0]
        function_counts[prefix] = function_counts.get(prefix, 0) + 1

    return {
        "total_entries": len(_cache),
        "active_entries": active_entries,
        "expired_entries": expired_entries,
        "function_counts": function_counts,
    }


def cleanup_expired_cache() -> int:
    """
    Очищает просроченные записи кэша.

    Returns:
        Количество удаленных записей
    """
    global _cache

    current_time = time.time()
    expired_keys = [k for k, (_, ts, ttl) in _cache.items() if current_time - ts >= ttl]

    for k in expired_keys:
        del _cache[k]

    logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    return len(expired_keys)


async def start_cache_cleanup(interval: float = 300.0) -> None:
    """
    Запускает периодическую очистку просроченных записей кэша.

    Args:
        interval: Интервал очистки в секундах
    """
    logger.info(f"Starting cache cleanup task with interval {interval} seconds")

    while True:
        try:
            await asyncio.sleep(interval)
            count = cleanup_expired_cache()
            if count > 0:
                logger.info(f"Cache cleanup removed {count} expired entries")
        except asyncio.CancelledError:
            logger.info("Cache cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cache cleanup: {str(e)}")
            await asyncio.sleep(interval)
