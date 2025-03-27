"""
Модуль для кэширования данных в памяти и на диске.
Предоставляет функции для эффективного хранения и извлечения часто используемых данных.
"""

import time
import pickle
import os
from typing import Any, Dict, Optional, Callable, Union, Set, List
import functools

from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Глобальный кэш в памяти
_cache = {}
_cache_expiration = {}
_cache_hits = {}
_cache_misses = {}


def cache(ttl=60, max_size=1000, key_prefix=""):
    """
    Декоратор для кэширования результатов функции в памяти.
    
    Args:
        ttl: Время жизни кэша в секундах (0 - бессрочно)
        max_size: Максимальный размер кэша для данной функции
        key_prefix: Префикс для ключей кэша
    
    Returns:
        Декоратор
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Генерируем ключ кэша
            key = f"{key_prefix}{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Проверяем наличие данных в кэше
            cached_result = get_from_cache(key)
            if cached_result is not None:
                _cache_hits[key] = _cache_hits.get(key, 0) + 1
                logger.debug("Кэш-попадание для %s (всего: %d)", key, _cache_hits.get(key, 0))
                return cached_result
                
            # Иначе вызываем функцию и кэшируем результат
            _cache_misses[key] = _cache_misses.get(key, 0) + 1
            result = await func(*args, **kwargs)
            
            # Кэшируем результат
            put_in_cache(key, result, ttl)
            
            # Проверяем размер кэша и очищаем при необходимости
            if max_size > 0 and len(_cache) > max_size:
                cleanup_cache(max_size // 2)  # Очищаем половину
                
            logger.debug("Кэш-промах для %s (всего: %d)", key, _cache_misses.get(key, 0))
            return result
        
        # Добавляем функцию для инвалидации кэша
        wrapper.invalidate_cache = lambda: invalidate_function_cache(key_prefix + func.__name__)
        return wrapper
    
    return decorator


def sync_cache(ttl=60, max_size=1000, key_prefix=""):
    """
    Декоратор для кэширования результатов синхронной функции.
    
    Args:
        ttl: Время жизни кэша в секундах (0 - бессрочно)
        max_size: Максимальный размер кэша для данной функции
        key_prefix: Префикс для ключей кэша
    
    Returns:
        Декоратор
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Генерируем ключ кэша
            key = f"{key_prefix}{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Проверяем наличие данных в кэше
            cached_result = get_from_cache(key)
            if cached_result is not None:
                _cache_hits[key] = _cache_hits.get(key, 0) + 1
                logger.debug("Кэш-попадание для %s (всего: %d)", key, _cache_hits.get(key, 0))
                return cached_result
                
            # Иначе вызываем функцию и кэшируем результат
            _cache_misses[key] = _cache_misses.get(key, 0) + 1
            result = func(*args, **kwargs)
            
            # Кэшируем результат
            put_in_cache(key, result, ttl)
            
            # Проверяем размер кэша и очищаем при необходимости
            if max_size > 0 and len(_cache) > max_size:
                cleanup_cache(max_size // 2)  # Очищаем половину
                
            logger.debug("Кэш-промах для %s (всего: %d)", key, _cache_misses.get(key, 0))
            return result
            
        # Добавляем функцию для инвалидации кэша
        wrapper.invalidate_cache = lambda: invalidate_function_cache(key_prefix + func.__name__)
        return wrapper
        
    return decorator


def get_from_cache(key: str) -> Any:
    """
    Получает данные из кэша по ключу.
    
    Args:
        key: Ключ кэша
    
    Returns:
        Данные из кэша или None, если данных нет или они устарели
    """
    global _cache
    
    # Проверяем наличие данных в кэше
    if key not in _cache:
        return None
        
    # Проверяем срок действия
    if key in _cache_expiration:
        expiration_time = _cache_expiration[key]
        if expiration_time > 0 and time.time() > expiration_time:
            # Данные устарели
            del _cache[key]
            del _cache_expiration[key]
            return None
            
    # Возвращаем данные из кэша
    return _cache[key]


def put_in_cache(key: str, value: Any, ttl: int = 60) -> None:
    """
    Помещает данные в кэш.
    
    Args:
        key: Ключ кэша
        value: Значение для кэширования
        ttl: Время жизни в секундах (0 - бессрочно)
    """
    global _cache
    
    _cache[key] = value
    
    # Устанавливаем срок действия
    if ttl > 0:
        _cache_expiration[key] = time.time() + ttl
    else:
        _cache_expiration[key] = 0  # Бессрочно
        
    logger.debug("Данные сохранены в кэш: %s (ttl=%d)", key, ttl)


def invalidate_cache(key_prefix: str = "") -> int:
    """
    Инвалидирует кэш по префиксу ключа.
    
    Args:
        key_prefix: Префикс ключа для инвалидации (пустая строка для всего кэша)
    
    Returns:
        Количество удаленных элементов
    """
    global _cache
    
    count = 0
    if not key_prefix:
        # Очистка всего кэша
        count = len(_cache)
        _cache = {}
        _cache_expiration.clear()
        logger.info("Кэш полностью очищен (%d элементов)", count)
    else:
        # Очистка по префиксу
        keys_to_remove = [key for key in _cache if key.startswith(key_prefix)]
        count = len(keys_to_remove)
        
        for key in keys_to_remove:
            del _cache[key]
            if key in _cache_expiration:
                del _cache_expiration[key]
                
        logger.info("Кэш очищен по префиксу '%s' (%d элементов)", key_prefix, count)
        
    return count


def invalidate_function_cache(func_prefix: str) -> int:
    """
    Инвалидирует кэш для конкретной функции.
    
    Args:
        func_prefix: Префикс функции
    
    Returns:
        Количество удаленных элементов
    """
    return invalidate_cache(func_prefix)


def cleanup_cache(max_items_to_keep: int = 100) -> int:
    """
    Очищает кэш, оставляя только указанное количество самых свежих элементов.
    
    Args:
        max_items_to_keep: Максимальное количество элементов для сохранения
    
    Returns:
        Количество удаленных элементов
    """
    global _cache
    
    # Если кэш меньше указанного размера, ничего не делаем
    if len(_cache) <= max_items_to_keep:
        return 0
        
    # Сортируем элементы по времени истечения
    items = list(_cache.items())
    items.sort(key=lambda x: _cache_expiration.get(x[0], 0))
    
    # Определяем, сколько элементов нужно удалить
    items_to_remove = items[:-max_items_to_keep] if max_items_to_keep > 0 else items
    
    # Удаляем старые элементы
    count = len(items_to_remove)
    for key, _ in items_to_remove:
        del _cache[key]
        if key in _cache_expiration:
            del _cache_expiration[key]
            
    logger.info("Кэш очищен, удалено %d элементов, осталось %d", count, len(_cache))
    return count


def get_cache_stats() -> Dict[str, int]:
    """
    Возвращает статистику использования кэша.
    
    Returns:
        Словарь со статистикой
    """
    total_hits = sum(_cache_hits.values())
    total_misses = sum(_cache_misses.values())
    total_requests = total_hits + total_misses
    
    return {
        "size": len(_cache),
        "hits": total_hits,
        "misses": total_misses,
        "total_requests": total_requests,
        "hit_rate": total_hits / total_requests if total_requests > 0 else 0,
    }


def save_cache_to_disk(filepath: str) -> bool:
    """
    Сохраняет кэш на диск.
    
    Args:
        filepath: Путь к файлу для сохранения
    
    Returns:
        True, если сохранение успешно
    """
    try:
        cache_data = {
            "cache": _cache,
            "expiration": _cache_expiration,
            "timestamp": time.time()
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(cache_data, f)
            
        logger.info("Кэш успешно сохранен на диск: %s (%d элементов)", filepath, len(_cache))
        return True
    except Exception as e:
        logger.error("Ошибка при сохранении кэша на диск: %s", str(e))
        return False


def load_cache_from_disk(filepath: str) -> bool:
    """
    Загружает кэш с диска.
    
    Args:
        filepath: Путь к файлу для загрузки
    
    Returns:
        True, если загрузка успешна
    """
    global _cache, _cache_expiration
    
    try:
        if not os.path.exists(filepath):
            logger.warning("Файл кэша не найден: %s", filepath)
            return False
            
        with open(filepath, "rb") as f:
            cache_data = pickle.load(f)
            
        # Проверяем формат данных
        if not isinstance(cache_data, dict) or "cache" not in cache_data or "expiration" not in cache_data:
            logger.warning("Некорректный формат файла кэша")
            return False
            
        # Восстанавливаем кэш
        _cache = cache_data["cache"]
        _cache_expiration = cache_data["expiration"]
        
        logger.info("Кэш успешно загружен с диска: %s (%d элементов)", filepath, len(_cache))
        
        # Очищаем просроченные элементы
        cleanup_expired_items()
        
        return True
    except Exception as e:
        logger.error("Ошибка при загрузке кэша с диска: %s", str(e))
        return False


def cleanup_expired_items() -> int:
    """
    Очищает просроченные элементы кэша.
    
    Returns:
        Количество удаленных элементов
    """
    global _cache
    
    current_time = time.time()
    keys_to_remove = [
        key for key, exp_time in _cache_expiration.items()
        if exp_time > 0 and current_time > exp_time
    ]
    
    count = len(keys_to_remove)
    
    for key in keys_to_remove:
        del _cache[key]
        del _cache_expiration[key]
        
    if count > 0:
        logger.debug("Удалено %d просроченных элементов кэша", count)
        
    return count
