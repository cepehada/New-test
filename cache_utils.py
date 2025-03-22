"""
Модуль кэширования.
Использует Redis для хранения данных в виде JSON.
"""

import redis
import json
from project.config import load_config

config = load_config()
_cache = redis.Redis(
    host=config["system"]["redis"]["host"],
    port=int(config["system"]["redis"]["port"]),
    db=1,
    decode_responses=True
)

class CacheManager:
    """
    Класс для работы с кэшем.
    Использует Redis для хранения кэшированных данных.
    """

    def __init__(self) -> None:
        self.ttl = config["cache_ttl"]

    def set(self, key: str, value: dict) -> None:
        """
        Сохраняет значение в кэш с заданным TTL.

        Args:
            key (str): Ключ для кэша.
            value (dict): Значение для сохранения.
        """
        _cache.set(key, json.dumps(value), ex=self.ttl)

    def get(self, key: str) -> dict:
        """
        Получает значение из кэша.

        Args:
            key (str): Ключ для поиска.

        Returns:
            dict: Словарь или None, если данных нет.
        """
        data = _cache.get(key)
        return json.loads(data) if data else None
