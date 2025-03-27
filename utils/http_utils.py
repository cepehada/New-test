import asyncio
import hashlib
import json
import os
import tempfile
import time
from typing import Any, Dict, Optional

import httpx
from project.utils.logging_utils import setup_logger

logger = setup_logger("http_utils")


class RateLimiter:
    """Класс для ограничения частоты запросов к API"""

    def __init__(self, calls: int, period: float, retry_after: float = 1.0):
        """
        Инициализирует ограничитель частоты запросов

        Args:
            calls: Максимальное количество запросов
            period: Период в секундах
            retry_after: Время ожидания после превышения лимита в секундах
        """
        self.calls = calls
        self.period = period
        self.retry_after = retry_after
        self.timestamps = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """
        Ожидает, пока не будет доступен слот для запроса
        """
        async with self._lock:
            now = time.time()

            # Удаляем устаревшие метки времени
            self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]

            # Проверяем, не превышен ли лимит
            while len(self.timestamps) >= self.calls:
                # Ждем, пока не истечет время для самого старого запроса
                oldest = self.timestamps[0]
                wait_time = self.period - (now - oldest)

                if wait_time <= 0:
                    # Удаляем устаревшие метки времени и проверяем снова
                    self.timestamps = [
                        ts for ts in self.timestamps if now - ts < self.period
                    ]
                else:
                    # Ждем указанное время и пробуем снова
                    await asyncio.sleep(max(wait_time, self.retry_after))
                    now = time.time()
                    # Удаляем устаревшие метки времени
                    self.timestamps = [
                        ts for ts in self.timestamps if now - ts < self.period
                    ]

            # Добавляем текущую метку времени
            self.timestamps.append(now)

    def reset(self):
        """Сбрасывает ограничитель"""
        self.timestamps = []


class HttpCache:
    """Класс для кеширования HTTP-запросов"""

    def __init__(
        self, cache_dir: str = None, max_size_mb: int = 100, default_ttl: int = 3600
    ):
        """
        Инициализирует кеш HTTP-запросов

        Args:
            cache_dir: Директория для хранения кеша. Если None, используется временная директория
            max_size_mb: Максимальный размер кеша в МБ
            default_ttl: Время жизни кеша по умолчанию в секундах
        """
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "http_cache")
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self._create_cache_dir()
        self._cache_metadata = {}
        self._load_metadata()

        logger.info("HTTP cache initialized in {self.cache_dir}" %)

    def _create_cache_dir(self):
        """Создает директорию для кеша, если она не существует"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

    def _load_metadata(self):
        """Загружает метаданные кеша"""
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    self._cache_metadata = json.load(f)
            except Exception as e:
                logger.error("Error loading cache metadata: {str(e)}" %)
                self._cache_metadata = {}

    def _save_metadata(self):
        """Сохраняет метаданные кеша"""
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        try:
            with open(metadata_path, "w") as f:
                json.dump(self._cache_metadata, f)
        except Exception as e:
            logger.error("Error saving cache metadata: {str(e)}" %)

    def _get_cache_key(
        self, url: str, params: Dict = None, headers: Dict = None
    ) -> str:
        """
        Генерирует ключ кеша для запроса

        Args:
            url: URL запроса
            params: Параметры запроса
            headers: Заголовки запроса, которые влияют на ответ

        Returns:
            str: Ключ кеша
        """
        # Отфильтровываем заголовки, которые влияют на ответ
        cache_headers = {}
        if headers:
            for key, value in headers.items():
                if key.lower() in ["accept", "accept-language", "content-type"]:
                    cache_headers[key] = value

        # Создаем строку для хеширования
        key_data = {"url": url, "params": params or {}, "headers": cache_headers}

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        """
        Возвращает путь к файлу кеша

        Args:
            cache_key: Ключ кеша

        Returns:
            str: Путь к файлу кеша
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Проверяет, действителен ли кеш

        Args:
            cache_key: Ключ кеша

        Returns:
            bool: True, если кеш действителен, иначе False
        """
        if cache_key not in self._cache_metadata:
            return False

        metadata = self._cache_metadata[cache_key]
        expires_at = metadata.get("expires_at", 0)

        # Проверяем время истечения
        return time.time() < expires_at

    def _clean_cache_if_needed(self):
        """Очищает кеш, если превышен максимальный размер"""
        # Получаем размер всех файлов кеша
        total_size = 0
        for filename in os.listdir(self.cache_dir):
            file_path = os.path.join(self.cache_dir, filename)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)

        # Если размер превышен, удаляем старые файлы
        if total_size > self.max_size_bytes:
            logger.info(
                f"Cache size ({total_size / 1024 / 1024:.2f} MB) exceeds maximum ({self.max_size_bytes / 1024 / 1024:.2f} MB). Cleaning up..."
            )

            # Сортируем метаданные по времени истечения
            sorted_metadata = sorted(
                self._cache_metadata.items(), key=lambda x: x[1].get("expires_at", 0)
            )

            # Удаляем файлы, начиная с самых старых, пока размер не станет меньше максимального
            for cache_key, _ in sorted_metadata:
                file_path = self._get_cache_file_path(cache_key)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    del self._cache_metadata[cache_key]
                    total_size -= file_size

                    if (
                        total_size <= self.max_size_bytes * 0.8
                    ):  # Уменьшаем до 80% от максимального размера
                        break

            # Сохраняем обновленные метаданные
            self._save_metadata()

            logger.info(
                f"Cache cleaned up. New size: {total_size / 1024 / 1024:.2f} MB"
            )

    async def get(
        self, url: str, params: Dict = None, headers: Dict = None
    ) -> Optional[Dict]:
        """
        Получает данные из кеша

        Args:
            url: URL запроса
            params: Параметры запроса
            headers: Заголовки запроса

        Returns:
            Optional[Dict]: Данные из кеша или None, если кеш не найден или недействителен
        """
        cache_key = self._get_cache_key(url, params, headers)

        if not self._is_cache_valid(cache_key):
            return None

        cache_file_path = self._get_cache_file_path(cache_key)

        try:
            if os.path.exists(cache_file_path):
                with open(cache_file_path, "r") as f:
                    cache_data = json.load(f)
                    return cache_data.get("response")
        except Exception as e:
            logger.error("Error reading cache file: {str(e)}" %)

        return None

    async def set(
        self,
        url: str,
        params: Dict = None,
        headers: Dict = None,
        response: Dict = None,
        ttl: int = None,
    ) -> None:
        """
        Сохраняет данные в кеш

        Args:
            url: URL запроса
            params: Параметры запроса
            headers: Заголовки запроса
            response: Ответ для кеширования
            ttl: Время жизни кеша в секундах. Если None, используется значение по умолчанию
        """
        if response is None:
            return

        ttl = ttl or self.default_ttl
        cache_key = self._get_cache_key(url, params, headers)
        expires_at = time.time() + ttl

        # Сохраняем метаданные
        self._cache_metadata[cache_key] = {
            "url": url,
            "created_at": time.time(),
            "expires_at": expires_at,
            "ttl": ttl,
        }

        # Сохраняем данные
        cache_file_path = self._get_cache_file_path(cache_key)

        try:
            cache_data = {
                "response": response,
                "metadata": self._cache_metadata[cache_key],
            }

            with open(cache_file_path, "w") as f:
                json.dump(cache_data, f)

            # Проверяем, не превышен ли максимальный размер кеша
            self._clean_cache_if_needed()

            # Сохраняем обновленные метаданные
            self._save_metadata()
        except Exception as e:
            logger.error("Error writing cache file: {str(e)}" %)

    async def clear(self, url: str = None, older_than: int = None) -> None:
        """
        Очищает кеш

        Args:
            url: URL для очистки кеша. Если None, очищает весь кеш
            older_than: Очищает записи старше указанного количества секунд
        """
        if url:
            # Очищаем кеш для конкретного URL
            keys_to_delete = []

            for cache_key, metadata in self._cache_metadata.items():
                if metadata.get("url") == url:
                    if (
                        older_than is None
                        or (time.time() - metadata.get("created_at", 0)) > older_than
                    ):
                        keys_to_delete.append(cache_key)

            for cache_key in keys_to_delete:
                file_path = self._get_cache_file_path(cache_key)
                if os.path.exists(file_path):
                    os.remove(file_path)
                del self._cache_metadata[cache_key]
        elif older_than:
            # Очищаем старые записи
            keys_to_delete = []

            for cache_key, metadata in self._cache_metadata.items():
                if (time.time() - metadata.get("created_at", 0)) > older_than:
                    keys_to_delete.append(cache_key)

            for cache_key in keys_to_delete:
                file_path = self._get_cache_file_path(cache_key)
                if os.path.exists(file_path):
                    os.remove(file_path)
                del self._cache_metadata[cache_key]
        else:
            # Очищаем весь кеш
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            self._cache_metadata = {}

        # Сохраняем обновленные метаданные
        self._save_metadata()

        logger.info("Cache cleared")


class HttpClient:
    """HTTP-клиент с поддержкой ограничения частоты запросов и кеширования"""

    def __init__(
        self,
        rate_limit_calls: int = 10,
        rate_limit_period: float = 1.0,
        retry_attempts: int = 3,
        retry_backoff: float = 1.5,
        cache_enabled: bool = True,
        cache_dir: str = None,
        cache_max_size_mb: int = 100,
        cache_default_ttl: int = 3600,
        user_agent: str = "Trading Bot/1.0",
        timeout: float = 30.0,
    ):
        """
        Инициализирует HTTP-клиент

        Args:
            rate_limit_calls: Максимальное количество запросов за период
            rate_limit_period: Период ограничения в секундах
            retry_attempts: Количество попыток повтора при ошибке
            retry_backoff: Множитель времени ожидания между попытками
            cache_enabled: Включено ли кеширование
            cache_dir: Директория для хранения кеша
            cache_max_size_mb: Максимальный размер кеша в МБ
            cache_default_ttl: Время жизни кеша по умолчанию в секундах
            user_agent: User-Agent для запросов
            timeout: Таймаут запросов в секундах
        """
        self.rate_limiter = RateLimiter(rate_limit_calls, rate_limit_period)
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff
        self.cache_enabled = cache_enabled
        self.user_agent = user_agent
        self.timeout = timeout

        # Создаем кеш, если включен
        self.cache = (
            HttpCache(cache_dir, cache_max_size_mb, cache_default_ttl)
            if cache_enabled
            else None
        )

        # HTTP-клиент
        self.client = None

        logger.info("HTTP client initialized")

    async def start(self):
        """Запускает HTTP-клиент"""
        if self.client is not None:
            logger.warning("HTTP client is already running")
            return

        # Создаем HTTP-клиент
        self.client = httpx.AsyncClient(
            timeout=self.timeout, headers={"User-Agent": self.user_agent}
        )

        logger.info("HTTP client started")

    async def stop(self):
        """Останавливает HTTP-клиент"""
        if self.client is None:
            logger.warning("HTTP client is not running")
            return

        # Закрываем HTTP-клиент
        await self.client.aclose()
        self.client = None

        logger.info("HTTP client stopped")

    async def get(
        self,
        url: str,
        params: Dict = None,
        headers: Dict = None,
        use_cache: bool = True,
        cache_ttl: int = None,
        skip_rate_limit: bool = False,
    ) -> Dict:
        """
        Выполняет GET-запрос

        Args:
            url: URL запроса
            params: Параметры запроса
            headers: Заголовки запроса
            use_cache: Использовать ли кеш
            cache_ttl: Время жизни кеша в секундах
            skip_rate_limit: Пропустить ограничение частоты запросов

        Returns:
            Dict: Ответ сервера в виде словаря

        Raises:
            httpx.HTTPError: Если произошла ошибка HTTP
            Exception: Если произошла другая ошибка
        """
        if self.client is None:
            await self.start()

        # Проверяем кеш, если включен и требуется использовать
        if self.cache_enabled and use_cache:
            cached_response = await self.cache.get(url, params, headers)
            if cached_response:
                logger.debug("Cache hit for {url}" %)
                return cached_response

        # Добавляем заголовки по умолчанию
        request_headers = {"User-Agent": self.user_agent}
        if headers:
            request_headers.update(headers)

        # Выполняем запрос с учетом ограничения частоты и повторных попыток
        attempt = 0
        last_error = None

        while attempt < self.retry_attempts:
            try:
                # Ожидаем, если нужно соблюдать ограничение частоты
                if not skip_rate_limit:
                    await self.rate_limiter.acquire()

                # Выполняем запрос
                response = await self.client.get(
                    url, params=params, headers=request_headers
                )

                # Проверяем статус ответа
                response.raise_for_status()

                # Парсим ответ
                response_data = (
                    response.json()
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else {"text": response.text}
                )

                # Сохраняем в кеш, если включен и требуется использовать
                if self.cache_enabled and use_cache:
                    await self.cache.set(url, params, headers, response_data, cache_ttl)

                return response_data

            except httpx.HTTPStatusError as e:
                last_error = e

                # Некоторые ошибки не имеет смысла повторять
                if e.response.status_code in [400, 401, 403, 404, 422]:
                    logger.warning(
                        f"HTTP error {e.response.status_code} for {url}: {e.response.text}"
                    )
                    raise

                logger.warning(
                    f"HTTP error {e.response.status_code} for {url}, attempt {attempt+1}/{self.retry_attempts}"
                )

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(
                    f"Request error for {url}, attempt {attempt+1}/{self.retry_attempts}: {str(e)}"
                )

            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error for {url}, attempt {attempt+1}/{self.retry_attempts}: {str(e)}"
                )

            # Увеличиваем счетчик попыток
            attempt += 1

            # Если это не последняя попытка, ждем перед следующей
            if attempt < self.retry_attempts:
                wait_time = self.retry_backoff**attempt
                await asyncio.sleep(wait_time)

        # Если все попытки неудачны, выбрасываем последнюю ошибку
        if last_error:
            raise last_error
        else:
            raise Exception(
                f"Failed to get response from {url} after {self.retry_attempts} attempts"
            )

    async def post(
        self,
        url: str,
        data: Any = None,
        json_data: Dict = None,
        headers: Dict = None,
        skip_rate_limit: bool = False,
    ) -> Dict:
        """
        Выполняет POST-запрос

        Args:
            url: URL запроса
            data: Данные запроса
            json_data: JSON-данные запроса
            headers: Заголовки запроса
            skip_rate_limit: Пропустить ограничение частоты запросов

        Returns:
            Dict: Ответ сервера в виде словаря

        Raises:
            httpx.HTTPError: Если произошла ошибка HTTP
            Exception: Если произошла другая ошибка
        """
        if self.client is None:
            await self.start()

        # Добавляем заголовки по умолчанию
        request_headers = {"User-Agent": self.user_agent}
        if headers:
            request_headers.update(headers)

        # Выполняем запрос с учетом ограничения частоты и повторных попыток
        attempt = 0
        last_error = None

        while attempt < self.retry_attempts:
            try:
                # Ожидаем, если нужно соблюдать ограничение частоты
                if not skip_rate_limit:
                    await self.rate_limiter.acquire()

                # Выполняем запрос
                response = await self.client.post(
                    url, data=data, json=json_data, headers=request_headers
                )

                # Проверяем статус ответа
                response.raise_for_status()

                # Парсим ответ
                response_data = (
                    response.json()
                    if response.headers.get("content-type", "").startswith(
                        "application/json"
                    )
                    else {"text": response.text}
                )

                return response_data

            except httpx.HTTPStatusError as e:
                last_error = e

                # Некоторые ошибки не имеет смысла повторять
                if e.response.status_code in [400, 401, 403, 404, 422]:
                    logger.warning(
                        f"HTTP error {e.response.status_code} for {url}: {e.response.text}"
                    )
                    raise

                logger.warning(
                    f"HTTP error {e.response.status_code} for {url}, attempt {attempt+1}/{self.retry_attempts}"
                )

            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(
                    f"Request error for {url}, attempt {attempt+1}/{self.retry_attempts}: {str(e)}"
                )

            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error for {url}, attempt {attempt+1}/{self.retry_attempts}: {str(e)}"
                )

            # Увеличиваем счетчик попыток
            attempt += 1

            # Если это не последняя попытка, ждем перед следующей
            if attempt < self.retry_attempts:
                wait_time = self.retry_backoff**attempt
                await asyncio.sleep(wait_time)

        # Если все попытки неудачны, выбрасываем последнюю ошибку
        if last_error:
            raise last_error
        else:
            raise Exception(
                f"Failed to get response from {url} after {self.retry_attempts} attempts"
            )


# Глобальный экземпляр HTTP-клиента
_http_client = None


def init_http_client(
    rate_limit_calls: int = 10,
    rate_limit_period: float = 1.0,
    retry_attempts: int = 3,
    retry_backoff: float = 1.5,
    cache_enabled: bool = True,
    cache_dir: str = None,
    cache_max_size_mb: int = 100,
    cache_default_ttl: int = 3600,
    user_agent: str = "Trading Bot/1.0",
    timeout: float = 30.0,
) -> HttpClient:
    """
    Инициализирует глобальный HTTP-клиент

    Args:
        rate_limit_calls: Максимальное количество запросов за период
        rate_limit_period: Период ограничения в секундах
        retry_attempts: Количество попыток повтора при ошибке
        retry_backoff: Множитель времени ожидания между попытками
        cache_enabled: Включено ли кеширование
        cache_dir: Директория для хранения кеша
        cache_max_size_mb: Максимальный размер кеша в МБ
        cache_default_ttl: Время жизни кеша по умолчанию в секундах
        user_agent: User-Agent для запросов
        timeout: Таймаут запросов в секундах

    Returns:
        HttpClient: HTTP-клиент
    """
    global _http_client

    _http_client = HttpClient(
        rate_limit_calls=rate_limit_calls,
        rate_limit_period=rate_limit_period,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
        cache_max_size_mb=cache_max_size_mb,
        cache_default_ttl=cache_default_ttl,
        user_agent=user_agent,
        timeout=timeout,
    )

    return _http_client


async def get_http_client() -> HttpClient:
    """
    Возвращает глобальный HTTP-клиент. Если клиент не инициализирован, создает его

    Returns:
        HttpClient: HTTP-клиент
    """
    global _http_client

    if _http_client is None:
        _http_client = HttpClient()

    if _http_client.client is None:
        await _http_client.start()

    return _http_client


async def rate_limited_request(
    url: str,
    method: str = "GET",
    params: Dict = None,
    data: Any = None,
    json_data: Dict = None,
    headers: Dict = None,
    use_cache: bool = True,
    cache_ttl: int = None,
    skip_rate_limit: bool = False,
) -> Dict:
    """
    Выполняет запрос с учетом ограничения частоты и кеширования

    Args:
        url: URL запроса
        method: Метод запроса (GET, POST)
        params: Параметры запроса
        data: Данные запроса
        json_data: JSON-данные запроса
        headers: Заголовки запроса
        use_cache: Использовать ли кеш
        cache_ttl: Время жизни кеша в секундах
        skip_rate_limit: Пропустить ограничение частоты запросов

    Returns:
        Dict: Ответ сервера в виде словаря

    Raises:
        httpx.HTTPError: Если произошла ошибка HTTP
        Exception: Если произошла другая ошибка
    """
    client = await get_http_client()

    if method.upper() == "GET":
        return await client.get(
            url, params, headers, use_cache, cache_ttl, skip_rate_limit
        )
    elif method.upper() == "POST":
        return await client.post(url, data, json_data, headers, skip_rate_limit)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
