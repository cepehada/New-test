# Standard imports
import asyncio
import hashlib
import hmac
import inspect
import json
import ssl
import time
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import certifi

# Third party imports
import websockets

# Local imports
from project.utils.logging_utils import setup_logger

logger = setup_logger("websocket_client")


class WebSocketClient:
    """Клиент WebSocket с обработкой переподключений и авторизацией"""

    def __init__(
        self,
        url: str,
        on_message: Optional[Callable] = None,
        on_connect: Optional[Callable] = None,
        on_disconnect: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 2.0,
        ping_interval: Optional[float] = 30.0,
        ping_timeout: float = 10.0,
        compression: Optional[str] = None,
        ssl_verify: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Инициализирует WebSocket клиент

        Args:
            url: URL WebSocket сервера
            on_message: Коллбэк для обработки сообщений
            on_connect: Коллбэк при подключении
            on_disconnect: Коллбэк при отключении
            on_error: Коллбэк при ошибке
            auto_reconnect: Включить автоматическое переподключение
            max_reconnect_attempts: Максимальное количество попыток переподключения
            reconnect_delay: Задержка между попытками переподключения в секундах
            ping_interval: Интервал отправки пинг-сообщений в секундах
            ping_timeout: Таймаут пинг-сообщений в секундах
            compression: Тип сжатия ('deflate' или None)
            ssl_verify: Проверять SSL-сертификат
            headers: Заголовки для подключения
        """
        self.url = url
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.on_error = on_error
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.compression = compression
        self.ssl_verify = ssl_verify
        self.headers = headers or {}

        # Внутренние переменные состояния
        self.ws = None
        self.connected = False
        self.reconnect_attempts = 0
        self.last_ping_time = 0
        self.last_pong_time = 0
        self.subscription_messages = []
        self.message_handlers = {}
        self.running = False
        self.tasks = []
        self._ping_task = None
        self._recv_task = None
        self._disconnect_event = asyncio.Event()
        self._user_disconnect = False
        self._auth_provider = None
        self._auth_params = None

        # WebSocket SSL-контекст
        ssl_context = None
        if self.ssl_verify:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context = ssl_context

        # Временные параметры
        self.message_id = 0
        self.buffer = bytearray()

        logger.debug(f"WebSocket client initialized for {self.url}")

    async def connect(self) -> bool:
        """
        Устанавливает соединение с WebSocket сервером

        Returns:
            bool: True, если соединение установлено, иначе False
        """
        if self.connected:
            logger.warning("Already connected to WebSocket server")
            return True

        self.running = True
        self._user_disconnect = False

        try:
            logger.info(f"Connecting to WebSocket server: {self.url}")

            # Выполняем авторизацию, если настроена
            if self._auth_provider and self._auth_params:
                # Обновляем заголовки авторизации
                headers = await self._auth_provider(self._auth_params, self.url)
                if headers:
                    self.headers.update(headers)

            # Устанавливаем соединение
            self.ws = await websockets.connect(
                self.url,
                extra_headers=self.headers,
                ping_interval=None,  # Отключаем встроенные пинги
                ping_timeout=None,  # Отключаем встроенные пинги
                ssl=self.ssl_context,
                compression=self.compression,
                max_size=2**24,  # 16MB
                close_timeout=10.0,
            )

            # Обновляем состояние
            self.connected = True
            self.reconnect_attempts = 0
            self._disconnect_event.clear()

            # Отправляем сообщения подписки
            if self.subscription_messages:
                for msg in self.subscription_messages:
                    await self._send_raw(msg)
                    await asyncio.sleep(0.1)  # Небольшая пауза между подписками

            # Запускаем задачи
            await self._start_tasks()

            # Вызываем коллбэк подключения
            if self.on_connect:
                if inspect.iscoroutinefunction(self.on_connect):
                    await self.on_connect()
                else:
                    self.on_connect()

            logger.info(f"Connected to WebSocket server: {self.url}")
            return True

        except (
            websockets.exceptions.WebSocketException,
            ConnectionRefusedError,
            asyncio.TimeoutError,
            OSError,
        ) as e:
            # Обработка ошибок подключения
            self.connected = False
            err_msg = f"Failed to connect to WebSocket server: {str(e)}"
            logger.error(err_msg)

            # Вызываем коллбэк ошибки
            if self.on_error:
                if inspect.iscoroutinefunction(self.on_error):
                    await self.on_error(err_msg)
                else:
                    self.on_error(err_msg)

            return False
        except Exception as e:
            # Прочие ошибки
            self.connected = False
            err_msg = f"Unexpected error connecting to WebSocket server: {str(e)}"
            logger.exception(err_msg)

            # Вызываем коллбэк ошибки
            if self.on_error:
                if inspect.iscoroutinefunction(self.on_error):
                    await self.on_error(err_msg)
                else:
                    self.on_error(err_msg)

            return False

    async def disconnect(self, code: int = 1000, reason: str = "Client disconnect"):
        """
        Закрывает соединение с WebSocket сервером

        Args:
            code: Код закрытия
            reason: Причина закрытия
        """
        self._user_disconnect = True
        self.running = False

        if not self.connected or not self.ws:
            logger.debug("Not connected to WebSocket server")
            return

        logger.info(f"Disconnecting from WebSocket server: {self.url}")

        # Останавливаем задачи
        await self._stop_tasks()

        try:
            # Закрываем соединение
            await self.ws.close(code=code, reason=reason)
        except Exception as e:
            logger.error(f"Error closing WebSocket connection: {str(e)}")

        # Сбрасываем флаги
        self.connected = False
        self._disconnect_event.set()

        # Вызываем коллбэк отключения
        if self.on_disconnect:
            if inspect.iscoroutinefunction(self.on_disconnect):
                await self.on_disconnect()
            else:
                self.on_disconnect()

        logger.info(f"Disconnected from WebSocket server: {self.url}")

    async def _start_tasks(self):
        """Запускает фоновые задачи"""
        # Задача получения сообщений
        self._recv_task = asyncio.create_task(self._receive_messages())
        self.tasks.append(self._recv_task)

        # Задача отправки пингов, если настроена
        if self.ping_interval:
            self._ping_task = asyncio.create_task(self._ping_pong())
            self.tasks.append(self._ping_task)

    async def _stop_tasks(self):
        """Останавливает фоновые задачи"""
        # Отменяем все задачи
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Ждем завершения задач
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

        # Очищаем список задач
        self.tasks = []
        self._ping_task = None
        self._recv_task = None

    async def _receive_messages(self):
        """Получает и обрабатывает сообщения от сервера"""
        if not self.ws:
            return

        try:
            async for message in self.ws:
                try:
                    # Декодируем и обрабатываем сообщение
                    decoded_message = self._decode_message(message)

                    # Обрабатываем pong сообщения
                    if isinstance(decoded_message, str) and decoded_message == "pong":
                        self.last_pong_time = time.time()
                        continue

                    # Обрабатываем сообщение
                    await self._handle_message(decoded_message)

                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {str(e)}")

        except (
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.ConnectionClosedError,
        ) as e:
            # Обработка закрытия соединения
            if self.connected:
                self.connected = False

                # Выводим сообщение с кодом ошибки
                logger.warning(f"WebSocket connection closed: {str(e)}")

                # Вызываем коллбэк отключения
                if self.on_disconnect:
                    if inspect.iscoroutinefunction(self.on_disconnect):
                        await self.on_disconnect()
                    else:
                        self.on_disconnect()

                # Переподключаемся, если необходимо
                if self.auto_reconnect and not self._user_disconnect:
                    asyncio.create_task(self._reconnect())

        except asyncio.CancelledError:
            # Задача отменена, выходим
            pass

        except Exception as e:
            # Прочие ошибки
            logger.exception(f"Unexpected error in WebSocket receive loop: {str(e)}")

            if self.connected:
                self.connected = False

                # Вызываем коллбэк отключения
                if self.on_disconnect:
                    if inspect.iscoroutinefunction(self.on_disconnect):
                        await self.on_disconnect()
                    else:
                        self.on_disconnect()

                # Переподключаемся, если необходимо
                if self.auto_reconnect and not self._user_disconnect:
                    asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        """Выполняет переподключение к серверу"""
        # Проверяем количество попыток
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(
                f"Maximum reconnection attempts reached ({self.max_reconnect_attempts})"
            )
            self.running = False
            self._disconnect_event.set()
            return

        # Увеличиваем счетчик попыток
        self.reconnect_attempts += 1

        # Вычисляем задержку с экспоненциальным ростом
        delay = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
        delay = min(delay, 60.0)  # Максимальная задержка 60 секунд

        logger.info(
            f"Reconnecting to WebSocket server (attempt {
                self.reconnect_attempts}/{
                self.max_reconnect_attempts}) after {
                delay:.1f} seconds")

        # Ждем указанное время
        await asyncio.sleep(delay)

        # Пытаемся переподключиться
        if self.running and not self._user_disconnect:
            success = await self.connect()

            if not success and self.running and not self._user_disconnect:
                # Если не удалось, пробуем еще раз
                asyncio.create_task(self._reconnect())

    async def _ping_pong(self):
        """Отправляет ping-сообщения для проверки соединения"""
        while self.connected and self.running:
            try:
                current_time = time.time()

                # Проверяем, нужно ли отправлять пинг
                if current_time - self.last_ping_time >= self.ping_interval:
                    # Отправляем пинг
                    await self._send_raw("ping")
                    self.last_ping_time = current_time

                    # Ждем понг
                    for _ in range(int(self.ping_timeout / 0.5)):
                        await asyncio.sleep(0.5)

                        if not self.connected or not self.running:
                            break

                        # Проверяем, получен ли понг
                        if self.last_pong_time > self.last_ping_time:
                            break
                    else:
                        # Если понг не получен, закрываем соединение
                        logger.warning("Ping timeout, closing connection")
                        self.connected = False
                        if self.ws:
                            await self.ws.close(code=1001, reason="Ping timeout")
                        break

                # Небольшая пауза перед следующей проверкой
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                # Задача отменена, выходим
                break

            except Exception as e:
                logger.error(f"Error in ping-pong loop: {str(e)}")
                await asyncio.sleep(1.0)

    async def _handle_message(self, message: Any):
        """
        Обрабатывает полученное сообщение

        Args:
            message: Полученное сообщение
        """
        # Проверяем, есть ли обработчик для этого типа сообщений
        message_type = self._get_message_type(message)

        if message_type and message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
                return
            except Exception as e:
                logger.error(
                    f"Error in message handler for type {message_type}: {str(e)}"
                )

        # Если нет обработчика или произошла ошибка, используем общий обработчик
        if self.on_message:
            try:
                if inspect.iscoroutinefunction(self.on_message):
                    await self.on_message(message)
                else:
                    self.on_message(message)
            except Exception as e:
                logger.error(f"Error in general message handler: {str(e)}")

    def _get_message_type(self, message: Any) -> Optional[str]:
        """
        Определяет тип сообщения

        Args:
            message: Полученное сообщение

        Returns:
            Optional[str]: Тип сообщения или None
        """
        if isinstance(message, dict):
            # Пробуем разные ключи, которые могут указывать на тип сообщения
            for key in ["type", "event", "e", "method", "op", "channel"]:
                if key in message:
                    return str(message[key])

        return None

    def _decode_message(self, message: Union[str, bytes]) -> Any:
        """
        Декодирует сообщение

        Args:
            message: Сырое сообщение

        Returns:
            Any: Декодированное сообщение
        """
        # Если сообщение уже является строкой, пробуем распарсить его как JSON
        if isinstance(message, str):
            if message == "ping":
                # Отвечаем на пинг
                asyncio.create_task(self._send_raw("pong"))
                return "ping"
            elif message == "pong":
                return "pong"

            try:
                return json.loads(message)
            except json.JSONDecodeError:
                return message

        # Если сообщение является байтами, пробуем распаковать его
        elif isinstance(message, bytes):
            # Пробуем разные форматы сжатия
            if self.compression == "deflate":
                try:
                    # Используем zlib для распаковки
                    decompressed = zlib.decompress(message)
                    return self._decode_message(decompressed.decode("utf-8"))
                except Exception as e:
                    logger.error(f"Error decompressing message: {str(e)}")
                    return message
            else:
                # Пробуем декодировать байты как UTF-8 строку
                try:
                    return self._decode_message(message.decode("utf-8"))
                except UnicodeDecodeError:
                    return message

        return message

    async def send(self, message: Union[Dict, List, str]) -> bool:
        """
        Отправляет сообщение на сервер

        Args:
            message: Сообщение для отправки

        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        if not self.connected or not self.ws:
            logger.warning("Cannot send message: not connected to WebSocket server")
            return False

        try:
            # Преобразуем сообщение в строку JSON
            if isinstance(message, (dict, list)):
                message_str = json.dumps(message)
            else:
                message_str = str(message)

            # Отправляем сообщение
            await self.ws.send(message_str)
            return True

        except (websockets.exceptions.WebSocketException, ConnectionError) as e:
            logger.error(f"Error sending message: {str(e)}")

            # Если соединение закрыто, пробуем переподключиться
            if self.auto_reconnect and self.connected:
                self.connected = False
                await self._reconnect()

            return False

    async def _send_raw(self, message: Union[str, bytes]) -> bool:
        """
        Отправляет сырое сообщение на сервер

        Args:
            message: Сырое сообщение для отправки

        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        if not self.connected or not self.ws:
            logger.warning("Cannot send raw message: not connected to WebSocket server")
            return False

        try:
            # Отправляем сообщение
            await self.ws.send(message)
            return True

        except (websockets.exceptions.WebSocketException, ConnectionError) as e:
            logger.error(f"Error sending raw message: {str(e)}")

            # Если соединение закрыто, пробуем переподключиться
            if self.auto_reconnect and self.connected:
                self.connected = False
                await self._reconnect()

            return False

    def add_subscription(self, subscription_message: Union[Dict, List, str]):
        """
        Добавляет сообщение подписки, которое будет отправлено при подключении

        Args:
            subscription_message: Сообщение подписки
        """
        # Преобразуем сообщение в строку JSON
        if isinstance(subscription_message, (dict, list)):
            message_str = json.dumps(subscription_message)
        else:
            message_str = str(subscription_message)

        # Добавляем в список подписок
        if message_str not in self.subscription_messages:
            self.subscription_messages.append(message_str)

            # Если уже подключены, отправляем сообщение сразу
            if self.connected:
                asyncio.create_task(self._send_raw(message_str))

    def remove_subscription(self, subscription_message: Union[Dict, List, str]):
        """
        Удаляет сообщение подписки

        Args:
            subscription_message: Сообщение подписки
        """
        # Преобразуем сообщение в строку JSON
        if isinstance(subscription_message, (dict, list)):
            message_str = json.dumps(subscription_message)
        else:
            message_str = str(subscription_message)

        # Удаляем из списка подписок
        if message_str in self.subscription_messages:
            self.subscription_messages.remove(message_str)

    def add_message_handler(self, message_type: str, handler: Callable):
        """
        Добавляет обработчик для определенного типа сообщений

        Args:
            message_type: Тип сообщения
            handler: Функция-обработчик
        """
        self.message_handlers[message_type] = handler

    def remove_message_handler(self, message_type: str):
        """
        Удаляет обработчик для определенного типа сообщений

        Args:
            message_type: Тип сообщения
        """
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]

    def set_auth_provider(self, auth_provider: Callable, auth_params: Dict = None):
        """
        Устанавливает провайдер авторизации

        Args:
            auth_provider: Функция для генерации заголовков авторизации
            auth_params: Параметры авторизации
        """
        self._auth_provider = auth_provider
        self._auth_params = auth_params or {}

    async def wait_for_disconnect(self):
        """Ожидает отключения от сервера"""
        await self._disconnect_event.wait()


class WebSocketPool:
    """Пул WebSocket соединений для управления несколькими соединениями"""

    def __init__(self):
        """Инициализирует пул WebSocket соединений"""
        self.connections = {}
        self.running = False
        self._cleanup_task = None

        logger.debug("WebSocket pool initialized")

    async def start(self):
        """Запускает пул соединений"""
        if self.running:
            logger.warning("WebSocket pool is already running")
            return

        self.running = True

        # Запускаем задачу очистки закрытых соединений
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("WebSocket pool started")

    async def stop(self):
        """Останавливает пул соединений"""
        if not self.running:
            logger.warning("WebSocket pool is not running")
            return

        self.running = False

        # Отменяем задачу очистки
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        # Закрываем все соединения
        close_tasks = []
        for conn_id, client in list(self.connections.items()):
            close_tasks.append(client.disconnect())

        # Ждем закрытия всех соединений
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Очищаем словарь соединений
        self.connections.clear()

        logger.info("WebSocket pool stopped")

    async def _cleanup_loop(self):
        """Периодически очищает закрытые соединения"""
        while self.running:
            try:
                # Находим закрытые соединения
                closed_connections = []

                for conn_id, client in list(self.connections.items()):
                    if not client.connected:
                        closed_connections.append(conn_id)

                # Удаляем закрытые соединения
                for conn_id in closed_connections:
                    del self.connections[conn_id]

                # Ждем перед следующей проверкой
                await asyncio.sleep(30.0)

            except asyncio.CancelledError:
                # Задача отменена, выходим
                break

            except Exception as e:
                logger.error(f"Error in WebSocket pool cleanup loop: {str(e)}")
                await asyncio.sleep(10.0)

    async def create_connection(
        self, url: str, conn_id: str = None, **kwargs
    ) -> Tuple[str, WebSocketClient]:
        """
        Создает новое WebSocket соединение

        Args:
            url: URL WebSocket сервера
            conn_id: Идентификатор соединения
            **kwargs: Дополнительные параметры для WebSocketClient

        Returns:
            Tuple[str, WebSocketClient]: Идентификатор соединения и клиент
        """
        # Генерируем идентификатор, если не указан
        if not conn_id:
            parts = urlparse(url)
            host = parts.netloc
            path = parts.path

            # Создаем идентификатор на основе хоста и пути
            if path:
                conn_id = f"{host}{path}"
            else:
                conn_id = host

        # Создаем клиент
        client = WebSocketClient(url, **kwargs)

        # Добавляем в словарь соединений
        self.connections[conn_id] = client

        logger.info(f"Created WebSocket connection: {conn_id}")

        return conn_id, client

    async def connect(self, conn_id: str) -> bool:
        """
        Устанавливает соединение с WebSocket сервером

        Args:
            conn_id: Идентификатор соединения

        Returns:
            bool: True, если соединение установлено, иначе False
        """
        if conn_id not in self.connections:
            logger.warning(f"Connection not found: {conn_id}")
            return False

        return await self.connections[conn_id].connect()

    async def disconnect(self, conn_id: str) -> bool:
        """
        Закрывает соединение с WebSocket сервером

        Args:
            conn_id: Идентификатор соединения

        Returns:
            bool: True, если соединение закрыто, иначе False
        """
        if conn_id not in self.connections:
            logger.warning(f"Connection not found: {conn_id}")
            return False

        await self.connections[conn_id].disconnect()
        return True

    async def send(self, conn_id: str, message: Union[Dict, List, str]) -> bool:
        """
        Отправляет сообщение на WebSocket сервер

        Args:
            conn_id: Идентификатор соединения
            message: Сообщение для отправки

        Returns:
            bool: True, если сообщение успешно отправлено, иначе False
        """
        if conn_id not in self.connections:
            logger.warning(f"Connection not found: {conn_id}")
            return False

        return await self.connections[conn_id].send(message)

    def add_subscription(
        self, conn_id: str, subscription_message: Union[Dict, List, str]
    ) -> bool:
        """
        Добавляет сообщение подписки

        Args:
            conn_id: Идентификатор соединения
            subscription_message: Сообщение подписки

        Returns:
            bool: True, если подписка добавлена, иначе False
        """
        if conn_id not in self.connections:
            logger.warning(f"Connection not found: {conn_id}")
            return False

        self.connections[conn_id].add_subscription(subscription_message)
        return True

    def remove_subscription(
        self, conn_id: str, subscription_message: Union[Dict, List, str]
    ) -> bool:
        """
        Удаляет сообщение подписки

        Args:
            conn_id: Идентификатор соединения
            subscription_message: Сообщение подписки

        Returns:
            bool: True, если подписка удалена, иначе False
        """
        if conn_id not in self.connections:
            logger.warning(f"Connection not found: {conn_id}")
            return False

        self.connections[conn_id].remove_subscription(subscription_message)
        return True

    def add_message_handler(
        self, conn_id: str, message_type: str, handler: Callable
    ) -> bool:
        """
        Добавляет обработчик для определенного типа сообщений

        Args:
            conn_id: Идентификатор соединения
            message_type: Тип сообщения
            handler: Функция-обработчик

        Returns:
            bool: True, если обработчик добавлен, иначе False
        """
        if conn_id not in self.connections:
            logger.warning(f"Connection not found: {conn_id}")
            return False

        self.connections[conn_id].add_message_handler(message_type, handler)
        return True

    def remove_message_handler(self, conn_id: str, message_type: str) -> bool:
        """
        Удаляет обработчик для определенного типа сообщений

        Args:
            conn_id: Идентификатор соединения
            message_type: Тип сообщения

        Returns:
            bool: True, если обработчик удален, иначе False
        """
        if conn_id not in self.connections:
            logger.warning(f"Connection not found: {conn_id}")
            return False

        self.connections[conn_id].remove_message_handler(message_type)
        return True

    def get_connection(self, conn_id: str) -> Optional[WebSocketClient]:
        """
        Возвращает соединение по идентификатору

        Args:
            conn_id: Идентификатор соединения

        Returns:
            Optional[WebSocketClient]: Клиент или None, если не найден
        """
        return self.connections.get(conn_id)

    def get_all_connections(self) -> Dict[str, WebSocketClient]:
        """
        Возвращает все соединения

        Returns:
            Dict[str, WebSocketClient]: Словарь соединений
        """
        return self.connections.copy()

    def is_connected(self, conn_id: str) -> bool:
        """
        Проверяет, установлено ли соединение

        Args:
            conn_id: Идентификатор соединения

        Returns:
            bool: True, если соединение установлено, иначе False
        """
        if conn_id not in self.connections:
            return False

        return self.connections[conn_id].connected


class WebSocketMessageProcessor:
    """Процессор для обработки сообщений WebSocket"""

    def __init__(
        self,
        buffer_size: int = 1000,
        process_interval: float = 0.01,
        num_worker_threads: int = 1,
        stats_interval: float = 60.0,
    ):
        """
        Инициализирует процессор сообщений

        Args:
            buffer_size: Размер буфера сообщений
            process_interval: Интервал обработки сообщений в секундах
            num_worker_threads: Количество потоков обработки
            stats_interval: Интервал сбора статистики в секундах
        """
        self.buffer_size = buffer_size
        self.process_interval = process_interval
        self.num_worker_threads = num_worker_threads
        self.stats_interval = stats_interval

        # Очереди сообщений для каждого типа
        self.message_queues = {}

        # Обработчики сообщений для каждого типа
        self.message_handlers = {}

        # Задачи обработки
        self.processing_tasks = []

        # Флаг запуска
        self.running = False

        # Статистика
        self.stats = {
            "processed_messages": 0,
            "dropped_messages": 0,
            "last_process_time": 0.0,
            "avg_process_time": 0.0,
            "queue_size": {},
            "message_types": {},
        }

        # Задача сбора статистики
        self._stats_task = None

        logger.debug("WebSocket message processor initialized")

    async def start(self):
        """Запускает процессор сообщений"""
        if self.running:
            logger.warning("Message processor is already running")
            return

        self.running = True

        # Запускаем задачи обработки
        for _ in range(self.num_worker_threads):
            task = asyncio.create_task(self._process_messages())
            self.processing_tasks.append(task)

        # Запускаем задачу сбора статистики
        self._stats_task = asyncio.create_task(self._collect_stats())

        logger.info(
            f"Message processor started with {self.num_worker_threads} worker threads"
        )

    async def stop(self):
        """Останавливает процессор сообщений"""
        if not self.running:
            logger.warning("Message processor is not running")
            return

        self.running = False

        # Отменяем задачи обработки
        for task in self.processing_tasks:
            task.cancel()

        # Отменяем задачу сбора статистики
        if self._stats_task:
            self._stats_task.cancel()

        # Ждем завершения задач
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        if self._stats_task:
            try:
                await self._stats_task
            except asyncio.CancelledError:
                pass

        # Очищаем задачи
        self.processing_tasks = []
        self._stats_task = None

        logger.info("Message processor stopped")

    def register_handler(self, message_type: str, handler: Callable):
        """
        Регистрирует обработчик для определенного типа сообщений

        Args:
            message_type: Тип сообщения
            handler: Функция-обработчик
        """
        self.message_handlers[message_type] = handler

        # Создаем очередь для этого типа сообщений
        if message_type not in self.message_queues:
            self.message_queues[message_type] = asyncio.Queue(maxsize=self.buffer_size)

        logger.debug(f"Registered handler for message type: {message_type}")

    def unregister_handler(self, message_type: str):
        """
        Удаляет обработчик для определенного типа сообщений

        Args:
            message_type: Тип сообщения
        """
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]

        logger.debug(f"Unregistered handler for message type: {message_type}")

    async def process_message(self, message: Any, message_type: str = None):
        """
        Обрабатывает сообщение

        Args:
            message: Сообщение для обработки
            message_type: Тип сообщения
        """
        # Определяем тип сообщения, если не указан
        if message_type is None:
            message_type = self._get_message_type(message)

        # Если тип не определен или нет обработчика, игнорируем
        if not message_type or message_type not in self.message_handlers:
            return

        # Добавляем сообщение в очередь
        try:
            # Используем неблокирующий put, чтобы не блокировать поток при переполнении очереди
            queue = self.message_queues.get(message_type)
            if queue:
                if queue.full():
                    # Очередь переполнена, увеличиваем счетчик отброшенных сообщений
                    self.stats["dropped_messages"] += 1
                else:
                    await queue.put(message)
        except Exception as e:
            logger.error(f"Error adding message to queue: {str(e)}")

    def _get_message_type(self, message: Any) -> Optional[str]:
        """
        Определяет тип сообщения

        Args:
            message: Сообщение

        Returns:
            Optional[str]: Тип сообщения или None
        """
        if isinstance(message, dict):
            # Пробуем разные ключи, которые могут указывать на тип сообщения
            for key in ["type", "event", "e", "method", "op", "channel"]:
                if key in message:
                    return str(message[key])

        return None

    async def _process_messages(self):
        """Обрабатывает сообщения из очередей"""
        while self.running:
            try:
                # Проверяем все очереди
                for message_type, queue in self.message_queues.items():
                    if message_type not in self.message_handlers:
                        continue

                    # Если в очереди есть сообщения, обрабатываем их
                    if not queue.empty():
                        message = await queue.get()

                        # Засекаем время обработки
                        start_time = time.time()

                        try:
                            # Вызываем обработчик
                            handler = self.message_handlers[message_type]
                            if inspect.iscoroutinefunction(handler):
                                await handler(message)
                            else:
                                handler(message)

                            # Обновляем статистику
                            self.stats["processed_messages"] += 1

                            # Обновляем время обработки
                            process_time = time.time() - start_time
                            self.stats["last_process_time"] = process_time

                            # Обновляем среднее время обработки
                            alpha = 0.05  # Коэффициент сглаживания
                            self.stats["avg_process_time"] = (1 - alpha) * self.stats[
                                "avg_process_time"
                            ] + alpha * process_time

                            # Обновляем статистику по типам сообщений
                            if message_type not in self.stats["message_types"]:
                                self.stats["message_types"][message_type] = 0
                            self.stats["message_types"][message_type] += 1

                        except Exception as e:
                            logger.error(
                                f"Error processing message of type {message_type}: {str(e)}"
                            )

                        finally:
                            # Отмечаем задачу как выполненную
                            queue.task_done()

                # Небольшая пауза перед следующей проверкой
                await asyncio.sleep(self.process_interval)

            except asyncio.CancelledError:
                # Задача отменена, выходим
                break

            except Exception as e:
                logger.error(f"Error in message processing loop: {str(e)}")
                await asyncio.sleep(1.0)

    async def _collect_stats(self):
        """Собирает статистику обработки сообщений"""
        while self.running:
            try:
                # Обновляем размеры очередей
                for message_type, queue in self.message_queues.items():
                    self.stats["queue_size"][message_type] = queue.qsize()

                # Логируем статистику
                if (
                    self.stats["processed_messages"] > 0
                    or self.stats["dropped_messages"] > 0
                ):
                    logger.debug(
                        f"Message stats: processed={self.stats['processed_messages']}, "
                        f"dropped={self.stats['dropped_messages']}, "
                        f"avg_time={self.stats['avg_process_time']:.6f}s"
                    )

                # Ждем перед следующим сбором статистики
                await asyncio.sleep(self.stats_interval)

            except asyncio.CancelledError:
                # Задача отменена, выходим
                break

            except Exception as e:
                logger.error(f"Error in stats collection loop: {str(e)}")
                await asyncio.sleep(10.0)

    def get_stats(self) -> Dict:
        """
        Возвращает статистику обработки сообщений

        Returns:
            Dict: Словарь со статистикой
        """
        # Копируем статистику
        return dict(self.stats)

    def clear_stats(self):
        """Очищает статистику обработки сообщений"""
        self.stats = {
            "processed_messages": 0,
            "dropped_messages": 0,
            "last_process_time": 0.0,
            "avg_process_time": 0.0,
            "queue_size": {},
            "message_types": {},
        }

        logger.debug("Message stats cleared")


# Глобальные экземпляры
_ws_pool = None
_message_processor = None


def get_ws_pool() -> WebSocketPool:
    """
    Возвращает глобальный пул WebSocket соединений

    Returns:
        WebSocketPool: Пул WebSocket соединений
    """
    global _ws_pool

    if _ws_pool is None:
        _ws_pool = WebSocketPool()

    return _ws_pool


def get_message_processor() -> WebSocketMessageProcessor:
    """
    Возвращает глобальный процессор сообщений

    Returns:
        WebSocketMessageProcessor: Процессор сообщений
    """
    global _message_processor

    if _message_processor is None:
        _message_processor = WebSocketMessageProcessor()

    return _message_processor


async def initialize_websocket_services():
    """Инициализирует все WebSocket сервисы"""
    # Инициализируем пул соединений
    pool = get_ws_pool()
    await pool.start()

    # Инициализируем процессор сообщений
    processor = get_message_processor()
    await processor.start()

    logger.info("WebSocket services initialized")


async def shutdown_websocket_services():
    """Останавливает все WebSocket сервисы"""
    # Останавливаем процессор сообщений
    global _message_processor
    if _message_processor:
        await _message_processor.stop()
        _message_processor = None

    # Останавливаем пул соединений
    global _ws_pool
    if _ws_pool:
        await _ws_pool.stop()
        _ws_pool = None

    logger.info("WebSocket services shutdown complete")


async def create_websocket_connection(
    url: str, **kwargs
) -> Tuple[str, WebSocketClient]:
    """
    Создает новое WebSocket соединение

    Args:
        url: URL WebSocket сервера
        **kwargs: Дополнительные параметры для WebSocketClient

    Returns:
        Tuple[str, WebSocketClient]: Идентификатор соединения и клиент
    """
    pool = get_ws_pool()
    return await pool.create_connection(url, **kwargs)


def generate_binance_signature(api_secret: str, query_string: str) -> str:
    """
    Генерирует подпись для API Binance

    Args:
        api_secret: API Secret
        query_string: Строка запроса

    Returns:
        str: Подпись
    """
    signature = hmac.new(
        api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return signature


async def binance_auth_provider(auth_params: Dict, url: str) -> Dict[str, str]:
    """
    Провайдер авторизации для Binance WebSocket API

    Args:
        auth_params: Параметры авторизации
        url: URL WebSocket

    Returns:
        Dict[str, str]: Заголовки авторизации
    """
    api_key = auth_params.get("api_key")
    api_secret = auth_params.get("api_secret")

    if not api_key or not api_secret:
        logger.warning("Missing API credentials for Binance authentication")
        return {}

    # Формируем строку запроса с временной меткой
    query_string = f"timestamp={int(time.time() * 1000)}"

    # Генерируем подпись
    signature = generate_binance_signature(api_secret, query_string)

    # Формируем заголовки
    headers = {"X-MBX-APIKEY": api_key}

    return headers
