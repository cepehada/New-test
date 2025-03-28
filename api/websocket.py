"""
Модуль для WebSocket API.
Предоставляет интерфейс для получения данных в реальном времени.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import websockets
from websockets.server import WebSocketServerProtocol

from project.config.configuration import get_config
from project.infrastructure.message_broker import MessageBroker
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Тип для обработчиков событий WebSocket
WSEventHandler = Callable[[Dict[str, Any], WebSocketServerProtocol], Awaitable[None]]


class WebSocketServer:
    """
    Сервер WebSocket для передачи данных в реальном времени.
    """

    def __init__(self, host: str = None, port: int = None):
        """
        Инициализирует сервер WebSocket.

        Args:
            host: Хост для запуска сервера (None для использования из конфигурации)
            port: Порт для запуска сервера (None для использования из конфигурации)
        """
        self.config = get_config()
        self.host = host or self.config.WS_HOST
        self.port = port or self.config.WS_PORT
        
        # Набор активных подключений
        self.connections: Set[WebSocketServerProtocol] = set()
        
        # Словарь для отслеживания подписок на темы
        self.subscriptions: Dict[str, Set[WebSocketServerProtocol]] = {}
        
        # Обработчики событий
        self.event_handlers: Dict[str, WSEventHandler] = {}
        
        # Брокер сообщений для интеграции с остальной системой
        self.message_broker = MessageBroker.get_instance()
        
        # Серверная задача
        self.server_task = None
        
        # Флаг, указывающий, запущен ли сервер
        self.is_running = False
        
        logger.debug(f"WebSocket server initialized at {self.host}:{self.port}")

    async def start(self) -> None:
        """
        Запускает сервер WebSocket.
        """
        if self.is_running:
            logger.warning("WebSocket server is already running")
            return

        # Инициализируем брокер сообщений
        await self.message_broker.initialize()
        
        # Создаем сервер
        try:
            server = await websockets.serve(self._handle_connection, self.host, self.port)
            self.is_running = True
            
            # Сохраняем серверную задачу
            self.server_task = asyncio.create_task(self._keep_alive())
            
            logger.info(f"WebSocket server started at {self.host}:{self.port}")
            
            # Подключаемся к брокеру сообщений для получения обновлений
            await self._subscribe_to_message_broker()
            
            # Ожидаем завершения сервера (не завершается сам по себе)
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {str(e)}")
            self.is_running = False
            raise

    async def stop(self) -> None:
        """
        Останавливает сервер WebSocket.
        """
        if not self.is_running:
            logger.warning("WebSocket server is not running")
            return

        # Отменяем задачу поддержания соединения
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
            self.server_task = None

        # Закрываем все соединения
        close_tasks = []
        for ws in list(self.connections):
            close_tasks.append(self._close_connection(ws))
        
        if close_tasks:
            await asyncio.gather(*close_tasks)
        
        # Очищаем структуры данных
        self.connections.clear()
        self.subscriptions.clear()
        
        self.is_running = False
        logger.info("WebSocket server stopped")

    @async_handle_error
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """
        Обрабатывает новое WebSocket-соединение.

        Args:
            websocket: WebSocket-соединение
            path: Путь запроса
        """
        # Добавляем соединение в набор
        self.connections.add(websocket)
        
        # Инициализируем клиентский ID
        client_id = str(uuid.uuid4())
        
        logger.info(f"New WebSocket connection: {client_id} ({websocket.remote_address})")
        
        try:
            # Отправляем приветственное сообщение
            await websocket.send(json.dumps({
                "type": "welcome",
                "client_id": client_id,
                "timestamp": int(time.time()),
                "message": "Connected to Trading Bot WebSocket Server"
            }))
            
            # Обрабатываем сообщения от клиента
            async for message in websocket:
                try:
                    # Парсим JSON
                    data = json.loads(message)
                    
                    # Обрабатываем запрос
                    await self._process_client_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON",
                        "timestamp": int(time.time())
                    }))
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {str(e)}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Internal server error",
                        "message": str(e),
                        "timestamp": int(time.time())
                    }))
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_id}")
        finally:
            # Удаляем соединение из всех структур данных
            await self._close_connection(websocket)

    async def _process_client_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Обрабатывает сообщение от клиента.

        Args:
            websocket: WebSocket-соединение
            data: Данные сообщения
        """
        message_type = data.get("type")
        
        if message_type == "subscribe":
            # Обрабатываем подписку на тему
            await self._handle_subscribe(websocket, data)
        elif message_type == "unsubscribe":
            # Обрабатываем отписку от темы
            await self._handle_unsubscribe(websocket, data)
        elif message_type == "ping":
            # Отвечаем на пинг
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": int(time.time())
            }))
        elif message_type == "event":
            # Обрабатываем пользовательское событие
            await self._handle_event(websocket, data)
        else:
            # Неизвестный тип сообщения
            await websocket.send(json.dumps({
                "type": "error",
                "error": "Unknown message type",
                "timestamp": int(time.time())
            }))

    async def _handle_subscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Обрабатывает запрос на подписку.

        Args:
            websocket: WebSocket-соединение
            data: Данные запроса
        """
        topic = data.get("topic")
        
        if not topic:
            await websocket.send(json.dumps({
                "type": "error",
                "error": "Topic not specified",
                "timestamp": int(time.time())
            }))
            return
        
        # Добавляем соединение в список подписчиков темы
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        
        self.subscriptions[topic].add(websocket)
        
        # Отправляем подтверждение
        await websocket.send(json.dumps({
            "type": "subscribed",
            "topic": topic,
            "timestamp": int(time.time())
        }))
        
        logger.debug(f"Client subscribed to {topic}")

    async def _handle_unsubscribe(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Обрабатывает запрос на отписку.

        Args:
            websocket: WebSocket-соединение
            data: Данные запроса
        """
        topic = data.get("topic")
        
        if not topic:
            await websocket.send(json.dumps({
                "type": "error",
                "error": "Topic not specified",
                "timestamp": int(time.time())
            }))
            return
        
        # Удаляем соединение из списка подписчиков темы
        if topic in self.subscriptions and websocket in self.subscriptions[topic]:
            self.subscriptions[topic].remove(websocket)
            
            # Если больше нет подписчиков, удаляем тему
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
        
        # Отправляем подтверждение
        await websocket.send(json.dumps({
            "type": "unsubscribed",
            "topic": topic,
            "timestamp": int(time.time())
        }))
        
        logger.debug(f"Client unsubscribed from {topic}")

    async def _handle_event(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> None:
        """
        Обрабатывает пользовательское событие.

        Args:
            websocket: WebSocket-соединение
            data: Данные события
        """
        event = data.get("event")
        
        if not event:
            await websocket.send(json.dumps({
                "type": "error",
                "error": "Event not specified",
                "timestamp": int(time.time())
            }))
            return
        
        # Проверяем наличие обработчика для события
        if event in self.event_handlers:
            try:
                # Вызываем обработчик
                await self.event_handlers[event](data, websocket)
            except Exception as e:
                logger.error(f"Error in event handler for '{event}': {str(e)}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Error processing event '{event}'",
                    "message": str(e),
                    "timestamp": int(time.time())
                }))
        else:
            # Неизвестное событие
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Unknown event '{event}'",
                "timestamp": int(time.time())
            }))

    async def _close_connection(self, websocket: WebSocketServerProtocol) -> None:
        """
        Закрывает и удаляет WebSocket-соединение из всех структур данных.

        Args:
            websocket: WebSocket-соединение
        """
        # Удаляем из набора соединений
        if websocket in self.connections:
            self.connections.remove(websocket)
        
        # Удаляем из всех подписок
        for topic in list(self.subscriptions.keys()):
            if websocket in self.subscriptions[topic]:
                self.subscriptions[topic].remove(websocket)
                
                # Если больше нет подписчиков, удаляем тему
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]
        
        # Закрываем соединение, если оно еще открыто
        try:
            await websocket.close()
        except:
            pass

    async def _keep_alive(self) -> None:
        """
        Периодически отправляет проверочные сообщения для поддержания соединения.
        """
        while True:
            try:
                # Ожидаем 30 секунд
                await asyncio.sleep(30)
                
                # Отправляем пинг всем соединениям
                ping_message = json.dumps({
                    "type": "ping",
                    "timestamp": int(time.time())
                })
                
                for ws in list(self.connections):
                    try:
                        await ws.send(ping_message)
                    except:
                        # Соединение, вероятно, закрыто, удаляем его
                        await self._close_connection(ws)
            except asyncio.CancelledError:
                # Задача отменена, выходим из цикла
                break
            except Exception as e:
                logger.error(f"Error in keep-alive task: {str(e)}")

    async def _subscribe_to_message_broker(self) -> None:
        """
        Подписывается на сообщения от брокера сообщений.
        """
        # Получаем список всех тем
        topics = self.message_broker.get_all_topics()
        
        # Подписываемся на каждую тему
        for topic in topics:
            await self.message_broker.subscribe(topic, self._on_message_broker_message)

    async def _on_message_broker_message(self, message: Dict[str, Any]) -> None:
        """
        Обрабатывает сообщение от брокера сообщений.

        Args:
            message: Сообщение
        """
        # Получаем тему сообщения
        topic = message.get("topic")
        
        if not topic:
            logger.warning("Received message without topic from message broker")
            return
        
        # Проверяем наличие подписчиков
        if topic not in self.subscriptions:
            # Нет подписчиков, пропускаем
            return
        
        # Создаем сообщение для отправки
        ws_message = {
            "type": "message",
            "topic": topic,
            "data": message.get("data", {}),
            "timestamp": message.get("timestamp", int(time.time()))
        }
        
        # Сериализуем сообщение
        ws_message_json = json.dumps(ws_message)
        
        # Отправляем сообщение всем подписчикам
        for ws in list(self.subscriptions[topic]):
            try:
                await ws.send(ws_message_json)
            except:
                # Соединение, вероятно, закрыто, удаляем его
                await self._close_connection(ws)

    def register_event_handler(self, event: str, handler: WSEventHandler) -> None:
        """
        Регистрирует обработчик события.

        Args:
            event: Название события
            handler: Обработчик события
        """
        self.event_handlers[event] = handler
        logger.debug(f"Registered handler for event '{event}'")

    def unregister_event_handler(self, event: str) -> None:
        """
        Удаляет обработчик события.

        Args:
            event: Название события
        """
        if event in self.event_handlers:
            del self.event_handlers[event]
            logger.debug(f"Unregistered handler for event '{event}'")

    @async_handle_error
    async def broadcast(self, topic: str, data: Dict[str, Any]) -> int:
        """
        Отправляет сообщение всем подписчикам темы.

        Args:
            topic: Тема сообщения
            data: Данные сообщения

        Returns:
            int: Количество клиентов, получивших сообщение
        """
        if not self.is_running:
            logger.warning("WebSocket server is not running, cannot broadcast")
            return 0
        
        # Проверяем наличие подписчиков
        if topic not in self.subscriptions:
            # Нет подписчиков
            return 0
        
        # Создаем сообщение
        message = {
            "type": "message",
            "topic": topic,
            "data": data,
            "timestamp": int(time.time())
        }
        
        # Сериализуем сообщение
        message_json = json.dumps(message)
        
        # Счетчик отправленных сообщений
        sent_count = 0
        
        # Отправляем сообщение всем подписчикам
        for ws in list(self.subscriptions[topic]):
            try:
                await ws.send(message_json)
                sent_count += 1
            except:
                # Соединение, вероятно, закрыто, удаляем его
                await self._close_connection(ws)
        
        return sent_count

    @async_handle_error
    async def send_to(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]) -> bool:
        """
        Отправляет сообщение конкретному WebSocket-соединению.

        Args:
            websocket: WebSocket-соединение
            data: Данные сообщения

        Returns:
            bool: True в случае успешной отправки, иначе False
        """
        if not self.is_running:
            logger.warning("WebSocket server is not running, cannot send message")
            return False
        
        # Проверяем, активно ли соединение
        if websocket not in self.connections:
            return False
        
        try:
            # Отправляем сообщение
            await websocket.send(json.dumps(data))
            return True
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {str(e)}")
            # Соединение, вероятно, закрыто, удаляем его
            await self._close_connection(websocket)
            return False


# Глобальный экземпляр WebSocket-сервера
_ws_server = None


def get_websocket_server() -> WebSocketServer:
    """
    Получает глобальный экземпляр WebSocket-сервера.

    Returns:
        WebSocketServer: Экземпляр WebSocket-сервера
    """
    global _ws_server
    
    if _ws_server is None:
        config = get_config()
        _ws_server = WebSocketServer(config.WS_HOST, config.WS_PORT)
    
    return _ws_server
