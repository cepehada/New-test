"""
Модуль для WebSocket API.
Предоставляет интерфейс для получения данных в реальном времени.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Set

import aiohttp
import jwt
from aiohttp import web
from project.bots.bot_manager import BotManager
from project.bots.strategies.strategy_manager import StrategyManager
from project.config import get_config
from project.data.market_data import MarketData
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Глобальные объекты
market_data = MarketData.get_instance()
bot_manager = BotManager.get_instance()
strategy_manager = StrategyManager.get_instance()
config = get_config()

# Настройки JWT
JWT_SECRET = config.JWT_SECRET or "super-secret-key-change-this"
JWT_ALGORITHM = "HS256"

# Активные клиенты
active_clients: Dict[str, Set[web.WebSocketResponse]] = {
    "market": set(),
    "orders": set(),
    "bots": set(),
    "strategies": set(),
    "alerts": set(),
    "system": set(),
}

# Кэш последних отправленных сообщений
message_cache: Dict[str, Dict[str, Any]] = {}


# Аутентификация для WebSocket
async def authenticate_ws(message, ws):
    """
    Аутентифицирует WebSocket-соединение.

    Args:
        message: JSON-сообщение с токеном
        ws: WebSocket-соединение

    Returns:
        True, если аутентификация прошла успешно, иначе False
    """
    try:
        # Извлекаем токен из сообщения
        token = message.get("token")

        if not token:
            await ws.send_json(
                {
                    "type": "error",
                    "code": "auth_required",
                    "message": "Authentication required",
                }
            )
            return False

        try:
            # Проверяем токен
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

            # Проверяем срок действия
            if "exp" in payload and time.time() > payload["exp"]:
                await ws.send_json(
                    {
                        "type": "error",
                        "code": "token_expired",
                        "message": "Token expired",
                    }
                )
                return False

            # Сохраняем пользователя в атрибутах соединения
            ws.username = payload.get("username")

            await ws.send_json(
                {"type": "auth", "status": "success", "username": ws.username}
            )

            return True

        except jwt.PyJWTError:
            await ws.send_json(
                {"type": "error", "code": "invalid_token", "message": "Invalid token"}
            )
            return False

    except Exception as e:
        logger.error("Authentication error: {str(e)}" %)
        await ws.send_json(
            {"type": "error", "code": "auth_error", "message": "Authentication error"}
        )
        return False


# Обработчик WebSocket-соединений
async def websocket_handler(request):
    """
    Обрабатывает WebSocket-соединения.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Инициализируем атрибуты
    ws.subscriptions = set()
    ws.authenticated = False
    ws.username = None

    try:
        logger.debug("WebSocket connection established")

        # Отправляем приветственное сообщение
        await ws.send_json(
            {
                "type": "welcome",
                "message": "Welcome to Trading Bot WebSocket API",
                "timestamp": time.time(),
            }
        )

        # Обрабатываем сообщения
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    # Обрабатываем аутентификацию
                    if msg_type == "auth":
                        ws.authenticated = await authenticate_ws(data, ws)
                        continue

                    # Проверяем аутентификацию для всех остальных типов сообщений
                    if not ws.authenticated:
                        await ws.send_json(
                            {
                                "type": "error",
                                "code": "auth_required",
                                "message": "Authentication required",
                            }
                        )
                        continue

                    # Обрабатываем подписки
                    if msg_type == "subscribe":
                        await handle_subscription(ws, data)

                    # Обрабатываем отписки
                    elif msg_type == "unsubscribe":
                        await handle_unsubscription(ws, data)

                    # Обрабатываем пинг
                    elif msg_type == "ping":
                        await ws.send_json({"type": "pong", "timestamp": time.time()})

                    # Неизвестный тип сообщения
                    else:
                        await ws.send_json(
                            {
                                "type": "error",
                                "code": "unknown_type",
                                "message": f"Unknown message type: {msg_type}",
                            }
                        )

                except json.JSONDecodeError:
                    await ws.send_json(
                        {
                            "type": "error",
                            "code": "invalid_json",
                            "message": "Invalid JSON format",
                        }
                    )
                except Exception as e:
                    logger.error("WebSocket message error: {str(e)}" %)
                    await ws.send_json(
                        {
                            "type": "error",
                            "code": "message_error",
                            "message": "Error processing message",
                        }
                    )

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(
                    f"WebSocket connection closed with exception: {ws.exception()}"
                )

    finally:
        # Удаляем соединение из всех каналов
        for channel in active_clients:
            active_clients[channel].discard(ws)

        logger.debug("WebSocket connection closed")

    return ws


# Обработчик подписок
async def handle_subscription(ws, data):
    """
    Обрабатывает запрос на подписку.

    Args:
        ws: WebSocket-соединение
        data: Данные запроса
    """
    channels = data.get("channels", [])

    if not channels:
        await ws.send_json(
            {
                "type": "error",
                "code": "channels_required",
                "message": "Channels are required for subscription",
            }
        )
        return

    # Обрабатываем подписки на каналы
    subscribed = []
    invalid = []

    for channel in channels:
        # Проверяем, существует ли канал
        if channel not in active_clients:
            invalid.append(channel)
            continue

        # Добавляем клиента в канал
        active_clients[channel].add(ws)
        ws.subscriptions.add(channel)
        subscribed.append(channel)

        # Отправляем последнее сообщение из кэша, если оно есть
        if channel in message_cache:
            await ws.send_json(message_cache[channel])

    # Отправляем подтверждение подписки
    await ws.send_json(
        {
            "type": "subscription",
            "status": "success",
            "subscribed": subscribed,
            "invalid": invalid,
            "timestamp": time.time(),
        }
    )


# Обработчик отписок
async def handle_unsubscription(ws, data):
    """
    Обрабатывает запрос на отписку.

    Args:
        ws: WebSocket-соединение
        data: Данные запроса
    """
    channels = data.get("channels", [])

    if not channels:
        # Отписываем от всех каналов
        for channel in list(ws.subscriptions):
            active_clients[channel].discard(ws)

        unsubscribed = list(ws.subscriptions)
        ws.subscriptions.clear()
    else:
        # Отписываем от указанных каналов
        unsubscribed = []

        for channel in channels:
            if channel in active_clients and channel in ws.subscriptions:
                active_clients[channel].discard(ws)
                ws.subscriptions.discard(channel)
                unsubscribed.append(channel)

    # Отправляем подтверждение отписки
    await ws.send_json(
        {
            "type": "unsubscription",
            "status": "success",
            "unsubscribed": unsubscribed,
            "timestamp": time.time(),
        }
    )


# Отправка сообщений в каналы
async def broadcast_to_channel(channel: str, message: Dict[str, Any]) -> None:
    """
    Отправляет сообщение всем подписчикам канала.

    Args:
        channel: Название канала
        message: Сообщение для отправки
    """
    if channel not in active_clients:
        logger.warning("Unknown channel: {channel}" %)
        return

    # Добавляем тип и timestamp, если их нет
    if "type" not in message:
        message["type"] = channel

    if "timestamp" not in message:
        message["timestamp"] = time.time()

    # Кэшируем сообщение
    message_cache[channel] = message

    # Отправляем сообщение всем клиентам канала
    disconnected = set()

    for ws in active_clients[channel]:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error("Error sending message to client: {str(e)}" %)
            disconnected.add(ws)

    # Удаляем отключенные соединения
    for ws in disconnected:
        active_clients[channel].discard(ws)
        logger.debug("Removed disconnected client from channel: {channel}" %)


# Обновление рыночных данных
async def update_market_data() -> None:
    """
    Периодически обновляет и отправляет рыночные данные.
    """
    try:
        while True:
            # Проверяем, есть ли подписчики
            if active_clients["market"]:
                # Список популярных символов
                symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
                exchange_id = "binance"

                # Получаем тикеры
                tickers = {}
                for symbol in symbols:
                    ticker = await market_data.get_ticker(exchange_id, symbol)
                    if ticker:
                        tickers[symbol] = ticker

                # Отправляем обновление
                if tickers:
                    await broadcast_to_channel(
                        "market",
                        {
                            "type": "market_update",
                            "exchange": exchange_id,
                            "tickers": tickers,
                            "timestamp": time.time(),
                        },
                    )

            # Ждем перед следующим обновлением
            await asyncio.sleep(5)

    except Exception as e:
        logger.error("Error in market data update: {str(e)}" %)


# Обновление состояния ботов
async def update_bots_state() -> None:
    """
    Периодически обновляет и отправляет состояние ботов.
    """
    try:
        while True:
            # Проверяем, есть ли подписчики
            if active_clients["bots"]:
                # Получаем список ботов
                bots = bot_manager.get_bots()

                # Преобразуем в JSON-совместимый формат
                bot_list = []
                for bot_id, bot in bots.items():
                    bot_info = {
                        "id": bot_id,
                        "name": bot.name,
                        "status": bot.get_status(),
                        "type": bot.__class__.__name__,
                        "exchange": bot.exchange_id,
                        "symbols": bot.symbols,
                        "stats": bot.stats,
                        "started_at": bot.start_time,
                        "uptime": (
                            time.time() - bot.start_time if bot.start_time > 0 else 0
                        ),
                    }
                    bot_list.append(bot_info)

                # Отправляем обновление
                await broadcast_to_channel(
                    "bots",
                    {
                        "type": "bots_update",
                        "bots": bot_list,
                        "count": len(bot_list),
                        "timestamp": time.time(),
                    },
                )

            # Ждем перед следующим обновлением
            await asyncio.sleep(10)

    except Exception as e:
        logger.error("Error in bots state update: {str(e)}" %)


# Обновление состояния стратегий
async def update_strategies_state() -> None:
    """
    Периодически обновляет и отправляет состояние стратегий.
    """
    try:
        while True:
            # Проверяем, есть ли подписчики
            if active_clients["strategies"]:
                # Получаем список стратегий
                running_strategies = strategy_manager.get_running_strategies()

                # Отправляем обновление
                await broadcast_to_channel(
                    "strategies",
                    {
                        "type": "strategies_update",
                        "strategies": running_strategies,
                        "count": len(running_strategies),
                        "timestamp": time.time(),
                    },
                )

            # Ждем перед следующим обновлением
            await asyncio.sleep(10)

    except Exception as e:
        logger.error("Error in strategies state update: {str(e)}" %)


# Обновление системной информации
async def update_system_info() -> None:
    """
    Периодически обновляет и отправляет системную информацию.
    """
    try:
        import psutil

        while True:
            # Проверяем, есть ли подписчики
            if active_clients["system"]:
                # Собираем системную информацию
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                # Отправляем обновление
                await broadcast_to_channel(
                    "system",
                    {
                        "type": "system_info",
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_used": memory.used,
                        "memory_total": memory.total,
                        "timestamp": time.time(),
                    },
                )
                
            # Ждем перед следующим обновлением
            await asyncio.sleep(10)

    except ImportError:
        logger.warning("psutil not installed, system info updates disabled")
    except Exception as e:
        logger.error("Error in system info update: {str(e)}" %)


# Запуск WebSocket-сервера
async def start_websocket_server(app: web.Application) -> None:
    """
    Настраивает WebSocket-сервер в приложении.

    Args:
        app: Приложение aiohttp
    """
    # Добавляем обработчик WebSocket
    app.router.add_get("/api/ws", websocket_handler)

    # Запускаем фоновые задачи
    app["market_task"] = asyncio.create_task(update_market_data())
    app["bots_task"] = asyncio.create_task(update_bots_state())
    app["strategies_task"] = asyncio.create_task(update_strategies_state())
    app["system_task"] = asyncio.create_task(update_system_info())

    # Обработчик закрытия приложения
    async def on_shutdown(app):
        # Отменяем фоновые задачи
        app["market_task"].cancel()
        app["bots_task"].cancel()
        app["strategies_task"].cancel()
        app["system_task"].cancel()

        # Закрываем все WebSocket-соединения
        for channel in active_clients:
            for ws in list(active_clients[channel]):
                await ws.close(code=1000, message=b"Server shutdown")
            active_clients[channel].clear()

    # Добавляем обработчик закрытия
    app.on_shutdown.append(on_shutdown)

    logger.info("WebSocket server configured")


# Функция для отправки оповещений
async def send_alert(
    alert_type: str, message: str, data: Dict[str, Any] = None
) -> None:
    """
    Отправляет оповещение всем подписчикам канала alerts.

    Args:
        alert_type: Тип оповещения
        message: Текст оповещения
        data: Дополнительные данные
    """
    alert = {
        "type": "alert",
        "alert_type": alert_type,
        "message": message,
        "timestamp": time.time(),
    }

    if data:
        alert["data"] = data

    await broadcast_to_channel("alerts", alert)


async def broadcast_updates(symbol, data_type, data):
    """Рассылает обновления всем подписанным клиентам"""
    
    # Используем .items() для перебора словаря
    for client_id, subscriptions in active_subscriptions.items():
        if symbol in subscriptions and data_type in subscriptions[symbol]:
            # ...existing code...
