"""
Модуль для приема вебхуков от TradingView.
Позволяет получать и обрабатывать торговые сигналы от TradingView.
"""

import hashlib
import hmac
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aiohttp import web
from project.bots.strategies.strategy_manager import StrategyManager
from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)

# Тип для обработчиков вебхуков
WebhookHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class TradingViewWebhooks:
    """
    Класс для приема и обработки вебхуков от TradingView.
    """

    def __init__(
        self, host: str = "0.0.0.0", port: int = 8081, secret: Optional[str] = None
    ):
        """
        Инициализирует обработчик вебхуков TradingView.

        Args:
            host: Хост для запуска сервера вебхуков
            port: Порт для запуска сервера вебхуков
            secret: Секретный ключ для валидации вебхуков (None для использования из конфигурации)
        """
        self.config = get_config()
        self.host = host
        self.port = port
        self.secret = secret or self.config.ENCRYPTION_KEY
        self.app = web.Application()
        self.runner = None
        self.site = None

        # Регистрируем обработчики URL
        self.app.router.add_post("/webhook", self.handle_webhook)
        self.app.router.add_get("/webhook/status", self.status)

        # Словарь для хранения обработчиков вебхуков
        self.handlers: Dict[str, List[WebhookHandler]] = {}

        logger.debug(f"TradingView вебхуки настроены на {self.host}:{self.port}")

    def register_handler(self, alert_name: str, handler: WebhookHandler) -> None:
        """
        Регистрирует обработчик для конкретного типа оповещения.

        Args:
            alert_name: Имя оповещения
            handler: Асинхронная функция-обработчик вебхука
        """
        if alert_name not in self.handlers:
            self.handlers[alert_name] = []

        if handler not in self.handlers[alert_name]:
            self.handlers[alert_name].append(handler)
            logger.debug(f"Зарегистрирован обработчик для оповещения {alert_name}")

    def unregister_handler(self, alert_name: str, handler: WebhookHandler) -> None:
        """
        Удаляет обработчик для конкретного типа оповещения.

        Args:
            alert_name: Имя оповещения
            handler: Обработчик для удаления
        """
        if alert_name in self.handlers and handler in self.handlers[alert_name]:
            self.handlers[alert_name].remove(handler)
            logger.debug(f"Удален обработчик для оповещения {alert_name}")

            # Удаляем ключ, если больше нет обработчиков
            if not self.handlers[alert_name]:
                del self.handlers[alert_name]

    @async_handle_error
    async def start(self) -> None:
        """
        Запускает сервер вебхуков.
        """
        if self.runner:
            logger.warning("Сервер вебхуков TradingView уже запущен")
            return

        # Запускаем сервер
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        logger.info(
            f"Сервер вебхуков TradingView запущен на http://{self.host}:{self.port}"
        )

    async def stop(self) -> None:
        """
        Останавливает сервер вебхуков.
        """
        if self.site:
            await self.site.stop()
            self.site = None

        if self.runner:
            await self.runner.cleanup()
            self.runner = None

        logger.info("Сервер вебхуков TradingView остановлен")

    async def _validate_webhook(self, request: web.Request) -> Optional[Dict[str, Any]]:
        """
        Проверяет валидность вебхука от TradingView.

        Args:
            request: HTTP-запрос

        Returns:
            Данные вебхука или None, если вебхук невалиден
        """
        # Проверяем наличие необходимых заголовков
        if "X-Tradingview-Hmac-SHA256" not in request.headers:
            logger.warning("Отсутствует заголовок X-Tradingview-Hmac-SHA256")
            return None

        try:
            # Получаем данные запроса
            data_bytes = await request.read()
            data_str = data_bytes.decode("utf-8")

            # Проверяем подпись
            signature = request.headers["X-Tradingview-Hmac-SHA256"]
            computed_signature = hmac.new(
                self.secret.encode("utf-8"), data_bytes, hashlib.sha256
            ).hexdigest()

            if signature != computed_signature:
                logger.warning("Невалидная подпись вебхука")
                return None

            # Парсим JSON
            try:
                data = json.loads(data_str)
                return data
            except json.JSONDecodeError:
                logger.error(f"Невалидный JSON: {data_str}")
                return None

        except Exception as e:
            logger.error(f"Ошибка при проверке вебхука: {str(e)}")
            return None

    async def handle_webhook(self, request: web.Request) -> web.Response:
        """
        Обрабатывает вебхук от TradingView.

        Args:
            request: HTTP-запрос

        Returns:
            HTTP-ответ
        """
        # Проверяем валидность вебхука
        data = await self._validate_webhook(request)
        if not data:
            return web.json_response({"error": "Invalid webhook signature"}, status=401)

        # Получаем данные вебхука
        try:
            # TradingView отправляет данные в специфическом формате
            alert_name = data.get("strategy", {}).get("alert_name", "") or data.get(
                "alert_name", ""
            )
            ticker = data.get("ticker", "")
            action = data.get("strategy", {}).get("action", "") or data.get(
                "action", ""
            )
            price = data.get("strategy", {}).get("price", 0) or data.get("price", 0)

            # Логируем полученный вебхук
            logger.info(
                f"Получен вебхук от TradingView: {alert_name}, {ticker}, {action}, {price}"
            )

            # Отправляем уведомление о сигнале
            await send_trading_signal(
                f"Получен сигнал от TradingView: {alert_name} - {ticker} - {action} по цене {price}"
            )

            # Обрабатываем вебхук в соответствии с его типом
            if alert_name in self.handlers:
                for handler in self.handlers[alert_name]:
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(
                            f"Ошибка в обработчике вебхука {handler.__name__}: {str(e)}"
                        )
            else:
                # Если нет специфического обработчика, используем обработчик по умолчанию
                await self._default_handler(data)

            return web.json_response({"status": "success"})

        except Exception as e:
            logger.error(f"Ошибка при обработке вебхука: {str(e)}")
            return web.json_response(
                {"error": "Internal server error", "message": str(e)}, status=500
            )

    async def status(self, request: web.Request) -> web.Response:
        """
        Возвращает статус сервера вебхуков.

        Args:
            request: HTTP-запрос

        Returns:
            HTTP-ответ
        """
        return web.json_response(
            {
                "status": "ok",
                "timestamp": time.time(),
                "handlers": list(self.handlers.keys()),
            }
        )

    async def _default_handler(self, data: Dict[str, Any]) -> None:
        """
        Обработчик вебхука по умолчанию.

        Args:
            data: Данные вебхука
        """
        # Получаем необходимые данные из вебхука
        alert_name = data.get("strategy", {}).get("alert_name", "") or data.get(
            "alert_name", ""
        )
        ticker = data.get("ticker", "")
        action = data.get("strategy", {}).get("action", "") or data.get("action", "")
        price = data.get("strategy", {}).get("price", 0) or data.get("price", 0)

        logger.info(
            f"Обработка вебхука по умолчанию: {alert_name}, {ticker}, {action}, {price}"
        )

        # Если действие - это торговый сигнал, пытаемся выполнить соответствующую операцию
        if action.lower() in ["buy", "sell", "long", "short", "exit"]:
            try:
                # Получаем менеджер стратегий
                strategy_manager = StrategyManager.get_instance()

                # Создаем сигнал для обработки стратегиями
                signal = {
                    "source": "tradingview",
                    "alert_name": alert_name,
                    "ticker": ticker,
                    "action": action.lower(),
                    "price": float(price),
                    "timestamp": time.time(),
                }

                # Отправляем сигнал всем активным стратегиям
                await self.router.process_signal(signal)

                logger.info(
                    f"Сигнал от TradingView отправлен стратегиям: {action} {ticker}"
                )

            except Exception as e:
                logger.error(f"Ошибка при обработке торгового сигнала: {str(e)}")
