"""
Модуль для мониторинга ликвидаций на Binance.
Отслеживает принудительные ликвидации позиций и отправляет уведомления.
"""

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import websockets
from project.config import get_config
from project.infrastructure.database import Database
from project.utils.error_handler import async_handle_error, async_with_retry
from project.utils.logging_utils import get_logger
from project.utils.notify import send_alert

logger = get_logger(__name__)

# Тип для обработчиков ликвидаций
LiquidationHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class BinanceLiquidationMonitor:
    """
    Монитор ликвидаций на Binance Futures.
    """

    def __init__(self):
        """
        Инициализирует монитор ликвидаций.
        """
        self.config = get_config()
        self.websocket_url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        self.connection = None
        self.is_running = False
        self.task = None
        self.handlers: List[LiquidationHandler] = []
        self.monitored_symbols: Set[str] = set()
        self.min_amount_threshold = (
            1000.0  # Минимальная сумма ликвидации для уведомления
        )
        logger.debug("Создан монитор ликвидаций Binance")

    def add_handler(self, handler: LiquidationHandler) -> None:
        """
        Добавляет обработчик ликвидаций.

        Args:
            handler: Асинхронная функция-обработчик ликвидаций
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
            logger.debug("Добавлен обработчик ликвидаций: {handler.__name__}")

    def remove_handler(self, handler: LiquidationHandler) -> None:
        """
        Удаляет обработчик ликвидаций.

        Args:
            handler: Обработчик для удаления
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
            logger.debug("Удален обработчик ликвидаций: {handler.__name__}")

    def add_symbol(self, symbol: str) -> None:
        """
        Добавляет символ для мониторинга.

        Args:
            symbol: Символ в формате Binance (например, BTCUSDT)
        """
        self.monitored_symbols.add(symbol.upper())

    def remove_symbol(self, symbol: str) -> None:
        """
        Удаляет символ из мониторинга.

        Args:
            symbol: Символ в формате Binance
        """
        if symbol.upper() in self.monitored_symbols:
            self.monitored_symbols.remove(symbol.upper())

    def set_amount_threshold(self, threshold: float) -> None:
        """
        Устанавливает минимальную сумму ликвидации для уведомления.

        Args:
            threshold: Пороговое значение в USD
        """
        self.min_amount_threshold = threshold

    @async_handle_error
    async def start_monitoring(self) -> None:
        """
        Запускает мониторинг ликвидаций.
        """
        if self.is_running:
            logger.warning("Монитор ликвидаций уже запущен")
            return

        self.is_running = True
        self.task = asyncio.create_task(self._monitoring_task())
        logger.info("Запущен мониторинг ликвидаций Binance")

    async def stop_monitoring(self) -> None:
        """
        Останавливает мониторинг ликвидаций.
        """
        if not self.is_running:
            logger.warning("Монитор ликвидаций не запущен")
            return

        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

        if self.connection:
            await self.connection.close()
            self.connection = None

        logger.info("Мониторинг ликвидаций Binance остановлен")

    @async_with_retry(max_retries=5, retry_delay=5.0)
    async def _connect_websocket(self) -> websockets.WebSocketClientProtocol:
        """
        Устанавливает соединение с websocket API Binance.

        Returns:
            Установленное соединение
        """
        logger.debug("Подключение к Binance WebSocket: {self.websocket_url}")
        return await websockets.connect(self.websocket_url)

    async def _monitoring_task(self) -> None:
        """
        Основная задача для мониторинга ликвидаций.
        """
        retry_count = 0
        max_retries = 10
        retry_delay = 5.0

        while self.is_running:
            try:
                # Устанавливаем соединение
                self.connection = await self._connect_websocket()
                logger.info("Соединение с Binance WebSocket установлено")
                retry_count = 0

                # Основной цикл чтения сообщений
                async for message in self.connection:
                    if not self.is_running:
                        break

                    try:
                        data = json.loads(message)
                        await self._process_liquidation(data)
                    except json.JSONDecodeError as e:
                        logger.error("Ошибка декодирования JSON: {str(e)}")
                    except Exception as e:
                        logger.error("Ошибка обработки сообщения: {str(e)}")

            except asyncio.CancelledError:
                logger.info("Задача мониторинга ликвидаций отменена")
                break
            except Exception as e:
                retry_count += 1
                logger.error("Ошибка соединения с Binance WebSocket: {str(e)}")

                if retry_count > max_retries:
                    logger.critical(
                        f"Превышено максимальное количество попыток подключения ({max_retries})"
                    )
                    self.is_running = False
                    break

                logger.warning(
                    f"Повторное подключение через {retry_delay} секунд (попытка {retry_count}/{max_retries})")
                await asyncio.sleep(retry_delay)

        # Закрываем соединение при выходе из цикла
        if self.connection:
            await self.connection.close()
            self.connection = None

        logger.info("Задача мониторинга ликвидаций завершена")

    async def _process_liquidation(self, data: Dict[str, Any]) -> None:
        """
        Обрабатывает данные о ликвидации.

        Args:
            data: Данные о ликвидации от Binance
        """
        try:
            # Проверяем формат данных
            if "data" not in data or "o" not in data["data"]:
                return

            # Извлекаем данные о ликвидации
            liq_data = data["data"]["o"]
            symbol = liq_data.get("s", "")
            side = liq_data.get("S", "")
            quantity = float(liq_data.get("q", 0))
            price = float(liq_data.get("p", 0))
            timestamp = int(liq_data.get("T", 0))

            # Проверяем, интересует ли нас этот символ
            if self.monitored_symbols and symbol not in self.monitored_symbols:
                return

            # Рассчитываем сумму ликвидации
            amount_usd = quantity * price

            # Проверяем порог суммы
            if amount_usd < self.min_amount_threshold:
                return

            # Создаем структурированные данные о ликвидации
            liquidation_data = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "amount_usd": amount_usd,
                "timestamp": timestamp,
                "time": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(timestamp / 1000)
                ),
            }

            logger.info(
                f"Обнаружена ликвидация: {symbol} {side} {quantity} по цене {price} "
                f"(${amount_usd:.2f})"
            )

            # Сохраняем данные о ликвидации в БД
            await self._store_liquidation(liquidation_data)

            # Отправляем уведомление о крупных ликвидациях
            if amount_usd > self.min_amount_threshold * 10:
                await self._notify_liquidation(liquidation_data)

            # Вызываем все обработчики
            for handler in self.handlers:
                try:
                    await handler(liquidation_data)
                except Exception as e:
                    logger.error(
                        f"Ошибка в обработчике ликвидаций {handler.__name__}: {str(e)}"
                    )

        except Exception as e:
            logger.error("Ошибка обработки данных о ликвидации: {str(e)}")

    async def _store_liquidation(self, data: Dict[str, Any]) -> None:
        """
        Сохраняет данные о ликвидации в базу данных.

        Args:
            data: Данные о ликвидации
        """
        try:
            db = Database.get_instance()
            await db.insert("liquidations", data)
            logger.debug("Данные о ликвидации сохранены в БД: {data['symbol']}")
        except Exception as e:
            logger.error("Ошибка сохранения данных о ликвидации в БД: {str(e)}")

    async def _notify_liquidation(self, data: Dict[str, Any]) -> None:
        """
        Отправляет уведомление о ликвидации.

        Args:
            data: Данные о ликвидации
        """
        try:
            message = (
                f"🔥 *Крупная ликвидация на Binance*\n"
                f"Символ: `{data['symbol']}`\n"
                f"Сторона: `{'LONG' if data['side'] == 'SELL' else 'SHORT'}`\n"
                f"Количество: `{data['quantity']}`\n"
                f"Цена: `{data['price']}`\n"
                f"Сумма: `${data['amount_usd']:.2f}`\n"
                f"Время: `{data['time']}`"
            )

            await send_alert(message, channel="telegram")
            logger.info(
                f"Отправлено уведомление о крупной ликвидации: {
                    data['symbol']} ${
                    data['amount_usd']:.2f}")
        except Exception as e:
            logger.error("Ошибка отправки уведомления о ликвидации: {str(e)}")

    async def get_recent_liquidations(
        self, symbol: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Получает последние ликвидации из базы данных.

        Args:
            symbol: Символ для фильтрации (None для всех символов)
            limit: Максимальное количество результатов

        Returns:
            Список ликвидаций
        """
        try:
            db = Database.get_instance()

            if symbol:
                query = """
                SELECT * FROM liquidations
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT $2
                """
                return await db.fetch(query, symbol, limit)
            else:
                query = """
                SELECT * FROM liquidations
                ORDER BY timestamp DESC
                LIMIT $1
                """
                return await db.fetch(query, limit)
        except Exception as e:
            logger.error("Ошибка получения данных о ликвидациях: {str(e)}")
            return []
