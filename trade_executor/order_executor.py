"""
Модуль для исполнения торговых ордеров.
Предоставляет функции для создания и управления ордерами на биржах.
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import get_config
from project.data.symbol_manager import SymbolManager
from project.infrastructure.database import Database
from project.utils.ccxt_exchanges import (
    cancel_order,
    create_order,
    fetch_open_orders,
    fetch_order,
    fetch_ticker,
)
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)


@dataclass
class OrderResult:
    """
    Результат выполнения ордера.
    """

    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    status: str = "unknown"
    fees: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class OrderExecutor:
    """
    Класс для исполнения торговых ордеров на биржах.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "OrderExecutor":
        """
        Получает глобальный экземпляр OrderExecutor.
        
        Returns:
            OrderExecutor: Экземпляр класса OrderExecutor
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self, config=None):
        """
        Инициализирует исполнитель ордеров.
        
        Args:
            config (Dict, optional): Конфигурация. Defaults to None.
        """
        self.config = config or get_config()
        self.symbol_manager = SymbolManager.get_instance()
        self._db = None
        self._last_orders = {}
        self._order_cache = {}
        self._order_history = {}
        self._active_orders = {}
        self._failed_orders = {}
        self._stats = {"created": 0, "filled": 0, "canceled": 0, "failed": 0}
        
    @async_handle_error
    async def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None,
        exchange_id: str = "binance",
        **kwargs,
    ) -> OrderResult:
        """
        Выполняет торговый ордер на указанной бирже.

        Args:
            symbol: Символ торговой пары
            side: Сторона ордера (buy/sell)
            amount: Количество
            order_type: Тип ордера (market/limit/etc)
            price: Цена (для лимитных ордеров)
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры ордера

        Returns:
            Результат выполнения ордера
        """
        try:
            # Генерируем уникальный client_order_id
            client_order_id = (
                kwargs.get("client_order_id") or f"bot_{uuid.uuid4().hex[:16]}"
            )

            # Нормализуем параметры
            symbol = await self._normalize_symbol(exchange_id, symbol)
            side = side.lower()
            order_type = order_type.lower()
            amount = await self._normalize_amount(exchange_id, symbol, amount)

            if order_type == "limit" and price is None:
                # Для лимитных ордеров требуется цена
                ticker = await fetch_ticker(exchange_id, symbol)
                price = ticker.get("last", 0)
                logger.warning(
                    "Для лимитного ордера не указана цена, используем текущую цену %s", price
                )

            if price is not None:
                price = await self._normalize_price(exchange_id, symbol, price)

            # Добавляем client_order_id к параметрам
            params = kwargs.get("params", {})
            params["clientOrderId"] = client_order_id

            # Логируем информацию о создаваемом ордере
            logger.info(
                "Создаем ордер на %s: %s %s %s %s%s",
                exchange_id, side, order_type, amount, symbol,
                f" по цене {price}" if price else ""
            )

            # Отправляем уведомление о создании ордера
            await send_trading_signal(
                f"Создание ордера: {side.upper()} {amount} {symbol}"
                f"{f' по цене {price}' if price else ''} ({order_type})"
            )

            # Создаем ордер на бирже
            order = await create_order(
                exchange_id=exchange_id,
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )

            # Сохраняем ордер в базе данных
            await self._save_order_to_db(exchange_id, order)

            # Формируем результат
            result = OrderResult(
                success=True,
                order_id=order.get("id"),
                client_order_id=client_order_id,
                filled_quantity=float(order.get("filled", 0)),
                average_price=(
                    float(order.get("price", 0)) if order.get("price") else None
                ),
                status=order.get("status", "open"),
                fees=order.get("fee"),
                raw_response=order,
            )

            return result

        except Exception as e:
            error_msg = f"Ошибка при выполнении ордера {side} {order_type} {amount} {symbol}: {str(e)}"
            logger.error("%s", error_msg)

            # Отправляем уведомление об ошибке
            await send_trading_signal(
                f"Ошибка создания ордера: {side.upper()} {amount} {symbol} - {str(e)}"
            )

            return OrderResult(success=False, error=str(e))

    @async_handle_error
    async def cancel_order(
        self, order_id: str, symbol: str, exchange_id: str = "binance", **kwargs
    ) -> OrderResult:
        """
        Отменяет ордер на указанной бирже.

        Args:
            order_id: ID ордера
            symbol: Символ торговой пары
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры

        Returns:
            Результат отмены ордера
        """
        try:
            # Нормализуем параметры
            symbol = await self._normalize_symbol(exchange_id, symbol)

            # Логируем информацию об отменяемом ордере
            logger.info("Отменяем ордер %s на %s для %s", order_id, exchange_id, symbol)

            # Отправляем уведомление об отмене ордера
            await send_trading_signal(f"Отмена ордера: {order_id} ({symbol})")

            # Отменяем ордер на бирже
            cancelled_order = await cancel_order(
                exchange_id=exchange_id,
                order_id=order_id,
                symbol=symbol,
                params=kwargs.get("params", {}),
            )

            # Получаем актуальный статус ордера
            order = await fetch_order(exchange_id, order_id, symbol)

            # Обновляем ордер в базе данных
            await self._update_order_in_db(exchange_id, order)

            return OrderResult(
                success=True,
                order_id=order_id,
                client_order_id=order.get("clientOrderId"),
                filled_quantity=float(order.get("filled", 0)),
                average_price=(
                    float(order.get("price", 0)) if order.get("price") else None
                ),
                status=order.get("status", "canceled"),
                raw_response=order,
            )

        except Exception as e:
            error_msg = (
                f"Ошибка при отмене ордера {order_id} на {exchange_id}: {str(e)}"
            )
            logger.error("%s", error_msg)

            # Отправляем уведомление об ошибке
            await send_trading_signal(
                f"Ошибка отмены ордера: {order_id} ({symbol}) - {str(e)}"
            )

            return OrderResult(success=False, order_id=order_id, error=str(e))

    @async_handle_error
    async def check_order_status(
        self, order_id: str, symbol: str, exchange_id: str = "binance", **kwargs
    ) -> OrderResult:
        """
        Проверяет статус ордера на указанной бирже.

        Args:
            order_id: ID ордера
            symbol: Символ торговой пары
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры

        Returns:
            Результат проверки статуса ордера
        """
        try:
            # Нормализуем параметры
            symbol = await self._normalize_symbol(exchange_id, symbol)

            # Получаем статус ордера на бирже
            order = await fetch_order(
                exchange_id=exchange_id,
                order_id=order_id,
                symbol=symbol,
                params=kwargs.get("params", {}),
            )

            # Обновляем ордер в базе данных
            await self._update_order_in_db(exchange_id, order)

            # Логируем информацию о статусе ордера
            logger.debug(
                "Статус ордера %s на %s: %s", 
                order_id, exchange_id, order.get('status', 'unknown')
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                client_order_id=order.get("clientOrderId"),
                filled_quantity=float(order.get("filled", 0)),
                average_price=(
                    float(order.get("price", 0)) if order.get("price") else None
                ),
                status=order.get("status", "unknown"),
                fees=order.get("fee"),
                raw_response=order,
            )

        except Exception as e:
            logger.error("Ошибка при проверке статуса ордера %s: %s", order_id, str(e))
            return OrderResult(
                success=False,
                order_id=order_id,
                error=str(e)
            )

    @async_handle_error
    async def get_open_orders(
        self, symbol: Optional[str] = None, exchange_id: str = "binance", **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Получает список открытых ордеров.

        Args:
            symbol: Символ торговой пары (None для всех символов)
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры

        Returns:
            Список открытых ордеров
        """
        try:
            # Нормализуем параметры
            if symbol:
                symbol = await self._normalize_symbol(exchange_id, symbol)

            # Получаем открытые ордера на бирже
            orders = await fetch_open_orders(
                exchange_id=exchange_id, symbol=symbol, params=kwargs.get("params", {})
            )

            # Логируем информацию об открытых ордерах
            logger.debug(
                "Получено %d открытых ордеров на %s%s",
                len(orders), exchange_id, f" для {symbol}" if symbol else ""
            )

            return orders

        except Exception as e:
            error_msg = (
                f"Ошибка при получении открытых ордеров на {exchange_id}: {str(e)}"
            )
            logger.error(error_msg)
            return []

    @async_handle_error
    async def market_buy(
        self, symbol: str, amount: float, exchange_id: str = "binance", **kwargs
    ) -> OrderResult:
        """
        Выполняет рыночный ордер на покупку.

        Args:
            symbol: Символ торговой пары
            amount: Количество
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры

        Returns:
            Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side="buy",
            amount=amount,
            order_type="market",
            exchange_id=exchange_id,
            **kwargs,
        )

    @async_handle_error
    async def market_sell(
        self, symbol: str, amount: float, exchange_id: str = "binance", **kwargs
    ) -> OrderResult:
        """
        Выполняет рыночный ордер на продажу.

        Args:
            symbol: Символ торговой пары
            amount: Количество
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры

        Returns:
            Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side="sell",
            amount=amount,
            order_type="market",
            exchange_id=exchange_id,
            **kwargs,
        )

    @async_handle_error
    async def limit_buy(
        self,
        symbol: str,
        amount: float,
        price: float,
        exchange_id: str = "binance",
        **kwargs,
    ) -> OrderResult:
        """
        Выполняет лимитный ордер на покупку.

        Args:
            symbol: Символ торговой пары
            amount: Количество
            price: Цена
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры

        Returns:
            Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side="buy",
            amount=amount,
            order_type="limit",
            price=price,
            exchange_id=exchange_id,
            **kwargs,
        )

    @async_handle_error
    async def limit_sell(
        self,
        symbol: str,
        amount: float,
        price: float,
        exchange_id: str = "binance",
        **kwargs,
    ) -> OrderResult:
        """
        Выполняет лимитный ордер на продажу.

        Args:
            symbol: Символ торговой пары
            amount: Количество
            price: Цена
            exchange_id: Идентификатор биржи
            **kwargs: Дополнительные параметры

        Returns:
            Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side="sell",
            amount=amount,
            order_type="limit",
            price=price,
            exchange_id=exchange_id,
            **kwargs,
        )

    async def _normalize_symbol(self, exchange_id: str, symbol: str) -> str:
        """Нормализует символ торговой пары для указанной биржи."""
        return await self.symbol_manager.normalize_symbol(symbol, exchange_id)

    async def _normalize_amount(
        self, exchange_id: str, symbol: str, amount: float
    ) -> float:
        """
        Нормализует количество для указанной биржи и символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары
            amount: Количество

        Returns:
            Нормализованное количество
        """
        return await self.symbol_manager.normalize_amount(exchange_id, symbol, amount)

    async def _normalize_price(
        self, exchange_id: str, symbol: str, price: float
    ) -> float:
        """
        Нормализует цену для указанной биржи и символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары
            price: Цена

        Returns:
            Нормализованная цена
        """
        return await self.symbol_manager.normalize_price(exchange_id, symbol, price)

    async def _save_order_to_db(self, exchange_id: str, order: Dict[str, Any]) -> None:
        """
        Сохраняет информацию об ордере в базу данных.

        Args:
            exchange_id: Идентификатор биржи
            order: Данные ордера
        """
        try:
            from project.data.database import Database
            
            if self._db is None:
                self._db = Database.get_instance()
            
            # Сохраняем информацию об ордере в базе данных
            logger.debug("Сохраняем ордер в базе данных: %s", order.get("id", "unknown"))
            
            # Здесь код для сохранения ордера в БД
            
        except Exception as e:
            logger.error("Ошибка при сохранении ордера в БД: %s", str(e))

    async def _update_order_in_db(
        self, exchange_id: str, order: Dict[str, Any]
    ) -> None:
        """
        Обновляет информацию об ордере в базе данных.

        Args:
            exchange_id: Идентификатор биржи
            order: Данные ордера
        """
        await self._save_order_to_db(exchange_id, order)
