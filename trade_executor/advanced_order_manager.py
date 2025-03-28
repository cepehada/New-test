import asyncio
import json
import logging
import math
import random
import time
import uuid
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from project.data.database import Database
from project.data.market_data import MarketDataProvider
from project.exchange.exchange_manager import ExchangeManager
from project.risk_management.position_sizer import PositionSizer
from project.trade_executor.order_executor import OrderExecutor
from project.trade_executor.order_tracker import OrderTracker
from project.utils.ccxt_exchanges import get_exchange_instance
from project.utils.error_handler import ExchangeErrorHandler, handle_exchange_errors
from project.utils.logging_utils import setup_logger
from project.utils.notify import send_notification
from project.trade_executor.strategy_manager import StrategyManager
from project.trade_executor.capital_manager import CapitalManager

logger = setup_logger("advanced_order_manager")


class OrderType(Enum):
    """Типы ордеров"""

    MARKET = "market"  # Рыночный ордер
    LIMIT = "limit"  # Лимитный ордер
    STOP_LOSS = "stop_loss"  # Стоп-лосс ордер
    TAKE_PROFIT = "take_profit"  # Тейк-профит ордер
    STOP_LIMIT = "stop_limit"  # Стоп-лимит ордер
    TRAILING_STOP = "trailing_stop"  # Трейлинг-стоп ордер
    OCO = "oco"  # Ордер OCO (One-Cancels-the-Other)
    ICEBERG = "iceberg"  # Ордер айсберг (частичное исполнение)
    TWAP = "twap"  # Ордер TWAP (Time-Weighted Average Price)
    VWAP = "vwap"  # Ордер VWAP (Volume-Weighted Average Price)
    BRACKET = "bracket"  # Брекет-ордер (вход + стоп-лосс + тейк-профит)
    SCALED = "scaled"  # Масштабируемый ордер (несколько ордеров по разным ценам)
    FILL_OR_KILL = "fok"  # Fill-or-Kill ордер
    IMMEDIATE_OR_CANCEL = "ioc"  # Immediate-or-Cancel ордер
    POST_ONLY = "post_only"  # Только для мейкера
    HEDGE = "hedge"  # Хеджирующий ордер
    CONDITIONAL = "conditional"  # Условный ордер


class OrderStatus(Enum):
    """Статусы ордеров"""

    PENDING = "pending"  # Ордер создан, но еще не отправлен на биржу
    OPEN = "open"  # Открытый ордер
    PARTIALLY_FILLED = "partially_filled"  # Частично исполненный ордер
    FILLED = "filled"  # Полностью исполненный ордер
    CANCELED = "canceled"  # Отмененный ордер
    EXPIRED = "expired"  # Истекший ордер
    REJECTED = "rejected"  # Отклоненный ордер
    ERROR = "error"  # Ошибка при создании или исполнении ордера
    WAITING = "waiting"  # Ожидание других условий (для условных ордеров)


class HedgeType(Enum):
    """Типы хеджирования"""

    DIRECT = "direct"  # Прямое хеджирование на той же бирже
    CROSS_EXCHANGE = "cross_exchange"  # Хеджирование на другой бирже


class OrderSide(Enum):
    """Стороны ордера"""

    BUY = "buy"  # Покупка
    SELL = "sell"  # Продажа


class TimeInForce(Enum):
    """Время действия ордера"""

    GTC = "gtc"  # Good Till Cancelled - до отмены
    IOC = "ioc"  # Immediate Or Cancel - исполнить немедленно или отменить
    FOK = "fok"  # Fill Or Kill - исполнить полностью или отменить
    GTD = "gtd"  # Good Till Date - до определенной даты
    DAY = "day"  # До конца дня


class AdvancedOrder:
    """Класс для представления расширенного ордера"""

    def __init__(
        self,
        order_id: str = None,
        symbol: str = None,
        exchange_id: str = None,
        order_type: OrderType = None,
        side: OrderSide = None,
        amount: float = None,
        price: float = None,
        stop_price: float = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        post_only: bool = False,
        reduce_only: bool = False,
        status: OrderStatus = OrderStatus.PENDING,
        created_at: datetime = None,
        updated_at: datetime = None,
        executed_amount: float = 0.0,
        average_price: float = 0.0,
        leverage: float = 1.0,
        parent_id: str = None,
        related_orders: List[str] = None,
        params: Dict[str, Any] = None,
        client_order_id: str = None,
        exchange_order_id: str = None,
        strategy_id: str = None,
        error: str = None,
        hedge_params: Dict[str, Any] = None,
    ):
        self.order_id = order_id or str(uuid.uuid4())
        self.symbol = symbol
        self.exchange_id = exchange_id
        self.order_type = order_type
        self.side = side
        self.amount = amount
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.post_only = post_only
        self.reduce_only = reduce_only
        self.status = status
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.executed_amount = executed_amount
        self.average_price = average_price
        self.leverage = leverage
        self.parent_id = parent_id
        self.related_orders = related_orders or []
        self.params = params or {}
        self.client_order_id = client_order_id or f"ao_{self.order_id[:8]}"
        self.exchange_order_id = exchange_order_id
        self.strategy_id = strategy_id
        self.error = error
        self.hedge_params = hedge_params or {}

        # Дополнительные атрибуты для сложных ордеров
        self.sub_orders: List[AdvancedOrder] = []
        self.fills: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.execution_attempts = 0
        self.is_hedged = False
        self.hedge_orders: List[AdvancedOrder] = []

    def update_status(self, status: OrderStatus, exchange_data: Dict[str, Any] = None):
        """Обновляет статус ордера и дополнительные данные от биржи"""
        self.status = status
        self.updated_at = datetime.now()

        if exchange_data:
            # Обновляем информацию об исполнении
            if "filled" in exchange_data:
                self.executed_amount = float(exchange_data["filled"])

            if "average" in exchange_data and exchange_data["average"]:
                self.average_price = float(exchange_data["average"])
            elif (
                "price" in exchange_data
                and exchange_data["price"]
                and self.executed_amount > 0
            ):
                self.average_price = float(exchange_data["price"])

            # Обновляем ID ордера на бирже
            if "id" in exchange_data and exchange_data["id"]:
                self.exchange_order_id = exchange_data["id"]

            # Сохраняем информацию о сделках
            if "trades" in exchange_data and exchange_data["trades"]:
                for trade in exchange_data["trades"]:
                    self.fills.append(trade)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует ордер в словарь для сериализации"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "exchange_id": self.exchange_id,
            "order_type": self.order_type.value if self.order_type else None,
            "side": self.side.value if self.side else None,
            "amount": self.amount,
            "price": self.price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value if self.time_in_force else None,
            "post_only": self.post_only,
            "reduce_only": self.reduce_only,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "executed_amount": self.executed_amount,
            "average_price": self.average_price,
            "leverage": self.leverage,
            "parent_id": self.parent_id,
            "related_orders": self.related_orders,
            "params": self.params,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "strategy_id": self.strategy_id,
            "error": self.error,
            "sub_orders": [order.to_dict() for order in self.sub_orders],
            "fills": self.fills,
            "metadata": self.metadata,
            "execution_attempts": self.execution_attempts,
            "is_hedged": self.is_hedged,
            "hedge_orders": [order.to_dict() for order in self.hedge_orders],
            "hedge_params": self.hedge_params,
        }

    def is_active(self) -> bool:
        """Проверяет, является ли ордер активным"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.WAITING,
        ]

    def calc_fill_ratio(self) -> float:
        """Рассчитывает процент исполнения ордера"""
        if not self.amount or self.amount == 0:
            return 0.0
        return self.executed_amount / self.amount

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdvancedOrder":
        """Создает объект ордера из словаря"""
        order = cls(
            order_id=data.get("order_id"),
            symbol=data.get("symbol"),
            exchange_id=data.get("exchange_id"),
            order_type=(
                OrderType(data.get("order_type")) if data.get("order_type") else None
            ),
            side=OrderSide(data.get("side")) if data.get("side") else None,
            amount=data.get("amount"),
            price=data.get("price"),
            stop_price=data.get("stop_price"),
            time_in_force=(
                TimeInForce(data.get("time_in_force"))
                if data.get("time_in_force")
                else TimeInForce.GTC
            ),
            post_only=data.get("post_only", False),
            reduce_only=data.get("reduce_only", False),
            status=(
                OrderStatus(data.get("status"))
                if data.get("status")
                else OrderStatus.PENDING
            ),
            created_at=(
                datetime.fromisoformat(data.get("created_at"))
                if data.get("created_at")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data.get("updated_at"))
                if data.get("updated_at")
                else None
            ),
            executed_amount=data.get("executed_amount", 0.0),
            average_price=data.get("average_price", 0.0),
            leverage=data.get("leverage", 1.0),
            parent_id=data.get("parent_id"),
            related_orders=data.get("related_orders", []),
            params=data.get("params", {}),
            client_order_id=data.get("client_order_id"),
            exchange_order_id=data.get("exchange_order_id"),
            strategy_id=data.get("strategy_id"),
            error=data.get("error"),
            hedge_params=data.get("hedge_params", {}),
        )

        # Восстанавливаем дополнительные данные
        order.fills = data.get("fills", [])
        order.metadata = data.get("metadata", {})
        order.execution_attempts = data.get("execution_attempts", 0)
        order.is_hedged = data.get("is_hedged", False)

        # Восстанавливаем суб-ордера
        if "sub_orders" in data:
            order.sub_orders = [
                AdvancedOrder.from_dict(sub_order) for sub_order in data["sub_orders"]
            ]

        # Восстанавливаем хедж-ордера
        if "hedge_orders" in data:
            order.hedge_orders = [
                AdvancedOrder.from_dict(hedge_order)
                for hedge_order in data["hedge_orders"]
            ]

        return order


class OrderCreator:
    """Класс для создания ордеров на бирже"""

    def __init__(self, exchange_manager, market_data_provider, db=None):
        self.exchange_manager = exchange_manager
        self.market_data_provider = market_data_provider
        self.db = db
        self.logger = setup_logger("order_creator")

    async def create_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        amount: float,
        price: Optional[float],
        exchange_id: str,
        params: Dict[str, Any] = None,
    ) -> AdvancedOrder:
        """
        Создает ордер на бирже

        Args:
            symbol: Торговая пара
            order_type: Тип ордера
            side: Сторона ордера
            amount: Количество
            price: Цена (для лимитных ордеров)
            exchange_id: ID биржи
            params: Дополнительные параметры

        Returns:
            AdvancedOrder: Созданный ордер
        """
        params = params or {}

        # Создаем объект ордера
        order = AdvancedOrder(
            symbol=symbol,
            exchange_id=exchange_id,
            order_type=order_type,
            side=side,
            amount=amount,
            price=price,
            params=params,
        )

        # Адаптируем цену для лимитного ордера
        if price and order_type == OrderType.LIMIT:
            order.price = await self.adapt_limit_price(symbol, exchange_id, side, price)

        try:
            # Получаем экземпляр биржи
            exchange = await get_exchange_instance(exchange_id)

            # Подготовка параметров для CCXT
            ccxt_order_type = self._map_to_ccxt_order_type(order_type)
            ccxt_side = order.side.value

            # Объединяем с параметрами из запроса
            extra_params = {"clientOrderId": order.client_order_id}
            extra_params.update(params)

            with ExchangeErrorHandler():
                # Создаем ордер на бирже
                if order_type == OrderType.MARKET:
                    response = await exchange.create_market_order(
                        symbol, ccxt_side, amount, extra_params
                    )
                elif order_type == OrderType.LIMIT:
                    response = await exchange.create_limit_order(
                        symbol, ccxt_side, amount, order.price, extra_params
                    )
                else:
                    # Другие типы ордеров
                    response = await self._create_special_order_type(
                        exchange,
                        order_type,
                        symbol,
                        ccxt_side,
                        amount,
                        order.price,
                        order.stop_price,
                        extra_params,
                    )

                # Обновляем статус и данные ордера
                order.update_status(
                    (
                        OrderStatus.FILLED
                        if response.get("status") == "closed"
                        else OrderStatus.OPEN
                    ),
                    response,
                )

            # Закрываем соединение с биржей
            await exchange.close()

            self.logger.info(
                f"Создан ордер {order.order_id} ({order.order_type.value}) для {order.symbol}"
            )
            return order

        except Exception as e:
            # Обрабатываем ошибку создания ордера
            error_msg = f"Ошибка при создании ордера: {str(e)}"
            self.logger.error(error_msg)

            # Отмечаем ордер как ошибочный
            order.status = OrderStatus.ERROR
            order.error = error_msg
            order.updated_at = datetime.now()

            raise

    def _map_to_ccxt_order_type(self, order_type: OrderType) -> str:
        """Преобразует наш тип ордера в тип CCXT"""
        type_map = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "stop",
            OrderType.TAKE_PROFIT: "take_profit",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TRAILING_STOP: "trailing_stop",
            OrderType.FILL_OR_KILL: "fok",
            OrderType.IMMEDIATE_OR_CANCEL: "ioc",
            OrderType.POST_ONLY: "limit",  # Post-only это лимитный ордер с доп. параметром
        }
        return type_map.get(order_type, "limit")

    async def _create_special_order_type(
        self, exchange, order_type, symbol, side, amount, price, stop_price, params
    ):
        """Создает специальные типы ордеров (стоп, тейк-профит и т.д.)"""
        if order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
            if not exchange.has.get("createStopOrder", False):
                raise ValueError(f"Биржа {exchange.id} не поддерживает стоп-ордера")

            stop_type = "stop" if order_type == OrderType.STOP_LOSS else "stop_limit"
            params["stopPrice"] = stop_price or price

            return await exchange.create_order(
                symbol, stop_type, side, amount, price, params
            )

        elif order_type == OrderType.TAKE_PROFIT:
            if not exchange.has.get("createStopOrder", False):
                raise ValueError(
                    f"Биржа {exchange.id} не поддерживает тейк-профит ордера"
                )

            params["stopPrice"] = stop_price or price

            return await exchange.create_order(
                symbol, "take_profit", side, amount, price, params
            )

        elif order_type == OrderType.TRAILING_STOP:
            if not exchange.has.get("createTrailingStopOrder", False):
                raise ValueError(
                    f"Биржа {exchange.id} не поддерживает трейлинг-стоп ордера"
                )

            callback_rate = params.get("callbackRate", 1.0)

            return await exchange.create_order(
                symbol,
                "trailing_stop",
                side,
                amount,
                price,
                {"callbackRate": callback_rate, **params},
            )

        else:
            raise ValueError(f"Неподдерживаемый тип ордера: {order_type}")

    async def adapt_limit_price(
        self, symbol: str, exchange_id: str, side: OrderSide, base_price: float
    ) -> float:
        """Адаптирует цену лимитного ордера в зависимости от текущего состояния рынка"""
        try:
            # Получаем текущую книгу ордеров
            order_book = await self.market_data_provider.get_order_book(
                symbol, exchange_id, limit=5
            )

            if not order_book or "bids" not in order_book or "asks" not in order_book:
                return base_price

            # Получаем лучшие цены
            best_bid = order_book["bids"][0][0] if order_book["bids"] else None
            best_ask = order_book["asks"][0][0] if order_book["asks"] else None

            if not best_bid or not best_ask:
                return base_price

            # Текущий спред
            spread = best_ask - best_bid

            # Подстраиваем цену в зависимости от стороны
            if side == OrderSide.BUY:
                # Для покупки устанавливаем цену чуть выше бида, но ниже аска
                if base_price < best_bid or base_price > best_ask:
                    # Если указанная цена вне текущего спреда, используем цену внутри спреда
                    adjusted_price = best_bid + spread * 0.2  # 20% от спреда выше бида
                else:
                    # Если цена в спреде, оставляем как есть
                    adjusted_price = base_price
            else:  # SELL
                # Для продажи устанавливаем цену чуть ниже аска, но выше бида
                if base_price < best_bid or base_price > best_ask:
                    # Если указанная цена вне текущего спреда, используем цену внутри спреда
                    adjusted_price = best_ask - spread * 0.2  # 20% от спреда ниже аска
                else:
                    # Если цена в спреде, оставляем как есть
                    adjusted_price = base_price

            return adjusted_price
        except Exception as e:
            self.logger.warning(
                f"Ошибка при адаптации цены для {symbol}: {str(e)}. Используется исходная цена."
            )
            return base_price
