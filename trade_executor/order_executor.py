"""
Модуль для выполнения торговых ордеров на различных биржах.
Обеспечивает унифицированный интерфейс для размещения ордеров и контроля их статуса.
"""

# Стандартные импорты
from project.utils.notify import send_trading_signal
from project.infrastructure.database import Database
from project.data.symbol_manager import SymbolManager
from ..config import get_config
from typing import Any, Dict, List, Optional
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Внутренние импорты
from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.ccxt_exchanges import (
    get_exchange,
    fetch_balance,
    create_order,
    cancel_order,
    fetch_order,
    fetch_ticker,  # Added fetch_ticker import
    fetch_open_orders,  # Added fetch_open_orders import
)

logger = get_logger(__name__)

# Константы для типов ордеров и сторон
ORDER_TYPE_MARKET = "market"
ORDER_TYPE_LIMIT = "limit"
SIDE_BUY = "buy"
SIDE_SELL = "sell"


@dataclass
class OrderResult:
    """Класс для представления результата выполнения ордера"""

    success: bool
    order_id: str = ""
    status: str = ""
    filled: float = 0.0
    cost: float = 0.0
    price: float = 0.0
    error: str = ""
    raw_data: Dict = None


class OrderExecutor:
    """
    Класс для выполнения торговых ордеров на различных биржах.
    """

    _instance = None

    def __init__(self, config=None):
        """
        Инициализирует исполнитель заказов.

        Args:
            config: Конфигурация исполнителя
        """
        self.config = config or get_config()
        self.default_order_type = self.config.get(
            "DEFAULT_ORDER_TYPE", ORDER_TYPE_MARKET
        )
        self.default_exchange = self.config.get("DEFAULT_EXCHANGE", "binance")
        self.default_timeout = self.config.get(
            "DEFAULT_ORDER_TIMEOUT", 60
        )  # в секундах
        self.default_retry_attempts = self.config.get("DEFAULT_RETRY_ATTEMPTS", 3)
        self.default_retry_delay = self.config.get(
            "DEFAULT_RETRY_DELAY", 1
        )  # в секундах
        self.enable_logging = self.config.get("ENABLE_ORDER_LOGGING", True)
        self.paper_trading = self.config.get("ENABLE_PAPER_TRADING", True)

        logger.info(
            "Инициализирован исполнитель ордеров. Бумажная торговля: %s",
            "Да" if self.paper_trading else "Нет",
        )

    @classmethod
    def get_instance(cls, config=None):
        """
        Получает экземпляр исполнителя ордеров (singleton).

        Args:
            config: Конфигурация для нового экземпляра

        Returns:
            Экземпляр OrderExecutor
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @async_handle_error
    async def execute_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        order_type: str = None,
        price: float = None,
        exchange_id: str = None,
        params: Dict = None,
    ) -> OrderResult:
        """
        Выполняет торговый ордер.

        Args:
            symbol: Торговая пара
            side: Сторона ордера (buy/sell)
            amount: Объем
            order_type: Тип ордера (market/limit)
            price: Цена (для лимитных ордеров)
            exchange_id: Идентификатор биржи
            params: Дополнительные параметры

        Returns:
            OrderResult: Результат выполнения ордера
        """
        order_type = order_type or self.default_order_type
        exchange_id = exchange_id or self.default_exchange
        params = params or {}

        try:
            # Валидация параметров
            if not symbol or not side or amount <= 0:
                logger.warning(
                    "Неверные параметры ордера: symbol=%s, side=%s, amount=%f",
                    symbol,
                    side,
                    amount,
                )
                return OrderResult(success=False, error="Неверные параметры ордера")

            if order_type == ORDER_TYPE_LIMIT and not price:
                logger.warning(
                    "Не указана цена для лимитного ордера: %s %s %f",
                    symbol,
                    side,
                    amount,
                )
                return OrderResult(
                    success=False, error="Не указана цена для лимитного ордера"
                )

            # Логирование начала выполнения ордера
            logger.info(
                "Выполнение ордера: %s %s %s %.8f @ %s",
                exchange_id,
                symbol,
                side,
                amount,
                f"{price:.8f}" if price else "рыночная цена",
            )

            # Для бумажной торговли используем эмуляцию
            if self.paper_trading:
                return await self._execute_paper_order(
                    symbol, side, amount, order_type, price, exchange_id
                )

            # Выполняем реальный ордер
            exchange = await get_exchange(exchange_id)
            if not exchange:
                return OrderResult(
                    success=False,
                    error=f"Не удалось подключиться к бирже {exchange_id}",
                )

            # Создаем ордер
            order = await create_order(
                exchange_id=exchange_id,
                symbol=symbol,
                type_order=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params,
            )

            if not order:
                return OrderResult(
                    success=False, error=f"Не удалось создать ордер на {exchange_id}"
                )

            # Получаем сведения о выполненном ордере
            order_id = order.get("id", "")
            status = order.get("status", "unknown")
            filled = order.get("filled", 0)
            cost = order.get("cost", 0)
            actual_price = order.get("price", price or 0)

            # Логирование результата
            logger.info(
                "Ордер выполнен: %s %s %s, ID: %s, Статус: %s, Цена: %.8f",
                exchange_id,
                symbol,
                side,
                order_id,
                status,
                actual_price,
            )

            # Возвращаем результат
            return OrderResult(
                success=True,
                order_id=order_id,
                status=status,
                filled=filled,
                cost=cost,
                price=actual_price,
                raw_data=order,
            )

        except Exception as e:
            logger.error("Ошибка при выполнении ордера: %s", str(e))
            return OrderResult(
                success=False, error=f"Ошибка при выполнении ордера: {str(e)}"
            )

    @async_handle_error
    async def _execute_paper_order(
        self, symbol, side, amount, order_type, price, exchange_id
    ) -> OrderResult:
        """
        Эмуляция выполнения ордера в режиме бумажной торговли.

        Args:
            symbol: Торговая пара
            side: Сторона ордера
            amount: Объем
            order_type: Тип ордера
            price: Цена
            exchange_id: Биржа

        Returns:
            OrderResult: Результат эмуляции
        """
        try:
            # Для эмуляции получаем текущие рыночные данные
            from project.data.market_data import MarketData

            market_data = MarketData.get_instance()

            # Получаем тикер для определения цены
            ticker = await market_data.get_ticker(exchange_id, symbol)
            if not ticker:
                return OrderResult(
                    success=False,
                    error=f"Не удалось получить рыночные данные для {symbol}",
                )

            # Определяем цену исполнения
            execution_price = price
            if order_type == ORDER_TYPE_MARKET or not execution_price:
                if side == SIDE_BUY:
                    execution_price = ticker.get("ask", ticker.get("last", 0))
                else:  # SELL
                    execution_price = ticker.get("bid", ticker.get("last", 0))

            # Рассчитываем стоимость
            cost = amount * execution_price

            # Генерируем фиктивный ID ордера
            order_id = f"paper_{uuid.uuid4().hex[:16]}"

            # Логирование результата эмуляции
            logger.info(
                "БУМАЖНАЯ ТОРГОВЛЯ: %s %s %s %.8f @ %.8f (ID: %s)",
                exchange_id,
                symbol,
                side,
                amount,
                execution_price,
                order_id,
            )

            # Сохраняем информацию о бумажной сделке в БД
            await self._save_paper_trade(
                exchange_id=exchange_id,
                symbol=symbol,
                side=side,
                amount=amount,
                price=execution_price,
                cost=cost,
                order_id=order_id,
            )

            # Возвращаем результат
            return OrderResult(
                success=True,
                order_id=order_id,
                status="FILLED",  # Всегда считаем исполненным
                filled=amount,
                cost=cost,
                price=execution_price,
                raw_data={
                    "id": order_id,
                    "symbol": symbol,
                    "side": side,
                    "type": order_type,
                    "amount": amount,
                    "price": execution_price,
                    "cost": cost,
                    "filled": amount,
                    "status": "FILLED",
                    "timestamp": int(time.time() * 1000),
                    "datetime": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "fee": {"cost": cost * 0.001, "currency": symbol.split("/")[1]},
                    "trades": [],
                },
            )

        except Exception as e:
            logger.error("Ошибка при эмуляции ордера: %s", str(e))
            return OrderResult(
                success=False, error=f"Ошибка при эмуляции ордера: {str(e)}"
            )

    @async_handle_error
    async def market_buy(
        self, symbol: str, amount: float, exchange_id: str = None, params: Dict = None
    ) -> OrderResult:
        """
        Выполняет рыночный ордер на покупку.

        Args:
            symbol: Торговая пара
            amount: Объем
            exchange_id: Биржа
            params: Дополнительные параметры

        Returns:
            OrderResult: Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side=SIDE_BUY,
            amount=amount,
            order_type=ORDER_TYPE_MARKET,
            exchange_id=exchange_id,
            params=params,
        )

    @async_handle_error
    async def market_sell(
        self, symbol: str, amount: float, exchange_id: str = None, params: Dict = None
    ) -> OrderResult:
        """
        Выполняет рыночный ордер на продажу.

        Args:
            symbol: Торговая пара
            amount: Объем
            exchange_id: Биржа
            params: Дополнительные параметры

        Returns:
            OrderResult: Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side=SIDE_SELL,
            amount=amount,
            order_type=ORDER_TYPE_MARKET,
            exchange_id=exchange_id,
            params=params,
        )

    @async_handle_error
    async def limit_buy(
        self,
        symbol: str,
        amount: float,
        price: float,
        exchange_id: str = None,
        params: Dict = None,
    ) -> OrderResult:
        """
        Выполняет лимитный ордер на покупку.

        Args:
            symbol: Торговая пара
            amount: Объем
            price: Цена
            exchange_id: Биржа
            params: Дополнительные параметры

        Returns:
            OrderResult: Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side=SIDE_BUY,
            amount=amount,
            order_type=ORDER_TYPE_LIMIT,
            price=price,
            exchange_id=exchange_id,
            params=params,
        )

    @async_handle_error
    async def limit_sell(
        self,
        symbol: str,
        amount: float,
        price: float,
        exchange_id: str = None,
        params: Dict = None,
    ) -> OrderResult:
        """
        Выполняет лимитный ордер на продажу.

        Args:
            symbol: Торговая пара
            amount: Объем
            price: Цена
            exchange_id: Биржа
            params: Дополнительные параметры

        Returns:
            OrderResult: Результат выполнения ордера
        """
        return await self.execute_order(
            symbol=symbol,
            side=SIDE_SELL,
            amount=amount,
            order_type=ORDER_TYPE_LIMIT,
            price=price,
            exchange_id=exchange_id,
            params=params,
        )

    @async_handle_error
    async def get_balance(self, exchange_id: str = None) -> Dict:
        """
        Получает баланс аккаунта.

        Args:
            exchange_id: Биржа

        Returns:
            Dict: Данные о балансе
        """
        exchange_id = exchange_id or self.default_exchange

        try:
            balance = await fetch_balance(exchange_id)
            return balance or {}
        except Exception as e:
            logger.error("Ошибка при получении баланса: %s", str(e))
            return {}

    @async_handle_error
    async def cancel_order(
        self, order_id: str, symbol: str, exchange_id: str = None
    ) -> OrderResult:
        """
        Отменяет открытый ордер.

        Args:
            order_id: Идентификатор ордера
            symbol: Торговая пара
            exchange_id: Биржа

        Returns:
            OrderResult: Результат отмены ордера
        """
        exchange_id = exchange_id or self.default_exchange

        try:
            # Для бумажной торговли эмулируем отмену
            if self.paper_trading and order_id.startswith("paper_"):
                logger.info(
                    "БУМАЖНАЯ ТОРГОВЛЯ: Отмена ордера %s для %s на %s",
                    order_id,
                    symbol,
                    exchange_id,
                )
                return OrderResult(success=True, order_id=order_id, status="canceled")

            # Отменяем реальный ордер
            result = await cancel_order(exchange_id, order_id, symbol)

            if not result:
                return OrderResult(
                    success=False, error=f"Не удалось отменить ордер {order_id}"
                )

            logger.info(
                "Ордер успешно отменен: %s для %s на %s", order_id, symbol, exchange_id
            )

            return OrderResult(
                success=True, order_id=order_id, status="canceled", raw_data=result
            )

        except Exception as e:
            logger.error("Ошибка при отмене ордера %s: %s", order_id, str(e))
            return OrderResult(
                success=False, error=f"Ошибка при отмене ордера: {str(e)}"
            )

    @async_handle_error
    async def get_order_status(
        self, order_id: str, symbol: str, exchange_id: str = None
    ) -> OrderResult:
        """
        Получает статус ордера.

        Args:
            order_id: Идентификатор ордера
            symbol: Торговая пара
            exchange_id: Биржа

        Returns:
            OrderResult: Данные о статусе ордера
        """
        exchange_id = exchange_id or self.default_exchange

        try:
            # Для бумажной торговли возвращаем фиктивный статус
            if self.paper_trading and order_id.startswith("paper_"):
                logger.debug(
                    "БУМАЖНАЯ ТОРГОВЛЯ: Проверка статуса ордера %s для %s",
                    order_id,
                    symbol,
                )

                # В реальном приложении здесь бы загружались данные о бумажном ордере из хранилища
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status="FILLED",
                    filled=1.0,
                    price=0.0,
                    cost=0.0,
                )

            # Получаем статус реального ордера
            order = await fetch_order(exchange_id, order_id, symbol)

            if not order:
                return OrderResult(
                    success=False, error=f"Не удалось получить статус ордера {order_id}"
                )

            # Извлекаем данные
            status = order.get("status", "unknown")
            filled = order.get("filled", 0)
            price = order.get("price", 0)
            cost = order.get("cost", 0)

            logger.debug(
                "Статус ордера %s: %s, исполнено: %.8f", order_id, status, filled
            )

            return OrderResult(
                success=True,
                order_id=order_id,
                status=status,
                filled=filled,
                price=price,
                cost=cost,
                raw_data=order,
            )

        except Exception as e:
            logger.error("Ошибка при получении статуса ордера %s: %s", order_id, str(e))
            return OrderResult(
                success=False, error=f"Ошибка при получении статуса ордера: {str(e)}"
            )

    @async_handle_error
    async def get_open_orders(
        self, exchange_id: str = None, symbol: str = None
    ) -> List[Dict]:
        """
        Получает список открытых ордеров.

        Args:
            exchange_id: Биржа
            symbol: Торговая пара (опционально)

        Returns:
            List[Dict]: Список открытых ордеров
        """
        exchange_id = exchange_id or self.default_exchange

        try:
            from project.utils.ccxt_exchanges import fetch_open_orders

            orders = await fetch_open_orders(exchange_id, symbol)
            return orders or []

        except Exception as e:
            logger.error("Ошибка при получении открытых ордеров: %s", str(e))
            return []

    @async_handle_error
    async def _save_paper_trade(
        self,
        exchange_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        cost: float,
        order_id: str,
    ) -> bool:
        """
        Сохраняет информацию о бумажной сделке в базе данных.

        Args:
            exchange_id: Биржа
            symbol: Торговая пара
            side: Сторона (buy/sell)
            amount: Объем
            price: Цена
            cost: Стоимость
            order_id: Идентификатор ордера

        Returns:
            bool: Успешность сохранения
        """
        try:
            # Импортируем базу данных
            from project.data.database import Database

            database = Database.get_instance()

            # Сохраняем данные о сделке
            await database.execute(
                """
                INSERT INTO paper_trades
                (exchange, symbol, side, amount, price, cost, order_id, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (exchange_id, symbol, side, amount, price, cost, order_id, time.time()),
            )

            logger.info(
                "Бумажная сделка сохранена: %s %s %s %.8f @ %.8f",
                exchange_id,
                symbol,
                side,
                amount,
                price,
            )
            return True
        except Exception as e:
            logger.error("Ошибка при сохранении бумажной сделки: %s", str(e))
            return False


"""
Модуль для исполнения торговых ордеров.
Предоставляет функции для создания и управления ордерами на биржах.
"""


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
                    "Для лимитного ордера не указана цена, используем текущую цену %s",
                    price,
                )

            if price is not None:
                price = await self._normalize_price(exchange_id, symbol, price)

            # Добавляем client_order_id к параметрам
            params = kwargs.get("params", {})
            params["clientOrderId"] = client_order_id

            # Логируем информацию о создаваемом ордере
            logger.info(
                "Создаем ордер на %s: %s %s %s %s%s",
                exchange_id,
                side,
                order_type,
                amount,
                symbol,
                f" по цене {price}" if price else "",
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
            error_msg = f"Ошибка при выполнении ордера {side} {order_type} {amount} {symbol}: {
                str(e)}"
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
                order_id,
                exchange_id,
                order.get("status", "unknown"),
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
            return OrderResult(success=False, order_id=order_id, error=str(e))

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
                len(orders),
                exchange_id,
                f" для {symbol}" if symbol else "",
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
            logger.debug(
                "Сохраняем ордер в базе данных: %s", order.get("id", "unknown")
            )

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
