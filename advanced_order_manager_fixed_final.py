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

logger = setup_logger("advanced_order_manager")


class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "market"            # Рыночный ордер
    LIMIT = "limit"              # Лимитный ордер
    STOP_LOSS = "stop_loss"      # Стоп-лосс ордер
    TAKE_PROFIT = "take_profit"  # Тейк-профит ордер
    STOP_LIMIT = "stop_limit"    # Стоп-лимит ордер
    TRAILING_STOP = "trailing_stop"  # Трейлинг-стоп ордер
    OCO = "oco"                  # Ордер OCO (One-Cancels-the-Other)
    ICEBERG = "iceberg"          # Ордер айсберг (частичное исполнение)
    TWAP = "twap"                # Ордер TWAP (Time-Weighted Average Price)
    VWAP = "vwap"                # Ордер VWAP (Volume-Weighted Average Price)
    BRACKET = "bracket"          # Брекет-ордер (вход + стоп-лосс + тейк-профит)
    SCALED = "scaled"            # Масштабируемый ордер (несколько ордеров по разным ценам)
    FILL_OR_KILL = "fok"         # Fill-or-Kill ордер
    IMMEDIATE_OR_CANCEL = "ioc"  # Immediate-or-Cancel ордер
    POST_ONLY = "post_only"      # Только для мейкера
    HEDGE = "hedge"              # Хеджирующий ордер
    CONDITIONAL = "conditional"  # Условный ордер


class OrderStatus(Enum):
    """Статусы ордеров"""
    PENDING = "pending"          # Ордер создан, но еще не отправлен на биржу
    OPEN = "open"                # Открытый ордер
    PARTIALLY_FILLED = "partially_filled"  # Частично исполненный ордер
    FILLED = "filled"            # Полностью исполненный ордер
    CANCELED = "canceled"        # Отмененный ордер
    EXPIRED = "expired"          # Истекший ордер
    REJECTED = "rejected"        # Отклоненный ордер
    ERROR = "error"              # Ошибка при создании или исполнении ордера
    WAITING = "waiting"          # Ожидание других условий (для условных ордеров)


class HedgeType(Enum):
    """Типы хеджирования"""
    DIRECT = "direct"            # Прямое хеджирование на той же бирже
    CROSS_EXCHANGE = "cross_exchange"  # Хеджирование на другой бирже

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedOrder':
        """Создает объект ордера из словаря"""
        order = cls(
            order_id=data.get('order_id'),
            symbol=data.get('symbol'),
            exchange_id=data.get('exchange_id'),
            order_type=OrderType(data.get('order_type')) if data.get('order_type') else None,
            side=OrderSide(data.get('side')) if data.get('side') else None,
            amount=data.get('amount'),
            price=data.get('price'),
            stop_price=data.get('stop_price'),
            time_in_force=TimeInForce(data.get('time_in_force')) if data.get('time_in_force') else TimeInForce.GTC,
            post_only=data.get('post_only', False),
            reduce_only=data.get('reduce_only', False),
            status=OrderStatus(data.get('status')) if data.get('status') else OrderStatus.PENDING,
            created_at=datetime.fromisoformat(data.get('created_at')) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data.get('updated_at')) if data.get('updated_at') else None,
            executed_amount=data.get('executed_amount', 0.0),
            average_price=data.get('average_price', 0.0),
            leverage=data.get('leverage', 1.0),
            parent_id=data.get('parent_id'),
            related_orders=data.get('related_orders', []),
            params=data.get('params', {}),
            client_order_id=data.get('client_order_id'),
            exchange_order_id=data.get('exchange_order_id'),
            strategy_id=data.get('strategy_id'),
            error=data.get('error'),
            hedge_params=data.get('hedge_params', {})
        )
        
        # Восстанавливаем дополнительные данные
        order.fills = data.get('fills', [])
        order.metadata = data.get('metadata', {})
        order.execution_attempts = data.get('execution_attempts', 0)
        order.is_hedged = data.get('is_hedged', False)
        
        # Восстанавливаем суб-ордера
        if 'sub_orders' in data:
            order.sub_orders = [AdvancedOrder.from_dict(sub_order) for sub_order in data['sub_orders']]
            
        # Восстанавливаем хедж-ордера
        if 'hedge_orders' in data:
            order.hedge_orders = [AdvancedOrder.from_dict(hedge_order) for hedge_order in data['hedge_orders']]
            
        return order
    
    def is_active(self) -> bool:
        """Проверяет, является ли ордер активным"""
        return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.WAITING]
        
    def calc_fill_ratio(self) -> float:
        """Рассчитывает процент исполнения ордера"""
        if not self.amount or self.amount == 0:
            return 0.0
        return self.executed_amount / self.amount

class AdvancedOrderManager:
    """
    Менеджер для создания and управления расширенными ордерами, включая хеджирование
    and сложные ордера на разных биржах
    """
    
    def __init__(self, config: Dict = None):
        """
        Инициализирует менеджер расширенных ордеров
        
        Args:
            config: Конфигурация менеджера
        """
        self.config = config or {}
        
        # Базовый исполнитель ордеров
        self.order_executor = OrderExecutor(self.config)
        
        # Трекер ордеров
        self.order_tracker = OrderTracker(self.config)
        
        # Провайдер рыночных данных
        self.market_data_provider = MarketDataProvider()
        
        # Менеджер бирж
        self.exchange_manager = ExchangeManager(self.config)
        
        # Расчет размера позиции
        self.position_sizer = PositionSizer()
        
        # База данных
        self.db = Database() if self.config.get("use_database", True) else None
        
        # Словарь для хранения активных ордеров
        self.active_orders: Dict[str, AdvancedOrder] = {}
        
        # Словарь для хранения истории ордеров
        self.order_history: Dict[str, AdvancedOrder] = {}
        
        # История хеджированных позиций
        self.hedge_history: Dict[str, Dict[str, Any]] = {}
        
        # Максимальное количество попыток исполнения ордера
        self.max_attempts = self.config.get("max_execution_attempts", 3)
        
        # Время ожидания между попытками (в секундах)
        self.retry_delay = self.config.get("retry_delay", 5)
        
        # Погрешность для сравнения цен in процентах
        self.price_tolerance = self.config.get("price_tolerance", 0.1)
        
        # Максимальное отклонение для хеджирования in процентах
        self.hedge_max_deviation = self.config.get("hedge_max_deviation", 0.5)
        
        # Максимальная задержка для хеджирования in секундах
        self.hedge_max_delay = self.config.get("hedge_max_delay", 5.0)
        
        # Список бирж для хеджирования по приоритету
        self.hedge_exchanges = self.config.get("hedge_exchanges", ["binance", "bybit", "kucoin"])
        
        # Флаг для включения/отключения хеджирования
        self.enable_hedging = self.config.get("enable_hedging", False)
        
        # Включение автоматической адаптации цены для лимитных ордеров
        self.enable_price_adaptation = self.config.get("enable_price_adaptation", True)
        
        # Автоматическое отслеживание and обновление ордеров
        self.enable_auto_tracking = self.config.get("enable_auto_tracking", True)
        
        # Процент скольжения для рыночных ордеров
        self.market_slippage = self.config.get("market_slippage", 0.1)
        
        # Флаг для остановки фоновых задач
        self._stop_requested = False
        
        # Задача для отслеживания ордеров
        self._tracking_task = None
        
        # Количество ордеров, обрабатываемых параллельно
        self.max_concurrent_orders = self.config.get("max_concurrent_orders", 10)
        
        # Семафор для ограничения параллельных операций
        self._order_semaphore = asyncio.Semaphore(self.max_concurrent_orders)
        
        # Функции обратного вызова для обработки ордеров
        self.on_order_created: Optional[Callable] = None
        self.on_order_filled: Optional[Callable] = None
        self.on_order_canceled: Optional[Callable] = None
        self.on_order_error: Optional[Callable] = None
        
        logger.info(f"Менеджер расширенных ордеров инициализирован")
    
    async def start(self):
        """Запускает фоновые задачи менеджера"""
        if self._tracking_task is not None:
            logger.warning("Менеджер расширенных ордеров уже запущен")
            return
            
        # Запускаем трекер ордеров
        await self.order_tracker.start()
        
        # Загружаем активные ордера из БД
        if self.db:
            await self._load_active_orders()
            
        # Запускаем фоновую задачу отслеживания
        if self.enable_auto_tracking:
            self._stop_requested = False
            self._tracking_task = asyncio.create_task(self._tracking_loop())
            logger.info("Задача отслеживания ордеров запущена")
            
        # Запускаем менеджер бирж
        await self.exchange_manager.start()
        
        logger.info("Менеджер расширенных ордеров запущен")
        
    async def stop(self):
        """Останавливает фоновые задачи менеджера"""
        if self._tracking_task is None:
            logger.warning("Менеджер расширенных ордеров не запущен")
            return
            
        self._stop_requested = True
        
        # Останавливаем трекер ордеров
        await self.order_tracker.stop()
        
        # Останавливаем задачу отслеживания
        if self._tracking_task:
            self._tracking_task.cancel()
            try:
                await self._tracking_task
            except asyncio.CancelledError:
                pass
            self._tracking_task = None
            
        # Останавливаем менеджер бирж
        await self.exchange_manager.stop()
        
        # Сохраняем активные ордера in БД
        if self.db:
            await self._save_active_orders()
            
        logger.info("Менеджер расширенных ордеров остановлен")
        
    async def _tracking_loop(self):
        """Фоновая задача для отслеживания статуса ордеров"""
        while not self._stop_requested:
            try:
                # Обновляем статус всех активных ордеров
                orders_to_check = list(self.active_orders.values())
                
                for order in orders_to_check:
                    if not order.is_active():
                        continue
                        
                    try:
                        # Проверяем верхнеуровневый ордер and все суб-ордера
                        await self._check_order_status(order)
                        
                        # Если ордер с хеджированием, проверяем and хедж-ордера
                        if order.is_hedged:
                            for hedge_order in order.hedge_orders:
                                await self._check_order_status(hedge_order)
                    except Exception as e:
                        logger.error("Ошибка при отслеживании ордера {order.order_id}: {str(e)}")
                        
                # Пауза между обновлениями
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                logger.info(f"Задача отслеживания ордеров отменена")
                break
            except Exception as e:
                logger.error("Ошибка in цикле отслеживания ордеров: {str(e)}")
                await asyncio.sleep(5)
                
    async def _check_order_status(self, order: AdvancedOrder):
        """
        Проверяет статус ордера на бирже and обновляет локальное состояние
        
        Args:
            order: Ордер для проверки
        """
        if not order.exchange_order_id or order.status not in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
            return
            
        try:
            # Получаем информацию об ордере с биржи
            exchange = await get_exchange_instance(order.exchange_id)
            
            with ExchangeErrorHandler():
                # Получаем ордер по ID
                exchange_order = await exchange.fetch_order(order.exchange_order_id, order.symbol)
                
                # Преобразуем статус биржи in наш формат
                if exchange_order:
                    ccxt_status = exchange_order['status']
                    our_status = self._map_exchange_status(ccxt_status)
                    
                    # Обновляем статус and данные ордера
                    if our_status != order.status:
                        order.update_status(our_status, exchange_order)
                        
                        # Выполняем соответствующие колбэки
                        if our_status == OrderStatus.FILLED and self.on_order_filled:
                            await self.on_order_filled(order)
                        elif our_status in [OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED] and self.on_order_canceled:
                            await self.on_order_canceled(order)
                            
                        # Сохраняем изменения in БД
                        if self.db:
                            await self.db.save_order(order.to_dict())
                            
                        # Обновляем списки активных/исторических ордеров
                        if not order.is_active():
                            if order.order_id in self.active_orders:
                                del self.active_orders[order.order_id]
                            self.order_history[order.order_id] = order
            
            # Закрываем соединение с биржей
            await exchange.close()
            
        except Exception as e:
            logger.error(f"Ошибка при проверке статуса ордера {order.order_id}: {str(e)}")
            
            # Если ошибка указывает на то, что ордер не найден, отмечаем как ошибку
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                order.update_status(OrderStatus.ERROR)
                order.error = f"Order not found on exchange: {str(e)}"
                
                # Сохраняем изменения in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
                # Обновляем списки активных/исторических ордеров
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]
                self.order_history[order.order_id] = order    
class OrderSide(Enum):
    """Стороны ордера"""
    BUY = "buy"                  # Покупка
    SELL = "sell"                # Продажа

class TimeInForce(Enum):
    """Время действия ордера"""
    GTC = "gtc"                  # Good Till Cancelled - до отмены
    IOC = "ioc"                  # Immediate Or Cancel - исполнить немедленно или отменить
    FOK = "fok"                  # Fill Or Kill - исполнить полностью или отменить
    GTD = "gtd"                  # Good Till Date - до определенной даты
    DAY = "day"                  # До конца дня

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
        hedge_params: Dict[str, Any] = None
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
        """Обновляет статус ордера and дополнительные данные от биржи"""
        self.status = status
        self.updated_at = datetime.now()
        
        if exchange_data:
            # Обновляем информацию об исполнении
            if 'filled' in exchange_data:
                self.executed_amount = float(exchange_data['filled'])
                
            if 'average' in exchange_data and exchange_data['average']:
                self.average_price = float(exchange_data['average'])
            elif 'price' in exchange_data and exchange_data['price'] and self.executed_amount > 0:
                self.average_price = float(exchange_data['price'])
                
            # Обновляем ID ордера на бирже
            if 'id' in exchange_data and exchange_data['id']:
                self.exchange_order_id = exchange_data['id']
                
            # Сохраняем информацию о сделках
            if 'trades' in exchange_data and exchange_data['trades']:
                for trade in exchange_data['trades']:
                    self.fills.append(trade)
                    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует ордер in словарь для сериализации"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'exchange_id': self.exchange_id,
            'order_type': self.order_type.value if self.order_type else None,
            'side': self.side.value if self.side else None,
            'amount': self.amount,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force.value if self.time_in_force else None,
            'post_only': self.post_only,
            'reduce_only': self.reduce_only,
            'status': self.status.value if self.status else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'executed_amount': self.executed_amount,
            'average_price': self.average_price,
            'leverage': self.leverage,
            'parent_id': self.parent_id,
            'related_orders': self.related_orders,
            'params': self.params,
            'client_order_id': self.client_order_id,
            'exchange_order_id': self.exchange_order_id,
            'strategy_id': self.strategy_id,
            'error': self.error,
            'sub_orders': [order.to_dict() for order in self.sub_orders],
            'fills': self.fills,
            'metadata': self.metadata,
            'execution_attempts': self.execution_attempts,
            'is_hedged': self.is_hedged,
            'hedge_orders': [order.to_dict() for order in self.hedge_orders],
            'hedge_params': self.hedge_params
        }
def _map_exchange_status(self, exchange_status: str) -> OrderStatus:
        """
        Преобразует статус ордера с биржи in наш формат
        
        Args:
            exchange_status: Статус ордера с биржи
            
        Returns:
            OrderStatus: Наш статус ордера
        """
        # Сопоставление статусов CCXT с нашими
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED,
            'partially_filled': OrderStatus.PARTIALLY_FILLED
        }
        
        return status_map.get(exchange_status.lower(), OrderStatus.OPEN)
    
        self, 
    async def create_order(
        order_type: Union[OrderType, str],
        side: Union[OrderSide, str],
        amount: float,
        price: float = None,
        exchange_id: str = None,
        params: Dict[str, Any] = None,
        enable_hedging: bool = None
    ) -> AdvancedOrder:
        """
        Создает and отправляет ордер на биржу
        
        Args:
            symbol: Торговая пара
            order_type: Тип ордера
            side: Сторона ордера (покупка/продажа)
            amount: Количество
            price: Цена (для лимитных ордеров)
            exchange_id: ID биржи
            params: Дополнительные параметры ордера
            enable_hedging: Включить хеджирование для этого ордера
            
        Returns:
            AdvancedOrder: Созданный ордер
        """
        params = params or {}
        
        # Конвертируем строковые типы in перечисления
        if isinstance(order_type, str):
            order_type = OrderType(order_type)
        if isinstance(side, str):
            side = OrderSide(side)
            
        # Определяем биржу, если не указана
        if not exchange_id:
            # Автоматически выбираем биржу
            exchange_id, _ = await self.exchange_manager.select_best_exchange(
                symbol,
                required_features=["createOrder", "fetchOrder"]
            )
            
            if not exchange_id:
                raise ValueError(f"Не удалось найти подходящую биржу для {symbol}")
                
        # Адаптируем цену для лимитного ордера, если включено
        adjusted_price = price
        if self.enable_price_adaptation and price and order_type == OrderType.LIMIT:
            adjusted_price = await self._adapt_limit_price(symbol, exchange_id, side, price)
            
        # Создаем объект ордера
        order = AdvancedOrder(
            symbol=symbol,
            exchange_id=exchange_id,
            order_type=order_type,
            side=side,
            amount=amount,
            price=adjusted_price,
            params=params,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Обрабатываем специальные типы ордеров
        if order_type in [OrderType.BRACKET, OrderType.OCO, OrderType.SCALED, OrderType.TWAP, OrderType.VWAP]:
            return await self._create_complex_order(order)
            
        # Отправляем ордер на биржу
        async with self._order_semaphore:
            try:
                # Получаем экземпляр биржи
                exchange = await get_exchange_instance(exchange_id)
                
                # Подготовка параметров для CCXT
                ccxt_order_type = self._map_to_ccxt_order_type(order_type)
                ccxt_side = order.side.value
                
                # Добавляем дополнительные параметры
                extra_params = {}
                
                # Добавляем клиентский ID ордера
                if 'clientOrderId' not in params:
                    extra_params['clientOrderId'] = order.client_order_id
                    
                # Объединяем с параметрами из запроса
                extra_params.update(params)
                
                with ExchangeErrorHandler():
                    # Выполняем рыночный ордер
                    if order_type == OrderType.MARKET:
                        response = await exchange.create_order(
                            symbol,
                            'market',
                            ccxt_side,
                            amount,
                            None,
                            extra_params
                        )
                    # Выполняем лимитный ордер
                    elif order_type == OrderType.LIMIT:
                        response = await exchange.create_order(
                            symbol,
                            'limit',
                            ccxt_side,
                            amount,
                            adjusted_price,
                            extra_params
                        )
                    # Выполняем стоп-ордер
                    elif order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
                        # Проверяем поддержку стоп-ордеров
                        if not exchange.has.get('createStopOrder', False):
                            raise ValueError(f"Биржа {exchange_id} не поддерживает стоп-ордера")
                            
                        stop_type = 'stop' if order_type == OrderType.STOP_LOSS else 'stop_limit'
                        stop_price = params.get('stopPrice', price)
                        
                        response = await exchange.create_order(
                            symbol,
                            stop_type,
                            ccxt_side,
                            amount,
                            adjusted_price,
                            {'stopPrice': stop_price, **extra_params}
                        )
                    # Выполняем тейк-профит ордер
                    elif order_type == OrderType.TAKE_PROFIT:
                        # Проверяем поддержку тейк-профит ордеров
                        if not exchange.has.get('createStopOrder', False):
                            raise ValueError(f"Биржа {exchange_id} не поддерживает тейк-профит ордера")
                            
                        take_profit_price = params.get('takeProfitPrice', price)
                        
                        response = await exchange.create_order(
                            symbol,
                            'take_profit',
                            ccxt_side,
                            amount,
                            adjusted_price,
                            {'stopPrice': take_profit_price, **extra_params}
                        )
                    # Трейлинг-стоп ордер
                    elif order_type == OrderType.TRAILING_STOP:
                        # Проверяем поддержку трейлинг-стоп ордеров
                        if not exchange.has.get('createTrailingStopOrder', False):
                            raise ValueError(f"Биржа {exchange_id} не поддерживает трейлинг-стоп ордера")
                            
                        callback_rate = params.get('callbackRate', 1.0)  # Значение in процентах
                        
                        response = await exchange.create_order(
                            symbol,
                            'trailing_stop',
                            ccxt_side,
                            amount,
                            adjusted_price,
                            {'callbackRate': callback_rate, **extra_params}
                        )
                    else:
                        raise ValueError(f"Неподдерживаемый тип ордера: {order_type}")
                        
                    # Обновляем статус and данные ордера
                    order.update_status(
                        OrderStatus.FILLED if response.get('status') == 'closed' else OrderStatus.OPEN,
                        response
                    )
                    
                    # Если ордер сразу исполнился
                    if order.status == OrderStatus.FILLED and self.on_order_filled:
                        await self.on_order_filled(order)
                
                # Закрываем соединение с биржей
                await exchange.close()
                
                # Сохраняем ордер in словаре активных ордеров
                if order.is_active():
                    self.active_orders[order.order_id] = order
                else:
                    self.order_history[order.order_id] = order
                    
                # Сохраняем ордер in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
                # Вызываем колбэк создания ордера
                if self.on_order_created:
                    await self.on_order_created(order)
                    
                # Если требуется хеджирование
                should_hedge = enable_hedging if enable_hedging is not None else self.enable_hedging
                if should_hedge and order.status in [OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED]:
                    await self._hedge_order(order)
                    
                logger.info(f"Создан ордер {order.order_id} ({order.order_type.value}) для {order.symbol} на {order.exchange_id}")
                
                return order
                
            except Exception as e:
                # Обрабатываем ошибку создания ордера
                error_msg = f"Ошибка при создании ордера: {str(e)}"
                logger.error(error_msg)
                
                # Отмечаем ордер как ошибочный
                order.status = OrderStatus.ERROR
                order.error = error_msg
                order.updated_at = datetime.now()
                
                # Сохраняем ордер in истории
                self.order_history[order.order_id] = order
                
                # Сохраняем ордер in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
                # Вызываем колбэк ошибки
                if self.on_order_error:
                    await self.on_order_error(order, str(e))
                    
                raise
    
    async def _create_complex_order(self, order: AdvancedOrder) -> AdvancedOrder:
        """
        Создает сложный ордер (брекет, OCO, TWAP, VWAP and т.д.)
        
        Args:
            order: Базовый ордер
            
        Returns:
            AdvancedOrder: Обновленный ордер со всеми суб-ордерами
        """
        # Проверяем, что все условия выполнены
        if all_conditions_met:
            self.logger.info(f"Все условия для ордера {order_id} выполнены, выполняем действие")
            # Выполняем действие
            await self._execute_order_action(order_id, order_data)
        else:
            self.logger.debug(f"Не все условия для ордера {order_id} выполнены")
        # Проверяем тип ордера
        if order.order_type == OrderType.BRACKET:
            return await self._create_bracket_order(order)
        elif order.order_type == OrderType.OCO:
            return await self._create_oco_order(order)
        elif order.order_type == OrderType.SCALED:
            return await self._create_scaled_order(order)
        elif order.order_type == OrderType.TWAP:
            return await self._create_twap_order(order)
        elif order.order_type == OrderType.VWAP:
            return await self._create_vwap_order(order)
        else:
            raise ValueError(f"Неподдерживаемый тип сложного ордера: {order.order_type}")
    
    async def _execute_twap(self, order: AdvancedOrder, amount_per_slice: float, interval: float, num_slices: int):
        """
        Выполняет алгоритм TWAP, создавая ордера через равные промежутки времени
        
        Args:
            order: TWAP ордер
            amount_per_slice: Объем для каждого ордера
            interval: Интервал между ордерами in секундах
            num_slices: Общее количество частей
        """
        order.status = OrderStatus.OPEN
        
        for i in range(num_slices):
            if self._stop_requested или order.status != OrderStatus.OPEN:
                break
                
            try:
                # Создаем ордер для текущей части
                order_type = OrderType.LIMIT если order.price еще OrderType.MARKET
                sub_order = await self.create_order(
                    order.symbol,
                    order_type,
                    order.side,
                    amount_per_slice,
                    order.price,
                    order.exchange_id,
                    {k: v for k, v in order.params.items() if k not in ['duration', 'numSlices']}
                )
                
                # Привязываем суб-ордер
                sub_order.parent_id = order.order_id
                order.sub_orders.append(sub_order)
                
                # Обновляем прогресс выполнения
                order.metadata['progress'] = (i + 1) / num_slices
                
                # Сохраняем обновление in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
            except Exception as e:
                logger.error(f"Ошибка при создании {i+1}-й части TWAP ордера {order.order_id}: {str(e)}")
                
                # Добавляем информацию об ошибке
                order.metadata['last_error'] = str(e)
                order.metadata['error_slice'] = i + 1
                
                # Если критическая ошибка, прерываем выполнение
                if "insufficient balance" in str(e).lower() или "not enough balance" in str(e).lower():
                    logger.error(f"Недостаточно средств для продолжения TWAP ордера {order.order_id}")
                    order.error = f"Недостаточно средств: {str(e)}"
                    break
            
            # Ждем до следующего интервала, если это не последняя часть
            if i < num_slices - 1:
                await asyncio.sleep(interval)
        
        # Обновляем статус TWAP ордера
        if all(sub.status == OrderStatus.FILLED for sub in order.sub_orders):
            order.status = OrderStatus.FILLED
        elif any(sub.status == OrderStatus.FILLED for sub in order.sub_orders):
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.ERROR
            
        # Обновляем информацию о выполнении
        order.executed_amount = sum(sub.executed_amount for sub in order.sub_orders)
        
        if order.executed_amount > 0:
            # Вычисляем средневзвешенную цену исполнения
            total_value = sum(sub.executed_amount * sub.average_price for sub in order.sub_orders если sub.average_price)
            order.average_price = total_value / order.executed_amount если order.executed_amount > 0 еще 0
        
        # Обновляем время последнего обновления
        order.updated_at = datetime.now()
        
        # Сохраняем окончательный результат in БД
        if self.db:
            await self.db.save_order(order.to_dict())
            
        # Обновляем списки ордеров
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        self.order_history[order.order_id] = order
        
        logger.info(f"TWAP ордер {order.order_id} завершен, исполнено: {order.executed_amount}/{order.amount}")
    
    async def _execute_vwap(self, order: AdvancedOrder, amounts: List[float], interval: float):
        """
        Выполняет алгоритм VWAP, создавая ордера согласно объемному профилю
        
        Args:
            order: VWAP ордер
            amounts: Список объемов для каждого ордера
            interval: Интервал между ордерами in секундах
        """
        order.status = OrderStatus.OPEN
        
        for i, amount in enumerate(amounts):
            if self._stop_requested или order.status != OrderStatus.OPEN:
                break
                
            try:
                # Получаем текущую рыночную цену для адаптации
                current_price = None
                if order.price and self.enable_price_adaptation:
                    ticker = await self.market_data_provider.get_ticker(order.symbol, order.exchange_id)
                    if ticker:
                        if order.side == OrderSide.BUY:
                            current_price = ticker.get('ask')
                        else:
                            current_price = ticker.get('bid')
                
                # Используем адаптированную цену или исходную
                price_to_use = current_price или order.price
                
                # Создаем ордер для текущей части
                order_type = OrderType.LIMIT если price_to_use еще OrderType.MARKET
                sub_order = await self.create_order(
                    order.symbol,
                    order_type,
                    order.side,
                    amount,
                    price_to_use,
                    order.exchange_id,
                    {k: v for k, v in order.params.items() if k not in ['duration', 'numSlices', 'useHistoricalProfile']}
                )
                
                # Привязываем суб-ордер
                sub_order.parent_id = order.order_id
                order.sub_orders.append(sub_order)
                
                # Обновляем прогресс выполнения
                order.metadata['progress'] = (i + 1) / len(amounts)
                
                # Сохраняем обновление in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
            except Exception as e:
                logger.error(f"Ошибка при создании {i+1}-й части VWAP ордера {order.order_id}: {str(e)}")
                
                # Добавляем информацию об ошибке
                order.metadata['last_error'] = str(e)
                order.metadata['error_slice'] = i + 1
                
                # Если критическая ошибка, прерываем выполнение
                if "insufficient balance" in str(e).lower() или "not enough balance" in str(e).lower():
                    logger.error(f"Недостаточно средств для продолжения VWAP ордера {order.order_id}")
                    order.error = f"Недостаточно средств: {str(e)}"
                    break
            
            # Ждем до следующего интервала, если это не последняя часть
            if i < len(amounts) - 1:
                await asyncio.sleep(interval)
        
        # Обновляем статус VWAP ордера
        if all(sub.status == OrderStatus.FILLED for sub in order.sub_orders):
            order.status = OrderStatus.FILLED
        elif any(sub.status == OrderStatus.FILLED for sub in order.sub_orders):
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.ERROR
            
        # Обновляем информацию о выполнении
        order.executed_amount = sum(sub.executed_amount for sub in order.sub_orders)
        
        if order.executed_amount > 0:
            # Вычисляем средневзвешенную цену исполнения
            total_value = sum(sub.executed_amount * sub.average_price for sub in order.sub_orders если sub.average_price)
            order.average_price = total_value / order.executed_amount если order.executed_amount > 0 еще 0
        
        # Обновляем время последнего обновления
        order.updated_at = datetime.now()
        
        # Сохраняем окончательный результат in БД
        if self.db:
            await self.db.save_order(order.to_dict())
            
        # Обновляем списки ордеров
        if order.order_id in self.active_orders:
            del self.active_orders[order.order_id]
        self.order_history[order.order_id] = order
        
        logger.info(f"VWAP ордер {order.order_id} завершен, исполнено: {order.executed_amount}/{order.amount}")
    
    async def _adapt_limit_price(self, symbol: str, exchange_id: str, side: OrderSide, base_price: float) -> float:
        """
        Адаптирует цену лимитного ордера in зависимости от текущего состояния рынка
        
        Args:
            symbol: Торговая пара
            exchange_id: ID биржи
            side: Сторона ордера
            base_price: Базовая цена
            
        Returns:
            float: Адаптированная цена для лимитного ордера
        """
        try:
            # Получаем текущую книгу ордеров
            order_book = await self.market_data_provider.get_order_book(symbol, exchange_id, limit=5)
            
            if not order_book или 'bids' не in order_book или 'asks' не in order_book:
                return base_price
                
            # Получаем лучшие цены
            best_bid = order_book['bids'][0][0] если order_book['bids'] еще None
            best_ask = order_book['asks'][0][0] если order_book['asks'] еще None
            
            if not best_bid или not best_ask:
                return base_price
                
            # Текущий спред
            spread = best_ask - best_bid
            
            # Подстраиваем цену in зависимости от стороны
            if side == OrderSide.BUY:
                # Для покупки устанавливаем цену чуть выше бида, но ниже аска
                if base_price < best_bid или base_price > best_ask:
                    # Если указанная цена вне текущего спреда, используем цену внутри спреда
                    adjusted_price = best_bid + spread * 0.2  # 20% от спреда выше бида
                else:
                    # Если цена in спреде, оставляем как есть
                    adjusted_price = base_price
            else:  # SELL
                # Для продажи устанавливаем цену чуть ниже аска, но выше бида
                if base_price < best_bid или base_price > best_ask:
                    # Если указанная цена вне текущего спреда, используем цену внутри спреда
                    adjusted_price = best_ask - spread * 0.2  # 20% от спреда ниже аска
                else:
                    # Если цена in спреде, оставляем как есть
                    adjusted_price = base_price
                    
            return adjusted_price
        except Exception as e:
            logger.warning(f"Ошибка при адаптации цены для {symbol}: {str(e)}. Используется исходная цена.")
            return base_price
    
    async def _hedge_order(self, order: AdvancedOrder) -> bool:
        """
        Создает хеджирующий ордер для исходного ордера
        
        Args:
            order: Исходный ордер
            
        Returns:
            bool: True, если хеджирование успешно, False in противном случае
        """
        if not self.enable_hedging:
            return False
            
        # Параметры хеджирования из конфигурации или ордера
        hedge_type = order.hedge_params.get('type', HedgeType.DIRECT.value)
        hedge_exchange = order.hedge_params.get('exchange')
        hedge_symbol = order.hedge_params.get('symbol')
        
        # Для хеджирующего ордера используем противоположную сторону
        hedge_side = OrderSide.SELL если order.side == OrderSide.BUY еще OrderSide.BUY
        
        # Объем для хеджирования (может быть меньше или равен исходному)
        hedge_amount = order.hedge_params.get('amount', order.amount)
        
        try:
            if hedge_type == HedgeType.CROSS_EXCHANGE.value:
                # Хеджирование на другой бирже
                if not hedge_exchange:
                    # Автоматически выбираем биржу для хеджирования
                    for ex_id in self.hedge_exchanges:
                        if ex_id != order.exchange_id:
                            # Проверяем доступность символа на этой бирже
                            mapped_symbol = await self.exchange_manager.map_symbol_across_exchanges(
                                order.symbol, order.exchange_id, ex_id
                            )
                            if mapped_symbol:
                                hedge_exchange = ex_id
                                hedge_symbol = mapped_symbol
                                break
                                
                    if not hedge_exchange:
                        logger.warning(f"Не удалось найти подходящую биржу для хеджирования {order.symbol}")
                        return False
                        
                # Если символ не указан, используем сопоставление
                if not hedge_symbol:
                    hedge_symbol = await self.exchange_manager.map_symbol_across_exchanges(
                        order.symbol, order.exchange_id, hedge_exchange
                    )
                    
                    if not hedge_symbol:
                        logger.warning(f"Не удалось сопоставить символ {order.symbol} для биржи {hedge_exchange}")
                        return False
                        
                # Получаем текущую цену на целевой бирже
                ticker = await self.market_data_provider.get_ticker(hedge_symbol, hedge_exchange)
                if not ticker:
                    logger.warning(f"Не удалось получить текущую цену для {hedge_symbol} на {hedge_exchange}")
                    return False
                    
                hedge_price = ticker.get('bid') если hedge_side == OrderSide.SELL еще ticker.get('ask')
                
                # Проверяем отклонение цены между биржами
                if order.average_price and hedge_price:
                    price_deviation = abs(hedge_price / order.average_price - 1) * 100
                    if price_deviation > self.hedge_max_deviation:
                        logger.warning(f"Слишком большое отклонение цены для хеджирования {order.symbol}: {price_deviation}%")
                        return False
                        
                # Создаем хеджирующий ордер на другой бирже
                hedge_order = await self.create_order(
                    hedge_symbol,
                    OrderType.MARKET,  # Используем рыночный ордер для быстрого хеджирования
                    hedge_side,
                    hedge_amount,
                    None,
                    hedge_exchange,
                    {'reduceOnly': False}  # Для хеджирования не используем reduceOnly
                )
                
                # Привязываем хеджирующий ордер к исходному
                hedge_order.metadata['hedge_for'] = order.order_id
                order.hedge_orders.append(hedge_order)
                order.is_hedged = True
                
                # Сохраняем информацию о хеджировании
                self.hedge_history[order.order_id] = {
                    'original_order': order.to_dict(),
                    'hedge_order': hedge_order.to_dict(),
                    'timestamp': datetime.now().isoformat(),
                    'price_deviation': price_deviation если order.average_price and hedge_price еще None
                }
                
                logger.info(f"Создан кросс-биржевой хеджирующий ордер {hedge_order.order_id} для {order.order_id}")
                return True
                
            else:  # HedgeType.DIRECT
                # Хеджирование на той же бирже
                
                # Создаем хеджирующий ордер на той же бирже
                hedge_order = await self.create_order(
                    order.symbol,
                    OrderType.MARKET,  # Используем рыночный ордер для быстрого хеджирования
                    hedge_side,
                    hedge_amount,
                    None,
                    order.exchange_id,
                    {'reduceOnly': False}  # Для хеджирования не используем reduceOnly
                )
                
                # Привязываем хеджирующий ордер к исходному
                hedge_order.metadata['hedge_for'] = order.order_id
                order.hedge_orders.append(hedge_order)
                order.is_hedged = True
                
                # Сохраняем информацию о хеджировании
                self.hedge_history[order.order_id] = {
                    'original_order': order.to_dict(),
                    'hedge_order': hedge_order.to_dict(),
                    'timestamp': datetime.now().isoformat(),
                    'price_deviation': 0.0  # На той же бирже отклонения нет
                }
                
                logger.info(f"Создан хеджирующий ордер {hedge_order.order_id} для {order.order_id}")
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при создании хеджирующего ордера для {order.order_id}: {str(e)}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Отменяет ордер
        
        Args:
            order_id: ID ордера
            
        Returns:
            bool: True, если ордер успешно отменен, False in противном случае
        """
        if order_id не in self.active_orders:
            logger.warning(f"Ордер {order_id} не найден in активных ордерах")
            return False
            
        order = self.active_orders[order_id]
        
        # Если ордер уже не активен, просто возвращаем True
        if not order.is_active():
            return True
            
        # Если это сложный ордер с суб-ордерами, отменяем каждый суб-ордер
        if order.sub_orders:
            all_cancelled = True
            for sub_order in order.sub_orders:
                if sub_order.is_active():
                    sub_cancelled = await self.cancel_order(sub_order.order_id)
                    all_cancelled = all_cancelled and sub_cancelled
            
            # Обновляем статус родительского ордера
            if all_cancelled:
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now()
                
                # Обновляем списки ордеров
                del self.active_orders[order_id]
                self.order_history[order_id] = order
                
                # Сохраняем ордер in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
                return True
            else:
                return False
                
        # Если ордер не имеет exchange_order_id, значит он не был отправлен на биржу
        if not order.exchange_order_id:
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.now()
            
            # Обновляем списки ордеров
            del self.active_orders[order_id]
            self.order_history[order_id] = order
            
            # Сохраняем ордер in БД
            if self.db:
                await self.db.save_order(order.to_dict())
                
            return True
            
        # Отменяем ордер на бирже
        try:
            # Получаем экземпляр биржи
            exchange = await get_exchange_instance(order.exchange_id)
            
            with ExchangeErrorHandler():
                # Отменяем ордер
                response = await exchange.cancel_order(order.exchange_order_id, order.symbol)
                
                # Обновляем статус ордера
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now()
                
                # Если получена информация об исполнении перед отменой
                if 'filled' in response and response['filled']:
                    order.executed_amount = float(response['filled'])
                    
                if 'average' in response and response['average']:
                    order.average_price = float(response['average'])
                    
                # Вызываем колбэк отмены ордера
                if self.on_order_canceled:
                    await self.on_order_canceled(order)
                    
                # Обновляем списки ордеров
                del self.active_orders[order_id]
                self.order_history[order_id] = order
                
                # Сохраняем ордер in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
                logger.info(f"Ордер {order_id} успешно отменен")
                
                # Закрываем соединение с биржей
                await exchange.close()
                
                return True
                
        except Exception as e:
            logger.error(f"Ошибка при отмене ордера {order_id}: {str(e)}")
            
            # Для ошибок типа "ордер не найден" или "уже отменен" считаем успешным завершением
            if "not found" in str(e).lower() или "already" in str(e).lower() and "cancel" in str(e).lower():
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.now()
                order.error = str(e)
                
                # Обновляем списки ордеров
                if order.order_id in self.active_orders:
                    del self.active_orders[order_id]
                self.order_history[order_id] = order
                
                # Сохраняем ордер in БД
                if self.db:
                    await self.db.save_order(order.to_dict())
                    
                return True
            
            return False
    
    def _map_to_ccxt_order_type(self, order_type: OrderType) -> str:
        """
        Преобразует наш тип ордера in тип CCXT
        
        Args:
            order_type: Наш тип ордера
            
        Returns:
            str: Тип ордера CCXT
        """
        # Сопоставление типов ордеров
        type_map = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP_LOSS: 'stop',
            OrderType.TAKE_PROFIT: 'take_profit',
            OrderType.STOP_LIMIT: 'stop_limit',
            OrderType.TRAILING_STOP: 'trailing_stop',
            OrderType.FILL_OR_KILL: 'fok',
            OrderType.IMMEDIATE_OR_CANCEL: 'ioc',
            OrderType.POST_ONLY: 'limit'  # Post-only это лимитный ордер с доп. параметром
        }
        
        return type_map.get(order_type, 'limit')
    
    def _map_from_ccxt_order_type(self, ccxt_type: str) -> OrderType:
        """
        Преобразует тип ордера CCXT in наш тип
        
        Args:
            ccxt_type: Тип ордера CCXT
            
        Returns:
            OrderType: Наш тип ордера
        """
        # Сопоставление типов ордеров
        type_map = {
            'market': OrderType.MARKET,
            'limit': OrderType.LIMIT,
            'stop': OrderType.STOP_LOSS,
            'stop_loss': OrderType.STOP_LOSS,
            'take_profit': OrderType.TAKE_PROFIT,
            'stop_limit': OrderType.STOP_LIMIT,
            'trailing_stop': OrderType.TRAILING_STOP,
            'fok': OrderType.FILL_OR_KILL,
            'ioc': OrderType.IMMEDIATE_OR_CANCEL
        }
        
        return type_map.get(ccxt_type.lower(), OrderType.LIMIT)