from datetime import datetime
from typing import Dict, List, Optional

from project.utils.logging_utils import setup_logger

logger = setup_logger("portfolio")


class Position:
    """Класс для представления торговой позиции"""

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        amount: float,
        open_time: datetime = None,
        close_time: datetime = None,
        close_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        params: Dict = None,
    ):
        """
        Инициализирует торговую позицию

        Args:
            symbol: Торговая пара
            direction: Направление (long, short)
            entry_price: Цена входа
            amount: Объем
            open_time: Время открытия
            close_time: Время закрытия
            close_price: Цена закрытия
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            params: Дополнительные параметры
        """
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.amount = amount
        self.open_time = open_time or datetime.now()
        self.close_time = close_time
        self.close_price = close_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.params = params or {}

        # Уникальный идентификатор позиции
        self.id = f"{symbol}_{direction}_{int(self.open_time.timestamp())}"

        # Связанные ордера
        self.orders = []

        # Расчетные значения
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.fees = 0.0

    def is_open(self) -> bool:
        """
        Проверяет, открыта ли позиция

        Returns:
            bool: True, если позиция открыта, иначе False
        """
        return self.close_time is None

    def update_price(self, current_price: float):
        """
        Обновляет нереализованную прибыль/убыток

        Args:
            current_price: Текущая цена
        """
        if not self.is_open():
            return

        # Рассчитываем нереализованную прибыль/убыток
        if self.direction == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.amount
        else:  # short
            self.unrealized_pnl = (self.entry_price - current_price) * self.amount

    def close(self, close_price: float, close_time: datetime = None):
        """
        Закрывает позицию

        Args:
            close_price: Цена закрытия
            close_time: Время закрытия
        """
        if not self.is_open():
            return

        self.close_price = close_price
        self.close_time = close_time or datetime.now()

        # Рассчитываем реализованную прибыль/убыток
        if self.direction == "long":
            self.realized_pnl = (self.close_price - self.entry_price) * self.amount
        else:  # short
            self.realized_pnl = (self.entry_price - self.close_price) * self.amount

        # Сбрасываем нереализованную прибыль/убыток
        self.unrealized_pnl = 0.0

    def to_dict(self) -> Dict:
        """
        Преобразует позицию в словарь

        Returns:
            Dict: Словарь с данными позиции
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "open_time": self.open_time.isoformat(),
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "close_price": self.close_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "fees": self.fees,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Position":
        """
        Создает позицию из словаря

        Args:
            data: Словарь с данными позиции

        Returns:
            Position: Созданная позиция
        """
        position = cls(
            symbol=data.get("symbol"),
            direction=data.get("direction"),
            entry_price=data.get("entry_price"),
            amount=data.get("amount"),
            open_time=(
                datetime.fromisoformat(data.get("open_time"))
                if data.get("open_time")
                else None
            ),
            close_time=(
                datetime.fromisoformat(data.get("close_time"))
                if data.get("close_time")
                else None
            ),
            close_price=data.get("close_price"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            params=data.get("params", {}),
        )

        position.unrealized_pnl = data.get("unrealized_pnl", 0.0)
        position.realized_pnl = data.get("realized_pnl", 0.0)
        position.fees = data.get("fees", 0.0)

        return position

    def __str__(self) -> str:
        """
        Строковое представление позиции

        Returns:
            str: Строковое представление
        """
        status = "OPEN" if self.is_open() else "CLOSED"
        pnl = self.unrealized_pnl if self.is_open() else self.realized_pnl
        return f"Position({
            self.symbol}, {
            self.direction}, {
            self.amount}, {
                self.entry_price}, {status}, PnL: {
                    pnl:.2f})"


class Trade:
    """Класс для представления сделки"""

    def __init__(
        self,
        symbol: str,
        direction: str,
        price: float,
        amount: float,
        timestamp: datetime = None,
        trade_type: str = "market",
        fee: float = 0.0,
        order_id: str = None,
        position_id: str = None,
    ):
        """
        Инициализирует сделку

        Args:
            symbol: Торговая пара
            direction: Направление (buy, sell)
            price: Цена
            amount: Объем
            timestamp: Временная метка
            trade_type: Тип сделки
            fee: Комиссия
            order_id: ID связанного ордера
            position_id: ID связанной позиции
        """
        self.symbol = symbol
        self.direction = direction
        self.price = price
        self.amount = amount
        self.timestamp = timestamp or datetime.now()
        self.trade_type = trade_type
        self.fee = fee
        self.order_id = order_id
        self.position_id = position_id

        # Уникальный идентификатор сделки
        self.id = f"{symbol}_{direction}_{int(self.timestamp.timestamp())}"

        # Стоимость сделки
        self.cost = price * amount

    def to_dict(self) -> Dict:
        """
        Преобразует сделку в словарь

        Returns:
            Dict: Словарь с данными сделки
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "price": self.price,
            "amount": self.amount,
            "timestamp": self.timestamp.isoformat(),
            "trade_type": self.trade_type,
            "fee": self.fee,
            "cost": self.cost,
            "order_id": self.order_id,
            "position_id": self.position_id,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Trade":
        """
        Создает сделку из словаря

        Args:
            data: Словарь с данными сделки

        Returns:
            Trade: Созданная сделка
        """
        trade = cls(
            symbol=data.get("symbol"),
            direction=data.get("direction"),
            price=data.get("price"),
            amount=data.get("amount"),
            timestamp=(
                datetime.fromisoformat(data.get("timestamp"))
                if data.get("timestamp")
                else None
            ),
            trade_type=data.get("trade_type", "market"),
            fee=data.get("fee", 0.0),
            order_id=data.get("order_id"),
            position_id=data.get("position_id"),
        )

        trade.id = data.get("id", trade.id)
        trade.cost = data.get("cost", trade.cost)

        return trade

    def __str__(self) -> str:
        """
        Строковое представление сделки

        Returns:
            str: Строковое представление
        """
        return f"Trade({
            self.symbol}, {
            self.direction}, {
            self.amount}, {
                self.price}, {
                    self.timestamp})"


class Order:
    """Класс для представления ордера"""

    def __init__(
        self,
        symbol: str,
        direction: str,
        price: float,
        amount: float,
        timestamp: datetime = None,
        order_type: str = "market",
        status: str = "open",
        filled: float = 0.0,
        remaining: float = None,
        position_id: str = None,
    ):
        """
        Инициализирует ордер

        Args:
            symbol: Торговая пара
            direction: Направление (buy, sell)
            price: Цена
            amount: Объем
            timestamp: Временная метка
            order_type: Тип ордера
            status: Статус ордера
            filled: Исполненный объем
            remaining: Оставшийся объем
            position_id: ID связанной позиции
        """
        self.symbol = symbol
        self.direction = direction
        self.price = price
        self.amount = amount
        self.timestamp = timestamp or datetime.now()
        self.order_type = order_type
        self.status = status
        self.filled = filled
        self.remaining = remaining if remaining is not None else amount
        self.position_id = position_id

        # Уникальный идентификатор ордера
        self.id = f"{symbol}_{direction}_{int(self.timestamp.timestamp())}"

        # Время обновления
        self.last_update = self.timestamp

        # Связанные сделки
        self.trades = []

    def update(self, filled: float = None, status: str = None):
        """
        Обновляет ордер

        Args:
            filled: Исполненный объем
            status: Статус ордера
        """
        if filled is not None:
            self.filled = filled
            self.remaining = self.amount - filled

        if status is not None:
            self.status = status

        self.last_update = datetime.now()

    def is_active(self) -> bool:
        """
        Проверяет, активен ли ордер

        Returns:
            bool: True, если ордер активен, иначе False
        """
        return self.status in ["open", "partially_filled"]

    def to_dict(self) -> Dict:
        """
        Преобразует ордер в словарь

        Returns:
            Dict: Словарь с данными ордера
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "price": self.price,
            "amount": self.amount,
            "timestamp": self.timestamp.isoformat(),
            "order_type": self.order_type,
            "status": self.status,
            "filled": self.filled,
            "remaining": self.remaining,
            "position_id": self.position_id,
            "last_update": self.last_update.isoformat(),
            "trades": [
                trade.to_dict() if hasattr(trade, "to_dict") else trade
                for trade in self.trades
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Order":
        """
        Создает ордер из словаря

        Args:
            data: Словарь с данными ордера

        Returns:
            Order: Созданный ордер
        """
        order = cls(
            symbol=data.get("symbol"),
            direction=data.get("direction"),
            price=data.get("price"),
            amount=data.get("amount"),
            timestamp=(
                datetime.fromisoformat(data.get("timestamp"))
                if data.get("timestamp")
                else None
            ),
            order_type=data.get("order_type", "market"),
            status=data.get("status", "open"),
            filled=data.get("filled", 0.0),
            remaining=data.get("remaining"),
            position_id=data.get("position_id"),
        )

        order.id = data.get("id", order.id)

        if data.get("last_update"):
            order.last_update = datetime.fromisoformat(data.get("last_update"))

        if data.get("trades"):
            order.trades = [
                Trade.from_dict(trade) if isinstance(trade, dict) else trade
                for trade in data.get("trades", [])
            ]

        return order

    def __str__(self) -> str:
        """
        Строковое представление ордера

        Returns:
            str: Строковое представление
        """
        return f"Order({self.symbol}, {self.direction}, {self.amount}, {self.price}, {self.status})"


class Portfolio:
    """Класс для управления торговым портфелем"""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0001,
        position_size_pct: float = 1.0,
        max_positions: int = 1,
        enable_fractional: bool = True,
        enable_shorting: bool = False,
        enable_compounding: bool = True,
    ):
        """
        Инициализирует портфель

        Args:
            initial_balance: Начальный баланс
            commission: Комиссия (в долях от стоимости сделки)
            slippage: Проскальзывание (в долях от цены)
            position_size_pct: Размер позиции (в процентах от баланса)
            max_positions: Максимальное количество позиций
            enable_fractional: Разрешить дробные объемы
            enable_shorting: Разрешить короткие позиции
            enable_compounding: Разрешить реинвестирование прибыли
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.enable_fractional = enable_fractional
        self.enable_shorting = enable_shorting
        self.enable_compounding = enable_compounding

        # Позиции
        self.positions = {}

        # История сделок
        self.trades = []

        # Ордера
        self.orders = {}

        # Метрики
        self.metrics = {
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_fees": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "total_trades": 0,
        }

        # Хронология изменений капитала
        self.equity_history = []

        logger.debug(f"Portfolio initialized with balance: {initial_balance}")

    async def open_position(
        self,
        direction: str,
        price: float,
        amount: float,
        timestamp: datetime = None,
        stop_loss: float = None,
        take_profit: float = None,
        symbol: str = "default",
    ) -> Optional[Position]:
        """
        Открывает новую позицию

        Args:
            direction: Направление (long, short)
            price: Цена
            amount: Объем
            timestamp: Временная метка
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            symbol: Торговая пара

        Returns:
            Optional[Position]: Открытая позиция или None, если открытие не удалось
        """
        if not self.can_open_position():
            logger.warning("Cannot open position: maximum positions reached")
            return None

        # Проверяем, что короткие позиции разрешены
        if direction == "short" and not self.enable_shorting:
            logger.warning("Cannot open short position: shorting is disabled")
            return None

        # Применяем проскальзывание
        slippage_price = self._apply_slippage(price, direction)

        # Рассчитываем стоимость позиции
        position_cost = slippage_price * amount

        # Рассчитываем комиссию
        fee = position_cost * self.commission

        # Проверяем, достаточно ли баланса
        if self.balance < position_cost + fee:
            logger.warning(
                f"Cannot open position: insufficient balance ({
                    self.balance} < {
                    position_cost + fee})"
            )
            return None

        # Обновляем баланс
        self.balance -= position_cost + fee

        # Создаем позицию
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=slippage_price,
            amount=amount,
            open_time=timestamp or datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Добавляем комиссию
        position.fees += fee

        # Сохраняем позицию
        self.positions[position.id] = position

        # Создаем сделку
        trade_direction = "buy" if direction == "long" else "sell"
        trade = Trade(
            symbol=symbol,
            direction=trade_direction,
            price=slippage_price,
            amount=amount,
            timestamp=timestamp or datetime.now(),
            fee=fee,
            position_id=position.id,
        )

        # Сохраняем сделку
        self.trades.append(trade)

        # Обновляем метрики
        self.metrics["total_fees"] += fee
        self.metrics["total_trades"] += 1

        # Обновляем хронологию капитала
        self._update_equity_history()

        logger.debug(f"Opened position: {position}")

        return position

    async def close_position(
        self, position_id: str, price: float, timestamp: datetime = None
    ) -> Optional[Position]:
        """
        Закрывает позицию

        Args:
            position_id: ID позиции
            price: Цена закрытия
            timestamp: Временная метка

        Returns:
            Optional[Position]: Закрытая позиция или None, если позиция не найдена
        """
        # Проверяем, существует ли позиция
        if position_id not in self.positions:
            logger.warning(f"Cannot close position: position not found ({position_id})")
            return None

        # Получаем позицию
        position = self.positions[position_id]

        # Проверяем, открыта ли позиция
        if not position.is_open():
            logger.warning(
                f"Cannot close position: position already closed ({position_id})"
            )
            return None

        # Применяем проскальзывание
        trade_direction = "sell" if position.direction == "long" else "buy"
        slippage_price = self._apply_slippage(price, trade_direction)

        # Рассчитываем стоимость закрытия
        close_cost = slippage_price * position.amount

        # Рассчитываем комиссию
        fee = close_cost * self.commission

        # Закрываем позицию
        position.close(slippage_price, timestamp or datetime.now())

        # Добавляем комиссию
        position.fees += fee

        # Обновляем баланс
        # Для длинной позиции возвращаем стоимость закрытия за вычетом комиссии
        # Для короткой позиции возвращаем объем позиции за вычетом комиссии и прибыли/убытка
        if position.direction == "long":
            self.balance += close_cost - fee
        else:  # short
            self.balance += (
                position.amount * position.entry_price + position.realized_pnl - fee
            )

        # Создаем сделку
        trade = Trade(
            symbol=position.symbol,
            direction=trade_direction,
            price=slippage_price,
            amount=position.amount,
            timestamp=timestamp or datetime.now(),
            fee=fee,
            position_id=position.id,
        )

        # Сохраняем сделку
        self.trades.append(trade)

        # Обновляем метрики
        self.metrics["total_fees"] += fee
        self.metrics["total_trades"] += 1
        self.metrics["realized_pnl"] += position.realized_pnl
        self.metrics["total_pnl"] += position.realized_pnl

        if position.realized_pnl > 0:
            self.metrics["win_count"] += 1
        else:
            self.metrics["loss_count"] += 1

        # Обновляем хронологию капитала
        self._update_equity_history()

        logger.debug(f"Closed position: {position}")

        return position

    def calculate_position_size(
        self, price: float, position_size_pct: float = None
    ) -> float:
        """
        Рассчитывает размер позиции

        Args:
            price: Цена
            position_size_pct: Размер позиции (в процентах от баланса)

        Returns:
            float: Размер позиции
        """
        # Используем переданный размер позиции или значение по умолчанию
        use_position_size_pct = (
            position_size_pct
            if position_size_pct is not None
            else self.position_size_pct
        )

        # Рассчитываем размер позиции
        position_value = self.balance * use_position_size_pct

        # Рассчитываем объем
        amount = position_value / price

        # Округляем объем, если дробные объемы не разрешены
        if not self.enable_fractional:
            amount = int(amount)

        return amount

    def can_open_position(self) -> bool:
        """
        Проверяет, можно ли открыть новую позицию

        Returns:
            bool: True, если можно открыть позицию, иначе False
        """
        # Получаем количество открытых позиций
        open_positions = sum(1 for pos in self.positions.values() if pos.is_open())

        # Проверяем, не превышено ли максимальное количество позиций
        return open_positions < self.max_positions

    def get_position_value(self, current_price: float = None) -> float:
        """
        Возвращает стоимость открытых позиций

        Args:
            current_price: Текущая цена (для всех позиций)

        Returns:
            float: Стоимость открытых позиций
        """
        value = 0.0

        for position in self.positions.values():
            if position.is_open():
                if current_price is not None:
                    value += position.amount * current_price
                else:
                    value += position.amount * position.entry_price

        return value

    def get_equity(self, current_price: float = None) -> float:
        """
        Возвращает общий капитал (баланс + стоимость позиций)

        Args:
            current_price: Текущая цена (для всех позиций)

        Returns:
            float: Общий капитал
        """
        return self.balance + self.get_position_value(current_price)

    def get_open_positions(self) -> List[Position]:
        """
        Возвращает список открытых позиций

        Returns:
            List[Position]: Список открытых позиций
        """
        return [pos for pos in self.positions.values() if pos.is_open()]

    def get_position_by_symbol(
        self, symbol: str, direction: str = None
    ) -> Optional[Position]:
        """
        Возвращает позицию по символу и направлению

        Args:
            symbol: Торговая пара
            direction: Направление (long, short)

        Returns:
            Optional[Position]: Позиция или None, если позиция не найдена
        """
        for position in self.positions.values():
            if position.is_open() and position.symbol == symbol:
                if direction is None or position.direction == direction:
                    return position

        return None

    def _apply_slippage(self, price: float, direction: str) -> float:
        """
        Применяет проскальзывание к цене

        Args:
            price: Исходная цена
            direction: Направление (buy, sell)

        Returns:
            float: Цена с учетом проскальзывания
        """
        # Для покупки цена увеличивается, для продажи - уменьшается
        if direction == "buy":
            return price * (1 + self.slippage)
        else:  # sell
            return price * (1 - self.slippage)

    def _update_equity_history(self):
        """Обновляет хронологию капитала"""
        self.equity_history.append(
            {
                "timestamp": datetime.now(),
                "balance": self.balance,
                "equity": self.get_equity(),
                "unrealized_pnl": self.metrics["unrealized_pnl"],
                "realized_pnl": self.metrics["realized_pnl"],
                "open_positions": len(self.get_open_positions()),
            }
        )

    def to_dict(self) -> Dict:
        """
        Преобразует портфель в словарь

        Returns:
            Dict: Словарь с данными портфеля
        """
        return {
            "initial_balance": self.initial_balance,
            "balance": self.balance,
            "commission": self.commission,
            "slippage": self.slippage,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "enable_fractional": self.enable_fractional,
            "enable_shorting": self.enable_shorting,
            "enable_compounding": self.enable_compounding,
            "positions": {
                pos_id: pos.to_dict() for pos_id, pos in self.positions.items()
            },
            "trades": [trade.to_dict() for trade in self.trades],
            "orders": {
                order_id: order.to_dict() for order_id, order in self.orders.items()
            },
            "metrics": self.metrics,
            "equity_history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "balance": entry["balance"],
                    "equity": entry["equity"],
                    "unrealized_pnl": entry["unrealized_pnl"],
                    "realized_pnl": entry["realized_pnl"],
                    "open_positions": entry["open_positions"],
                }
                for entry in self.equity_history
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Portfolio":
        """
        Создает портфель из словаря

        Args:
            data: Словарь с данными портфеля

        Returns:
            Portfolio: Созданный портфель
        """
        portfolio = cls(
            initial_balance=data.get("initial_balance", 10000.0),
            commission=data.get("commission", 0.001),
            slippage=data.get("slippage", 0.0001),
            position_size_pct=data.get("position_size_pct", 1.0),
            max_positions=data.get("max_positions", 1),
            enable_fractional=data.get("enable_fractional", True),
            enable_shorting=data.get("enable_shorting", False),
            enable_compounding=data.get("enable_compounding", True),
        )

        portfolio.balance = data.get("balance", portfolio.initial_balance)
        portfolio.metrics = data.get("metrics", portfolio.metrics)

        # Загружаем позиции
        if data.get("positions"):
            portfolio.positions = {
                pos_id: Position.from_dict(pos_data)
                for pos_id, pos_data in data.get("positions", {}).items()
            }

        # Загружаем сделки
        if data.get("trades"):
            portfolio.trades = [
                Trade.from_dict(trade_data) for trade_data in data.get("trades", [])
            ]

        # Загружаем ордера
        if data.get("orders"):
            portfolio.orders = {
                order_id: Order.from_dict(order_data)
                for order_id, order_data in data.get("orders", {}).items()
            }

        # Загружаем хронологию капитала
        if data.get("equity_history"):
            portfolio.equity_history = [
                {
                    "timestamp": (
                        datetime.fromisoformat(entry.get("timestamp"))
                        if isinstance(entry.get("timestamp"), str)
                        else entry.get("timestamp")
                    ),
                    "balance": entry.get("balance"),
                    "equity": entry.get("equity"),
                    "unrealized_pnl": entry.get("unrealized_pnl"),
                    "realized_pnl": entry.get("realized_pnl"),
                    "open_positions": entry.get("open_positions"),
                }
                for entry in data.get("equity_history", [])
            ]

        return portfolio

    def update(self, current_price: float = None):
        """
        Обновляет состояние портфеля

        Args:
            current_price: Текущая цена (для всех позиций)
        """
        # Обновляем нереализованную прибыль/убыток
        unrealized_pnl = 0.0

        for position in self.positions.values():
            if position.is_open():
                if current_price is not None:
                    position.update_price(current_price)

                unrealized_pnl += position.unrealized_pnl

        # Обновляем метрики
        self.metrics["unrealized_pnl"] = unrealized_pnl
        self.metrics["total_pnl"] = self.metrics["realized_pnl"] + unrealized_pnl

        # Обновляем хронологию капитала
        self._update_equity_history()

    def reset(self):
        """Сбрасывает портфель в исходное состояние"""
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.orders = {}
        self.metrics = {
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_fees": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "total_trades": 0,
        }
        self.equity_history = []

        logger.debug("Portfolio reset to initial state")


"""
Модуль для управления портфелем во время бэктестинга.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from project.utils.logging_utils import setup_logger

logger = setup_logger("backtest_portfolio")


class BacktestPortfolio:
    """Класс для отслеживания состояния портфеля во время бэктестинга"""
    
    def __init__(self, initial_balance: float = 10000, commission: float = 0.001):
        pass
