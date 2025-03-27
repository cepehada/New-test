"""
Базовый модуль для реализации торговых стратегий.
Содержит классы для создания сигналов, управления позициями
и абстрактный класс стратегии.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List

import pandas as pd
from project.utils.logging_utils import setup_logger

logger = setup_logger("strategy_base")


class Signal:
    """Класс для представления торгового сигнала"""

    def __init__(
        self,
        symbol: str,
        direction: str,
        strength: float = 1.0,
        price: float = None,
        timestamp: datetime = None,
        expiration: datetime = None,
        params: Dict = None,
    ):
        """
        Инициализирует торговый сигнал

        Args:
            symbol: Торговая пара
            direction: Направление (buy, sell, close)
            strength: Сила сигнала (от 0 до 1)
            price: Цена сигнала
            timestamp: Время создания сигнала
            expiration: Время истечения сигнала
            params: Дополнительные параметры
        """
        self.symbol = symbol
        self.direction = direction
        self.strength = max(0.0, min(1.0, strength))  # Ограничиваем от 0 до 1
        self.price = price
        self.timestamp = timestamp or datetime.now()
        self.expiration = expiration
        self.params = params or {}

        # Уникальный идентификатор сигнала
        self.id = f"{symbol}_{direction}_{int(self.timestamp.timestamp())}"

    def is_valid(self) -> bool:
        """
        Проверяет, действителен ли сигнал

        Returns:
            bool: True, если сигнал действителен, иначе False
        """
        # Проверяем истечение срока действия
        if self.expiration and datetime.now() > self.expiration:
            return False

        return True

    def to_dict(self) -> Dict:
        """
        Преобразует сигнал в словарь

        Returns:
            Dict: Словарь с данными сигнала
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Signal":
        """
        Создает сигнал из словаря

        Args:
            data: Словарь с данными сигнала

        Returns:
            Signal: Созданный сигнал
        """
        return cls(
            symbol=data.get("symbol"),
            direction=data.get("direction"),
            strength=data.get("strength", 1.0),
            price=data.get("price"),
            timestamp=(
                datetime.fromisoformat(data.get("timestamp"))
                if data.get("timestamp")
                else None
            ),
            expiration=(
                datetime.fromisoformat(data.get("expiration"))
                if data.get("expiration")
                else None
            ),
            params=data.get("params", {}),
        )

    def __str__(self) -> str:
        """
        Строковое представление сигнала

        Returns:
            str: Строковое представление
        """
        return f"Signal({self.symbol}, {self.direction}, {self.strength:.2f}, {self.price})"


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
        return f"Position({self.symbol}, {self.direction}, {self.amount}, {self.entry_price}, {status}, PnL: {pnl:.2f})"


class Strategy(ABC):
    """Базовый абстрактный класс для торговых стратегий"""

    def __init__(self, parameters: Dict = None):
        """
        Инициализирует стратегию

        Args:
            parameters: Параметры стратегии
        """
        self.parameters = parameters or {}

        # Имя стратегии
        self.name = self.__class__.__name__

        # Текущие данные
        self.data = pd.DataFrame()

        # Список сигналов
        self.signals = []

        # Текущие позиции
        self.positions = {}

        # Индикаторы и их значения
        self.indicators = {}

        # Статистика
        self.stats = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "last_update": None,
        }

        # Логгер
        self.logger = setup_logger(f"strategy.{self.name}")

        # Обратные вызовы
        self.on_signal = None
        self.on_position_opened = None
        self.on_position_closed = None

        # Инициализируем начальные параметры
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Инициализирует параметры стратегии значениями по умолчанию"""
        # Устанавливаем параметры по умолчанию, если они не указаны
        default_params = self.get_default_parameters()

        for param, value in default_params.items():
            if param not in self.parameters:
                self.parameters[param] = value

        # Логируем параметры
        self.logger.info("Strategy parameters: %s", self.parameters)

    @classmethod
    def get_default_parameters(cls) -> Dict:
        """
        Возвращает параметры стратегии по умолчанию

        Returns:
            Dict: Параметры по умолчанию
        """
        return {}

    @abstractmethod
    async def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Рассчитывает индикаторы для стратегии

        Args:
            data: Исторические данные

        Returns:
            Dict: Словарь с рассчитанными индикаторами
        """

    @abstractmethod
    async def generate_signals(
        self, data: pd.DataFrame, indicators: Dict
    ) -> List[Signal]:
        """
        Генерирует торговые сигналы

        Args:
            data: Исторические данные
            indicators: Рассчитанные индикаторы

        Returns:
            List[Signal]: Список сигналов
        """

    @abstractmethod
    async def on_data(self, data: pd.DataFrame) -> List[Signal]:
        """
        Обрабатывает новые данные

        Args:
            data: Новые данные

        Returns:
            List[Signal]: Список сигналов
        """

    async def update(self, data: pd.DataFrame) -> List[Signal]:
        """
        Обновляет состояние стратегии новыми данными

        Args:
            data: Новые данные

        Returns:
            List[Signal]: Список сигналов
        """
        try:
            # Сохраняем данные
            self.data = data

            # Рассчитываем индикаторы
            self.indicators = await self.calculate_indicators(data)

            # Генерируем сигналы
            self.signals = await self.generate_signals(data, self.indicators)

            # Вызываем метод обработки данных
            signals = await self.on_data(data)

            # Добавляем сигналы
            if signals:
                self.signals.extend(signals)

            # Вызываем обратный вызов для сигналов
            if self.on_signal and self.signals:
                for signal in self.signals:
                    await self.on_signal(signal)

            # Обновляем статистику
            self.stats["last_update"] = datetime.now()

            return self.signals

        except Exception as e:
            self.logger.error("Error updating strategy: %s", str(e))
            return []

    async def on_tick(self, tick: Dict) -> List[Signal]:
        """
        Обрабатывает новые тики

        Args:
            tick: Новый тик

        Returns:
            List[Signal]: Список сигналов
        """
        # По умолчанию ничего не делаем
        return []

    async def on_trade(self, trade: Dict) -> List[Signal]:
        """
        Обрабатывает новые сделки

        Args:
            trade: Новая сделка

        Returns:
            List[Signal]: Список сигналов
        """
        # По умолчанию ничего не делаем
        return []

    async def on_order_book(self, order_book: Dict) -> List[Signal]:
        """
        Обрабатывает обновления книги ордеров

        Args:
            order_book: Обновление книги ордеров

        Returns:
            List[Signal]: Список сигналов
        """
        # По умолчанию ничего не делаем
        return []

    async def on_position_update(self, position: Position):
        """
        Обрабатывает обновления позиции

        Args:
            position: Обновленная позиция
        """
        # Сохраняем позицию
        self.positions[position.id] = position

        # Вызываем обратный вызов
        if position.is_open() and self.on_position_opened:
            await self.on_position_opened(position)
        elif not position.is_open() and self.on_position_closed:
            await self.on_position_closed(position)

        # Обновляем статистику
        if not position.is_open():
            self.stats["total_trades"] += 1
            self.stats["total_pnl"] += position.realized_pnl

            if position.realized_pnl > 0:
                self.stats["winning_trades"] += 1
            else:
                self.stats["losing_trades"] += 1

            # Обновляем процент выигрышных сделок
            if self.stats["total_trades"] > 0:
                self.stats["win_rate"] = (
                    self.stats["winning_trades"] / self.stats["total_trades"]
                )

    def get_info(self) -> Dict:
        """
        Возвращает информацию о стратегии

        Returns:
            Dict: Информация о стратегии
        """
        return {
            "name": self.name,
            "parameters": self.parameters,
            "stats": self.stats,
            "positions": {
                pos_id: pos.to_dict() for pos_id, pos in self.positions.items()
            },
            "signals": [signal.to_dict() for signal in self.signals],
        }


class StrategyRegistry:
    """Реестр торговых стратегий"""

    _strategies = {}

    @classmethod
    def register(cls, strategy_class: type):
        """
        Регистрирует класс стратегии

        Args:
            strategy_class: Класс стратегии

        Returns:
            type: Класс стратегии
        """
        if not issubclass(strategy_class, Strategy):
            raise TypeError(f"{strategy_class.__name__} is not a subclass of Strategy")

        cls._strategies[strategy_class.__name__] = strategy_class
        logger.info("Registered strategy: %s", strategy_class.__name__)

        return strategy_class

    @classmethod
    def get_strategy_class(cls, strategy_name: str) -> type:
        """
        Возвращает класс стратегии по имени

        Args:
            strategy_name: Имя стратегии

        Returns:
            type: Класс стратегии
        """
        if strategy_name not in cls._strategies:
            raise ValueError(f"Strategy not found: {strategy_name}")

        return cls._strategies[strategy_name]

    @classmethod
    def get_all_strategies(cls) -> Dict[str, type]:
        """
        Возвращает словарь всех зарегистрированных стратегий

        Returns:
            Dict[str, type]: Словарь стратегий
        """
        return cls._strategies.copy()

    @classmethod
    def create_strategy(cls, strategy_name: str, parameters: Dict = None) -> Strategy:
        """
        Создает экземпляр стратегии

        Args:
            strategy_name: Имя стратегии
            parameters: Параметры стратегии

        Returns:
            Strategy: Экземпляр стратегии
        """
        strategy_class = cls.get_strategy_class(strategy_name)
        return strategy_class(parameters)


def register_strategy(cls):
    """
    Декоратор для регистрации стратегии

    Args:
        cls: Класс стратегии

    Returns:
        type: Класс стратегии
    """
    return StrategyRegistry.register(cls)
