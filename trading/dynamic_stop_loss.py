"""
Модуль для реализации динамического стоп-лосса, который адаптируется
к рыночным условиям и волатильности.
"""
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass

from project.utils.logging_utils import setup_logger

logger = setup_logger("dynamic_stop_loss")


class StopLossType(Enum):
    """Типы стоп-лосса"""
    FIXED = "fixed"                  # Фиксированный процент от цены входа
    TRAILING = "trailing"            # Скользящий стоп-лосс
    ATR = "atr"                      # На основе ATR (Average True Range)
    CHANDELIER = "chandelier"        # Chandelier Exit
    PARABOLIC = "parabolic"          # Parabolic SAR
    BOLLINGER = "bollinger"          # На основе Bollinger Bands
    SUPPORT_RESISTANCE = "support_resistance"  # На основе уровней поддержки/сопротивления
    VOLATILITY = "volatility"        # На основе волатильности рынка
    CUSTOM = "custom"                # Пользовательский стоп-лосс


@dataclass
class StopLossConfig:
    """Конфигурация стоп-лосса"""
    type: StopLossType
    value: float = 0.0
    trailing_value: Optional[float] = None
    atr_period: int = 14
    atr_multiplier: float = 3.0
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    parabolic_step: float = 0.02
    parabolic_max_step: float = 0.2
    volatility_window: int = 20
    volatility_multiplier: float = 2.0
    chandelier_period: int = 22
    chandelier_multiplier: float = 3.0
    custom_func: Optional[Callable] = None


class DynamicStopLoss:
    """Класс для управления динамическим стоп-лоссом"""

    def __init__(self, config: StopLossConfig):
        """
        Инициализирует объект динамического стоп-лосса

        Args:
            config: Конфигурация стоп-лосса
        """
        self.config = config
        self.stop_level: Optional[float] = None
        self.initial_price: Optional[float] = None
        self.highest_price: Optional[float] = None
        self.lowest_price: Optional[float] = None
        self.direction: Optional[str] = None
        self.is_active = False
        
        logger.debug(f"DynamicStopLoss initialized with type: {config.type.value}")

    def activate(self, price: float, direction: str):
        """
        Активирует стоп-лосс при открытии позиции

        Args:
            price: Цена входа
            direction: Направление позиции ("long" или "short")
        """
        self.initial_price = price
        self.highest_price = price
        self.lowest_price = price
        self.direction = direction
        self.is_active = True

        # Вычисляем начальный уровень стоп-лосса
        self.stop_level = self.calculate_stop_level(price, direction, None)

        logger.debug(
            f"StopLoss activated: direction={direction}, price={price}, "
            f"stop_level={self.stop_level}")

        return self.stop_level

    def update(self, price: float, data: Optional[pd.DataFrame] = None) -> float:
        """
        Обновляет уровень стоп-лосса на основе текущей цены

        Args:
            price: Текущая цена
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Новый уровень стоп-лосса
        """
        if not self.is_active:
            logger.warning("Attempted to update inactive stop loss")
            return 0.0

        # Обновляем максимальную/минимальную цену
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)

        # Вычисляем новый уровень стоп-лосса
        new_stop = self.calculate_stop_level(price, self.direction, data)

        # Для лонгов стоп-лосс может только повышаться
        if self.direction == "long" and new_stop > self.stop_level:
            self.stop_level = new_stop
        # Для шортов стоп-лосс может только понижаться
        elif self.direction == "short" and new_stop < self.stop_level:
            self.stop_level = new_stop

        logger.debug(
            f"StopLoss updated: price={price}, new_stop={self.stop_level}")

        return self.stop_level

    def calculate_stop_level(
        self, price: float, direction: str, data: Optional[pd.DataFrame]
    ) -> float:
        """
        Вычисляет уровень стоп-лосса на основе типа стоп-лосса

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Уровень стоп-лосса
        """
        stop_type = self.config.type

        if stop_type == StopLossType.FIXED:
            return self._calculate_fixed_stop(price, direction)
        elif stop_type == StopLossType.TRAILING:
            return self._calculate_trailing_stop(price, direction)
        elif stop_type == StopLossType.ATR:
            return self._calculate_atr_stop(price, direction, data)
        elif stop_type == StopLossType.CHANDELIER:
            return self._calculate_chandelier_stop(price, direction, data)
        elif stop_type == StopLossType.PARABOLIC:
            return self._calculate_parabolic_stop(price, direction, data)
        elif stop_type == StopLossType.BOLLINGER:
            return self._calculate_bollinger_stop(price, direction, data)
        elif stop_type == StopLossType.SUPPORT_RESISTANCE:
            return self._calculate_support_resistance_stop(price, direction, data)
        elif stop_type == StopLossType.VOLATILITY:
            return self._calculate_volatility_stop(price, direction, data)
        elif stop_type == StopLossType.CUSTOM and self.config.custom_func:
            return self.config.custom_func(price, direction, data, self)
        else:
            logger.warning(f"Unknown stop-loss type: {stop_type}")
            return self._calculate_fixed_stop(price, direction)

    def is_triggered(self, price: float) -> bool:
        """
        Проверяет, сработал ли стоп-лосс при текущей цене

        Args:
            price: Текущая цена

        Returns:
            bool: True, если стоп-лосс сработал, иначе False
        """
        if not self.is_active or self.stop_level is None:
            return False

        if self.direction == "long":
            return price <= self.stop_level
        elif self.direction == "short":
            return price >= self.stop_level

        return False

    def deactivate(self):
        """Деактивирует стоп-лосс"""
        self.is_active = False
        logger.debug("StopLoss deactivated")

    def _calculate_fixed_stop(self, price: float, direction: str) -> float:
        """
        Вычисляет фиксированный стоп-лосс

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")

        Returns:
            float: Уровень стоп-лосса
        """
        value = max(0.001, self.config.value)  # Мин. значение 0.1%

        if direction == "long":
            return self.initial_price * (1 - value)
        else:  # short
            return self.initial_price * (1 + value)

    def _calculate_trailing_stop(self, price: float, direction: str) -> float:
        """
        Вычисляет скользящий стоп-лосс

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")

        Returns:
            float: Уровень стоп-лосса
        """
        value = self.config.trailing_value or self.config.value
        value = max(0.001, value)  # Мин. значение 0.1%

        if direction == "long":
            return self.highest_price * (1 - value)
        else:  # short
            return self.lowest_price * (1 + value)

    def _calculate_atr_stop(
        self, price: float, direction: str, data: Optional[pd.DataFrame]
    ) -> float:
        """
        Вычисляет стоп-лосс на основе ATR (Average True Range)

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Уровень стоп-лосса
        """
        # Если нет данных для расчета ATR, используем фиксированный стоп-лосс
        if data is None or len(data) < self.config.atr_period:
            return self._calculate_fixed_stop(price, direction)

        # Вычисляем ATR
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        # Для первого значения используем только tr1
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        
        # Находим максимум из трех компонентов TR
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Вычисляем ATR как скользящее среднее TR
        atr = np.mean(tr[-self.config.atr_period:])
        
        # Вычисляем уровень стоп-лосса
        multiplier = self.config.atr_multiplier
        
        if direction == "long":
            return price - (atr * multiplier)
        else:  # short
            return price + (atr * multiplier)

    def _calculate_chandelier_stop(
        self, price: float, direction: str, data: Optional[pd.DataFrame]
    ) -> float:
        """
        Вычисляет стоп-лосс по методу Chandelier Exit

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Уровень стоп-лосса
        """
        # Если нет данных, используем скользящий стоп-лосс
        if data is None or len(data) < self.config.chandelier_period:
            return self._calculate_trailing_stop(price, direction)

        # Вычисляем ATR за указанный период
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-self.config.chandelier_period:])
        
        # Находим максимум/минимум за период
        period_high = np.max(high[-self.config.chandelier_period:])
        period_low = np.min(low[-self.config.chandelier_period:])
        
        # Вычисляем уровень стоп-лосса
        multiplier = self.config.chandelier_multiplier
        
        if direction == "long":
            return period_high - (atr * multiplier)
        else:  # short
            return period_low + (atr * multiplier)

    def _calculate_parabolic_stop(
        self, price: float, direction: str, data: Optional[pd.DataFrame]
    ) -> float:
        """
        Вычисляет стоп-лосс на основе Parabolic SAR

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Уровень стоп-лосса
        """
        # Если нет данных, используем скользящий стоп-лосс
        if data is None or len(data) < 10:  # Минимальное количество данных для SAR
            return self._calculate_trailing_stop(price, direction)

        # Упрощенный расчет Parabolic SAR
        af = self.config.parabolic_step  # Acceleration Factor
        max_af = self.config.parabolic_max_step  # Maximum AF
        
        high = data["high"].values
        low = data["low"].values
        
        if direction == "long":
            # Для лонга начальный SAR - минимум за период
            sar = np.min(low[-10:])
            ep = np.max(high[-10:])  # Extreme Point
            
            # Обновляем SAR по формуле
            sar = sar + af * (ep - sar)
            
            # Увеличиваем AF, если достигнут новый максимум
            if price > ep:
                af = min(af + self.config.parabolic_step, max_af)
                ep = price
        else:  # short
            # Для шорта начальный SAR - максимум за период
            sar = np.max(high[-10:])
            ep = np.min(low[-10:])  # Extreme Point
            
            # Обновляем SAR по формуле
            sar = sar - af * (sar - ep)
            
            # Увеличиваем AF, если достигнут новый минимум
            if price < ep:
                af = min(af + self.config.parabolic_step, max_af)
                ep = price
        
        return sar

    def _calculate_bollinger_stop(
        self, price: float, direction: str, data: Optional[pd.DataFrame]
    ) -> float:
        """
        Вычисляет стоп-лосс на основе полос Боллинджера

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Уровень стоп-лосса
        """
        # Если нет данных, используем фиксированный стоп-лосс
        if data is None or len(data) < self.config.bollinger_period:
            return self._calculate_fixed_stop(price, direction)

        # Вычисляем среднее и стандартное отклонение цен закрытия
        close = data["close"].values
        sma = np.mean(close[-self.config.bollinger_period:])
        std = np.std(close[-self.config.bollinger_period:], ddof=1)
        
        # Вычисляем верхнюю и нижнюю полосы Боллинджера
        upper_band = sma + (std * self.config.bollinger_std)
        lower_band = sma - (std * self.config.bollinger_std)
        
        # Для лонга используем нижнюю полосу, для шорта - верхнюю
        if direction == "long":
            return lower_band
        else:  # short
            return upper_band

    def _calculate_support_resistance_stop(
        self, price: float, direction: str, data: Optional[pd.DataFrame]
    ) -> float:
        """
        Вычисляет стоп-лосс на основе уровней поддержки и сопротивления

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Уровень стоп-лосса
        """
        # Если нет данных, используем фиксированный стоп-лосс
        if data is None or len(data) < 50:  # Нужно достаточно данных для определения уровней
            return self._calculate_fixed_stop(price, direction)

        # Простой алгоритм поиска уровней поддержки и сопротивления
        # На основе локальных максимумов и минимумов
        high = data["high"].values
        low = data["low"].values
        
        # Ищем локальные максимумы и минимумы
        # (упрощенная версия - можно использовать более сложные алгоритмы)
        window = 5  # Размер окна для поиска локальных экстремумов
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(high) - window):
            # Локальный максимум
            if all(high[i] > high[i-j] for j in range(1, window+1)) and all(high[i] > high[i+j] for j in range(1, window+1)):
                resistance_levels.append(high[i])
            
            # Локальный минимум
            if all(low[i] < low[i-j] for j in range(1, window+1)) and all(low[i] < low[i+j] for j in range(1, window+1)):
                support_levels.append(low[i])
        
        # Если не найдены уровни, используем фиксированный стоп-лосс
        if not resistance_levels or not support_levels:
            return self._calculate_fixed_stop(price, direction)
        
        # Находим ближайший уровень поддержки или сопротивления
        if direction == "long":
            # Ищем ближайший уровень поддержки ниже текущей цены
            valid_supports = [s for s in support_levels if s < price]
            if valid_supports:
                return max(valid_supports)
            else:
                return price * (1 - self.config.value)  # Используем фиксированный стоп-лосс, если нет подходящего уровня
        else:  # short
            # Ищем ближайший уровень сопротивления выше текущей цены
            valid_resistances = [r for r in resistance_levels if r > price]
            if valid_resistances:
                return min(valid_resistances)
            else:
                return price * (1 + self.config.value)  # Используем фиксированный стоп-лосс, если нет подходящего уровня

    def _calculate_volatility_stop(
        self, price: float, direction: str, data: Optional[pd.DataFrame]
    ) -> float:
        """
        Вычисляет стоп-лосс на основе волатильности рынка

        Args:
            price: Текущая цена
            direction: Направление позиции ("long" или "short")
            data: DataFrame с историческими данными (OHLC)

        Returns:
            float: Уровень стоп-лосса
        """
        # Если нет данных, используем фиксированный стоп-лосс
        if data is None or len(data) < self.config.volatility_window:
            return self._calculate_fixed_stop(price, direction)

        # Вычисляем волатильность как стандартное отклонение процентных изменений
        close = data["close"].values
        pct_changes = np.diff(close) / close[:-1]
        volatility = np.std(pct_changes[-self.config.volatility_window:], ddof=1)
        
        # Умножаем на множитель, чтобы получить размер стоп-лосса
        stop_size = volatility * self.config.volatility_multiplier
        
        # Вычисляем уровень стоп-лосса
        if direction == "long":
            return price * (1 - stop_size)
        else:  # short
            return price * (1 + stop_size)

    def get_info(self) -> Dict:
        """
        Возвращает информацию о текущем состоянии стоп-лосса

        Returns:
            Dict: Словарь с информацией о стоп-лоссе
        """
        return {
            "type": self.config.type.value,
            "is_active": self.is_active,
            "direction": self.direction,
            "initial_price": self.initial_price,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "stop_level": self.stop_level
        }


class DynamicStopLossManager:
    """Менеджер для управления несколькими стоп-лоссами"""

    def __init__(self):
        """Инициализирует менеджер стоп-лоссов"""
        self.stop_losses: Dict[str, DynamicStopLoss] = {}

    def add_stop_loss(self, position_id: str, config: StopLossConfig) -> DynamicStopLoss:
        """
        Добавляет стоп-лосс для указанной позиции

        Args:
            position_id: ID позиции
            config: Конфигурация стоп-лосса

        Returns:
            DynamicStopLoss: Объект стоп-лосса
        """
        stop_loss = DynamicStopLoss(config)
        self.stop_losses[position_id] = stop_loss
        return stop_loss

    def activate_stop_loss(
        self, position_id: str, price: float, direction: str
    ) -> Optional[float]:
        """
        Активирует стоп-лосс для указанной позиции

        Args:
            position_id: ID позиции
            price: Цена входа
            direction: Направление позиции ("long" или "short")

        Returns:
            Optional[float]: Уровень стоп-лосса или None, если стоп-лосс не найден
        """
        if position_id in self.stop_losses:
            return self.stop_losses[position_id].activate(price, direction)
        return None

    def update_stop_loss(
        self, position_id: str, price: float, data: Optional[pd.DataFrame] = None
    ) -> Optional[float]:
        """
        Обновляет стоп-лосс для указанной позиции

        Args:
            position_id: ID позиции
            price: Текущая цена
            data: DataFrame с историческими данными (OHLC)

        Returns:
            Optional[float]: Новый уровень стоп-лосса или None, если стоп-лосс не найден
        """
        if position_id in self.stop_losses:
            return self.stop_losses[position_id].update(price, data)
        return None

    def check_stop_loss(self, position_id: str, price: float) -> bool:
        """
        Проверяет, сработал ли стоп-лосс для указанной позиции

        Args:
            position_id: ID позиции
            price: Текущая цена

        Returns:
            bool: True, если стоп-лосс сработал, иначе False
        """
        if position_id in self.stop_losses:
            return self.stop_losses[position_id].is_triggered(price)
        return False

    def remove_stop_loss(self, position_id: str) -> bool:
        """
        Удаляет стоп-лосс для указанной позиции

        Args:
            position_id: ID позиции

        Returns:
            bool: True, если стоп-лосс был удален, иначе False
        """
        if position_id in self.stop_losses:
            self.stop_losses[position_id].deactivate()
            del self.stop_losses[position_id]
            return True
        return False

    def get_stop_loss(self, position_id: str) -> Optional[DynamicStopLoss]:
        """
        Возвращает объект стоп-лосса для указанной позиции

        Args:
            position_id: ID позиции

        Returns:
            Optional[DynamicStopLoss]: Объект стоп-лосса или None, если не найден
        """
        return self.stop_losses.get(position_id)

    def get_all_stop_losses(self) -> Dict[str, Dict]:
        """
        Возвращает информацию о всех стоп-лоссах

        Returns:
            Dict[str, Dict]: Словарь с информацией о всех стоп-лоссах
        """
        return {
            position_id: stop_loss.get_info()
            for position_id, stop_loss in self.stop_losses.items()
        }


# Создаем глобальный экземпляр менеджера стоп-лоссов
_stop_loss_manager = None


def get_stop_loss_manager() -> DynamicStopLossManager:
    """
    Возвращает глобальный экземпляр менеджера стоп-лоссов

    Returns:
        DynamicStopLossManager: Экземпляр менеджера стоп-лоссов
    """
    global _stop_loss_manager
    
    if _stop_loss_manager is None:
        _stop_loss_manager = DynamicStopLossManager()
    
    return _stop_loss_manager


def create_stop_loss(config: Union[Dict, StopLossConfig]) -> DynamicStopLoss:
    """
    Создает объект стоп-лосса с указанной конфигурацией

    Args:
        config: Конфигурация стоп-лосса (словарь или объект StopLossConfig)

    Returns:
        DynamicStopLoss: Объект стоп-лосса
    """
    if isinstance(config, dict):
        # Преобразуем словарь в объект StopLossConfig
        stop_type = StopLossType(config.get("type", "fixed"))
        
        stop_config = StopLossConfig(
            type=stop_type,
            value=config.get("value", 0.02),
            trailing_value=config.get("trailing_value"),
            atr_period=config.get("atr_period", 14),
            atr_multiplier=config.get("atr_multiplier", 3.0),
            bollinger_period=config.get("bollinger_period", 20),
            bollinger_std=config.get("bollinger_std", 2.0),
            parabolic_step=config.get("parabolic_step", 0.02),
            parabolic_max_step=config.get("parabolic_max_step", 0.2),
            volatility_window=config.get("volatility_window", 20),
            volatility_multiplier=config.get("volatility_multiplier", 2.0),
            chandelier_period=config.get("chandelier_period", 22),
            chandelier_multiplier=config.get("chandelier_multiplier", 3.0),
            custom_func=config.get("custom_func")
        )
    else:
        stop_config = config
    
    return DynamicStopLoss(stop_config)
