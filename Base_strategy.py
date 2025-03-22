"""
Базовый модуль стратегий.
Определяет абстрактный класс для всех торговых стратегий.
"""

import abc
import logging
from typing import Any, Dict, Optional, List, Set


class BaseStrategy(abc.ABC):
    """
    Абстрактный базовый класс стратегии.
    Каждая стратегия должна реализовывать метод run.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализирует стратегию.

        Args:
            name: Имя стратегии
            config: Конфигурация стратегии
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"Strategy.{name}")
        self.is_active = True
        self.required_data_keys: Set[str] = set()  # Ключи, необходимые в market_data
        self.performance_stats: Dict[str, Any] = {
            "runs": 0,
            "successful_runs": 0,
            "errors": 0,
            "signals_generated": 0
        }

    @abc.abstractmethod
    async def run(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет стратегию на основе данных рынка.

        Args:
            market_data: Словарь с данными рынка

        Returns:
            Словарь с результатами выполнения стратегии, который должен содержать:
            - "action": строка с действием ("buy", "sell", "hold")
            - "symbol": торговая пара или актив
            - "confidence": уровень уверенности от 0 до 1
            - "reason": причина сигнала
            - "parameters": параметры сигнала (цена, объем и т.д.)
        """
        pass

    def validate_input(self, market_data: Dict[str, Any]) -> bool:
        """
        Проверяет, содержит ли market_data все необходимые данные.

        Args:
            market_data: Данные рынка для проверки

        Returns:
            True, если все необходимые данные присутствуют
        """
        if not self.required_data_keys:
            return True  # Если не указаны требуемые ключи, считаем всё валидным
            
        missing_keys = self.required_data_keys - set(market_data.keys())
        if missing_keys:
            self.logger.warning(f"Missing required data keys: {missing_keys}")
            return False
        return True

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Устанавливает или обновляет конфигурацию стратегии.

        Args:
            config: Новая конфигурация
        """
        self.config.update(config)
        self.logger.debug(f"Updated configuration for {self.name}")

    def activate(self) -> None:
        """Активирует стратегию."""
        self.is_active = True
        self.logger.info(f"Strategy {self.name} activated")

    def deactivate(self) -> None:
        """Деактивирует стратегию."""
        self.is_active = False
        self.logger.info(f"Strategy {self.name} deactivated")

    async def evaluate(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Оценивает эффективность стратегии на исторических данных.

        Args:
            historical_data: Список исторических данных рынка

        Returns:
            Метрики эффективности стратегии
        """
        results = []
        signals = {"buy": 0, "sell": 0, "hold": 0}
        
        for data_point in historical_data:
            if not self.validate_input(data_point):
                continue
                
            try:
                result = await self.run(data_point)
                results.append(result)
                
                # Подсчитываем сигналы по типам
                action = result.get("action", "unknown")
                signals[action] = signals.get(action, 0) + 1
                
            except Exception as e:
                self.logger.error(f"Error evaluating strategy: {e}")
        
        # Рассчитываем статистику
        return {
            "strategy": self.name,
            "runs": len(historical_data),
            "successful_runs": len(results),
            "success_rate": len(results) / len(historical_data) if historical_data else 0,
            "signals": signals
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику производительности стратегии.
        
        Returns:
            Словарь со статистикой
        """
        return {
            "name": self.name,
            "is_active": self.is_active,
            **self.performance_stats
        }
    
    def reset_stats(self) -> None:
        """Сбрасывает статистику производительности."""
        self.performance_stats = {
            "runs": 0,
            "successful_runs": 0,
            "errors": 0,
            "signals_generated": 0
        }
        self.logger.debug(f"Reset performance stats for {self.name}")
