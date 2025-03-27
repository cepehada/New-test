from typing import Dict, Any, List

class StrategyManager:
    """Модуль для выбора и управления стратегиями"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = self._load_strategies()

    def _load_strategies(self) -> List[str]:
        """Загружает доступные стратегии из конфигурации"""
        return self.config.get("strategies", ["scalping", "trend_following", "arbitrage"])

    def select_best_strategy(self, market_data: Dict[str, Any]) -> str:
        """
        Выбирает лучшую стратегию на основе рыночных данных.

        Args:
            market_data: Данные о рынке.

        Returns:
            str: Название выбранной стратегии.
        """
        # Пример логики выбора стратегии
        if market_data["volatility"] > 0.05:
            return "scalping"
        elif market_data["trend"] == "uptrend":
            return "trend_following"
        else:
            return "arbitrage"
