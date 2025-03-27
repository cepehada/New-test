from typing import Dict, Any

class CapitalManager:
    """Модуль для управления капиталом"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_risk_per_trade = config.get("max_risk_per_trade", 0.02)  # 2% риска на сделку
        self.min_balance_threshold = config.get("min_balance_threshold", 100)  # Минимальный баланс

    def calculate_allocation(self, balance: float, volatility: float, slippage: float) -> float:
        """
        Рассчитывает сумму для торговли на основе баланса, волатильности и слиппейджа.

        Args:
            balance: Текущий баланс.
            volatility: Волатильность рынка.
            slippage: Слиппейдж.

        Returns:
            float: Рекомендуемая сумма для торговли.
        """
        # Учитываем риск и волатильность
        risk_adjusted_balance = balance * self.max_risk_per_trade
        volatility_factor = max(0.1, 1 - volatility)  # Чем выше волатильность, тем меньше сумма
        slippage_factor = max(0.1, 1 - slippage)  # Чем выше слиппейдж, тем меньше сумма

        allocation = risk_adjusted_balance * volatility_factor * slippage_factor

        # Убедимся, что сумма не меньше минимального порога
        return max(allocation, self.min_balance_threshold)
