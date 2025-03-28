from datetime import datetime
from typing import Dict, Tuple

import numpy as np
from project.utils.logging_utils import setup_logger

logger = setup_logger("position_sizer")


class PositionSizer:
    """Класс для управления размером позиции"""

    def __init__(self, config: Dict = None):
        """
        Инициализирует менеджер размера позиций

        Args:
            config: Конфигурация
        """
        self.config = config or {}

        # Базовый процент от капитала
        self.base_position_size = self.config.get(
            "base_position_size", 0.02
        )  # 2% по умолчанию

        # Минимальный и максимальный размер позиции (% от капитала)
        self.min_position_size = self.config.get("min_position_size", 0.01)
        self.max_position_size = self.config.get("max_position_size", 0.1)

        # Адаптивное управление размером (на основе производительности)
        self.adaptive_sizing = self.config.get("adaptive_sizing", True)

        # Параметры кривой адаптации
        self.win_multiplier = self.config.get(
            "win_multiplier", 1.1
        )  # Увеличение на 10% после выигрыша
        self.loss_multiplier = self.config.get(
            "loss_multiplier", 0.9
        )  # Уменьшение на 10% после проигрыша

        # Адаптация размера на основе волатильности
        self.volatility_sizing = self.config.get("volatility_sizing", False)
        self.volatility_lookback = self.config.get("volatility_lookback", 20)
        self.volatility_factor = self.config.get("volatility_factor", 1.0)

        # Адаптация на основе силы сигнала
        self.signal_sizing = self.config.get("signal_sizing", False)
        self.signal_multiplier = self.config.get("signal_multiplier", 1.0)

        # Управление риском
        self.risk_per_trade = self.config.get(
            "risk_per_trade", 0.01
        )  # 1% риска на сделку
        self.max_risk_multiplier = self.config.get("max_risk_multiplier", 2.0)

        # Мартингейл/Антимартингейл
        self.martingale = self.config.get("martingale", False)
        self.martingale_factor = self.config.get("martingale_factor", 2.0)
        self.max_martingale_level = self.config.get("max_martingale_level", 3)

        # Текущий мультипликатор размера позиции
        self.current_position_multiplier = 1.0

        # Текущий уровень мартингейла
        self.current_martingale_level = 0

        # История сделок и капитала для адаптации
        self.trade_history = []
        self.equity_history = []

        logger.info("PositionSizer initialized with adaptive sizing capabilities")

    def calculate_position_size(
        self,
        balance: float,
        price: float,
        signal_strength: float = 1.0,
        volatility: float = None,
        stop_loss_pct: float = None,
    ) -> Tuple[float, float]:
        """
        Рассчитывает размер позиции

        Args:
            balance: Текущий баланс
            price: Текущая цена
            signal_strength: Сила сигнала (от 0 до 1)
            volatility: Волатильность (например, ATR или стандартное отклонение)
            stop_loss_pct: Процент стоп-лосса (если используется риск-менеджмент)

        Returns:
            Tuple[float, float]: (Размер позиции в единицах актива, процент от баланса)
        """
        # Начинаем с базового размера
        position_size_pct = self.base_position_size

        # Применяем текущий мультипликатор
        position_size_pct *= self.current_position_multiplier

        # Адаптируем размер позиции на основе силы сигнала, если включено
        if self.signal_sizing and signal_strength is not None:
            position_size_pct *= 1.0 + (signal_strength - 0.5) * self.signal_multiplier

        # Адаптируем размер позиции на основе волатильности, если включено
        if self.volatility_sizing and volatility is not None:
            # Если волатильность выше среднего, уменьшаем размер позиции
            # Если ниже, увеличиваем
            if hasattr(self, "avg_volatility") and self.avg_volatility > 0:
                volatility_ratio = (
                    self.avg_volatility / volatility if volatility > 0 else 1.0
                )
                position_size_pct *= volatility_ratio * self.volatility_factor

        # Применяем мартингейл, если включен
        if self.martingale and self.current_martingale_level > 0:
            martingale_multiplier = (
                self.martingale_factor**self.current_martingale_level
            )
            position_size_pct *= martingale_multiplier

        # Рассчитываем размер позиции на основе риска, если указан стоп-лосс
        if stop_loss_pct is not None and stop_loss_pct > 0:
            risk_amount = balance * self.risk_per_trade
            max_loss_amount = balance * position_size_pct * stop_loss_pct

            if max_loss_amount > 0:
                risk_multiplier = risk_amount / max_loss_amount
                position_size_pct *= min(risk_multiplier, self.max_risk_multiplier)

        # Ограничиваем размер позиции
        position_size_pct = max(
            min(position_size_pct, self.max_position_size), self.min_position_size
        )

        # Рассчитываем размер позиции в единицах актива
        position_value = balance * position_size_pct
        position_size = position_value / price if price > 0 else 0

        logger.debug(
            f"Calculated position size: {position_size} ({position_size_pct * 100:.2f}% of balance)"
        )

        return position_size, position_size_pct

    def update_after_trade(self, trade_result: Dict):
        """
        Обновляет параметры после сделки

        Args:
            trade_result: Результат сделки
        """
        # Добавляем сделку в историю
        self.trade_history.append(trade_result)

        # Ограничиваем историю сделок
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        # Проверяем результат сделки
        is_profit = trade_result.get("realized_pnl", 0) > 0

        # Обновляем мультипликатор на основе результата сделки
        if self.adaptive_sizing:
            if is_profit:
                self.current_position_multiplier *= self.win_multiplier
                self.current_martingale_level = (
                    0  # Сбрасываем уровень мартингейла при выигрыше
                )
            else:
                self.current_position_multiplier *= self.loss_multiplier

                # Увеличиваем уровень мартингейла при проигрыше
                if self.martingale:
                    self.current_martingale_level = min(
                        self.current_martingale_level + 1, self.max_martingale_level
                    )

            # Ограничиваем мультипликатор
            max_multiplier = self.max_position_size / self.base_position_size
            min_multiplier = self.min_position_size / self.base_position_size
            self.current_position_multiplier = max(
                min(self.current_position_multiplier, max_multiplier), min_multiplier
            )

        logger.info(
            f"Position multiplier updated: {
                self.current_position_multiplier:.2f}, Martingale level: {
                self.current_martingale_level}")

    def update_equity(self, equity: float, timestamp: datetime = None):
        """
        Обновляет историю капитала

        Args:
            equity: Текущий капитал
            timestamp: Временная метка
        """
        # Создаем запись
        entry = {"equity": equity, "timestamp": timestamp or datetime.now()}

        # Добавляем запись в историю
        self.equity_history.append(entry)

        # Ограничиваем историю
        if len(self.equity_history) > 1000:
            self.equity_history = self.equity_history[-1000:]

    def update_volatility(self, volatility: float):
        """
        Обновляет текущую и среднюю волатильность

        Args:
            volatility: Текущая волатильность
        """
        # Если атрибут не существует, создаем его
        if not hasattr(self, "volatility_history"):
            self.volatility_history = []

        # Добавляем значение в историю
        self.volatility_history.append(volatility)

        # Ограничиваем историю
        if len(self.volatility_history) > self.volatility_lookback:
            self.volatility_history = self.volatility_history[
                -self.volatility_lookback:
            ]

        # Рассчитываем среднюю волатильность
        self.avg_volatility = np.mean(self.volatility_history)

    def calculate_kelly_criterion(self) -> float:
        """
        Рассчитывает оптимальную долю капитала по критерию Келли

        Returns:
            float: Оптимальная доля капитала
        """
        # Для расчета критерия Келли нам нужны:
        # - Вероятность выигрыша
        # - Отношение среднего выигрыша к среднему проигрышу

        # Если история сделок пуста, возвращаем базовый размер позиции
        if not self.trade_history:
            return self.base_position_size

        # Рассчитываем количество выигрышных и проигрышных сделок
        win_trades = [
            trade for trade in self.trade_history if trade.get("realized_pnl", 0) > 0
        ]
        loss_trades = [
            trade for trade in self.trade_history if trade.get("realized_pnl", 0) <= 0
        ]

        win_count = len(win_trades)
        loss_count = len(loss_trades)
        total_count = win_count + loss_count

        # Рассчитываем вероятность выигрыша
        win_probability = win_count / total_count if total_count > 0 else 0.5

        # Рассчитываем среднюю прибыль и убыток
        avg_win = (
            np.mean([trade.get("realized_pnl", 0) for trade in win_trades])
            if win_trades
            else 0
        )
        avg_loss = (
            abs(np.mean([trade.get("realized_pnl", 0) for trade in loss_trades]))
            if loss_trades
            else 0
        )

        # Рассчитываем отношение выигрыша к проигрышу
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Формула Келли: f* = p - (1-p)/r, где p - вероятность выигрыша, r -
        # отношение выигрыша к проигрышу
        kelly_pct = win_probability - (1 - win_probability) / win_loss_ratio

        # Ограничиваем размер позиции
        kelly_pct = max(min(kelly_pct, self.max_position_size), self.min_position_size)

        logger.debug(
            f"Calculated Kelly criterion: {
                kelly_pct *
                100:.2f}% (Win prob: {
                win_probability:.2f}, Win/Loss ratio: {
                win_loss_ratio:.2f})")

        return kelly_pct

    def calculate_optimal_f(self) -> float:
        """
        Рассчитывает оптимальную долю капитала по методу optimal f

        Returns:
            float: Оптимальная доля капитала
        """
        # Если история сделок пуста, возвращаем базовый размер позиции
        if not self.trade_history:
            return self.base_position_size

        # Получаем список результатов сделок в процентах от капитала
        returns = [trade.get("realized_pnl_pct", 0) for trade in self.trade_history]

        # Находим наибольший проигрыш
        worst_loss_pct = abs(min(returns)) if returns else 0

        # Если наибольший проигрыш равен 0, возвращаем базовый размер позиции
        if worst_loss_pct == 0:
            return self.base_position_size

        # Рассчитываем optimal f: f* = 1 / worst_loss_pct
        optimal_f = (
            0.5 / worst_loss_pct
        )  # Используем половину optimal f для безопасности

        # Ограничиваем размер позиции
        optimal_f = max(min(optimal_f, self.max_position_size), self.min_position_size)

        logger.debug(
            f"Calculated optimal f: {
                optimal_f *
                100:.2f}% (Worst loss: {
                worst_loss_pct *
                100:.2f}%)")

        return optimal_f

    def reset(self):
        """Сбрасывает параметры"""
        self.current_position_multiplier = 1.0
        self.current_martingale_level = 0
        self.trade_history = []
        self.equity_history = []
        if hasattr(self, "volatility_history"):
            self.volatility_history = []
            self.avg_volatility = 0

        logger.info("PositionSizer reset")

    def to_dict(self) -> Dict:
        """
        Преобразует состояние в словарь

        Returns:
            Dict: Состояние в виде словаря
        """
        return {
            "config": self.config,
            "current_position_multiplier": self.current_position_multiplier,
            "current_martingale_level": self.current_martingale_level,
            "trade_history": self.trade_history[
                -10:
            ],  # Сохраняем только последние 10 сделок
            "avg_volatility": getattr(self, "avg_volatility", 0),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PositionSizer":
        """
        Создает экземпляр из словаря

        Args:
            data: Данные

        Returns:
            PositionSizer: Новый экземпляр
        """
        sizer = cls(config=data.get("config", {}))
        sizer.current_position_multiplier = data.get("current_position_multiplier", 1.0)
        sizer.current_martingale_level = data.get("current_martingale_level", 0)
        sizer.trade_history = data.get("trade_history", [])

        if "avg_volatility" in data:
            sizer.avg_volatility = data["avg_volatility"]

        return sizer

    # Для обратной совместимости - поддержка старых вызовов
    def calculate_size(self, balance, risk, max_risk=None):
        """
        Обратная совместимость со старым API

        Args:
            balance: Баланс аккаунта
            risk: Риск в процентах
            max_risk: Максимальный риск

        Returns:
            float: Размер позиции
        """
        # Используем новый метод с базовыми параметрами
        self.base_position_size = risk / 100.0

        if max_risk:
            self.max_position_size = max_risk / 100.0

        # Вызываем новый метод с текущей ценой = 1, что даст размер в процентах
        size, _ = self.calculate_position_size(balance, 1.0)

        # Возвращаем процент от баланса
        return size * balance
