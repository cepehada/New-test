"""
Модуль с модификациями и улучшениями для стратегий.
Предоставляет функции для настройки и оптимизации стратегий.
"""

from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
from project.technicals.indicators import Indicators
from project.utils.error_handler import handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class StrategyModifications:
    """
    Класс с модификациями и улучшениями для стратегий.
    """

    @staticmethod
    @handle_error
    def add_trailing_stop(
        current_price: float,
        entry_price: float,
        highest_price: float,
        side: str,
        initial_stop_pct: float = 0.02,
        trailing_pct: float = 0.01,
    ) -> float:
        """
        Рассчитывает цену трейлинг-стопа.

        Args:
            current_price: Текущая цена
            entry_price: Цена входа
            highest_price: Наивысшая достигнутая цена для long, наименьшая для short
            side: Сторона позиции (long или short)
            initial_stop_pct: Начальный процент стоп-лосса
            trailing_pct: Процент трейлинга

        Returns:
            Цена трейлинг-стопа
        """
        # Для длинной позиции
        if side.lower() == "long":
            # Начальный стоп-лосс
            initial_stop = entry_price * (1 - initial_stop_pct)

            # Трейлинг-стоп
            trailing_stop = highest_price * (1 - trailing_pct)

            # Возвращаем максимальный из двух стопов
            return max(initial_stop, trailing_stop)

        # Для короткой позиции
        elif side.lower() == "short":
            # Начальный стоп-лосс
            initial_stop = entry_price * (1 + initial_stop_pct)

            # Трейлинг-стоп
            trailing_stop = highest_price * (1 + trailing_pct)

            # Возвращаем минимальный из двух стопов
            return min(initial_stop, trailing_stop)

        else:
            logger.error("Неизвестная сторона позиции: {side}" %)
            return entry_price

    @staticmethod
    @handle_error
    def add_partial_exits(
        entry_price: float,
        side: str,
        take_profit_levels: List[float] = None,
        exit_percentages: List[float] = None,
    ) -> List[Tuple[float, float]]:
        """
        Рассчитывает уровни для частичного закрытия позиции.

        Args:
            entry_price: Цена входа
            side: Сторона позиции (long или short)
            take_profit_levels: Уровни тейк-профита (в процентах)
            exit_percentages: Проценты закрытия позиции на каждом уровне

        Returns:
            Список кортежей (цена выхода, процент закрытия)
        """
        if take_profit_levels is None:
            take_profit_levels = [0.01, 0.02, 0.03]
        if exit_percentages is None:
            exit_percentages = [0.33, 0.33, 0.34]

        if len(take_profit_levels) != len(exit_percentages):
            logger.error(
                "Количество уровней тейк-профита должно совпадать с количеством процентов закрытия"
            )
            return []

        # Проверяем, что проценты закрытия в сумме дают 1 (или близко к 1)
        if not 0.99 <= sum(exit_percentages) <= 1.01:
            logger.warning(
                f"Сумма процентов закрытия ({sum(exit_percentages)}) не равна 1"
            )

        result = []

        # Для длинной позиции
        if side.lower() == "long":
            for level, pct in zip(take_profit_levels, exit_percentages):
                exit_price = entry_price * (1 + level)
                result.append((exit_price, pct))

        # Для короткой позиции
        elif side.lower() == "short":
            for level, pct in zip(take_profit_levels, exit_percentages):
                exit_price = entry_price * (1 - level)
                result.append((exit_price, pct))

        else:
            logger.error("Неизвестная сторона позиции: {side}" %)

        return result

    @staticmethod
    @handle_error
    def dynamic_position_sizing(
        account_balance: float,
        risk_per_trade: float,
        stop_loss_pct: float,
        win_rate: float,
        recent_trades: List[float],
        max_drawdown: float = 0.0,
    ) -> float:
        """
        Рассчитывает размер позиции с учетом динамических параметров.

        Args:
            account_balance: Баланс счета
            risk_per_trade: Риск на одну сделку (доля от баланса)
            stop_loss_pct: Процент стоп-лосса
            win_rate: Доля выигрышных сделок
            recent_trades: Результаты последних сделок (в процентах)
            max_drawdown: Текущая просадка (в процентах)

        Returns:
            Размер позиции (доля от баланса)
        """
        # Базовый размер позиции
        position_size = risk_per_trade

        # Корректируем размер позиции в зависимости от винрейта
        # Если винрейт выше 50%, увеличиваем размер; иначе уменьшаем
        if win_rate >= 0.5:
            position_size *= 1 + (win_rate - 0.5)
        else:
            position_size *= 1 - (0.5 - win_rate) * 2

        # Корректируем размер позиции в зависимости от последних сделок
        # Если последние сделки успешные, увеличиваем размер; иначе уменьшаем
        if recent_trades:
            avg_recent_trade = sum(recent_trades) / len(recent_trades)
            if avg_recent_trade > 0:
                position_size *= 1 + min(avg_recent_trade * 0.5, 0.5)
            else:
                position_size *= 1 + max(avg_recent_trade, -0.5)

        # Корректируем размер позиции в зависимости от просадки
        # Если просадка большая, уменьшаем размер
        if max_drawdown > 0:
            position_size *= 1 - min(max_drawdown, 0.5)

        # Ограничиваем размер позиции
        position_size = max(
            position_size, risk_per_trade * 0.5
        )  # не меньше 50% от базового
        position_size = min(
            position_size, risk_per_trade * 2.0
        )  # не больше 200% от базового

        return position_size

    @staticmethod
    @handle_error
    def adaptive_parameters(
        data: pd.DataFrame, base_params: Dict[str, Any], volatility_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Адаптирует параметры стратегии в зависимости от рыночных условий.

        Args:
            data: DataFrame с данными OHLCV
            base_params: Базовые параметры стратегии
            volatility_factor: Фактор для расчета волатильности

        Returns:
            Адаптированные параметры стратегии
        """
        if data.empty:
            return base_params

        # Копируем базовые параметры
        adapted_params = base_params.copy()

        # Рассчитываем волатильность
        atr = Indicators.average_true_range(data, 14)
        current_atr = atr.iloc[-1] if not atr.empty else 0

        # Рассчитываем среднюю цену
        average_price = data["close"].mean()

        # Рассчитываем относительную волатильность
        relative_volatility = current_atr / average_price if average_price > 0 else 0

        # Адаптируем параметры в зависимости от волатильности
        if "stop_loss_pct" in adapted_params:
            adapted_params["stop_loss_pct"] = max(
                base_params["stop_loss_pct"], relative_volatility * volatility_factor
            )

        if "take_profit_pct" in adapted_params:
            adapted_params["take_profit_pct"] = max(
                base_params["take_profit_pct"],
                relative_volatility * volatility_factor * 2,
            )

        if "rsi_oversold" in adapted_params and "rsi_overbought" in adapted_params:
            # Для высокой волатильности расширяем диапазон RSI
            volatility_adjustment = relative_volatility * 10 * volatility_factor
            adapted_params["rsi_oversold"] = max(
                10, base_params["rsi_oversold"] - volatility_adjustment
            )
            adapted_params["rsi_overbought"] = min(
                90, base_params["rsi_overbought"] + volatility_adjustment
            )

        return adapted_params

    @staticmethod
    @handle_error
    def calculate_risk_reward_adjustment(
        win_rate: float, risk_reward_ratio: float
    ) -> float:
        """
        Рассчитывает оптимальное соотношение риск/прибыль на основе винрейта.

        Args:
            win_rate: Доля выигрышных сделок
            risk_reward_ratio: Текущее соотношение риск/прибыль

        Returns:
            Оптимальное соотношение риск/прибыль
        """
        if win_rate <= 0 or win_rate >= 1:
            logger.warning("Некорректный винрейт: {win_rate}" %)
            return risk_reward_ratio

        # Оптимальное соотношение риск/прибыль = (1 - win_rate) / win_rate
        optimal_ratio = (1 - win_rate) / win_rate

        # Ограничиваем изменение соотношения
        if optimal_ratio < risk_reward_ratio * 0.5:
            return risk_reward_ratio * 0.5
        elif optimal_ratio > risk_reward_ratio * 2:
            return risk_reward_ratio * 2
        else:
            return optimal_ratio

    @staticmethod
    @handle_error
    def generate_adaptive_signal_thresholds(
        symbol: str, data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Генерирует адаптивные пороги для сигналов в зависимости от рыночных условий.

        Args:
            symbol: Символ для торговли
            data: DataFrame с данными OHLCV

        Returns:
            Словарь с порогами для разных индикаторов
        """
        if data.empty:
            logger.warning("Пустой DataFrame для {symbol}" %)
            return {
                "rsi_oversold": 30.0,
                "rsi_overbought": 70.0,
                "macd_threshold": 0.0,
                "bollinger_threshold": 0.0,
            }

        # Рассчитываем волатильность
        atr = Indicators.average_true_range(data, 14)
        current_atr = atr.iloc[-1] if not atr.empty else 0

        # Рассчитываем среднюю цену
        average_price = data["close"].mean()

        # Рассчитываем относительную волатильность
        relative_volatility = current_atr / average_price if average_price > 0 else 0

        # Рассчитываем пороги для RSI
        # При высокой волатильности расширяем диапазон
        volatility_adjustment = relative_volatility * 20
        rsi_oversold = max(20, 30 - volatility_adjustment)
        rsi_overbought = min(80, 70 + volatility_adjustment)

        # Рассчитываем порог для MACD
        # При высокой волатильности увеличиваем порог
        macd_threshold = relative_volatility * average_price * 0.001

        # Рассчитываем порог для Bollinger Bands
        # При высокой волатильности увеличиваем порог
        bollinger_threshold = relative_volatility * 0.5

        return {
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "macd_threshold": macd_threshold,
            "bollinger_threshold": bollinger_threshold,
        }

    @staticmethod
    @handle_error
    def calculate_weighted_signals(
        signals: Dict[str, str], weights: Dict[str, float]
    ) -> str:
        """
        Рассчитывает взвешенный сигнал на основе нескольких индикаторов.

        Args:
            signals: Словарь с сигналами от разных индикаторов
            weights: Словарь с весами для каждого индикатора

        Returns:
            Итоговый сигнал (buy, sell, hold)
        """
        if not signals or not weights:
            return "hold"

        # Подсчитываем взвешенные голоса
        buy_score = 0.0
        sell_score = 0.0

        for indicator, signal in signals.items():
            weight = weights.get(indicator, 1.0)

            if signal == "buy":
                buy_score += weight
            elif signal == "sell":
                sell_score += weight

        # Определяем итоговый сигнал
        if buy_score > sell_score and buy_score >= 0.5 * sum(weights.values()):
            return "buy"
        elif sell_score > buy_score and sell_score >= 0.5 * sum(weights.values()):
            return "sell"
        else:
            return "hold"

    @staticmethod
    @handle_error
    def generate_trade_probability(signals: Dict[str, Dict[str, Any]]) -> float:
        """
        Генерирует вероятность успешной сделки на основе сигналов.

        Args:
            signals: Словарь с сигналами от разных таймфреймов и индикаторов

        Returns:
            Вероятность успешной сделки (0.0 - 1.0)
        """
        if not signals:
            return 0.0

        # Подсчитываем общее количество сигналов и согласованных сигналов
        total_signals = 0
        agreeing_signals = 0

        # Определяем доминирующий сигнал
        buy_count = 0
        sell_count = 0

        for timeframe, timeframe_signals in signals.items():
            for indicator, signal in timeframe_signals.items():
                total_signals += 1

                if signal == "buy":
                    buy_count += 1
                elif signal == "sell":
                    sell_count += 1

        dominant_signal = (
            "buy"
            if buy_count > sell_count
            else "sell" if sell_count > buy_count else "hold"
        )

        if dominant_signal == "hold":
            return 0.0

        # Подсчитываем количество сигналов, согласующихся с доминирующим
        for timeframe, timeframe_signals in signals.items():
            for indicator, signal in timeframe_signals.items():
                if signal == dominant_signal:
                    agreeing_signals += 1

        # Рассчитываем вероятность
        if total_signals > 0:
            return agreeing_signals / total_signals
        else:
            return 0.0

    @staticmethod
    @handle_error
    def apply_volume_filter(
        signal: str, volume: float, average_volume: float, min_volume_ratio: float = 1.0
    ) -> str:
        """
        Применяет фильтр объема к сигналу.

        Args:
            signal: Исходный сигнал (buy, sell, hold)
            volume: Текущий объем
            average_volume: Средний объем
            min_volume_ratio: Минимальное отношение текущего объема к среднему

        Returns:
            Отфильтрованный сигнал
        """
        if signal == "hold" or average_volume <= 0:
            return signal

        # Рассчитываем отношение текущего объема к среднему
        volume_ratio = volume / average_volume

        # Если объем слишком маленький, изменяем сигнал на "hold"
        if volume_ratio < min_volume_ratio:
            return "hold"

        return signal

    @staticmethod
    @handle_error
    def apply_time_filter(
        signal: str, hour: int, allowed_hours: List[int] = None
    ) -> str:
        """
        Применяет фильтр времени к сигналу.

        Args:
            signal: Исходный сигнал (buy, sell, hold)
            hour: Текущий час (0-23)
            allowed_hours: Список разрешенных часов (None для всех часов)

        Returns:
            Отфильтрованный сигнал
        """
        if signal == "hold" or allowed_hours is None:
            return signal

        # Если текущий час не в списке разрешенных, изменяем сигнал на "hold"
        if hour not in allowed_hours:
            return "hold"

        return signal

    @staticmethod
    @handle_error
    def apply_filters(signal: str, filters: List[Callable[[str], str]]) -> str:
        """
        Применяет несколько фильтров к сигналу.

        Args:
            signal: Исходный сигнал (buy, sell, hold)
            filters: Список функций-фильтров

        Returns:
            Отфильтрованный сигнал
        """
        result = signal

        for filter_func in filters:
            result = filter_func(result)

            # Если сигнал изменился на "hold", прекращаем применение фильтров
            if result == "hold":
                break

        return result
