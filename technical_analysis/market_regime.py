"""
Модуль для определения рыночных режимов и адаптации стратегий
к текущим условиям рынка без использования машинного обучения.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from project.utils.logging_utils import setup_logger

logger = setup_logger("market_regime")


class MarketRegime(Enum):
    """Типы рыночных режимов"""
    TRENDING_UP = "trending_up"          # Восходящий тренд
    TRENDING_DOWN = "trending_down"      # Нисходящий тренд
    RANGING = "ranging"                  # Боковой рынок
    VOLATILE = "volatile"                # Высокая волатильность
    LOW_VOLATILITY = "low_volatility"    # Низкая волатильность
    BREAKOUT = "breakout"                # Пробой уровня
    REVERSAL = "reversal"                # Разворот тренда
    UNKNOWN = "unknown"                  # Неопределенный режим


class MarketRegimeDetector:
    """Детектор рыночных режимов на основе технических индикаторов"""

    def __init__(self, lookback_period: int = 20, volatility_window: int = 20):
        """
        Инициализирует детектор рыночных режимов

        Args:
            lookback_period: Период для анализа тренда
            volatility_window: Окно для расчета волатильности
        """
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        self.last_regimes = []  # Для отслеживания смены режимов
        
        logger.info(f"Market regime detector initialized with lookback={lookback_period}, volatility_window={volatility_window}")

    def detect_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Определяет текущий рыночный режим

        Args:
            data: DataFrame с OHLCV данными
            
        Returns:
            MarketRegime: Обнаруженный рыночный режим
        """
        if len(data) < max(self.lookback_period, self.volatility_window):
            logger.warning("Insufficient data for market regime detection")
            return MarketRegime.UNKNOWN
        
        # Получаем последние данные для анализа
        recent_data = data.iloc[-self.lookback_period:]
        
        # Определяем тренд
        trend = self._detect_trend(recent_data)
        
        # Определяем волатильность
        volatility_level = self._detect_volatility(data)
        
        # Определяем пробои и развороты
        breakout = self._detect_breakout(data)
        reversal = self._detect_reversal(data)
        
        # Приоритет режимов: сначала проверяем особые события, затем общие режимы
        if breakout:
            regime = MarketRegime.BREAKOUT
        elif reversal:
            regime = MarketRegime.REVERSAL
        elif volatility_level == "high":
            regime = MarketRegime.VOLATILE
        elif volatility_level == "low":
            regime = MarketRegime.LOW_VOLATILITY
        elif trend == "up":
            regime = MarketRegime.TRENDING_UP
        elif trend == "down":
            regime = MarketRegime.TRENDING_DOWN
        elif trend == "sideways":
            regime = MarketRegime.RANGING
        else:
            regime = MarketRegime.UNKNOWN
        
        # Сохраняем режим в историю
        self.last_regimes.append(regime)
        if len(self.last_regimes) > 10:  # Храним только последние 10 режимов
            self.last_regimes.pop(0)
            
        logger.debug(f"Detected market regime: {regime.value}")
        return regime

    def _detect_trend(self, data: pd.DataFrame) -> str:
        """
        Определяет направление тренда

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            str: "up", "down" или "sideways"
        """
        # Используем линейную регрессию для определения тренда
        closes = data['close'].values
        x = np.arange(len(closes))
        
        # Вычисляем коэффициент наклона
        if len(closes) > 1:
            slope, _ = np.polyfit(x, closes, 1)
            
            # Нормализуем наклон относительно цены
            norm_slope = slope / np.mean(closes)
            
            # Определяем силу тренда по нормализованному наклону
            if norm_slope > 0.001:  # Порог для восходящего тренда
                return "up"
            elif norm_slope < -0.001:  # Порог для нисходящего тренда
                return "down"
        
        # Если наклон незначительный, считаем рынок боковым
        return "sideways"

    def _detect_volatility(self, data: pd.DataFrame) -> str:
        """
        Определяет уровень волатильности

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            str: "high", "normal" или "low"
        """
        # Вычисляем историческую волатильность
        returns = data['close'].pct_change().dropna()
        
        if len(returns) < self.volatility_window:
            return "normal"
        
        # Получаем окно для текущей волатильности
        recent_returns = returns.iloc[-self.volatility_window:]
        current_vol = recent_returns.std()
        
        # Получаем историческую волатильность для сравнения
        if len(returns) >= self.volatility_window * 3:
            historical_windows = [returns.iloc[i:i+self.volatility_window].std() 
                                 for i in range(0, len(returns) - self.volatility_window * 2, self.volatility_window)]
            historical_vol = np.mean(historical_windows)
            
            # Сравниваем текущую волатильность с исторической
            ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            if ratio > 1.5:  # На 50% выше исторической
                return "high"
            elif ratio < 0.5:  # На 50% ниже исторической
                return "low"
        
        return "normal"

    def _detect_breakout(self, data: pd.DataFrame) -> bool:
        """
        Определяет наличие пробоя уровней

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            bool: True, если обнаружен пробой
        """
        if len(data) < self.lookback_period * 2:
            return False
        
        # Находим максимумы и минимумы за предыдущий период
        prev_data = data.iloc[-(self.lookback_period*2):-self.lookback_period]
        recent_data = data.iloc[-self.lookback_period:]
        
        prev_high = prev_data['high'].max()
        prev_low = prev_data['low'].min()
        
        # Проверяем, был ли пробой этих уровней
        if recent_data['close'].iloc[-1] > prev_high * 1.02:  # 2% запас для подтверждения пробоя
            return True
        
        if recent_data['close'].iloc[-1] < prev_low * 0.98:  # 2% запас для подтверждения пробоя
            return True
            
        return False

    def _detect_reversal(self, data: pd.DataFrame) -> bool:
        """
        Определяет наличие разворота тренда

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            bool: True, если обнаружен разворот
        """
        if len(data) < self.lookback_period * 2:
            return False
            
        # Проверяем изменение тренда
        prev_trend = self._detect_trend(data.iloc[-(self.lookback_period*2):-self.lookback_period])
        current_trend = self._detect_trend(data.iloc[-self.lookback_period:])
        
        # Если тренд изменился с восходящего на нисходящий или наоборот - это разворот
        return (prev_trend == "up" and current_trend == "down") or (prev_trend == "down" and current_trend == "up")

    def get_regime_statistics(self) -> Dict[str, int]:
        """
        Возвращает статистику по обнаруженным рыночным режимам

        Returns:
            Dict[str, int]: Статистика режимов
        """
        stats = {regime.value: 0 for regime in MarketRegime}
        
        for regime in self.last_regimes:
            stats[regime.value] += 1
            
        return stats


class StrategySelector:
    """Селектор стратегий на основе рыночных режимов"""

    def __init__(self):
        """Инициализирует селектор стратегий"""
        # Сопоставление режимов рынка с наиболее подходящими стратегиями
        self.regime_strategy_map = {
            MarketRegime.TRENDING_UP: ["trend_following", "momentum", "breakout"],
            MarketRegime.TRENDING_DOWN: ["trend_following", "mean_reversion", "breakout"],
            MarketRegime.RANGING: ["mean_reversion", "oscillator", "scalping"],
            MarketRegime.VOLATILE: ["volatility", "breakout", "options_straddle"],
            MarketRegime.LOW_VOLATILITY: ["mean_reversion", "carry", "grid"],
            MarketRegime.BREAKOUT: ["breakout", "momentum", "trend_following"],
            MarketRegime.REVERSAL: ["mean_reversion", "counter_trend", "swing"],
            MarketRegime.UNKNOWN: ["balanced", "diversified", "adaptive"]
        }
        
        # Обратное сопоставление: стратегии к подходящим режимам
        self.strategy_regime_map = self._build_strategy_regime_map()
        
        logger.info("Strategy selector initialized")

    def _build_strategy_regime_map(self) -> Dict[str, List[MarketRegime]]:
        """
        Строит обратное сопоставление стратегий к режимам

        Returns:
            Dict[str, List[MarketRegime]]: Сопоставление стратегий к режимам
        """
        result = {}
        
        # Перебираем все сопоставления режимов и стратегий
        for regime, strategies in self.regime_strategy_map.items():
            for strategy in strategies:
                if strategy not in result:
                    result[strategy] = []
                result[strategy].append(regime)
                
        return result

    def get_suitable_strategies(self, regime: MarketRegime) -> List[str]:
        """
        Возвращает список подходящих стратегий для указанного режима

        Args:
            regime: Рыночный режим

        Returns:
            List[str]: Список подходящих стратегий
        """
        return self.regime_strategy_map.get(regime, [])

    def get_suitable_regimes(self, strategy: str) -> List[MarketRegime]:
        """
        Возвращает список подходящих режимов для указанной стратегии

        Args:
            strategy: Название стратегии

        Returns:
            List[MarketRegime]: Список подходящих режимов
        """
        return self.strategy_regime_map.get(strategy, [])

    def select_strategy(self, regime: MarketRegime, available_strategies: List[str]) -> Optional[str]:
        """
        Выбирает лучшую стратегию из доступных для указанного режима

        Args:
            regime: Рыночный режим
            available_strategies: Список доступных стратегий

        Returns:
            Optional[str]: Название выбранной стратегии или None, если нет подходящих
        """
        # Получаем подходящие стратегии для режима
        suitable_strategies = self.get_suitable_strategies(regime)
        
        # Находим пересечение подходящих и доступных стратегий
        matching_strategies = [s for s in suitable_strategies if s in available_strategies]
        
        if matching_strategies:
            # Возвращаем самую подходящую стратегию (первую в списке)
            return matching_strategies[0]
        
        # Если нет подходящих стратегий, возвращаем любую из доступных
        return available_strategies[0] if available_strategies else None

    def adjust_strategy_parameters(self, strategy: str, regime: MarketRegime, 
                                  base_parameters: Dict) -> Dict:
        """
        Адаптирует параметры стратегии под текущий рыночный режим

        Args:
            strategy: Название стратегии
            regime: Рыночный режим
            base_parameters: Базовые параметры стратегии

        Returns:
            Dict: Адаптированные параметры стратегии
        """
        # Создаем копию базовых параметров
        adjusted_params = base_parameters.copy()
        
        # Адаптация параметров в зависимости от рыночного режима
        if regime == MarketRegime.VOLATILE:
            # Для волатильного рынка: уменьшаем размер позиций и делаем более жесткие стоп-лоссы
            if 'position_size' in adjusted_params:
                adjusted_params['position_size'] = adjusted_params['position_size'] * 0.7
                
            if 'stop_loss' in adjusted_params:
                adjusted_params['stop_loss'] = adjusted_params['stop_loss'] * 1.5
                
        elif regime == MarketRegime.LOW_VOLATILITY:
            # Для рынка с низкой волатильностью: увеличиваем размер позиций, но делаем более узкие тейк-профиты
            if 'position_size' in adjusted_params:
                adjusted_params['position_size'] = adjusted_params['position_size'] * 1.3
                
            if 'take_profit' in adjusted_params:
                adjusted_params['take_profit'] = adjusted_params['take_profit'] * 0.7
                
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Для трендового рынка: используем трейлинг-стоп вместо фиксированного
            if 'stop_loss_type' in adjusted_params:
                adjusted_params['stop_loss_type'] = 'trailing'
                
        elif regime == MarketRegime.RANGING:
            # Для бокового рынка: используем более короткие таймфреймы и уменьшаем тейк-профиты
            if 'timeframe' in adjusted_params:
                # Уменьшаем таймфрейм, если это возможно
                tf_map = {'1d': '4h', '4h': '1h', '1h': '15m', '15m': '5m', '5m': '1m'}
                adjusted_params['timeframe'] = tf_map.get(adjusted_params['timeframe'], adjusted_params['timeframe'])
                
            if 'take_profit' in adjusted_params:
                adjusted_params['take_profit'] = adjusted_params['take_profit'] * 0.6
        
        logger.debug(f"Adjusted parameters for {strategy} in {regime.value} regime")
        return adjusted_params


# Глобальные экземпляры
_regime_detector = None
_strategy_selector = None


def get_regime_detector(lookback_period: int = 20, volatility_window: int = 20) -> MarketRegimeDetector:
    """
    Возвращает глобальный экземпляр детектора рыночных режимов

    Args:
        lookback_period: Период для анализа тренда
        volatility_window: Окно для расчета волатильности

    Returns:
        MarketRegimeDetector: Экземпляр детектора рыночных режимов
    """
    global _regime_detector
    
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector(
            lookback_period=lookback_period,
            volatility_window=volatility_window
        )
    
    return _regime_detector


def get_strategy_selector() -> StrategySelector:
    """
    Возвращает глобальный экземпляр селектора стратегий

    Returns:
        StrategySelector: Экземпляр селектора стратегий
    """
    global _strategy_selector
    
    if _strategy_selector is None:
        _strategy_selector = StrategySelector()
    
    return _strategy_selector


def detect_market_regime(data: pd.DataFrame) -> MarketRegime:
    """
    Определяет текущий рыночный режим

    Args:
        data: DataFrame с OHLCV данными

    Returns:
        MarketRegime: Обнаруженный рыночный режим
    """
    detector = get_regime_detector()
    return detector.detect_regime(data)


def select_strategy_for_regime(regime: MarketRegime, available_strategies: List[str]) -> Optional[str]:
    """
    Выбирает подходящую стратегию для указанного режима

    Args:
        regime: Рыночный режим
        available_strategies: Список доступных стратегий

    Returns:
        Optional[str]: Название выбранной стратегии или None, если нет подходящих
    """
    selector = get_strategy_selector()
    return selector.select_strategy(regime, available_strategies)
