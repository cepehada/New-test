"""
Модуль для расчета показателя Value at Risk (VaR).
Предоставляет функции для оценки рисков портфеля и отдельных позиций.
"""

from typing import Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
from project.config import get_config
from project.utils.error_handler import handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VarCalculator:
    """
    Класс для расчета показателя Value at Risk (VaR).
    """

    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """
        Инициализирует калькулятор VaR.

        Args:
            confidence_level: Уровень доверия (0.9, 0.95, 0.99)
            time_horizon: Горизонт времени в днях
        """
        self.config = get_config()
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        logger.debug(
            f"Создан экземпляр VarCalculator (confidence_level={confidence_level}, time_horizon={time_horizon})"
        )

    @handle_error
    def calculate_historical_var(
        self, returns: pd.Series, portfolio_value: float
    ) -> float:
        """
        Рассчитывает исторический VaR на основе прошлых доходностей.

        Args:
            returns: Серия исторических доходностей
            portfolio_value: Текущая стоимость портфеля

        Returns:
            Значение VaR в абсолютных единицах
        """
        if len(returns) < 10:
            logger.warning("Недостаточно исторических данных для расчета VaR")
            return 0.0

        # Определяем квантиль для заданного уровня доверия
        var_percentile = 1 - self.confidence_level

        # Рассчитываем VaR как квантиль эмпирического распределения
        var_return = returns.quantile(var_percentile)

        # Масштабируем на горизонт времени
        var_return_scaled = var_return * np.sqrt(self.time_horizon)

        # Переводим в абсолютные единицы
        var_absolute = portfolio_value * abs(var_return_scaled)

        logger.debug(
            f"Рассчитан исторический VaR: {var_absolute} "
            f"(confidence={self.confidence_level}, horizon={self.time_horizon})"
        )

        return var_absolute

    @handle_error
    def calculate_parametric_var(
        self, returns: pd.Series, portfolio_value: float
    ) -> float:
        """
        Рассчитывает параметрический VaR, предполагая нормальное распределение.

        Args:
            returns: Серия исторических доходностей
            portfolio_value: Текущая стоимость портфеля

        Returns:
            Значение VaR в абсолютных единицах
        """
        if len(returns) < 10:
            logger.warning("Недостаточно исторических данных для расчета VaR")
            return 0.0

        # Рассчитываем среднее и стандартное отклонение доходностей
        mean_return = returns.mean()
        std_return = returns.std()

        # Определяем Z-значение для заданного уровня доверия
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # Рассчитываем VaR
        var_return = mean_return + z_score * std_return

        # Масштабируем на горизонт времени
        var_return_scaled = var_return * np.sqrt(self.time_horizon)

        # Переводим в абсолютные единицы
        var_absolute = portfolio_value * abs(var_return_scaled)

        logger.debug(
            f"Рассчитан параметрический VaR: {var_absolute} "
            f"(confidence={self.confidence_level}, horizon={self.time_horizon})"
        )

        return var_absolute

    @handle_error
    def calculate_monte_carlo_var(
        self, returns: pd.Series, portfolio_value: float, num_simulations: int = 10000
    ) -> float:
        """
        Рассчитывает VaR с использованием метода Монте-Карло.

        Args:
            returns: Серия исторических доходностей
            portfolio_value: Текущая стоимость портфеля
            num_simulations: Количество симуляций

        Returns:
            Значение VaR в абсолютных единицах
        """
        if len(returns) < 10:
            logger.warning("Недостаточно исторических данных для расчета VaR")
            return 0.0

        # Рассчитываем параметры распределения
        mean_return = returns.mean()
        std_return = returns.std()

        # Генерируем случайные доходности
        np.random.seed(42)  # Для воспроизводимости результатов
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)

        # Масштабируем на горизонт времени
        simulated_returns_scaled = simulated_returns * np.sqrt(self.time_horizon)

        # Определяем квантиль для заданного уровня доверия
        var_percentile = 1 - self.confidence_level

        # Рассчитываем VaR как квантиль симулированного распределения
        var_return = np.percentile(simulated_returns_scaled, var_percentile * 100)

        # Переводим в абсолютные единицы
        var_absolute = portfolio_value * abs(var_return)

        logger.debug(
            f"Рассчитан VaR методом Монте-Карло: {var_absolute} "
            f"(confidence={self.confidence_level}, horizon={self.time_horizon}, "
            f"simulations={num_simulations})"
        )

        return var_absolute

    @handle_error
    def calculate_conditional_var(
        self, returns: pd.Series, portfolio_value: float
    ) -> float:
        """
        Рассчитывает условный VaR (Expected Shortfall или CVaR).

        Args:
            returns: Серия исторических доходностей
            portfolio_value: Текущая стоимость портфеля

        Returns:
            Значение CVaR в абсолютных единицах
        """
        if len(returns) < 10:
            logger.warning("Недостаточно исторических данных для расчета CVaR")
            return 0.0

        # Определяем квантиль для заданного уровня доверия
        var_percentile = 1 - self.confidence_level

        # Рассчитываем VaR
        var_return = returns.quantile(var_percentile)

        # Рассчитываем CVaR как среднее всех доходностей, которые хуже VaR
        cvar_returns = returns[returns <= var_return]
        cvar_return = cvar_returns.mean()

        # Масштабируем на горизонт времени
        cvar_return_scaled = cvar_return * np.sqrt(self.time_horizon)

        # Переводим в абсолютные единицы
        cvar_absolute = portfolio_value * abs(cvar_return_scaled)

        logger.debug(
            f"Рассчитан условный VaR (CVaR): {cvar_absolute} "
            f"(confidence={self.confidence_level}, horizon={self.time_horizon})"
        )

        return cvar_absolute

    @handle_error
    def calculate_portfolio_var(
        self,
        asset_returns: Dict[str, pd.DataFrame],
        asset_weights: Dict[str, float],
        portfolio_value: float,
    ) -> float:
        """
        Рассчитывает VaR для портфеля с учетом корреляций активов.

        Args:
            asset_returns: Словарь с фреймами исторических доходностей для каждого актива
            asset_weights: Словарь с весами каждого актива в портфеле
            portfolio_value: Текущая стоимость портфеля

        Returns:
            Значение VaR для портфеля в абсолютных единицах
        """
        # Проверяем наличие данных
        if not asset_returns or not asset_weights:
            logger.warning("Недостаточно данных для расчета VaR портфеля")
            return 0.0

        # Создаем фрейм доходностей активов
        returns_data = {}
        for asset, returns in asset_returns.items():
            if asset in asset_weights:
                returns_data[asset] = returns

        if not returns_data:
            logger.warning("Нет пересечения между активами с доходностями и весами")
            return 0.0

        # Объединяем доходности в один фрейм
        portfolio_returns = pd.DataFrame(returns_data)

        # Создаем вектор весов
        weights = pd.Series(asset_weights).reindex(portfolio_returns.columns).fillna(0)
        weights = weights / weights.sum()  # Нормализуем веса

        # Рассчитываем ковариационную матрицу
        cov_matrix = portfolio_returns.cov()

        # Рассчитываем портфельную дисперсию
        portfolio_variance = weights.dot(cov_matrix).dot(weights)
        portfolio_std = np.sqrt(portfolio_variance)

        # Определяем Z-значение для заданного уровня доверия
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # Рассчитываем VaR
        var_return = z_score * portfolio_std

        # Масштабируем на горизонт времени
        var_return_scaled = var_return * np.sqrt(self.time_horizon)

        # Переводим в абсолютные единицы
        var_absolute = portfolio_value * abs(var_return_scaled)

        logger.debug(
            f"Рассчитан VaR портфеля: {var_absolute} "
            f"(confidence={self.confidence_level}, horizon={self.time_horizon})"
        )

        return var_absolute

    @handle_error
    def calculate_var_contribution(
        self,
        asset_returns: Dict[str, pd.DataFrame],
        asset_weights: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """
        Рассчитывает вклад каждого актива в общий VaR портфеля.

        Args:
            asset_returns: Словарь с фреймами исторических доходностей для каждого актива
            asset_weights: Словарь с весами каждого актива в портфеле
            portfolio_value: Текущая стоимость портфеля

        Returns:
            Словарь с вкладом каждого актива в VaR
        """
        # Проверяем наличие данных
        if not asset_returns or not asset_weights:
            logger.warning("Недостаточно данных для расчета вклада в VaR")
            return {}

        # Создаем фрейм доходностей активов
        returns_data = {}
        for asset, returns in asset_returns.items():
            if asset in asset_weights:
                returns_data[asset] = returns

        if not returns_data:
            logger.warning("Нет пересечения между активами с доходностями и весами")
            return {}

        # Объединяем доходности в один фрейм
        portfolio_returns = pd.DataFrame(returns_data)

        # Создаем вектор весов
        weights = pd.Series(asset_weights).reindex(portfolio_returns.columns).fillna(0)
        weights = weights / weights.sum()  # Нормализуем веса

        # Рассчитываем ковариационную матрицу
        cov_matrix = portfolio_returns.cov()

        # Рассчитываем портфельную дисперсию
        portfolio_variance = weights.dot(cov_matrix).dot(weights)
        portfolio_std = np.sqrt(portfolio_variance)

        # Рассчитываем вклад каждого актива в риск портфеля
        # Маржинальный вклад в риск = весь * ковариация с портфелем / стандартное отклонение портфеля
        marginal_contributions = {}
        for asset in weights.index:
            cov_with_portfolio = 0.0
            for other_asset in weights.index:
                cov_with_portfolio += (
                    weights[other_asset] * cov_matrix.loc[asset, other_asset]
                )

            marginal_contribution = weights[asset] * cov_with_portfolio / portfolio_std
            marginal_contributions[asset] = marginal_contribution

        # Определяем Z-значение для заданного уровня доверия
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # Рассчитываем VaR портфеля
        portfolio_var = (
            z_score * portfolio_std * np.sqrt(self.time_horizon) * portfolio_value
        )

        # Рассчитываем вклад каждого актива в VaR
        var_contributions = {}
        for asset, contribution in marginal_contributions.items():
            var_contributions[asset] = (
                contribution * z_score * np.sqrt(self.time_horizon) * portfolio_value
            )

        logger.debug("Рассчитаны вклады в VaR портфеля: {var_contributions}" %)

        return var_contributions

    @handle_error
    def calculate_stress_test_var(
        self, returns: pd.Series, portfolio_value: float, stress_factor: float = 1.5
    ) -> float:
        """
        Рассчитывает VaR в стрессовых условиях.

        Args:
            returns: Серия исторических доходностей
            portfolio_value: Текущая стоимость портфеля
            stress_factor: Множитель для увеличения волатильности

        Returns:
            Значение стресс-тестового VaR в абсолютных единицах
        """
        if len(returns) < 10:
            logger.warning(
                "Недостаточно исторических данных для расчета стресс-тестового VaR"
            )
            return 0.0

        # Рассчитываем среднее и стандартное отклонение доходностей
        mean_return = returns.mean()
        std_return = returns.std() * stress_factor  # Увеличиваем волатильность

        # Определяем Z-значение для заданного уровня доверия
        z_score = stats.norm.ppf(1 - self.confidence_level)

        # Рассчитываем VaR
        var_return = mean_return + z_score * std_return

        # Масштабируем на горизонт времени
        var_return_scaled = var_return * np.sqrt(self.time_horizon)

        # Переводим в абсолютные единицы
        var_absolute = portfolio_value * abs(var_return_scaled)

        logger.debug(
            f"Рассчитан стресс-тестовый VaR: {var_absolute} "
            f"(confidence={self.confidence_level}, horizon={self.time_horizon}, "
            f"stress_factor={stress_factor})"
        )

        return var_absolute
