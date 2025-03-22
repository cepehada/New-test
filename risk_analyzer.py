"""
Risk Analyzer.
Анализирует риск портфеля с использованием различных методологий:
- Value at Risk (VaR) с методами: Historical, Parametric, Monte Carlo
- Expected Shortfall (ES) / Conditional VaR (CVaR)
- Сценарный анализ с историческими кризисами
- Стресс-тестирование с различными уровнями шоков
- Анализ корреляций между активами
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from scipy import stats
import asyncio
import functools
import time
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("RiskAnalyzer")

# Настройка логирования
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class RiskAnalysisResult:
    """Хранит результаты анализа риска в структурированном виде."""
    var_historical: Optional[float] = None
    var_parametric: Optional[float] = None
    var_monte_carlo: Optional[float] = None
    es_historical: Optional[float] = None
    es_parametric: Optional[float] = None
    es_monte_carlo: Optional[float] = None
    stress_test_results: Optional[Dict[str, Any]] = None
    scenario_analysis: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результаты в словарь для сериализации."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def __str__(self) -> str:
        """Строковое представление результатов."""
        result = "=== Результаты анализа риска ===\n"
        
        if self.var_historical is not None:
            result += f"VaR (исторический, 95%): {self.var_historical:.2%}\n"
        if self.var_parametric is not None:
            result += f"VaR (параметрический, 95%): {self.var_parametric:.2%}\n"
        if self.var_monte_carlo is not None:
            result += f"VaR (Monte Carlo, 95%): {self.var_monte_carlo:.2%}\n"
            
        if self.es_historical is not None:
            result += f"Expected Shortfall (исторический, 95%): {self.es_historical:.2%}\n"
        if self.es_parametric is not None:
            result += f"Expected Shortfall (параметрический, 95%): {self.es_parametric:.2%}\n"
        if self.es_monte_carlo is not None:
            result += f"Expected Shortfall (Monte Carlo, 95%): {self.es_monte_carlo:.2%}\n"
        
        if self.stress_test_results:
            result += "\n--- Результаты стресс-тестирования ---\n"
            for scenario, data in self.stress_test_results.items():
                result += f"Сценарий '{scenario}':\n"
                result += f"  Последнее значение: {data['last_value']:.2f}\n"
                result += f"  Шок: {data['shock']:.2%}\n"
                result += f"  Потенциальные потери: {data['potential_loss']:.2f} ({data['loss_percentage']:.2%})\n"
        
        if self.scenario_analysis:
            result += "\n--- Сценарный анализ ---\n"
            for scenario, impact in self.scenario_analysis.items():
                result += f"Сценарий '{scenario}': {impact:.2%}\n"
                
        return result


class RiskAnalyzer:
    """
    Класс для комплексного анализа рисков портфеля.
    Поддерживает различные методы оценки VaR, стресс-тесты и сценарный анализ.
    """
    
    def __init__(self, cache_results: bool = True, cache_ttl: int = 3600):
        """
        Инициализация анализатора рисков.
        
        Args:
            cache_results: Включить кэширование результатов для оптимизации
            cache_ttl: Время жизни кэша в секундах
        """
        self.cache = {}
        self.cache_results = cache_results
        self.cache_ttl = cache_ttl
        self.last_run = {}
        
        # Определяем исторические сценарии кризисов и их шоки
        self.historical_scenarios = {
            "Black Monday 1987": -0.228,  # Падение S&P 500 на 22.8% 19 октября 1987
            "Dot-com Crash 2000": -0.49,  # Падение NASDAQ на 49% с марта по май 2000
            "2008 Financial Crisis": -0.57,  # Падение S&P 500 на 57% с 2007 по 2009
            "COVID-19 Crash 2020": -0.35,  # Падение S&P 500 на 35% в марте 2020
            "Crypto Winter 2022": -0.65   # Падение криптовалют на 65% в 2022
        }
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Получает кэшированный результат, если он не устарел."""
        if not self.cache_results:
            return None
            
        if cache_key in self.cache and cache_key in self.last_run:
            if time.time() - self.last_run[cache_key] < self.cache_ttl:
                logger.debug(f"Используем кэшированный результат для {cache_key}")
                return self.cache[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Кэширует результат с временной меткой."""
        if self.cache_results:
            self.cache[cache_key] = result
            self.last_run[cache_key] = time.time()
    
    def calculate_historical_var(
        self, returns: List[float], confidence: float = 0.95, cleanup: bool = True
    ) -> float:
        """
        Рассчитывает VaR на основе исторических данных.
        
        Args:
            returns: Список исторических доходностей
            confidence: Уровень доверия (например, 0.95)
            cleanup: Очистить данные от выбросов и NaN
            
        Returns:
            float: Значение VaR как отрицательное число (например, -0.05 для 5% потерь)
        """
        if not returns:
            logger.error("Список доходностей пуст")
            raise ValueError("Список доходностей пуст")
        
        # Преобразование в numpy массив для обработки
        returns_array = np.array(returns)
        
        if cleanup:
            # Удаление NaN и Infinity
            returns_array = returns_array[~np.isnan(returns_array)]
            returns_array = returns_array[~np.isinf(returns_array)]
            
            # Удаление выбросов (опционально)
            z_scores = np.abs(stats.zscore(returns_array))
            returns_array = returns_array[z_scores < 3]
        
        if len(returns_array) == 0:
            logger.error("После очистки не осталось валидных данных")
            raise ValueError("После очистки не осталось валидных данных")
        
        # Рассчет VaR как процентиль распределения
        var_value = np.percentile(returns_array, (1 - confidence) * 100)
        
        logger.info(f"Исторический VaR ({confidence*100}%): {var_value:.4f}")
        return var_value
    
    def calculate_parametric_var(
        self, returns: List[float], confidence: float = 0.95, cleanup: bool = True
    ) -> float:
        """
        Рассчитывает VaR с использованием параметрического (нормального) подхода.
        
        Args:
            returns: Список исторических доходностей
            confidence: Уровень доверия (например, 0.95)
            cleanup: Очистить данные от выбросов и NaN
            
        Returns:
            float: Значение VaR как отрицательное число
        """
        if not returns:
            logger.error("Список доходностей пуст")
            raise ValueError("Список доходностей пуст")
        
        # Преобразование в numpy массив для обработки
        returns_array = np.array(returns)
        
        if cleanup:
            # Удаление NaN и Infinity
            returns_array = returns_array[~np.isnan(returns_array)]
            returns_array = returns_array[~np.isinf(returns_array)]
            
            # Удаление выбросов (опционально)
            z_scores = np.abs(stats.zscore(returns_array))
            returns_array = returns_array[z_scores < 3]
        
        if len(returns_array) == 0:
            logger.error("После очистки не осталось валидных данных")
            raise ValueError("После очистки не осталось валидных данных")
        
        # Рассчет среднего и стандартного отклонения
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Используем несмещенную оценку
        
        # Рассчет VaR с использованием квантиля нормального распределения
        z_score = stats.norm.ppf(1 - confidence)
        var_value = mean_return + z_score * std_return
        
        logger.info(f"Параметрический VaR ({confidence*100}%): {var_value:.4f}")
        return var_value
    
    def monte_carlo_var(
        self, 
        returns: List[float], 
        num_simulations: int = 10000, 
        confidence: float = 0.95,
        dist_type: str = "normal",
        cleanup: bool = True,
        seed: Optional[int] = None
    ) -> float:
        """
        Рассчитывает VaR методом Monte Carlo моделирования.

        Генерирует случайные доходности на основе заданного распределения
        и вычисляет процентиль, отражающий риск потерь.

        Args:
            returns: Исторические дневные доходности
            num_simulations: Число симуляций (по умолчанию 10000)
            confidence: Уровень доверия (например, 0.95)
            dist_type: Тип распределения ("normal", "t", "bootstrap")
            cleanup: Очистить данные от выбросов и NaN
            seed: Семя для генератора случайных чисел

        Returns:
            float: Значение VaR (отрицательное число)
        """
        # Формирование ключа кэша
        cache_key = f"mc_var_{hash(tuple(returns))}_{num_simulations}_{confidence}_{dist_type}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        if not returns:
            logger.error("Список доходностей пуст")
            raise ValueError("Список доходностей пуст")
        
        # Установка семени для воспроизводимости результатов
        if seed is not None:
            np.random.seed(seed)
        
        # Преобразование в numpy массив для обработки
        returns_array = np.array(returns)
        
        if cleanup:
            # Удаление NaN и Infinity
            returns_array = returns_array[~np.isnan(returns_array)]
            returns_array = returns_array[~np.isinf(returns_array)]
            
            # Удаление выбросов (опционально)
            z_scores = np.abs(stats.zscore(returns_array))
            returns_array = returns_array[z_scores < 3]
        
        if len(returns_array) == 0:
            logger.error("После очистки не осталось валидных данных")
            raise ValueError("После очистки не осталось валидных данных")
        
        # Параметры распределения
        mean_ret = np.mean(returns_array)
        std_ret = np.std(returns_array, ddof=1)
        
        # Генерация случайных доходностей в зависимости от выбранного распределения
        if dist_type == "normal":
            # Нормальное распределение
            simulations = np.random.normal(mean_ret, std_ret, num_simulations)
        elif dist_type == "t":
            # t-распределение для учета "тяжелых хвостов"
            df = 5  # Степени свободы
            simulations = stats.t.rvs(df, loc=mean_ret, scale=std_ret, size=num_simulations)
        elif dist_type == "bootstrap":
            # Бутстрап - случайная выборка с возвращением из исторических данных
            indices = np.random.choice(len(returns_array), size=num_simulations, replace=True)
            simulations = returns_array[indices]
        else:
            raise ValueError(f"Неизвестный тип распределения: {dist_type}")
        
        # Рассчет VaR как процентиль симулированных доходностей
        percentile = (1 - confidence) * 100
        var_value = np.percentile(simulations, percentile)
        
        # Кэширование результата
        self._cache_result(cache_key, var_value)
        
        logger.info(f"Monte Carlo VaR ({confidence*100}%, {dist_type}): {var_value:.4f}")
        return var_value
    
    def calculate_expected_shortfall(
        self, 
        returns: List[float], 
        var_value: float,
        confidence: float = 0.95,
        method: str = "historical"
    ) -> float:
        """
        Рассчитывает Expected Shortfall (Conditional VaR) - среднее потерь, превышающих VaR.
        
        Args:
            returns: Список исторических доходностей
            var_value: Значение Value-at-Risk
            confidence: Уровень доверия
            method: Метод расчета ("historical", "parametric", "monte_carlo")
            
        Returns:
            float: Значение Expected Shortfall (отрицательное число)
        """
        if not returns:
            logger.error("Список доходностей пуст")
            raise ValueError("Список доходностей пуст")
        
        # Преобразование в numpy массив для обработки
        returns_array = np.array(returns)
        returns_array = returns_array[~np.isnan(returns_array)]
        
        if method == "historical":
            # Находим все доходности, которые хуже VaR
            tail_returns = returns_array[returns_array <= var_value]
            
            if len(tail_returns) == 0:
                logger.warning("Нет наблюдений, превышающих VaR")
                return var_value
            
            # Средняя доходность в хвосте распределения
            es_value = np.mean(tail_returns)
            
        elif method == "parametric":
            # Для нормального распределения
            mean_ret = np.mean(returns_array)
            std_ret = np.std(returns_array, ddof=1)
            z_score = stats.norm.ppf(1 - confidence)
            
            # Формула для ES при нормальном распределении
            es_value = mean_ret - std_ret * stats.norm.pdf(z_score) / (1 - confidence)
            
        elif method == "monte_carlo":
            # Используем Monte Carlo симуляции
            simulations = self._get_cached_result(f"mc_sims_{hash(tuple(returns))}")
            
            if simulations is None:
                # Если нет кэшированных симуляций, создаем их
                mean_ret = np.mean(returns_array)
                std_ret = np.std(returns_array, ddof=1)
                simulations = np.random.normal(mean_ret, std_ret, 10000)
                self._cache_result(f"mc_sims_{hash(tuple(returns))}", simulations)
            
            # Находим все симуляции, которые хуже VaR
            tail_simulations = simulations[simulations <= var_value]
            
            if len(tail_simulations) == 0:
                logger.warning("Нет симуляций, превышающих VaR")
                return var_value
            
            # Средняя доходность в хвосте распределения
            es_value = np.mean(tail_simulations)
            
        else:
            raise ValueError(f"Неизвестный метод расчета ES: {method}")
        
        logger.info(f"Expected Shortfall ({confidence*100}%, {method}): {es_value:.4f}")
        return es_value
    
    def stress_test_portfolio(
        self, 
        portfolio_values: List[float], 
        shocks: Optional[Union[float, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Проводит стресс-тест портфеля с различными уровнями шоков.

        Args:
            portfolio_values: История значений портфеля
            shocks: Шок или словарь шоков (имя:значение). По умолчанию использует стандартные шоки.

        Returns:
            Dict[str, Any]: Результаты стресс-теста по каждому сценарию
        """
        if not portfolio_values:
            logger.error("История портфеля пуста")
            raise ValueError("История портфеля пуста")
        
        # Предварительная обработка значений портфеля
        portfolio_array = np.array(portfolio_values)
        portfolio_array = portfolio_array[~np.isnan(portfolio_array)]
        
        if len(portfolio_array) == 0:
            logger.error("После очистки не осталось валидных значений портфеля")
            raise ValueError("После очистки не осталось валидных значений портфеля")
        
        last_value = portfolio_array[-1]
        
        # Определение шоков для тестирования
        if shocks is None:
            shocks = {
                "Легкий шок": 0.05,
                "Умеренный шок": 0.10,
                "Серьезный шок": 0.20,
                "Тяжелый шок": 0.30,
                "Экстремальный шок": 0.50
            }
        elif isinstance(shocks, (int, float)):
            shocks = {"Пользовательский шок": float(shocks)}
        
        # Применение шоков к портфелю
        results = {}
        for name, shock in shocks.items():
            potential_loss = last_value * shock
            results[name] = {
                "last_value": last_value,
                "shock": shock,
                "potential_loss": potential_loss,
                "loss_percentage": shock,
                "post_shock_value": last_value * (1 - shock)
            }
            logger.info(f"Стресс-тест '{name}': потенциальные потери {potential_loss:.2f} ({shock:.2%})")
        
        return results
    
    def scenario_analysis(
        self, 
        portfolio_values: List[float],
        asset_allocations: Optional[Dict[str, float]] = None,
        custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """
        Проводит сценарный анализ на основе исторических кризисов.
        
        Args:
            portfolio_values: История значений портфеля
            asset_allocations: Словарь распределения активов {класс_актива: доля}
            custom_scenarios: Пользовательские сценарии {имя: {класс_актива: шок}}
            
        Returns:
            Dict[str, float]: Результаты по каждому сценарию
        """
        if not portfolio_values:
            logger.error("История портфеля пуста")
            raise ValueError("История портфеля пуста")
        
        last_value = portfolio_values[-1]
        
        # Если распределение активов не задано, предполагаем один класс активов
        if asset_allocations is None:
            asset_allocations = {"default": 1.0}
        
        # Проверка корректности распределения активов
        if abs(sum(asset_allocations.values()) - 1.0) > 0.01:
            logger.warning("Сумма долей активов не равна 1.0")
        
        # Объединяем исторические и пользовательские сценарии
        scenarios = {}
        
        # Добавляем исторические сценарии
        for scenario_name, shock in self.historical_scenarios.items():
            # Предполагаем, что шок одинаково влияет на все классы активов
            scenarios[scenario_name] = {asset: shock for asset in asset_allocations}
        
        # Добавляем пользовательские сценарии
        if custom_scenarios:
            scenarios.update(custom_scenarios)
        
        # Применяем сценарии к портфелю
        results = {}
        for scenario_name, asset_shocks in scenarios.items():
            # Рассчитываем общий шок для портфеля с учетом распределения активов
            total_impact = 0.0
            
            for asset, allocation in asset_allocations.items():
                # Если для актива не определен шок в сценарии, используем 0
                asset_shock = asset_shocks.get(asset, 0.0)
                total_impact += allocation * asset_shock
            
            results[scenario_name] = total_impact
            logger.info(f"Сценарный анализ '{scenario_name}': воздействие на портфель {total_impact:.2%}")
        
        return results
    
    def analyze_risk(
        self,
        returns: List[float],
        portfolio_values: List[float],
        confidence: float = 0.95,
        num_simulations: int = 10000,
        shocks: Optional[Union[float, Dict[str, float]]] = None,
        asset_allocations: Optional[Dict[str, float]] = None,
        custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None,
        generate_report: bool = False,
        report_path: Optional[str] = None
    ) -> RiskAnalysisResult:
        """
        Проводит комплексный анализ риска портфеля.
        
        Args:
            returns: Исторические дневные доходности
            portfolio_values: История значений портфеля
            confidence: Уровень доверия для VaR и ES
            num_simulations: Число симуляций для Monte Carlo
            shocks: Шоки для стресс-тестирования
            asset_allocations: Распределение активов в портфеле
            custom_scenarios: Пользовательские сценарии для анализа
            generate_report: Создать ли файл отчета
            report_path: Путь для сохранения отчета
            
        Returns:
            RiskAnalysisResult: Результаты анализа риска
        """
        try:
            # Проверка входных данных
            if not returns or len(returns) < 2:
                logger.error("Недостаточно данных о доходностях")
                raise ValueError("Недостаточно данных о доходностях")
            
            if not portfolio_values or len(portfolio_values) < 1:
                logger.error("Недостаточно данных о портфеле")
                raise ValueError("Недостаточно данных о портфеле")
            
            # Инициализация результата
            result = RiskAnalysisResult()
            
            # Добавление метаданных
            result.metadata = {
                "analysis_time": datetime.now().isoformat(),
                "returns_count": len(returns),
                "portfolio_values_count": len(portfolio_values),
                "confidence_level": confidence,
                "simulation_count": num_simulations
            }
            
            # 1. Расчет VaR разными методами
            result.var_historical = self.calculate_historical_var(returns, confidence)
            result.var_parametric = self.calculate_parametric_var(returns, confidence)
            result.var_monte_carlo = self.monte_carlo_var(returns, num_simulations, confidence)
            
            # 2. Расчет Expected Shortfall (ES)
            result.es_historical = self.calculate_expected_shortfall(
                returns, result.var_historical, confidence, "historical"
            )
            result.es_parametric = self.calculate_expected_shortfall(
                returns, result.var_parametric, confidence, "parametric"
            )
            result.es_monte_carlo = self.calculate_expected_shortfall(
                returns, result.var_monte_carlo, confidence, "monte_carlo"
            )
            
            # 3. Стресс-тестирование
            result.stress_test_results = self.stress_test_portfolio(portfolio_values, shocks)
            
            # 4. Сценарный анализ
            result.scenario_analysis = self.scenario_analysis(
                portfolio_values, asset_allocations, custom_scenarios
            )
            
            # 5. Опционально: генерация отчета
            if generate_report:
                self._generate_report(result, report_path)
            
            return result
        
        except Exception as e:
            logger.error(f"Ошибка при анализе риска: {e}", exc_info=True)
            raise
    
    def _generate_report(
        self, result: RiskAnalysisResult, report_path: Optional[str] = None
    ) -> str:
        """
        Генерирует отчет о риске в HTML и/или PDF формате.
        
        Args:
            result: Результаты анализа риска
            report_path: Путь для сохранения отчета
            
        Returns:
            str: Путь к созданному отчету
        """
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"risk_report_{timestamp}.html"
        
        # Создание директории, если необходимо
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        
        # Генерация HTML отчета
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Отчет по анализу риска</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .negative {{ color: red; }}
                .positive {{ color: green; }}
                .section {{ margin-bottom: 30px; }}
                .metadata {{ font-size: 0.9em; color: #7f8c8d; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>Отчет по анализу риска портфеля</h1>
            
            <div class="metadata">
                <p>Дата анализа: {result.metadata['analysis_time']}</p>
                <p>Количество данных о доходностях: {result.metadata['returns_count']}</p>
                <p>Количество данных о портфеле: {result.metadata['portfolio_values_count']}</p>
                <p>Уровень доверия: {result.metadata['confidence_level']}</p>
                <p>Количество симуляций: {result.metadata['simulation_count']}</p>
            </div>
            
            <div class="section">
                <h2>Value at Risk (VaR)</h2>
                <table>
                    <tr>
                        <th>Метод</th>
                        <th>Значение</th>
                    </tr>
                    <tr>
                        <td>Исторический VaR</td>
                        <td class="negative">{result.var_historical:.2%}</td>
                    </tr>
                    <tr>
                        <td>Параметрический VaR</td>
                        <td class="negative">{result.var_parametric:.2%}</td>
                    </tr>
                    <tr>
                        <td>Monte Carlo VaR</td>
                        <td class="negative">{result.var_monte_carlo:.2%}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Expected Shortfall (ES)</h2>
                <table>
                    <tr>
                        <th>Метод</th>
                        <th>Значение</th>
                    </tr>
                    <tr>
                        <td>Исторический ES</td>
                        <td class="negative">{result.es_historical:.2%}</td>
                    </tr>
                    <tr>
                        <td>Параметрический ES</td>
                        <td class="negative">{result.es_parametric:.2%}</td>
                    </tr>
                    <tr>
                        <td>Monte Carlo ES</td>
                        <td class="negative">{result.es_monte_carlo:.2%}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Результаты стресс-тестирования</h2>
                <table>
                    <tr>
                        <th>Сценарий</th>
                        <th>Шок</th>
                        <th>Потенциальные потери</th>
                        <th>Значение после шока</th>
                    </tr>
        """
        
        for scenario, data in result.stress_test_results.items():
            html_content += f"""
                    <tr>
                        <td>{scenario}</td>
                        <td class="negative">{data['shock']:.2%}</td>
                        <td class="negative">{data['potential_loss']:.2f} ({data['shock']:.2%})</td>
                        <td>{data['post_shock_value']:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Сценарный анализ</h2>
                <table>
                    <tr>
                        <th>Сценарий</th>
                        <th>Влияние на портфель</th>
                    </tr>
        """
        
        for scenario, impact in result.scenario_analysis.items():
            html_content += f"""
                    <tr>
                        <td>{scenario}</td>
                        <td class="negative">{impact:.2%}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Выводы и рекомендации</h2>
                <p>На основе проведенного анализа риска можно сделать следующие выводы:</p>
                <ul>
        """
        
        # Автоматическое формирование выводов на основе результатов
        if abs(result.var_historical) > 0.1:
            html_content += """
                    <li>Портфель имеет <strong>высокий уровень риска</strong> на основе исторического VaR.</li>
            """
        elif abs(result.var_historical) > 0.05:
            html_content += """
                    <li>Портфель имеет <strong>средний уровень риска</strong> на основе исторического VaR.</li>
            """
        else:
            html_content += """
                    <li>Портфель имеет <strong>низкий уровень риска</strong> на основе исторического VaR.</li>
            """
        
        # Сравнение методов VaR
        max_var_diff = max(abs(result.var_historical - result.var_parametric), 
                           abs(result.var_historical - result.var_monte_carlo),
                           abs(result.var_parametric - result.var_monte_carlo))
        
        if max_var_diff > 0.03:
            html_content += """
                    <li>Наблюдается <strong>значительное расхождение</strong> между различными методами расчета VaR, что указывает на возможные отклонения от нормального распределения доходностей.</li>
            """
        
        # Анализ стресс-тестов
        worst_scenario = max(result.stress_test_results.items(), key=lambda x: x[1]['shock'])
        
        html_content += f"""
                    <li>В случае реализации сценария "{worst_scenario[0]}", потенциальные потери могут составить <strong>{worst_scenario[1]['potential_loss']:.2f}</strong>.</li>
        """
        
        # Рекомендации
        html_content += """
                </ul>
                
                <p>Рекомендации по управлению риском:</p>
                <ul>
                    <li>Регулярно проводить мониторинг риск-метрик портфеля.</li>
                    <li>Рассмотреть возможность диверсификации для снижения общего риска.</li>
                    <li>Установить стоп-лоссы для ограничения потенциальных убытков.</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>Примечание</h3>
                <p>Данный отчет создан автоматически. Результаты анализа риска основаны на исторических данных и предположениях о рыночных условиях. Фактические результаты могут отличаться.</p>
            </div>
        </body>
        </html>
        """
        
        # Сохранение HTML отчета
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Отчет по анализу риска сохранен: {report_path}")
        return report_path

    async def analyze_risk_async(self, *args, **kwargs) -> RiskAnalysisResult:
        """
        Асинхронная версия метода analyze_risk.
        
        Выполняет анализ риска в отдельном потоке для предотвращения
        блокировки асинхронного цикла событий.
        """
        # Выполнение блокирующего анализа риска в пуле потоков
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, functools.partial(self.analyze_risk, *args, **kwargs)
        )
        return result
        

async def run_risk_analysis_example():
    """Пример асинхронного использования RiskAnalyzer."""
    # Тестовые данные
    sample_returns = [
        0.01, -0.02, 0.005, -0.015, 0.02, -0.01, 0.003, 0.004, -0.005, 0.007,
        -0.03, 0.025, -0.018, 0.012, -0.008, 0.01, -0.02, 0.017, -0.015, 0.008
    ]
    sample_portfolio = [100, 102, 101, 105, 103, 107, 110, 108, 112, 115]
    
    # Распределение активов в портфеле
    asset_allocations = {
        "equities": 0.6,
        "bonds": 0.3,
        "crypto": 0.1,
    }
    
    # Пользовательские сценарии
    custom_scenarios = {
        "Высокая инфляция": {
            "equities": -0.15,
            "bonds": -0.25,
            "crypto": -0.05
        },
        "Геополитический кризис": {
            "equities": -0.25,
            "bonds": -0.05,
            "crypto": -0.40
        }
    }
    
    # Инициализация анализатора
    analyzer = RiskAnalyzer()
    
    # Асинхронный анализ риска
    risk_result = await analyzer.analyze_risk_async(
        returns=sample_returns,
        portfolio_values=sample_portfolio,
        confidence=0.95,
        num_simulations=20000,
        asset_allocations=asset_allocations,
        custom_scenarios=custom_scenarios,
        generate_report=True
    )
    
    # Вывод результатов
    print(risk_result)
    
    # Отдельный расчет метрик
    hist_var = analyzer.calculate_historical_var(sample_returns)
    print(f"Исторический VaR: {hist_var:.2%}")
    
    mc_var = analyzer.monte_carlo_var(
        sample_returns, dist_type="t", num_simulations=50000
    )
    print(f"Monte Carlo VaR (t-распределение): {mc_var:.2%}")
    
    # Стресс-тестирование с конкретными шоками
    stress_shocks = {
        "COVID-19 сценарий": 0.35,
        "Умеренная коррекция": 0.15
    }
    stress_results = analyzer.stress_test_portfolio(sample_portfolio, stress_shocks)
    
    for scenario, result in stress_results.items():
        print(f"Сценарий '{scenario}': потери {result['potential_loss']:.2f} ({result['shock']:.2%})")


if __name__ == "__main__":
    # Для синхронного запуска примера
    sample_returns = [
        0.01, -0.02, 0.005, -0.015, 0.02, -0.01, 0.003, 0.004, -0.005, 0.007,
        -0.03, 0.025, -0.018, 0.012, -0.008, 0.01, -0.02, 0.017, -0.015, 0.008
    ]
    sample_portfolio = [100, 102, 101, 105, 103, 107, 110, 108, 112, 115]
    
    # Комплексный анализ риска
    analyzer = RiskAnalyzer()
    risk_report = analyzer.analyze_risk(
        returns=sample_returns, 
        portfolio_values=sample_portfolio,
        generate_report=True
    )
    print("Risk Analysis Report:")
    print(risk_report)
    
    # Для асинхронного запуска примера
    # asyncio.run(run_risk_analysis_example())