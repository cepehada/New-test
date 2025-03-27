import asyncio
import random
import time
import traceback
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from project.backtesting.backtester import Backtester, BacktestResult
from project.data.database import Database
from project.trading.strategy_base import Strategy
from project.utils.logging_utils import setup_logger

logger = setup_logger("genetic_optimizer")


class Individual:
    """Класс для представления индивида в генетическом алгоритме"""

    def __init__(
        self,
        parameters: Dict,
        fitness: float = None,
        metrics: Dict = None,
        generation: int = 0,
    ):
        """
        Инициализирует индивида

        Args:
            parameters: Параметры индивида
            fitness: Значение функции приспособленности
            metrics: Метрики производительности
            generation: Поколение
        """
        self.parameters = parameters
        self.fitness = fitness
        self.metrics = metrics or {}
        self.generation = generation

        # Генерируем уникальный идентификатор
        self.id = str(uuid.uuid4())

    def __lt__(self, other: "Individual") -> bool:
        """
        Сравнивает индивидов по значению функции приспособленности

        Args:
            other: Другой индивид

        Returns:
            bool: True, если текущий индивид хуже другого
        """
        # Если значение функции приспособленности не определено, считаем индивида хуже
        if self.fitness is None:
            return True

        if other.fitness is None:
            return False

        return self.fitness < other.fitness

    def to_dict(self) -> Dict:
        """
        Преобразует индивида в словарь

        Returns:
            Dict: Словарь с данными индивида
        """
        return {
            "id": self.id,
            "parameters": self.parameters,
            "fitness": self.fitness,
            "metrics": self.metrics,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Individual":
        """
        Создает индивида из словаря

        Args:
            data: Словарь с данными индивида

        Returns:
            Individual: Созданный индивид
        """
        individual = cls(
            parameters=data.get("parameters", {}),
            fitness=data.get("fitness"),
            metrics=data.get("metrics", {}),
            generation=data.get("generation", 0),
        )

        individual.id = data.get("id", individual.id)

        return individual


class GeneticOptimizationStats:
    """Класс для сбора статистики оптимизации"""

    def __init__(self):
        """Инициализирует статистику оптимизации"""
        self.start_time = time.time()
        self.end_time = None

        # Статистика популяции
        self.population_stats = {
            "min_fitness": [],
            "max_fitness": [],
            "avg_fitness": [],
            "std_fitness": [],
        }

        # Статистика генераций
        self.generations = 0
        self.evaluations = 0
        self.total_evaluations = 0
        self.best_individual_per_generation = []

        # Лучший индивид
        self.best_fitness = None
        self.best_parameters = None
        self.best_metrics = None
        self.best_generation = 0
        self.best_age = 0

        # Прогресс
        self.progress = 0.0

    def update_population_stats(self, population: List[Individual]):
        """
        Обновляет статистику популяции

        Args:
            population: Список индивидов
        """
        # Собираем значения функции приспособленности
        fitness_values = [ind.fitness for ind in population if ind.fitness is not None]

        if fitness_values:
            self.population_stats["min_fitness"].append(min(fitness_values))
            self.population_stats["max_fitness"].append(max(fitness_values))
            self.population_stats["avg_fitness"].append(np.mean(fitness_values))
            self.population_stats["std_fitness"].append(np.std(fitness_values))

            # Обновляем лучший результат
            max_fitness = max(fitness_values)
            if self.best_fitness is None or max_fitness > self.best_fitness:
                best_individual = max(
                    population,
                    key=lambda ind: (
                        ind.fitness if ind.fitness is not None else float("-inf")
                    ),
                )
                self.best_fitness = best_individual.fitness
                self.best_parameters = best_individual.parameters
                self.best_metrics = best_individual.metrics
                self.best_generation = self.generations
                self.best_age = 0
            else:
                self.best_age += 1

            # Добавляем лучшего индивида текущего поколения
            self.best_individual_per_generation.append(
                max(
                    population,
                    key=lambda ind: (
                        ind.fitness if ind.fitness is not None else float("-inf")
                    ),
                ).to_dict()
            )

    def update_progress(self, progress: float):
        """
        Обновляет прогресс оптимизации

        Args:
            progress: Прогресс (от 0 до 1)
        """
        self.progress = progress

    def finish(self):
        """Завершает сбор статистики"""
        self.end_time = time.time()

    def get_stats(self) -> Dict:
        """
        Возвращает статистику оптимизации

        Returns:
            Dict: Статистика оптимизации
        """
        elapsed_time = (
            self.end_time if self.end_time else time.time()
        ) - self.start_time

        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": (
                datetime.fromtimestamp(self.end_time).isoformat()
                if self.end_time
                else None
            ),
            "elapsed_time": elapsed_time,
            "generations": self.generations,
            "evaluations": self.evaluations,
            "total_evaluations": self.total_evaluations,
            "best_fitness": self.best_fitness,
            "best_parameters": self.best_parameters,
            "best_metrics": self.best_metrics,
            "best_generation": self.best_generation,
            "best_age": self.best_age,
            "progress": self.progress,
            "population_stats": self.population_stats,
            "best_individual_per_generation": self.best_individual_per_generation,
        }


class GeneticOptimizer:
    """Класс для оптимизации параметров стратегии с использованием генетического алгоритма"""

    def __init__(self, config: Dict = None):
        """
        Инициализирует оптимизатор

        Args:
            config: Конфигурация оптимизатора
        """
        self.config = config or {}

        # Параметры генетического алгоритма
        self.population_size = self.config.get("population_size", 50)
        self.generations = self.config.get("generations", 10)
        self.crossover_rate = self.config.get("crossover_rate", 0.7)
        self.mutation_rate = self.config.get("mutation_rate", 0.2)
        self.elitism_rate = self.config.get("elitism_rate", 0.1)
        self.tournament_size = self.config.get("tournament_size", 3)

        # Параметры многопоточности
        self.max_workers = self.config.get("max_workers", 4)

        # Параметры оценки качества
        self.fitness_metrics = self.config.get(
            "fitness_metrics",
            {
                "total_return": 1.0,
                "sharpe_ratio": 1.0,
                "max_drawdown": -0.5,
                "win_rate": 0.3,
            },
        )

        # Настройки бэктестера
        self.backtest_settings = self.config.get(
            "backtest_settings",
            {"initial_balance": 10000.0, "commission": 0.001, "slippage": 0.0001},
        )

        # Настройки базы данных
        self.database = None
        self.save_results = self.config.get("save_results", True)

        # Флаг отмены оптимизации
        self._cancel_requested = False

        # Статистика оптимизации
        self.optimization_stats = GeneticOptimizationStats()

        # Бэктестер
        self.backtester = Backtester(self.backtest_settings)

        # Прогресс коллбэк
        self.progress_callback = None

        logger.info("Genetic optimizer initialized")

    async def optimize(
        self,
        strategy_class: type,
        parameter_ranges: Dict,
        data: pd.DataFrame,
        initial_population: List[Dict] = None,
        fitness_function: Callable = None,
        progress_callback: Callable = None,
    ) -> Dict:
        """
        Оптимизирует параметры стратегии

        Args:
            strategy_class: Класс стратегии
            parameter_ranges: Диапазоны параметров для оптимизации
            data: Исторические данные
            initial_population: Начальная популяция (список словарей с параметрами)
            fitness_function: Функция оценки качества стратегии
            progress_callback: Функция для обратного вызова при обновлении прогресса

        Returns:
            Dict: Результаты оптимизации
        """
        # Сбрасываем флаг отмены
        self._cancel_requested = False

        # Инициализируем статистику
        self.optimization_stats = GeneticOptimizationStats()

        # Устанавливаем функцию прогресса
        self.progress_callback = progress_callback

        # Устанавливаем прогресс коллбэк для бэктестера
        if progress_callback:
            self.backtester.progress_callback = self._backtest_progress_callback

        try:
            logger.info("Starting optimization for {strategy_class.__name__}" %)

            # Проверяем входные данные
            if not issubclass(strategy_class, Strategy):
                raise ValueError(
                    f"{strategy_class.__name__} is not a subclass of Strategy"
                )

            if not parameter_ranges:
                raise ValueError("Parameter ranges not specified")

            if data is None or len(data) == 0:
                raise ValueError("No data provided")

            # Создаем начальную популяцию
            population = await self._initialize_population(
                strategy_class, parameter_ranges, initial_population
            )

            # Оцениваем начальную популяцию
            await self._evaluate_population(
                population, strategy_class, data, fitness_function
            )

            # Обновляем статистику
            self.optimization_stats.update_population_stats(population)
            self.optimization_stats.generations = 1

            # Вызываем коллбэк прогресса
            await self._update_progress(1, self.generations)

            # Выполняем заданное количество поколений
            for generation in range(1, self.generations):
                # Проверяем, был ли запрос на отмену
                if self._cancel_requested:
                    logger.info("Optimization cancelled")
                    break

                # Создаем новое поколение
                new_population = await self._create_new_generation(
                    population, parameter_ranges
                )

                # Оцениваем новое поколение
                await self._evaluate_population(
                    new_population, strategy_class, data, fitness_function
                )

                # Заменяем старую популяцию новой
                population = new_population

                # Обновляем статистику
                self.optimization_stats.update_population_stats(population)
                self.optimization_stats.generations += 1

                # Вызываем коллбэк прогресса
                await self._update_progress(generation + 1, self.generations)

                logger.info("Generation {generation + 1}/{self.generations} completed" %)

            # Сортируем финальную популяцию
            population.sort(
                key=lambda ind: (
                    ind.fitness if ind.fitness is not None else float("-inf")
                ),
                reverse=True,
            )

            # Получаем лучшего индивида
            best_individual = population[0]

            # Завершаем сбор статистики
            self.optimization_stats.finish()

            # Формируем результаты
            results = {
                "best_parameters": best_individual.parameters,
                "best_fitness": best_individual.fitness,
                "best_metrics": best_individual.metrics,
                "final_population": [ind.to_dict() for ind in population],
                "optimization_stats": self.optimization_stats.get_stats(),
                "parameter_ranges": parameter_ranges,
                "strategy_class": strategy_class.__name__,
            }

            # Сохраняем результаты в базу данных
            if self.save_results and self.database:
                await self._save_optimization_results(results)

            logger.info(
                f"Optimization completed. Best fitness: {best_individual.fitness}"
            )

            return results

        except Exception as e:
            logger.error("Error during optimization: {str(e)}" %)
            logger.error(traceback.format_exc())

            # Завершаем сбор статистики
            self.optimization_stats.finish()

            # Формируем результаты с ошибкой
            results = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "optimization_stats": self.optimization_stats.get_stats(),
                "parameter_ranges": parameter_ranges,
                "strategy_class": strategy_class.__name__,
            }

            return results

    async def _initialize_population(
        self,
        strategy_class: type,
        parameter_ranges: Dict,
        initial_population: List[Dict] = None,
    ) -> List[Individual]:
        """
        Инициализирует начальную популяцию

        Args:
            strategy_class: Класс стратегии
            parameter_ranges: Диапазоны параметров для оптимизации
            initial_population: Начальная популяция (список словарей с параметрами)

        Returns:
            List[Individual]: Начальная популяция
        """
        population = []

        # Если указана начальная популяция, используем её
        if initial_population:
            for params in initial_population:
                # Валидируем параметры
                validated_params = self._validate_parameters(params, parameter_ranges)

                # Создаем индивида
                individual = Individual(validated_params)
                population.append(individual)

        # Если популяция всё еще неполная, добавляем случайных индивидов
        while len(population) < self.population_size:
            # Генерируем случайные параметры
            params = self._generate_random_parameters(parameter_ranges)

            # Создаем индивида
            individual = Individual(params)
            population.append(individual)

        logger.info("Initialized population with {len(population)} individuals" %)

        return population

    def _validate_parameters(self, params: Dict, parameter_ranges: Dict) -> Dict:
        """
        Проверяет, что параметры находятся в допустимых диапазонах

        Args:
            params: Параметры для проверки
            parameter_ranges: Диапазоны параметров

        Returns:
            Dict: Валидированные параметры
        """
        validated_params = {}

        for param_name, param_value in params.items():
            # Если параметр есть в диапазонах, проверяем его
            if param_name in parameter_ranges:
                param_range = parameter_ranges[param_name]

                # Проверяем тип диапазона
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    # Диапазон значений
                    min_val, max_val = param_range

                    # Проверяем, что значение в диапазоне
                    if isinstance(min_val, (int, float)) and isinstance(
                        max_val, (int, float)
                    ):
                        # Числовой диапазон
                        if param_value < min_val:
                            param_value = min_val
                        elif param_value > max_val:
                            param_value = max_val

                elif isinstance(param_range, list):
                    # Список допустимых значений
                    if param_value not in param_range:
                        param_value = random.choice(param_range)

            # Добавляем параметр в результат
            validated_params[param_name] = param_value

        # Проверяем, что все параметры из диапазонов присутствуют
        for param_name in parameter_ranges:
            if param_name not in validated_params:
                # Если параметр отсутствует, генерируем случайное значение
                validated_params[param_name] = self._generate_random_parameter(
                    param_name, parameter_ranges[param_name]
                )

        return validated_params

    def _generate_random_parameters(self, parameter_ranges: Dict) -> Dict:
        """
        Генерирует случайные параметры в заданных диапазонах

        Args:
            parameter_ranges: Диапазоны параметров

        Returns:
            Dict: Сгенерированные параметры
        """
        params = {}

        for param_name, param_range in parameter_ranges.items():
            params[param_name] = self._generate_random_parameter(
                param_name, param_range
            )

        return params

    def _generate_random_parameter(self, param_name: str, param_range: Any) -> Any:
        """
        Генерирует случайное значение параметра

        Args:
            param_name: Имя параметра
            param_range: Диапазон значений

        Returns:
            Any: Сгенерированное значение
        """
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Диапазон значений
            min_val, max_val = param_range

            # Проверяем тип значений
            if isinstance(min_val, int) and isinstance(max_val, int):
                # Целые числа
                return random.randint(min_val, max_val)
            elif isinstance(min_val, (int, float)) and isinstance(
                max_val, (int, float)
            ):
                # Числа с плавающей точкой
                return random.uniform(min_val, max_val)

        elif isinstance(param_range, list):
            # Список допустимых значений
            return random.choice(param_range)

        elif isinstance(param_range, bool) or param_range in [True, False]:
            # Булево значение
            return random.choice([True, False])

        else:
            # Неизвестный тип, возвращаем как есть
            return param_range

    async def _evaluate_population(
        self,
        population: List[Individual],
        strategy_class: type,
        data: pd.DataFrame,
        fitness_function: Callable = None,
    ):
        """
        Оценивает приспособленность индивидов в популяции

        Args:
            population: Список индивидов
            strategy_class: Класс стратегии
            data: Исторические данные
            fitness_function: Функция оценки качества стратегии
        """
        # Задания для выполнения
        tasks = []

        # Создаем задания для бэктестирования
        for individual in population:
            if individual.fitness is None:  # Оцениваем только новых индивидов
                task = self._evaluate_individual(
                    individual, strategy_class, data, fitness_function
                )
                tasks.append(task)

        # Обновляем счетчики оценок
        self.optimization_stats.evaluations = len(tasks)
        self.optimization_stats.total_evaluations += len(tasks)

        # Выполняем оценку с ограничением на количество одновременных задач
        batch_size = self.max_workers
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i: i + batch_size]
            await asyncio.gather(*batch)

    async def _evaluate_individual(
        self,
        individual: Individual,
        strategy_class: type,
        data: pd.DataFrame,
        fitness_function: Callable = None,
    ):
        """
        Оценивает приспособленность индивида

        Args:
            individual: Индивид
            strategy_class: Класс стратегии
            data: Исторические данные
            fitness_function: Функция оценки качества стратегии
        """
        try:
            # Создаем экземпляр стратегии с параметрами индивида
            strategy = strategy_class(parameters=individual.parameters)

            # Выполняем бэктест
            result_dict = await self.backtester.backtest(strategy, data)

            # Преобразуем результаты в экземпляр BacktestResult
            from project.backtesting.backtester import BacktestResult

            result = BacktestResult.from_dict(result_dict)

            # Рассчитываем значение функции приспособленности
            fitness = self._calculate_fitness(result, fitness_function)

            # Обновляем индивида
            individual.fitness = fitness
            individual.metrics = result.metrics

        except Exception as e:
            logger.error("Error evaluating individual: {str(e)}" %)
            individual.fitness = float("-inf")
            individual.metrics = {}

    def _calculate_fitness(
        self, result: "BacktestResult", fitness_function: Callable = None
    ) -> float:
        """
        Рассчитывает значение функции приспособленности

        Args:
            result: Результаты бэктеста
            fitness_function: Функция оценки качества стратегии

        Returns:
            float: Значение функции приспособленности
        """
        # Если указана пользовательская функция, используем её
        if fitness_function:
            return fitness_function(result)

        # Иначе используем встроенную функцию
        metrics = result.metrics

        # Если метрики пустые, возвращаем минимальное значение
        if not metrics:
            return float("-inf")

        # Рассчитываем взвешенную сумму метрик
        fitness = 0.0

        for metric_name, weight in self.fitness_metrics.items():
            if metric_name in metrics:
                # Умножаем метрику на вес
                fitness += metrics[metric_name] * weight

        return fitness

    async def _create_new_generation(
        self, population: List[Individual], parameter_ranges: Dict
    ) -> List[Individual]:
        """
        Создает новое поколение на основе текущей популяции

        Args:
            population: Текущая популяция
            parameter_ranges: Диапазоны параметров

        Returns:
            List[Individual]: Новое поколение
        """
        new_population = []

        # Сортируем популяцию по приспособленности
        population.sort(
            key=lambda ind: ind.fitness if ind.fitness is not None else float("-inf"),
            reverse=True,
        )

        # Элитизм - сохраняем лучших индивидов
        elite_count = int(self.population_size * self.elitism_rate)
        for i in range(elite_count):
            # Создаем копию индивида
            elite = Individual(
                parameters=deepcopy(population[i].parameters),
                fitness=population[i].fitness,
                metrics=(
                    deepcopy(population[i].metrics) if population[i].metrics else {}
                ),
                generation=population[i].generation + 1,
            )
            new_population.append(elite)

        # Создаем оставшихся индивидов с помощью кроссинговера и мутации
        while len(new_population) < self.population_size:
            # Выбираем родителей с помощью турнирного отбора
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Выполняем кроссинговер
            if random.random() < self.crossover_rate:
                child_params = self._crossover(parent1.parameters, parent2.parameters)
            else:
                # Если кроссинговер не выполняется, используем параметры первого родителя
                child_params = deepcopy(parent1.parameters)

            # Выполняем мутацию
            child_params = self._mutate(child_params, parameter_ranges)

            # Создаем нового индивида
            child = Individual(
                parameters=child_params, generation=population[0].generation + 1
            )

            # Добавляем в новую популяцию
            new_population.append(child)

        return new_population

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Выбирает индивида с помощью турнирного отбора

        Args:
            population: Популяция

        Returns:
            Individual: Выбранный индивид
        """
        # Выбираем случайных участников турнира
        tournament = random.sample(
            population, min(self.tournament_size, len(population))
        )

        # Выбираем лучшего участника
        return max(
            tournament,
            key=lambda ind: ind.fitness if ind.fitness is not None else float("-inf"),
        )

    def _crossover(self, params1: Dict, params2: Dict) -> Dict:
        """
        Выполняет кроссинговер двух наборов параметров

        Args:
            params1: Параметры первого родителя
            params2: Параметры второго родителя

        Returns:
            Dict: Параметры потомка
        """
        child_params = {}

        # Для каждого параметра выбираем значение от одного из родителей
        for param_name in params1.keys():
            if param_name in params2:
                # Выбираем значение случайным образом
                if random.random() < 0.5:
                    child_params[param_name] = params1[param_name]
                else:
                    child_params[param_name] = params2[param_name]
            else:
                # Если параметр есть только у первого родителя, берем его
                child_params[param_name] = params1[param_name]

        # Добавляем параметры, которые есть только у второго родителя
        for param_name in params2.keys():
            if param_name not in params1:
                child_params[param_name] = params2[param_name]

        return child_params

    def _mutate(self, params: Dict, parameter_ranges: Dict) -> Dict:
        """
        Выполняет мутацию параметров

        Args:
            params: Исходные параметры
            parameter_ranges: Диапазоны параметров

        Returns:
            Dict: Мутированные параметры
        """
        # Создаем копию параметров
        mutated_params = deepcopy(params)

        # Для каждого параметра с вероятностью mutation_rate выполняем мутацию
        for param_name, param_value in mutated_params.items():
            if random.random() < self.mutation_rate and param_name in parameter_ranges:
                # Выполняем мутацию
                param_range = parameter_ranges[param_name]

                # Выбираем тип мутации в зависимости от типа параметра
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    min_val, max_val = param_range

                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Для целых чисел
                        if random.random() < 0.5:
                            # Небольшое изменение
                            delta = random.randint(
                                1, max(1, int((max_val - min_val) * 0.1))
                            )
                            if random.random() < 0.5:
                                delta = -delta

                            new_value = param_value + delta
                            if new_value < min_val:
                                new_value = min_val
                            elif new_value > max_val:
                                new_value = max_val

                            mutated_params[param_name] = new_value
                        else:
                            # Случайное значение
                            mutated_params[param_name] = random.randint(
                                min_val, max_val
                            )

                    elif isinstance(min_val, (int, float)) and isinstance(
                        max_val, (int, float)
                    ):
                        # Для чисел с плавающей точкой
                        if random.random() < 0.5:
                            # Небольшое изменение
                            delta = random.uniform(0, (max_val - min_val) * 0.1)
                            if random.random() < 0.5:
                                delta = -delta

                            new_value = param_value + delta
                            if new_value < min_val:
                                new_value = min_val
                            elif new_value > max_val:
                                new_value = max_val

                            mutated_params[param_name] = new_value
                        else:
                            # Случайное значение
                            mutated_params[param_name] = random.uniform(
                                min_val, max_val
                            )

                elif isinstance(param_range, list):
                    # Для списка значений выбираем случайное значение
                    mutated_params[param_name] = random.choice(param_range)

                elif isinstance(param_range, bool) or param_range in [True, False]:
                    # Для булевых значений инвертируем значение
                    mutated_params[param_name] = not param_value

        return mutated_params

    async def _save_optimization_results(self, results: Dict):
        """
        Сохраняет результаты оптимизации в базу данных

        Args:
            results: Результаты оптимизации
        """
        try:
            # Проверяем, что база данных доступна
            if not self.database:
                logger.warning(
                    "Database not available, cannot save optimization results"
                )
                return

            # Генерируем ID оптимизации
            optimization_id = str(uuid.uuid4())

            # Создаем запись для сохранения
            optimization_record = {
                "optimization_id": optimization_id,
                "strategy_id": results.get("strategy_class"),
                "best_parameters": results.get("best_parameters", {}),
                "best_fitness": results.get("best_fitness"),
                "best_metrics": results.get("best_metrics", {}),
                "optimization_stats": results.get("optimization_stats", {}),
                "parameter_ranges": results.get("parameter_ranges", {}),
            }

            # Сохраняем результаты
            await self.database.save_optimization_result(optimization_record)

            logger.info("Saved optimization results with ID: {optimization_id}" %)

        except Exception as e:
            logger.error("Error saving optimization results: {str(e)}" %)

    async def _update_progress(self, current: int, total: int):
        """
        Обновляет прогресс оптимизации

        Args:
            current: Текущий прогресс
            total: Общий объем работы
        """
        # Рассчитываем прогресс
        progress = current / total if total > 0 else 0

        # Обновляем статистику
        self.optimization_stats.update_progress(progress)

        # Вызываем коллбэк прогресса
        if self.progress_callback:
            try:
                await self.progress_callback(
                    progress, self.optimization_stats.get_stats()
                )
            except Exception as e:
                logger.error("Error in progress callback: {str(e)}" %)

    async def _backtest_progress_callback(self, progress: float, info: Dict):
        """
        Коллбэк прогресса бэктеста

        Args:
            progress: Прогресс бэктеста
            info: Дополнительная информация
        """
        # Вызываем коллбэк прогресса оптимизации
        if self.progress_callback:
            # Рассчитываем общий прогресс
            current_generation = self.optimization_stats.generations
            total_generations = self.generations

            # Прогресс внутри текущего поколения
            generation_progress = self.optimization_stats.evaluations / max(
                1, self.population_size
            )

            # Общий прогресс
            total_progress = (
                current_generation - 1 + generation_progress
            ) / total_generations

            try:
                await self.progress_callback(
                    total_progress, self.optimization_stats.get_stats()
                )
            except Exception as e:
                logger.error("Error in progress callback: {str(e)}" %)

    async def cancel(self):
        """Отменяет выполнение оптимизации"""
        self._cancel_requested = True

        # Отменяем текущий бэктест, если он выполняется
        await self.backtester.cancel()

        logger.info("Optimization cancellation requested")

    def set_database(self, database: Database):
        """
        Устанавливает базу данных для сохранения результатов

        Args:
            database: База данных
        """
        self.database = database
