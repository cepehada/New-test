"""
Оптимизатор стратегий на основе перебора параметров по сетке.
Легковесная замена генетическому оптимизатору.
"""

import itertools
import logging
import time
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from project.utils.logging_utils import setup_logger

logger = setup_logger("grid_optimizer")


class GridOptimizer:
    """Оптимизатор на основе перебора параметров по сетке"""

    def __init__(self, config: Dict = None):
        """
        Инициализирует оптимизатор
        
        Args:
            config: Конфигурация оптимизатора
        """
        self.config = config or {}
        self.max_workers = self.config.get("max_workers", 4)
        self.use_threads = self.config.get("use_threads", True)
        
        # Параметры для функции оценки производительности
        self.fitness_metrics = self.config.get("fitness_metrics", {
            "net_profit": 1.0,
            "win_rate": 1.0, 
            "profit_factor": 0.5,
            "max_drawdown": -1.0,
            "sharpe_ratio": 0.5,
            "num_trades": 0.2
        })
        
        logger.info("Grid optimizer initialized")

    async def optimize(
        self, 
        strategy_class: Type, 
        param_ranges: Dict[str, Union[List, Tuple, range]], 
        data: pd.DataFrame, 
        **kwargs
    ) -> Dict:
        """
        Проводит оптимизацию стратегии
        
        Args:
            strategy_class: Класс стратегии
            param_ranges: Диапазоны значений параметров
            data: Исторические данные
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict: Результаты оптимизации
        """
        start_time = time.time()
        logger.info(f"Starting grid optimization for {strategy_class.__name__}")

        # Создаем сетку параметров
        param_grid = self._create_param_grid(param_ranges)
        total_combinations = len(param_grid)
        
        logger.info(f"Generated parameter grid with {total_combinations} combinations")
        
        # Создаем функцию для оценки комбинации параметров
        def evaluate_params(params):
            try:
                # Создаем экземпляр стратегии с текущими параметрами
                strategy_params = kwargs.copy()
                strategy_params.update(params)
                strategy = strategy_class(**strategy_params)
                
                # Запускаем бэктестирование
                strategy.backtest(data)
                
                # Получаем результаты
                result = strategy.get_results()
                
                # Рассчитываем итоговую оценку
                fitness = self._calculate_fitness(result)
                
                return {
                    "parameters": params,
                    "fitness": fitness,
                    "results": result
                }
            except Exception as e:
                logger.warning(f"Error evaluating parameters {params}: {str(e)}")
                return {
                    "parameters": params,
                    "fitness": -999999,
                    "results": {},
                    "error": str(e)
                }
        
        # Запускаем оценку всех комбинаций параметров
        all_results = []
        
        # Использовать многопоточность, если указано в конфигурации
        if self.max_workers > 1:
            executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
            with executor_class(max_workers=self.max_workers) as executor:
                futures = [executor.submit(evaluate_params, params) for params in param_grid]
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    all_results.append(result)
                    
                    # Логируем прогресс
                    if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
                        logger.info(f"Evaluated {i + 1}/{total_combinations} parameter combinations")
        else:
            # Последовательная оценка
            for i, params in enumerate(param_grid):
                result = evaluate_params(params)
                all_results.append(result)
                
                # Логируем прогресс
                if (i + 1) % 10 == 0 or (i + 1) == total_combinations:
                    logger.info(f"Evaluated {i + 1}/{total_combinations} parameter combinations")
        
        # Сортируем результаты по фитнесу (в убывающем порядке)
        all_results.sort(key=lambda x: x["fitness"], reverse=True)
        
        # Получаем лучший результат
        best_result = all_results[0] if all_results else None
        
        # Формируем итоговый результат
        result = {
            "best_params": best_result["parameters"] if best_result else None,
            "best_fitness": best_result["fitness"] if best_result else None,
            "best_result": best_result["results"] if best_result else None,
            "all_results": all_results,
            "execution_time": time.time() - start_time,
            "total_combinations": total_combinations
        }
        
        logger.info(
            f"Grid optimization completed in {result['execution_time']:.2f} seconds. "
            f"Best fitness: {result['best_fitness']}"
        )
        
        return result

    def _create_param_grid(self, param_ranges: Dict[str, Union[List, Tuple, range]]) -> List[Dict]:
        """
        Создает сетку параметров
        
        Args:
            param_ranges: Диапазоны значений параметров
            
        Returns:
            List[Dict]: Список всех комбинаций параметров
        """
        # Преобразуем все диапазоны в списки
        param_lists = {}
        
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, (list, tuple, range)):
                param_lists[param_name] = list(param_range)
            else:
                # Если передано одно значение, создаем список с одним элементом
                param_lists[param_name] = [param_range]
        
        # Получаем все комбинации параметров
        keys = param_lists.keys()
        values = param_lists.values()
        
        param_grid = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            param_grid.append(param_dict)
        
        return param_grid

    def _calculate_fitness(self, results: Dict) -> float:
        """
        Рассчитывает совокупную оценку результатов
        
        Args:
            results: Результаты бэктестирования
            
        Returns:
            float: Оценка производительности
        """
        if not results:
            return -999999
        
        fitness = 0.0
        
        # Учитываем каждую метрику с соответствующим весом
        for metric, weight in self.fitness_metrics.items():
            if metric in results:
                value = float(results[metric])
                fitness += value * weight
        
        return fitness
