import asyncio
import logging

import pandas as pd

# Импорт новых модулей: генетического оптимизатора и визуализатора
from project.optimizers.genetic_optimizer import GeneticOptimizer
from project.visualizers.data_visualizer import DataVisualizer

from config import get_config

logger = logging.getLogger(__name__)


async def run_optimization():
    config = get_config()
    # Извлекаем настройки оптимизатора
    optimizer_conf = config.GENETIC_OPTIMIZER_SETTINGS
    optimizer_config = {
        "population_size": optimizer_conf.POPULATION_SIZE,
        "generations": optimizer_conf.GENERATIONS,
        "crossover_rate": optimizer_conf.CROSSOVER_RATE,
        "mutation_rate": optimizer_conf.MUTATION_RATE,
        "elitism_rate": optimizer_conf.ELITISM_RATE,
        "tournament_size": optimizer_conf.TOURNAMENT_SIZE,
        "max_workers": optimizer_conf.MAX_WORKERS,
        "backtest_settings": {
            "initial_balance": optimizer_conf.BACKTEST_INITIAL_BALANCE,
            "commission": optimizer_conf.BACKTEST_COMMISSION,
            "slippage": optimizer_conf.BACKTEST_SLIPPAGE,
        },
        "fitness_metrics": optimizer_conf.FITNESS_METRICS,
    }

    genetic_optimizer = GeneticOptimizer(config=optimizer_config)

    # Для примера – используем базовый класс стратегии; замените на вашу конкретную стратегию
    from project.trading.strategy_base import Strategy

    strategy_class = Strategy

    # Пример диапазонов параметров
    parameter_ranges = {"param1": (0, 100), "param2": [1, 2, 3, 4]}

    # Пример исторических данных для бэктестинга
    data = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 150],
        },
        index=pd.date_range("2023-01-01", periods=3),
    )

    # Запуск оптимизации
    results = await genetic_optimizer.optimize(strategy_class, parameter_ranges, data)
    logger.info("Результаты оптимизации: {results}" %)


def run_visualization():
    config = get_config()
    visualizer_conf = config.DATA_VISUALIZER_SETTINGS
    visualizer = DataVisualizer(
        theme=visualizer_conf.THEME, figsize=visualizer_conf.FIGSIZE
    )

    # Пример данных для визуализации OHLC
    data = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 150],
        },
        index=pd.date_range("2023-01-01", periods=3),
    )

    # Вызов метода визуализации OHLC (возвращает base64-кодированное изображение)
    ohlc_img = visualizer.plot_ohlc(data, title="OHLC Chart")
    if ohlc_img:
        logger.info("OHLC график успешно построен")
    else:
        logger.error("Ошибка построения OHLC графика")

    # Дополнительно можно вызвать другие методы визуализации, например, для equity curve


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        # Запуск оптимизации в event loop
        asyncio.run(run_optimization())
    except Exception as e:
        logger.error("Ошибка в оптимизации: {e}" %)

    run_visualization()
