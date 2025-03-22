"""
Модуль backtest_engine.
Запускает тестирование торговых стратегий на исторических данных.
"""

import csv
import logging
from typing import List, Dict, Any

logger = logging.getLogger("BacktestEngine")


def load_historical_data(csv_file: str) -> List[Dict[str, Any]]:
    """
    Загружает исторические данные из CSV файла.

    Args:
        csv_file (str): Путь к файлу CSV с историческими данными.

    Returns:
        List[Dict[str, Any]]: Список исторических данных.
    """
    data = []
    try:
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = [row for row in reader]
        logger.info(f"Загружено данных: {len(data)} строк.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
    return data


def run_backtest(csv_file: str, strategy_func) -> Dict[str, Any]:
    """
    Прогоняет стратегию на исторических данных из CSV.

    Args:
        csv_file (str): Путь к файлу CSV.
        strategy_func (callable): Функция торговой стратегии.

    Returns:
        Dict[str, Any]: Результаты бэктеста.
    """
    data = load_historical_data(csv_file)
    results = strategy_func(data)
    logger.info("Бэктест завершён.")
    return results
