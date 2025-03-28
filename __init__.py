"""
Основной пакет алгоритмического торгового бота.
Включает в себя модули для управления стратегиями, исполнения ордеров,
управления рисками, технического анализа и интеграций с внешними API.
"""

__version__ = "1.0.0"
__author__ = "cepehada"
__email__ = "your.email@example.com"

# Инициализация логгера на самом верхнем уровне
import logging

from project.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.info("Initializing trading bot v{__version__}")
