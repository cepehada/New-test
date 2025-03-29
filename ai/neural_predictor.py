"""
Модуль для прогнозирования цен с использованием нейронных сетей.
"""

from typing import Dict, List, Any
import numpy as np
import tensorflow as tf

from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class NeuralPredictor:
    """Класс для прогнозирования цен с использованием нейронных сетей"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        logger.info("NeuralPredictor инициализирован")
