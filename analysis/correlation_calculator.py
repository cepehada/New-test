"""
Модуль для расчета корреляции между активами.
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np

from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CorrelationCalculator:
    """Класс для расчета корреляции между активами"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info("CorrelationCalculator инициализирован")
