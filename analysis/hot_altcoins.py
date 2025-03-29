"""
Модуль для анализа горячих альткоинов.
"""

from typing import Dict, List, Any
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HotAltcoinsAnalyzer:
    """Класс для анализа горячих альткоинов"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info("HotAltcoinsAnalyzer инициализирован")
