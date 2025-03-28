"""
Trading Bot System
A modular cryptocurrency trading platform with strategy backtesting and live trading.
"""

__version__ = "0.1.0"
__author__ = "Trading Bot Team"

# Main package exports
from project.config.configuration import get_config
from project.utils.logging_utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)
