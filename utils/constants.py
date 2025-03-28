"""
Модуль с константами проекта.
Содержит глобальные константы и конфигурационные значения.
"""

import logging

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# Общие константы
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30.0
REQUEST_HEADERS = {"User-Agent": "Trading Bot/1.0", "Content-Type": "application/json"}

# Константы для бирж
# Исправление оператора 'не в' на 'not in'
def is_order_active(order_id, active_orders):
    if order_id not in active_orders:  # Было: order_id не в self.active_orders
        logger.warning(f"Ордер {order_id} не найден в активных ордерах")
        return False
    return True
API_VERSION = "v1"  # Assuming this is the correct fix
EXCHANGE_TIMEOUT = 60.0
EXCHANGE_RATE_LIMIT_MARGIN = 0.8
MAX_ORDERBOOK_DEPTH = 100
DEFAULT_TIMEFRAME = "1h"

# Константы для работы с данными
MAX_OHLCV_LIMIT = 1000
MIN_DATA_POINTS = 20
MAX_DATA_POINTS = 5000
CACHE_EXPIRATION = 60  # секунды
# Константы для обработки данных
DATA_DIRECTORY = "data/"
PROCESSED_DATA_DIRECTORY = "data/processed/"
RAW_DATA_DIRECTORY = "data/raw/"
MODELS_DIRECTORY = "models/"
DEFAULT_CSV_DELIMITER = ","
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Константы для базы данных
DATABASE_URL = "sqlite:///trading_bot.db"
DB_POOL_SIZE = 5
DB_MAX_OVERFLOW = 10
DB_CONNECTION_TIMEOUT = 30

# Константы для бэктестинга
BACKTEST_START_DATE = "2020-01-01"
BACKTEST_END_DATE = "2023-01-01"
BACKTEST_INITIAL_CAPITAL = 10000
BACKTEST_COMMISSION = 0.001

# Константы управления рисками
MAX_POSITION_SIZE_RATIO = 0.1  # Максимальный размер позиции в % от капитала
STOP_LOSS_PERCENTAGE = 0.02  # Стоп-лосс 2%
TAKE_PROFIT_PERCENTAGE = 0.06  # Тейк-профит 6%
RISK_REWARD_RATIO = 3.0  # Соотношение риск/прибыль
# Константы для торговли
DEFAULT_FEE = 0.001
MAX_TRADE_AMOUNT = 10000
MIN_TRADE_AMOUNT = 10
TRADE_SLIPPAGE = 0.005
