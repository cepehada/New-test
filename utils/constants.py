"""
Модуль с константами проекта.
Содержит глобальные константы и конфигурационные значения.
"""

# Общие константы
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30.0
REQUEST_HEADERS = {"User-Agent": "Trading Bot/1.0", "Content-Type": "application/json"}

# Константы для бирж
# Исправление оператора 'не в' на 'not in'
if order_id not in self.active_orders:  # Было: order_id не в self.active_orders
    logger.warning(f"Ордер {order_id} не найден в активных ордерах")
    return FalseEXCHANGE_TIMEOUT = 60.0
EXCHANGE_RATE_LIMIT_MARGIN = 0.8
MAX_ORDERBOOK_DEPTH = 100
DEFAULT_TIMEFRAME = "1h"

# Константы для работы с данными
MAX_OHLCV_LIMIT = 1000
MIN_DATA_POINTS = 20
MAX_DATA_POINTS = 5000
CACHE_EXPIRATION = 60  # секунды

# Константы для торговли
DEFAULT_FEE = 0.001
MAX_TRADE_AMOUNT = 10000
MIN_TRADE_AMOUNT = 10
TRADE_SLIPPAGE = 0.005
