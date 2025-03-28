from typing import Dict, Any, Optional, Type, Tuple
import ccxt
from project.utils.logging_utils import setup_logger

logger = setup_logger("exchange_errors")

# Маппинг известных ошибок бирж к унифицированным ошибкам
ERROR_MAPPING = {
    # Общие ошибки CCXT
    ccxt.NetworkError: "network_error",
    ccxt.ExchangeError: "exchange_error",
    ccxt.InvalidOrder: "invalid_order",
    ccxt.InsufficientFunds: "insufficient_funds",
    ccxt.OrderNotFound: "order_not_found",
    ccxt.AuthenticationError: "auth_error",
    ccxt.RateLimitExceeded: "rate_limit",
    # Специфичные ошибки бирж
    # Binance
    "binance_error_1": "insufficient_balance",
    "binance_error_4": "invalid_price",
    # Bybit
    "bybit_error_10001": "param_error",
    "bybit_error_10002": "system_error",
    # HTX
    "htx_error_1002": "auth_error",
    # PHEMEX
    "phemex_error_11001": "invalid_symbol",
    # MEX (BitMEX)
    "mex_error_404": "not_found",
}


class ExchangeErrorHandler:
    """Обработчик ошибок бирж"""

    @staticmethod
    def handle_error(
        e: Exception, exchange_id: str = "unknown"
    ) -> Tuple[str, str, Any]:
        """
        Обрабатывает ошибку биржи

        Args:
            e: Исключение
            exchange_id: Идентификатор биржи

        Returns:
            Tuple[str, str, Any]: (код ошибки, сообщение, дополнительные данные)
        """
        error_code = "unknown_error"
        error_msg = str(e)
        error_data = None

        # Обработка ошибок CCXT
        if isinstance(e, ccxt.BaseError):
            for error_class, mapped_code in ERROR_MAPPING.items():
                if isinstance(e, error_class):
                    error_code = mapped_code
                    break

            # Извлекаем данные ошибки
            if hasattr(e, "json_response") and e.json_response:
                error_data = e.json_response

        # Обработка специфичных ошибок бирж
        elif exchange_id == "binance" and "code" in str(e):
            # Пример обработки ошибки Binance
            import re

            code_match = re.search(r"code\s*[:=]\s*(-?\d+)", str(e))
            if code_match:
                binance_code = f"binance_error_{code_match.group(1)}"
                error_code = ERROR_MAPPING.get(binance_code, "binance_error")

        # Логирование ошибки
        logger.error(f"{exchange_id} error: {error_code} - {error_msg}")

        return error_code, error_msg, error_data
