"""
Модуль для управления торговыми символами.
Предоставляет интерфейс для получения информации о символах и их параметрах.
"""

from typing import Any, Dict, List, Optional

from project.config import get_config
from project.utils.cache_utils import async_cache
from project.utils.ccxt_exchanges import connect_exchange
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SymbolManager:
    """
    Класс для управления информацией о торговых символах.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "SymbolManager":
        """
        Получает экземпляр класса SymbolManager (Singleton).

        Returns:
            Экземпляр класса SymbolManager
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Инициализирует менеджер символов.
        """
        self.config = get_config()
        self.symbols_cache: Dict[str, Dict[str, Any]] = {}
        self.markets_cache: Dict[str, Dict[str, Any]] = {}
        self.symbols_info: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.preferred_quote_assets = ["USDT", "BUSD", "USDC", "BTC", "ETH"]
        logger.debug("Создан экземпляр SymbolManager")

    @async_handle_error
    @async_cache(ttl=3600.0)  # Кэшируем на 1 час
    async def load_markets(self, exchange_id: str) -> Dict[str, Any]:
        """
        Загружает информацию о рынках для указанной биржи.

        Args:
            exchange_id: Идентификатор биржи

        Returns:
            Информация о рынках
        """
        try:
            exchange = await connect_exchange(exchange_id)
            markets = await exchange.load_markets()

            # Сохраняем информацию в кэш
            self.markets_cache[exchange_id] = markets
            logger.info(
                f"Загружена информация о {len(markets)} рынках на {exchange_id}"
            )

            # Извлекаем и сохраняем информацию о символах
            symbols_info = {}
            for symbol, market in markets.items():
                symbols_info[symbol] = {
                    "id": market.get("id", ""),
                    "base": market.get("base", ""),
                    "quote": market.get("quote", ""),
                    "active": market.get("active", True),
                    "precision": market.get("precision", {}),
                    "limits": market.get("limits", {}),
                    "type": market.get("type", "spot"),
                    "contract": market.get("contract", False),
                    "linear": market.get("linear", None),
                    "inverse": market.get("inverse", None),
                    "maker": market.get("maker", 0),
                    "taker": market.get("taker", 0),
                }

            self.symbols_info[exchange_id] = symbols_info
            return markets

        except Exception as e:
            logger.error(f"Ошибка при загрузке рынков для {exchange_id}: {str(e)}" )
            raise

    @async_handle_error
    async def get_symbol_info(
        self, exchange_id: str, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Получает информацию о символе.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Информация о символе или None, если символ не найден
        """
        # Проверяем, загружены ли рынки
        if exchange_id not in self.symbols_info:
            await self.load_markets(exchange_id)

        # Проверяем наличие символа в кэше
        if symbol in self.symbols_info.get(exchange_id, {}):
            return self.symbols_info[exchange_id][symbol]

        # Нормализуем символ и повторяем поиск
        normalized_symbol = symbol.replace("_", "/").upper()
        if normalized_symbol in self.symbols_info.get(exchange_id, {}):
            return self.symbols_info[exchange_id][normalized_symbol]

        logger.warning(f"Символ {symbol} не найден на {exchange_id}" )
        return None

    @async_handle_error
    async def get_active_symbols(
        self, exchange_id: str, quote_asset: Optional[str] = None
    ) -> List[str]:
        """
        Получает список активных символов на бирже.

        Args:
            exchange_id: Идентификатор биржи
            quote_asset: Валюта котировки для фильтрации (None для всех валют)

        Returns:
            Список активных символов
        """
        # Проверяем, загружены ли рынки
        if exchange_id not in self.symbols_info:
            await self.load_markets(exchange_id)

        symbols = []
        for symbol, info in self.symbols_info.get(exchange_id, {}).items():
            if info.get("active", True):
                if quote_asset:
                    if info.get("quote") == quote_asset:
                        symbols.append(symbol)
                else:
                    symbols.append(symbol)

        logger.debug(f"Получено {len(symbols)} активных символов на {exchange_id}" )
        return symbols

    @async_handle_error
    async def get_tradable_symbols(
        self,
        exchange_id: str,
        min_volume: float = 0.0,
        quote_assets: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Получает список торгуемых символов с учетом фильтров.

        Args:
            exchange_id: Идентификатор биржи
            min_volume: Минимальный объем торгов
            quote_assets: Список валют котировок для фильтрации (None для использования предпочтительных)

        Returns:
            Список торгуемых символов
        """
        from project.data.market_data import MarketData

        # Используем предпочтительные валюты котировок, если не указаны
        quote_assets = quote_assets or self.preferred_quote_assets

        # Загружаем рынки, если еще не загружены
        if exchange_id not in self.symbols_info:
            await self.load_markets(exchange_id)

        # Фильтруем символы по активности и валютам котировок
        active_symbols = []
        for symbol, info in self.symbols_info.get(exchange_id, {}).items():
            if info.get("active", True) and info.get("quote") in quote_assets:
                active_symbols.append(symbol)

        logger.debug(
            f"Найдено {len(active_symbols)} активных символов для {quote_assets} на {exchange_id}"
        )

        # Если нет минимального объема, возвращаем активные символы
        if min_volume <= 0:
            return active_symbols

        # Фильтруем по объему торгов
        tradable_symbols = []
        market_data = MarketData.get_instance()

        for symbol in active_symbols:
            try:
                ticker = await market_data.get_ticker(exchange_id, symbol)
                volume = ticker.get("quoteVolume", 0) or ticker.get("volume", 0)

                if volume >= min_volume:
                    tradable_symbols.append(symbol)
            except Exception as e:
                logger.warning(
                    f"Ошибка при получении тикера для {symbol} на {exchange_id}: {str(e)}"
                )

        logger.info(
            f"Найдено {
                len(tradable_symbols)} торгуемых символов с объемом >= {min_volume} на {exchange_id}")
        return tradable_symbols

    @async_handle_error
    async def get_precision(self, exchange_id: str, symbol: str) -> Dict[str, int]:
        """
        Получает данные о точности для указанного символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Словарь с данными о точности
        """
        symbol_info = await self.get_symbol_info(exchange_id, symbol)

        if not symbol_info or "precision" not in symbol_info:
            logger.warning(f"Нет данных о точности для {symbol} на {exchange_id}" )
            return {"amount": 8, "price": 8, "cost": 8}

        precision = symbol_info["precision"]

        # Нормализуем формат точности (разные биржи могут использовать разные форматы)
        if isinstance(precision, dict):
            return {
                "amount": precision.get("amount", 8),
                "price": precision.get("price", 8),
                "cost": precision.get("cost", 8) if "cost" in precision else 8,
            }
        else:
            # Некоторые биржи могут возвращать точность в виде числа
            return {"amount": precision, "price": precision, "cost": precision}

    @async_handle_error
    async def get_limits(
        self, exchange_id: str, symbol: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Получает данные о лимитах для указанного символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Словарь с данными о лимитах
        """
        symbol_info = await self.get_symbol_info(exchange_id, symbol)

        if not symbol_info or "limits" not in symbol_info:
            logger.warning(f"Нет данных о лимитах для {symbol} на {exchange_id}" )
            return {
                "amount": {"min": 0.0, "max": float("inf")},
                "price": {"min": 0.0, "max": float("inf")},
                "cost": {"min": 0.0, "max": float("inf")},
            }

        limits = symbol_info["limits"]

        # Нормализуем формат лимитов
        return {
            "amount": {
                "min": limits.get("amount", {}).get("min", 0.0),
                "max": limits.get("amount", {}).get("max", float("inf")),
            },
            "price": {
                "min": limits.get("price", {}).get("min", 0.0),
                "max": limits.get("price", {}).get("max", float("inf")),
            },
            "cost": {
                "min": limits.get("cost", {}).get("min", 0.0),
                "max": limits.get("cost", {}).get("max", float("inf")),
            },
        }

    @async_handle_error
    async def get_fees(self, exchange_id: str, symbol: str) -> Dict[str, float]:
        """
        Получает данные о комиссиях для указанного символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Словарь с данными о комиссиях
        """
        symbol_info = await self.get_symbol_info(exchange_id, symbol)

        if not symbol_info:
            logger.warning(f"Нет данных о комиссиях для {symbol} на {exchange_id}" )
            return {"maker": 0.001, "taker": 0.001}

        return {
            "maker": symbol_info.get("maker", 0.001),
            "taker": symbol_info.get("taker", 0.001),
        }

    @async_handle_error
    async def normalize_amount(
        self, exchange_id: str, symbol: str, amount: float
    ) -> float:
        """
        Нормализует количество для указанного символа с учетом точности и лимитов.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары
            amount: Исходное количество

        Returns:
            Нормализованное количество
        """
        precision = await self.get_precision(exchange_id, symbol)
        limits = await self.get_limits(exchange_id, symbol)

        # Округляем до указанной точности
        amount_precision = precision.get("amount", 8)
        if isinstance(amount_precision, int):
            # Если точность выражена числом десятичных знаков
            normalized_amount = round(amount, amount_precision)
        else:
            # Если точность выражена шагом
            normalized_amount = round(amount / amount_precision) * amount_precision

        # Ограничиваем минимальным и максимальным значениями
        amount_min = limits.get("amount", {}).get("min", 0.0)
        amount_max = limits.get("amount", {}).get("max", float("inf"))

        normalized_amount = max(normalized_amount, amount_min)
        normalized_amount = min(normalized_amount, amount_max)

        return normalized_amount

    @async_handle_error
    async def normalize_price(
        self, exchange_id: str, symbol: str, price: float
    ) -> float:
        """
        Нормализует цену для указанного символа с учетом точности и лимитов.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары
            price: Исходная цена

        Returns:
            Нормализованная цена
        """
        precision = await self.get_precision(exchange_id, symbol)
        limits = await self.get_limits(exchange_id, symbol)

        # Округляем до указанной точности
        price_precision = precision.get("price", 8)
        if isinstance(price_precision, int):
            # Если точность выражена числом десятичных знаков
            normalized_price = round(price, price_precision)
        else:
            # Если точность выражена шагом
            normalized_price = round(price / price_precision) * price_precision

        # Ограничиваем минимальным и максимальным значениями
        price_min = limits.get("price", {}).get("min", 0.0)
        price_max = limits.get("price", {}).get("max", float("inf"))

        normalized_price = max(normalized_price, price_min)
        normalized_price = min(normalized_price, price_max)

        return normalized_price
