"""
Базовые классы и функции для арбитражной торговли.
Предоставляет инструменты для поиска и анализа арбитражных возможностей.
"""

# Стандартные импорты
import time
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set

# Сторонние импорты
import pandas as pd

# Внутренние импорты
from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.ccxt_exchanges import fetch_ticker, fetch_order_book, fetch_balance
from project.data.market_data import MarketData


logger = get_logger(__name__)


@dataclass
class ArbitrageOpportunity:
    """
    Класс для представления арбитражной возможности между биржами.
    """

    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    volume: float
    profit_usd: float
    profit_margin_pct: float
    buy_fee_pct: float = 0.0
    sell_fee_pct: float = 0.0
    timestamp: float = 0.0
    is_valid: bool = True
    estimated_profit: float = 0.0
    validation_count: int = 0
    last_update: float = 0.0


class ArbitrageCore:
    """
    Основной класс для обнаружения и анализа арбитражных возможностей.
    """

    def __init__(self, config=None):
        """
        Инициализирует ядро арбитража.

        Args:
            config (Dict, optional): Конфигурация. По умолчанию None.
        """
        self.config = config or get_config()
        self.market_data = MarketData.get_instance()

        # Настройки арбитража
        arb_config = self.config.get("arbitrage", {})
        self.min_profit_pct = arb_config.get("min_profit_pct", 0.5)
        self.min_volume_usd = arb_config.get("min_volume_usd", 10.0)
        self.max_validation_age = arb_config.get("max_validation_age", 60.0)
        self.max_trade_execution_time = arb_config.get("max_trade_execution_time", 5.0)

        # Регистрация бирж и символов
        self.exchanges = arb_config.get("exchanges", ["binance", "kucoin", "okx"])
        self.symbols = arb_config.get("symbols", [])
        self.excluded_pairs = set(arb_config.get("excluded_pairs", []))

        # Данные о комиссиях
        self.fee_data = {}
        for exchange in self.exchanges:
            self.fee_data[exchange] = arb_config.get("fees", {}).get(exchange, 0.1)

        # Статистика
        self.stats = {
            "total_checks": 0,
            "opportunities_found": 0,
            "opportunities_verified": 0,
            "opportunities_executed": 0,
            "total_profit": 0.0,
            "start_time": time.time(),
        }

    @async_handle_error
    async def check_arbitrage_opportunities(
        self, symbols=None, exchanges=None, min_profit_pct=None, min_volume_usd=None
    ) -> List[ArbitrageOpportunity]:
        """
        Проверяет доступные арбитражные возможности для заданных символов и бирж.

        Args:
            symbols: Список символов для проверки
            exchanges: Список бирж для проверки
            min_profit_pct: Минимальный процент прибыли
            min_volume_usd: Минимальный объем в USD

        Returns:
            Список обнаруженных арбитражных возможностей
        """
        try:
            # Используем значения по умолчанию, если не заданы
            symbols = symbols or self.symbols
            exchanges = exchanges or self.exchanges
            min_profit_pct = min_profit_pct or self.min_profit_pct
            min_volume_usd = min_volume_usd or self.min_volume_usd

            # Обновляем статистику
            self.stats["total_checks"] += 1

            opportunities = []

            # Получаем данные по всем биржам для всех символов
            exchange_data = {}

            for symbol in symbols:
                if symbol in self.excluded_pairs:
                    continue

                tickers = {}

                for exchange in exchanges:
                    try:
                        ticker = await fetch_ticker(exchange=exchange, symbol=symbol)
                        if (
                            ticker
                            and ticker.get("last")
                            and ticker.get("bid")
                            and ticker.get("ask")
                        ):
                            tickers[exchange] = ticker
                    except Exception as e:
                        logger.debug(
                            "Ошибка при получении тикера %s на %s: %s",
                            symbol,
                            exchange,
                            str(e),
                        )

                # Если меньше двух бирж с данными, арбитраж невозможен
                if len(tickers) < 2:
                    continue

                exchange_data[symbol] = tickers

            # Анализируем и находим арбитражные возможности
            for symbol, tickers in exchange_data.items():
                exchanges_list = list(tickers.keys())

                # Перебираем все пары бирж
                for i in range(len(exchanges_list)):
                    for j in range(i + 1, len(exchanges_list)):
                        exchange1 = exchanges_list[i]
                        exchange2 = exchanges_list[j]

                        ticker1 = tickers[exchange1]
                        ticker2 = tickers[exchange2]

                        # Проверяем возможность арбитража в обе стороны

                        # exchange1 -> exchange2
                        buy_price = ticker1["ask"]
                        sell_price = ticker2["bid"]
                        fee1 = self.fee_data.get(exchange1, 0.1) / 100
                        fee2 = self.fee_data.get(exchange2, 0.1) / 100

                        # Расчет объема и прибыли
                        max_volume = min(
                            ticker1.get("askVolume", ticker1.get("volume", 0)),
                            ticker2.get("bidVolume", ticker2.get("volume", 0)),
                        )

                        # Расчет потенциальной прибыли с учетом комиссий
                        buy_cost = buy_price * max_volume * (1 + fee1)
                        sell_proceed = sell_price * max_volume * (1 - fee2)

                        profit_usd = sell_proceed - buy_cost
                        profit_margin_pct = (
                            (profit_usd / buy_cost) * 100 if buy_cost > 0 else 0
                        )

                        if (
                            profit_margin_pct > min_profit_pct
                            and max_volume * buy_price > min_volume_usd
                        ):

                            opportunity = ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=exchange1,
                                sell_exchange=exchange2,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                volume=max_volume,
                                profit_usd=profit_usd,
                                profit_margin_pct=profit_margin_pct,
                                buy_fee_pct=fee1 * 100,
                                sell_fee_pct=fee2 * 100,
                                timestamp=time.time(),
                            )
                            opportunities.append(opportunity)

                            self.stats["opportunities_found"] += 1

                        # exchange2 -> exchange1
                        buy_price = ticker2["ask"]
                        sell_price = ticker1["bid"]

                        # Расчет объема и прибыли
                        max_volume = min(
                            ticker2.get("askVolume", ticker2.get("volume", 0)),
                            ticker1.get("bidVolume", ticker1.get("volume", 0)),
                        )

                        buy_cost = buy_price * max_volume * (1 + fee2)
                        sell_proceed = sell_price * max_volume * (1 - fee1)

                        profit_usd = sell_proceed - buy_cost
                        profit_margin_pct = (
                            (profit_usd / buy_cost) * 100 if buy_cost > 0 else 0
                        )

                        if (
                            profit_margin_pct > min_profit_pct
                            and max_volume * buy_price > min_volume_usd
                        ):

                            opportunity = ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=exchange2,
                                sell_exchange=exchange1,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                volume=max_volume,
                                profit_usd=profit_usd,
                                profit_margin_pct=profit_margin_pct,
                                buy_fee_pct=fee2 * 100,
                                sell_fee_pct=fee1 * 100,
                                timestamp=time.time(),
                            )
                            opportunities.append(opportunity)

                            self.stats["opportunities_found"] += 1

            # Сортируем возможности по убыванию прибыли
            opportunities.sort(key=lambda x: x.profit_margin_pct, reverse=True)

            return opportunities

        except Exception as e:
            logger.warning("Ошибка при проверке арбитражных возможностей: %s", str(e))
            return []

    @async_handle_error
    async def verify_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        min_volume: float = 0.0,
        max_age_seconds: float = 60.0,
    ) -> bool:
        """
        Проверяет актуальность арбитражной возможности.

        Args:
            opportunity: Объект арбитражной возможности
            min_volume: Минимальный объем для торговли
            max_age_seconds: Максимальный возраст возможности в секундах

        Returns:
            True если возможность остается выгодной, иначе False
        """
        try:
            current_time = time.time()
            opportunity_age = current_time - opportunity.timestamp

            # Проверка возраста возможности
            if opportunity_age > max_age_seconds:
                logger.debug(
                    "Возможность устарела: %s - возраст: %.2f с (макс: %.2f с)",
                    opportunity.symbol,
                    opportunity_age,
                    max_age_seconds,
                )
                return False

            # Получаем актуальные данные
            buy_ticker = await fetch_ticker(
                exchange=opportunity.buy_exchange, symbol=opportunity.symbol
            )
            sell_ticker = await fetch_ticker(
                exchange=opportunity.sell_exchange, symbol=opportunity.symbol
            )

            # Проверяем наличие данных
            if not buy_ticker or not sell_ticker:
                return False

            # Обновляем цены
            buy_price = buy_ticker.get("ask", 0)
            sell_price = sell_ticker.get("bid", 0)

            if buy_price <= 0 or sell_price <= 0:
                return False

            # Пересчитываем потенциальную прибыль
            fee1 = self.fee_data.get(opportunity.buy_exchange, 0.1) / 100
            fee2 = self.fee_data.get(opportunity.sell_exchange, 0.1) / 100

            max_volume = min(
                buy_ticker.get("askVolume", buy_ticker.get("volume", 0)),
                sell_ticker.get("bidVolume", sell_ticker.get("volume", 0)),
            )

            if max_volume < min_volume:
                return False

            buy_cost = buy_price * max_volume * (1 + fee1)
            sell_proceed = sell_price * max_volume * (1 - fee2)

            profit_usd = sell_proceed - buy_cost
            profit_margin_pct = (profit_usd / buy_cost) * 100 if buy_cost > 0 else 0

            # Обновляем данные возможности
            updated_opportunity = dataclasses.replace(
                opportunity,
                buy_price=buy_price,
                sell_price=sell_price,
                volume=max_volume,
                profit_usd=profit_usd,
                profit_margin_pct=profit_margin_pct,
                validation_count=opportunity.validation_count + 1,
                last_update=current_time,
            )

            # Проверяем, остается ли возможность выгодной
            if updated_opportunity.profit_margin_pct > 0:
                # Обновляем статистику
                self.stats["opportunities_verified"] += 1
                return True

            logger.debug(
                "Возможность больше не выгодна: %s - прибыль: %.2f%%",
                opportunity.symbol,
                updated_opportunity.profit_margin_pct,
            )
            return False

        except Exception as e:
            logger.warning(
                "Ошибка при проверке возможности %s: %s", opportunity.symbol, str(e)
            )
            return False

    @async_handle_error
    async def check_balances(
        self, opportunity: ArbitrageOpportunity, min_trade_amount: float = 10.0
    ) -> Dict[str, float]:
        """
        Проверяет наличие достаточных балансов для торговли.

        Args:
            opportunity: Объект арбитражной возможности
            min_trade_amount: Минимальная сумма для торговли в USD

        Returns:
            Словарь с рассчитанными размерами сделок или пустой словарь,
            если балансы недостаточны
        """
        try:
            # Получаем символы базовой и котируемой валюты
            symbol_parts = opportunity.symbol.split("/")
            if len(symbol_parts) != 2:
                logger.warning("Некорректный формат символа: %s", opportunity.symbol)
                return {}

            base_currency, quote_currency = symbol_parts

            # Получаем балансы на биржах
            buy_balance = await fetch_balance(opportunity.buy_exchange)
            sell_balance = await fetch_balance(opportunity.sell_exchange)

            if not buy_balance or not sell_balance:
                logger.warning("Не удалось получить балансы")
                return {}

            # Получаем свободные балансы
            buy_quote_balance = buy_balance.get("free", {}).get(quote_currency, 0)
            sell_base_balance = sell_balance.get("free", {}).get(base_currency, 0)

            # Проверяем, достаточно ли балансов для минимальной торговли
            buy_amount_max = buy_quote_balance / opportunity.buy_price

            # Определяем максимальный размер сделки
            max_buy_amount = min(buy_amount_max, sell_base_balance, opportunity.volume)

            # Проверяем минимальные требования
            if max_buy_amount * opportunity.buy_price < min_trade_amount:
                logger.debug(
                    "Недостаточный баланс для арбитража %s: "
                    "максимальная сумма покупки %.2f USD, требуется %.2f USD",
                    opportunity.symbol,
                    max_buy_amount * opportunity.buy_price,
                    min_trade_amount,
                )
                return {}

            # Рассчитываем параметры сделки
            buy_amount = max_buy_amount
            sell_amount = buy_amount

            buy_fee = (
                buy_amount * opportunity.buy_price * (opportunity.buy_fee_pct / 100)
            )
            sell_fee = (
                sell_amount * opportunity.sell_price * (opportunity.sell_fee_pct / 100)
            )

            buy_cost = buy_amount * opportunity.buy_price + buy_fee
            sell_proceeds = sell_amount * opportunity.sell_price - sell_fee

            expected_profit = sell_proceeds - buy_cost

            # Возвращаем размеры сделок
            return {
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
                "buy_cost": buy_cost,
                "sell_proceeds": sell_proceeds,
                "expected_profit": expected_profit,
                "expected_profit_pct": (
                    (expected_profit / buy_cost) * 100 if buy_cost > 0 else 0
                ),
            }

        except Exception as e:
            logger.warning(
                "Ошибка при проверке балансов для %s: %s", opportunity.symbol, str(e)
            )
            return {}

    @async_handle_error
    async def calculate_trade_sizes(
        self, opportunity: ArbitrageOpportunity, account_balance: float = 0.0
    ) -> Dict[str, float]:
        """
        Расчет оптимальных размеров сделок для данной арбитражной возможности.

        Args:
            opportunity: Объект арбитражной возможности
            account_balance: Общий доступный баланс аккаунта

        Returns:
            Словарь с расчетами размеров сделок
        """
        try:
            symbol_parts = opportunity.symbol.split("/")
            if len(symbol_parts) != 2:
                logger.warning("Некорректный формат символа: %s", opportunity.symbol)
                return {}

            base_currency, quote_currency = symbol_parts

            # Получаем данные о балансах на обеих биржах
            buy_balance = await fetch_balance(opportunity.buy_exchange)
            sell_balance = await fetch_balance(opportunity.sell_exchange)

            if not buy_balance or not sell_balance:
                return {}

            # Получаем свободные балансы
            buy_quote_balance = buy_balance.get("free", {}).get(quote_currency, 0)
            sell_base_balance = sell_balance.get("free", {}).get(base_currency, 0)

            # Получаем данные об объемах на рынке
            buy_orderbook = await fetch_order_book(
                exchange=opportunity.buy_exchange, symbol=opportunity.symbol
            )
            sell_orderbook = await fetch_order_book(
                exchange=opportunity.sell_exchange, symbol=opportunity.symbol
            )

            # Определяем доступный объем на рынке
            market_buy_volume = 0
            if buy_orderbook and "asks" in buy_orderbook and buy_orderbook["asks"]:
                for price, volume in buy_orderbook["asks"]:
                    if price <= opportunity.buy_price * 1.01:  # +1% к цене
                        market_buy_volume += volume

            market_sell_volume = 0
            if sell_orderbook and "bids" in sell_orderbook and sell_orderbook["bids"]:
                for price, volume in sell_orderbook["bids"]:
                    if price >= opportunity.sell_price * 0.99:  # -1% к цене
                        market_sell_volume += volume

            # Определяем максимальный размер сделки
            max_buy_amount = min(
                buy_quote_balance / opportunity.buy_price,
                sell_base_balance,
                market_buy_volume,
                market_sell_volume,
                opportunity.volume,
            )

            # Ограничиваем процент от баланса счета
            max_trade_pct = (
                self.config.get("arbitrage", {}).get("max_balance_pct", 25) / 100
            )

            if account_balance > 0:
                max_account_trade = (
                    account_balance * max_trade_pct / opportunity.buy_price
                )
                max_buy_amount = min(max_buy_amount, max_account_trade)

            # Минимальная сумма сделки
            min_trade_amount = self.config.get("arbitrage", {}).get(
                "min_trade_amount", 10
            )

            if max_buy_amount * opportunity.buy_price < min_trade_amount:
                logger.debug(
                    "Слишком маленький размер сделки для %s: %.2f USD (мин: %.2f USD)",
                    opportunity.symbol,
                    max_buy_amount * opportunity.buy_price,
                    min_trade_amount,
                )
                return {}

            # Рассчитываем размеры и стоимость сделок
            buy_amount = max_buy_amount
            sell_amount = buy_amount

            buy_fee = (
                buy_amount * opportunity.buy_price * (opportunity.buy_fee_pct / 100)
            )
            sell_fee = (
                sell_amount * opportunity.sell_price * (opportunity.sell_fee_pct / 100)
            )

            buy_cost = buy_amount * opportunity.buy_price + buy_fee
            sell_proceeds = sell_amount * opportunity.sell_price - sell_fee

            expected_profit = sell_proceeds - buy_cost
            expected_profit_pct = (
                (expected_profit / buy_cost) * 100 if buy_cost > 0 else 0
            )

            # Проверяем, что прибыль положительная
            if expected_profit <= 0:
                logger.debug(
                    "Расчетная прибыль не положительная для %s: %.2f USD",
                    opportunity.symbol,
                    expected_profit,
                )
                return {}

            # Возвращаем расчеты
            return {
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
                "buy_cost": buy_cost,
                "sell_proceeds": sell_proceeds,
                "expected_profit": expected_profit,
                "expected_profit_pct": expected_profit_pct,
                "market_buy_volume": market_buy_volume,
                "market_sell_volume": market_sell_volume,
            }

        except Exception as e:
            logger.warning(
                "Ошибка при расчете размеров сделок для %s: %s",
                opportunity.symbol,
                str(e),
            )
            return {}

    async def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику арбитражной системы.

        Returns:
            Словарь со статистикой
        """
        stats = dict(self.stats)
        stats["uptime"] = time.time() - stats["start_time"]

        # Добавляем дополнительные показатели
        if stats["opportunities_found"] > 0:
            stats["verified_ratio"] = (
                stats["opportunities_verified"] / stats["opportunities_found"]
            )
        else:
            stats["verified_ratio"] = 0

        if stats["opportunities_verified"] > 0:
            stats["execution_ratio"] = (
                stats["opportunities_executed"] / stats["opportunities_verified"]
            )
        else:
            stats["execution_ratio"] = 0

        stats["avg_profit"] = (
            stats["total_profit"] / stats["opportunities_executed"]
            if stats["opportunities_executed"] > 0
            else 0
        )

        return stats


"""
Базовые классы и функции для арбитражной торговли.
Предоставляет инструменты для поиска и анализа арбитражных возможностей.
"""

# Стандартные импорты
import time
import logging
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set

# Сторонние импорты
import pandas as pd
import numpy as np

# Настройка логгера
logger = logging.getLogger(__name__)

# Внутренние импорты
from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.notify import send_trading_signal
from project.utils.ccxt_exchanges import fetch_ticker, fetch_order_book, fetch_balance
from project.data.market_data import MarketData


@dataclass
class ArbitrageOpportunity:
    """
    Класс для представления арбитражной возможности между биржами.
    """

    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    volume: float
    profit_usd: float
    profit_margin_pct: float
    buy_fee_pct: float = 0.0
    sell_fee_pct: float = 0.0
    timestamp: float = 0.0
    is_valid: bool = True
    estimated_profit: float = 0.0
    validation_count: int = 0
    last_update: float = 0.0


class ArbitrageCore:
    """
    Основной класс для обнаружения и анализа арбитражных возможностей.
    """

    def __init__(self, config=None):
        """
        Инициализирует ядро арбитража.

        Args:
            config (Dict, optional): Конфигурация. По умолчанию None.
        """
        self.config = config or get_config()
        self.market_data = MarketData.get_instance()

        # Настройки арбитража
        arb_config = self.config.get("arbitrage", {})
        self.min_profit_pct = arb_config.get("min_profit_pct", 0.5)
        self.min_volume_usd = arb_config.get("min_volume_usd", 10.0)
        self.max_validation_age = arb_config.get("max_validation_age", 60.0)
        self.max_trade_execution_time = arb_config.get("max_trade_execution_time", 5.0)

        # Регистрация бирж и символов
        self.exchanges = arb_config.get("exchanges", ["binance", "kucoin", "okx"])
        self.symbols = arb_config.get("symbols", [])
        self.excluded_pairs = set(arb_config.get("excluded_pairs", []))

        # Данные о комиссиях
        self.fee_data = {}
        for exchange in self.exchanges:
            self.fee_data[exchange] = arb_config.get("fees", {}).get(exchange, 0.1)

        # Статистика
        self.stats = {
            "total_checks": 0,
            "opportunities_found": 0,
            "opportunities_verified": 0,
            "opportunities_executed": 0,
            "total_profit": 0.0,
            "start_time": time.time(),
        }

    @async_handle_error
    async def check_arbitrage_opportunities(
        self, symbols=None, exchanges=None, min_profit_pct=None, min_volume_usd=None
    ) -> List[ArbitrageOpportunity]:
        """
        Проверяет доступные арбитражные возможности для заданных символов и бирж.

        Args:
            symbols: Список символов для проверки
            exchanges: Список бирж для проверки
            min_profit_pct: Минимальный процент прибыли
            min_volume_usd: Минимальный объем в USD

        Returns:
            Список обнаруженных арбитражных возможностей
        """
        try:
            # Используем значения по умолчанию, если не заданы
            symbols = symbols or self.symbols
            exchanges = exchanges or self.exchanges
            min_profit_pct = min_profit_pct or self.min_profit_pct
            min_volume_usd = min_volume_usd or self.min_volume_usd

            # Обновляем статистику
            self.stats["total_checks"] += 1

            opportunities = []

            # Получаем данные по всем биржам для всех символов
            exchange_data = {}

            for symbol in symbols:
                if symbol in self.excluded_pairs:
                    continue

                tickers = {}

                for exchange in exchanges:
                    try:
                        ticker = await fetch_ticker(exchange, symbol)
                        if (
                            ticker
                            and ticker.get("last")
                            and ticker.get("bid")
                            and ticker.get("ask")
                        ):
                            tickers[exchange] = ticker
                    except Exception as e:
                        logger.debug(
                            "Ошибка при получении тикера %s на %s: %s",
                            symbol,
                            exchange,
                            str(e),
                        )

                # Если меньше двух бирж с данными, арбитраж невозможен
                if len(tickers) < 2:
                    continue

                exchange_data[symbol] = tickers

            # Анализируем и находим арбитражные возможности
            for symbol, tickers in exchange_data.items():
                exchanges_list = list(tickers.keys())

                # Перебираем все пары бирж
                for i in range(len(exchanges_list)):
                    for j in range(i + 1, len(exchanges_list)):
                        exchange1 = exchanges_list[i]
                        exchange2 = exchanges_list[j]

                        ticker1 = tickers[exchange1]
                        ticker2 = tickers[exchange2]

                        # Проверяем возможность арбитража в обе стороны

                        # exchange1 -> exchange2
                        buy_price = ticker1["ask"]
                        sell_price = ticker2["bid"]
                        fee1 = self.fee_data.get(exchange1, 0.1) / 100
                        fee2 = self.fee_data.get(exchange2, 0.1) / 100

                        # Расчет объема и прибыли
                        max_volume = min(
                            ticker1.get("askVolume", ticker1.get("volume", 0)),
                            ticker2.get("bidVolume", ticker2.get("volume", 0)),
                        )

                        # Расчет потенциальной прибыли с учетом комиссий
                        buy_cost = buy_price * max_volume * (1 + fee1)
                        sell_proceed = sell_price * max_volume * (1 - fee2)

                        profit_usd = sell_proceed - buy_cost
                        profit_margin_pct = (
                            (profit_usd / buy_cost) * 100 if buy_cost > 0 else 0
                        )

                        if (
                            profit_margin_pct > min_profit_pct
                            and max_volume * buy_price > min_volume_usd
                        ):

                            opportunity = ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=exchange1,
                                sell_exchange=exchange2,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                volume=max_volume,
                                profit_usd=profit_usd,
                                profit_margin_pct=profit_margin_pct,
                                buy_fee_pct=fee1 * 100,
                                sell_fee_pct=fee2 * 100,
                                timestamp=time.time(),
                            )
                            opportunities.append(opportunity)

                            self.stats["opportunities_found"] += 1

                        # exchange2 -> exchange1
                        buy_price = ticker2["ask"]
                        sell_price = ticker1["bid"]

                        # Расчет объема и прибыли
                        max_volume = min(
                            ticker2.get("askVolume", ticker2.get("volume", 0)),
                            ticker1.get("bidVolume", ticker1.get("volume", 0)),
                        )

                        buy_cost = buy_price * max_volume * (1 + fee2)
                        sell_proceed = sell_price * max_volume * (1 - fee1)

                        profit_usd = sell_proceed - buy_cost
                        profit_margin_pct = (
                            (profit_usd / buy_cost) * 100 if buy_cost > 0 else 0
                        )

                        if (
                            profit_margin_pct > min_profit_pct
                            and max_volume * buy_price > min_volume_usd
                        ):

                            opportunity = ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=exchange2,
                                sell_exchange=exchange1,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                volume=max_volume,
                                profit_usd=profit_usd,
                                profit_margin_pct=profit_margin_pct,
                                buy_fee_pct=fee2 * 100,
                                sell_fee_pct=fee1 * 100,
                                timestamp=time.time(),
                            )
                            opportunities.append(opportunity)

                            self.stats["opportunities_found"] += 1

            # Сортируем возможности по убыванию прибыли
            opportunities.sort(key=lambda x: x.profit_margin_pct, reverse=True)

            return opportunities

        except Exception as e:
            logger.warning("Ошибка при проверке арбитражных возможностей: %s", str(e))
            return []

    @async_handle_error
    async def verify_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        min_volume: float = 0.0,
        max_age_seconds: float = 60.0,
    ) -> bool:
        """
        Проверяет актуальность арбитражной возможности.

        Args:
            opportunity: Объект арбитражной возможности
            min_volume: Минимальный объем для торговли
            max_age_seconds: Максимальный возраст возможности в секундах

        Returns:
            True если возможность остается выгодной, иначе False
        """
        try:
            current_time = time.time()
            opportunity_age = current_time - opportunity.timestamp

            # Проверка возраста возможности
            if opportunity_age > max_age_seconds:
                logger.debug(
                    "Возможность устарела: %s - возраст: %.2f с (макс: %.2f с)",
                    opportunity.symbol,
                    opportunity_age,
                    max_age_seconds,
                )
                return False

            # Получаем актуальные данные
            buy_ticker = await fetch_ticker(
                opportunity.buy_exchange, opportunity.symbol
            )
            sell_ticker = await fetch_ticker(
                opportunity.sell_exchange, opportunity.symbol
            )

            # Проверяем наличие данных
            if not buy_ticker or not sell_ticker:
                return False

            # Обновляем цены
            buy_price = buy_ticker.get("ask", 0)
            sell_price = sell_ticker.get("bid", 0)

            if buy_price <= 0 or sell_price <= 0:
                return False

            # Пересчитываем потенциальную прибыль
            fee1 = self.fee_data.get(opportunity.buy_exchange, 0.1) / 100
            fee2 = self.fee_data.get(opportunity.sell_exchange, 0.1) / 100

            max_volume = min(
                buy_ticker.get("askVolume", buy_ticker.get("volume", 0)),
                sell_ticker.get("bidVolume", sell_ticker.get("volume", 0)),
            )

            if max_volume < min_volume:
                return False

            buy_cost = buy_price * max_volume * (1 + fee1)
            sell_proceed = sell_price * max_volume * (1 - fee2)

            profit_usd = sell_proceed - buy_cost
            profit_margin_pct = (profit_usd / buy_cost) * 100 if buy_cost > 0 else 0

            # Обновляем данные возможности
            updated_opportunity = dataclasses.replace(
                opportunity,
                buy_price=buy_price,
                sell_price=sell_price,
                volume=max_volume,
                profit_usd=profit_usd,
                profit_margin_pct=profit_margin_pct,
                validation_count=opportunity.validation_count + 1,
                last_update=current_time,
            )

            # Проверяем, остается ли возможность выгодной
            if updated_opportunity.profit_margin_pct > 0:
                # Обновляем статистику
                self.stats["opportunities_verified"] += 1
                return True

            logger.debug(
                "Возможность больше не выгодна: %s - " "прибыль: %.2f%%",
                opportunity.symbol,
                updated_opportunity.profit_margin_pct,
            )
            return False

        except Exception as e:
            logger.warning(
                "Ошибка при проверке возможности %s: %s", opportunity.symbol, str(e)
            )
            return False

    @async_handle_error
    async def check_balances(
        self, opportunity: ArbitrageOpportunity, min_trade_amount: float = 10.0
    ) -> Dict[str, float]:
        """
        Проверяет наличие достаточных балансов для торговли.

        Args:
            opportunity: Объект арбитражной возможности
            min_trade_amount: Минимальная сумма для торговли в USD

        Returns:
            Словарь с рассчитанными размерами сделок или пустой словарь,
            если балансы недостаточны
        """
        try:
            # Получаем символы базовой и котируемой валюты
            symbol_parts = opportunity.symbol.split("/")
            if len(symbol_parts) != 2:
                logger.warning("Некорректный формат символа: %s", opportunity.symbol)
                return {}

            base_currency, quote_currency = symbol_parts

            # Получаем балансы на биржах
            buy_balance = await fetch_balance(opportunity.buy_exchange)
            sell_balance = await fetch_balance(opportunity.sell_exchange)

            if not buy_balance or not sell_balance:
                logger.warning("Не удалось получить балансы")
                return {}

            # Получаем свободные балансы
            buy_quote_balance = buy_balance.get("free", {}).get(quote_currency, 0)
            sell_base_balance = sell_balance.get("free", {}).get(base_currency, 0)

            # Проверяем, достаточно ли балансов для минимальной торговли
            buy_amount_max = buy_quote_balance / opportunity.buy_price

            # Определяем максимальный размер сделки
            max_buy_amount = min(buy_amount_max, sell_base_balance, opportunity.volume)

            # Проверяем минимальные требования
            if max_buy_amount * opportunity.buy_price < min_trade_amount:
                logger.debug(
                    "Недостаточный баланс для арбитража %s: "
                    "максимальная сумма покупки %.2f USD, требуется %.2f USD",
                    opportunity.symbol,
                    max_buy_amount * opportunity.buy_price,
                    min_trade_amount,
                )
                return {}

            # Рассчитываем параметры сделки
            buy_amount = max_buy_amount
            sell_amount = buy_amount

            buy_fee = (
                buy_amount * opportunity.buy_price * (opportunity.buy_fee_pct / 100)
            )
            sell_fee = (
                sell_amount * opportunity.sell_price * (opportunity.sell_fee_pct / 100)
            )

            buy_cost = buy_amount * opportunity.buy_price + buy_fee
            sell_proceeds = sell_amount * opportunity.sell_price - sell_fee

            expected_profit = sell_proceeds - buy_cost

            # Возвращаем размеры сделок
            return {
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
                "buy_cost": buy_cost,
                "sell_proceeds": sell_proceeds,
                "expected_profit": expected_profit,
                "expected_profit_pct": (
                    (expected_profit / buy_cost) * 100 if buy_cost > 0 else 0
                ),
            }

        except Exception as e:
            logger.warning(
                "Ошибка при проверке балансов для %s: %s", opportunity.symbol, str(e)
            )
            return {}

    @async_handle_error
    async def calculate_trade_sizes(
        self, opportunity: ArbitrageOpportunity, account_balance: float = 0.0
    ) -> Dict[str, float]:
        """
        Расчет оптимальных размеров сделок для данной арбитражной возможности.

        Args:
            opportunity: Объект арбитражной возможности
            account_balance: Общий доступный баланс аккаунта

        Returns:
            Словарь с расчетами размеров сделок
        """
        try:
            symbol_parts = opportunity.symbol.split("/")
            if len(symbol_parts) != 2:
                logger.warning("Некорректный формат символа: %s", opportunity.symbol)
                return {}

            base_currency, quote_currency = symbol_parts

            # Получаем данные о балансах на обеих биржах
            buy_balance = await fetch_balance(opportunity.buy_exchange)
            sell_balance = await fetch_balance(opportunity.sell_exchange)

            if not buy_balance or not sell_balance:
                return {}

            # Получаем свободные балансы
            buy_quote_balance = buy_balance.get("free", {}).get(quote_currency, 0)
            sell_base_balance = sell_balance.get("free", {}).get(base_currency, 0)

            # Получаем данные об объемах на рынке
            buy_orderbook = await fetch_order_book(
                opportunity.buy_exchange, opportunity.symbol
            )
            sell_orderbook = await fetch_order_book(
                opportunity.sell_exchange, opportunity.symbol
            )

            # Определяем доступный объем на рынке
            market_buy_volume = 0
            if buy_orderbook and "asks" in buy_orderbook and buy_orderbook["asks"]:
                for price, volume in buy_orderbook["asks"]:
                    if price <= opportunity.buy_price * 1.01:  # +1% к цене
                        market_buy_volume += volume

            market_sell_volume = 0
            if sell_orderbook and "bids" in sell_orderbook and sell_orderbook["bids"]:
                for price, volume in sell_orderbook["bids"]:
                    if price >= opportunity.sell_price * 0.99:  # -1% к цене
                        market_sell_volume += volume

            # Определяем максимальный размер сделки
            max_buy_amount = min(
                buy_quote_balance / opportunity.buy_price,
                sell_base_balance,
                market_buy_volume,
                market_sell_volume,
                opportunity.volume,
            )

            # Ограничиваем процент от баланса счета
            max_trade_pct = (
                self.config.get("arbitrage", {}).get("max_balance_pct", 25) / 100
            )

            if account_balance > 0:
                max_account_trade = (
                    account_balance * max_trade_pct / opportunity.buy_price
                )
                max_buy_amount = min(max_buy_amount, max_account_trade)

            # Минимальная сумма сделки
            min_trade_amount = self.config.get("arbitrage", {}).get(
                "min_trade_amount", 10
            )

            if max_buy_amount * opportunity.buy_price < min_trade_amount:
                logger.debug(
                    "Слишком маленький размер сделки для %s: "
                    "%.2f USD (мин: %.2f USD)",
                    opportunity.symbol,
                    max_buy_amount * opportunity.buy_price,
                    min_trade_amount,
                )
                return {}

            # Рассчитываем размеры и стоимость сделок
            buy_amount = max_buy_amount
            sell_amount = buy_amount

            buy_fee = (
                buy_amount * opportunity.buy_price * (opportunity.buy_fee_pct / 100)
            )
            sell_fee = (
                sell_amount * opportunity.sell_price * (opportunity.sell_fee_pct / 100)
            )

            buy_cost = buy_amount * opportunity.buy_price + buy_fee
            sell_proceeds = sell_amount * opportunity.sell_price - sell_fee

            expected_profit = sell_proceeds - buy_cost
            expected_profit_pct = (
                (expected_profit / buy_cost) * 100 if buy_cost > 0 else 0
            )

            # Проверяем, что прибыль положительная
            if expected_profit <= 0:
                logger.debug(
                    "Расчетная прибыль не положительная для %s: %.2f USD",
                    opportunity.symbol,
                    expected_profit,
                )
                return {}

            # Возвращаем расчеты
            return {
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
                "buy_cost": buy_cost,
                "sell_proceeds": sell_proceeds,
                "expected_profit": expected_profit,
                "expected_profit_pct": expected_profit_pct,
                "market_buy_volume": market_buy_volume,
                "market_sell_volume": market_sell_volume,
            }

        except Exception as e:
            logger.warning(
                "Ошибка при расчете размеров сделок для %s: %s",
                opportunity.symbol,
                str(e),
            )
            return {}

    async def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику арбитражной системы.

        Returns:
            Словарь со статистикой
        """
        stats = dict(self.stats)
        stats["uptime"] = time.time() - stats["start_time"]

        # Добавляем дополнительные показатели
        if stats["opportunities_found"] > 0:
            stats["verified_ratio"] = (
                stats["opportunities_verified"] / stats["opportunities_found"]
            )
        else:
            stats["verified_ratio"] = 0

        if stats["opportunities_verified"] > 0:
            stats["execution_ratio"] = (
                stats["opportunities_executed"] / stats["opportunities_verified"]
            )
        else:
            stats["execution_ratio"] = 0

        stats["avg_profit"] = (
            stats["total_profit"] / stats["opportunities_executed"]
            if stats["opportunities_executed"] > 0
            else 0
        )

        return stats
