"""
Основной модуль для арбитражной торговли.
Предоставляет функции для обнаружения и использования арбитражных возможностей.
"""

import asyncio
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass

from project.config import get_config
from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error, async_with_retry
from project.utils.notify import send_trading_signal
from project.data.market_data import MarketData
from project.utils.ccxt_exchanges import fetch_ticker, fetch_balance, create_order

logger = get_logger(__name__)


@dataclass
class ArbitrageOpportunity:
    """
    Класс для хранения информации об арбитражной возможности.
    """

    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    price_diff: float
    price_diff_pct: float
    buy_volume: float
    sell_volume: float
    timestamp: float
    profit_margin_pct: float = 0.0  # с учетом комиссий
    fee_adjusted: bool = False
    net_profit_estimate: float = 0.0
    trade_sizes: Dict[str, float] = None
    status: str = "detected"  # detected, executed, failed, expired


class ArbitrageCore:
    """
    Основной класс для арбитражной торговли между биржами.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "ArbitrageCore":
        """
        Получает экземпляр класса ArbitrageCore (Singleton).

        Returns:
            Экземпляр класса ArbitrageCore
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Инициализирует ядро арбитража.
        """
        self.config = get_config()
        self.market_data = MarketData.get_instance()
        self.supported_exchanges = [
            "binance",
            "kucoin",
            "huobi",
            "okex",
            "kraken",
            "bitfinex",
        ]
        self.fee_rates = {
            "binance": 0.001,  # 0.1%
            "kucoin": 0.001,
            "huobi": 0.002,
            "okex": 0.0015,
            "kraken": 0.0026,
            "bitfinex": 0.002,
        }
        self.opportunities: List[ArbitrageOpportunity] = []
        self.active_arbitrages: Dict[str, ArbitrageOpportunity] = (
            {}
        )  # ID -> возможность
        self.status_updates: List[Dict[str, Any]] = []
        self.last_scan_time = 0

        logger.debug("Создан экземпляр ArbitrageCore")

    @async_handle_error
    async def scan_opportunities(
        self, symbols: List[str], exchanges: List[str] = None
    ) -> List[ArbitrageOpportunity]:
        """
        Сканирует арбитражные возможности между биржами.

        Args:
            symbols: Список символов для сканирования
            exchanges: Список бирж для сканирования (None для всех поддерживаемых)

        Returns:
            Список арбитражных возможностей
        """
        # Проверяем и устанавливаем биржи
        if exchanges is None:
            exchanges = self.supported_exchanges
        else:
            exchanges = [e for e in exchanges if e in self.supported_exchanges]

        if not exchanges:
            logger.warning("Нет поддерживаемых бирж для сканирования")
            return []

        if not symbols:
            logger.warning("Не указаны символы для сканирования")
            return []

        # Обновляем время последнего сканирования
        self.last_scan_time = time.time()

        # Промежуточные данные
        exchange_prices: Dict[str, Dict[str, float]] = {}
        exchange_volumes: Dict[str, Dict[str, float]] = {}

        # Сбор цен и объемов со всех бирж
        for exchange in exchanges:
            exchange_prices[exchange] = {}
            exchange_volumes[exchange] = {}

            for symbol in symbols:
                try:
                    ticker = await fetch_ticker(exchange, symbol)

                    if ticker:
                        # Сохраняем цену и объем
                        price = ticker.get("last") or ticker.get("close")
                        volume = ticker.get("quoteVolume") or ticker.get("volume") or 0

                        if price and price > 0:
                            exchange_prices[exchange][symbol] = price
                            exchange_volumes[exchange][symbol] = volume

                except Exception as e:
                    logger.warning(
                        f"Ошибка при получении данных для {symbol} на {exchange}: {str(e)}"
                    )

        # Поиск возможностей
        new_opportunities = []

        for symbol in symbols:
            # Собираем все цены для данного символа
            prices_for_symbol = {}

            for exchange in exchanges:
                if symbol in exchange_prices[exchange]:
                    prices_for_symbol[exchange] = exchange_prices[exchange][symbol]

            # Если цены есть как минимум на двух биржах
            if len(prices_for_symbol) >= 2:
                # Находим биржу с минимальной и максимальной ценой
                buy_exchange = min(prices_for_symbol, key=prices_for_symbol.get)
                sell_exchange = max(prices_for_symbol, key=prices_for_symbol.get)

                buy_price = prices_for_symbol[buy_exchange]
                sell_price = prices_for_symbol[sell_exchange]

                # Если есть разница в цене
                if buy_price < sell_price:
                    # Рассчитываем разницу и процент
                    price_diff = sell_price - buy_price
                    price_diff_pct = price_diff / buy_price

                    # Получаем объемы
                    buy_volume = exchange_volumes[buy_exchange].get(symbol, 0)
                    sell_volume = exchange_volumes[sell_exchange].get(symbol, 0)

                    # Создаем объект возможности
                    opportunity = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=buy_exchange,
                        sell_exchange=sell_exchange,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        price_diff=price_diff,
                        price_diff_pct=price_diff_pct,
                        buy_volume=buy_volume,
                        sell_volume=sell_volume,
                        timestamp=time.time(),
                    )

                    # Рассчитываем маржу с учетом комиссий
                    opportunity = self._adjust_for_fees(opportunity)

                    # Если после учета комиссий остается прибыль
                    if opportunity.profit_margin_pct > 0:
                        new_opportunities.append(opportunity)
                        logger.debug(
                            f"Найдена арбитражная возможность: {symbol} - "
                            f"купить на {buy_exchange} за {buy_price:.8f}, "
                            f"продать на {sell_exchange} за {sell_price:.8f}, "
                            f"прибыль: {opportunity.profit_margin_pct:.2%}"
                        )

        # Сортируем возможности по убыванию прибыли
        new_opportunities.sort(key=lambda x: x.profit_margin_pct, reverse=True)

        # Обновляем список возможностей
        self.opportunities = new_opportunities

        return new_opportunities

    def _adjust_for_fees(
        self, opportunity: ArbitrageOpportunity
    ) -> ArbitrageOpportunity:
        """
        Корректирует арбитражную возможность с учетом комиссий.

        Args:
            opportunity: Объект арбитражной возможности

        Returns:
            Скорректированная арбитражная возможность
        """
        # Получаем комиссии для бирж
        buy_fee = self.fee_rates.get(
            opportunity.buy_exchange, 0.002
        )  # По умолчанию 0.2%
        sell_fee = self.fee_rates.get(opportunity.sell_exchange, 0.002)

        # Рассчитываем чистую прибыль с учетом комиссий
        # 1. Покупаем за 100 USD на первой бирже
        # 2. Получаем (100 - buy_fee*100) / buy_price = amount
        # 3. Продаем amount на второй бирже
        # 4. Получаем amount * sell_price * (1 - sell_fee)

        initial_amount = 100  # Условная сумма в USD

        # Количество купленных монет
        bought_amount = (initial_amount * (1 - buy_fee)) / opportunity.buy_price

        # Сумма после продажи
        sold_amount = bought_amount * opportunity.sell_price * (1 - sell_fee)

        # Чистая прибыль
        net_profit = sold_amount - initial_amount
        net_profit_pct = net_profit / initial_amount

        # Обновляем объект
        opportunity.profit_margin_pct = net_profit_pct
        opportunity.fee_adjusted = True
        opportunity.net_profit_estimate = net_profit

        return opportunity

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
            True, если возможность актуальна, иначе False
        """
        # Проверяем возраст возможности
        current_time = time.time()
        age = current_time - opportunity.timestamp

        if age > max_age_seconds:
            logger.debug(
                f"Возможность устарела: {opportunity.symbol} - возраст {age:.1f} секунд > {max_age_seconds} секунд"
            )
            return False

        # Проверяем минимальный объем
        if min_volume > 0:
            if (
                opportunity.buy_volume < min_volume
                or opportunity.sell_volume < min_volume
            ):
                logger.debug(
                    f"Недостаточный объем для {opportunity.symbol}: "
                    f"buy={opportunity.buy_volume}, sell={opportunity.sell_volume}, min={min_volume}"
                )
                return False

        # Обновляем цены
        try:
            buy_ticker = await fetch_ticker(
                opportunity.buy_exchange, opportunity.symbol
            )
            sell_ticker = await fetch_ticker(
                opportunity.sell_exchange, opportunity.symbol
            )

            if not buy_ticker or not sell_ticker:
                return False

            # Получаем актуальные цены
            new_buy_price = buy_ticker.get("last") or buy_ticker.get("close")
            new_sell_price = sell_ticker.get("last") or sell_ticker.get("close")

            if not new_buy_price or not new_sell_price:
                return False

            # Рассчитываем новую разницу
            new_price_diff = new_sell_price - new_buy_price
            new_price_diff_pct = new_price_diff / new_buy_price

            # Создаем обновленную возможность
            updated_opportunity = ArbitrageOpportunity(
                symbol=opportunity.symbol,
                buy_exchange=opportunity.buy_exchange,
                sell_exchange=opportunity.sell_exchange,
                buy_price=new_buy_price,
                sell_price=new_sell_price,
                price_diff=new_price_diff,
                price_diff_pct=new_price_diff_pct,
                buy_volume=opportunity.buy_volume,
                sell_volume=opportunity.sell_volume,
                timestamp=current_time,
            )

            # Пересчитываем маржу с учетом комиссий
            updated_opportunity = self._adjust_for_fees(updated_opportunity)

            # Если после обновления прибыль все еще положительная
            if updated_opportunity.profit_margin_pct > 0:
                # Обновляем оригинальный объект
                opportunity.buy_price = new_buy_price
                opportunity.sell_price = new_sell_price
                opportunity.price_diff = new_price_diff
                opportunity.price_diff_pct = new_price_diff_pct
                opportunity.timestamp = current_time
                opportunity.profit_margin_pct = updated_opportunity.profit_margin_pct
                opportunity.net_profit_estimate = (
                    updated_opportunity.net_profit_estimate
                )

                logger.debug(
                    f"Возможность актуализирована: {opportunity.symbol} - "
                    f"купить на {opportunity.buy_exchange} за {new_buy_price:.8f}, "
                    f"продать на {opportunity.sell_exchange} за {new_sell_price:.8f}, "
                    f"прибыль: {opportunity.profit_margin_pct:.2%}"
                )

                return True
            else:
                logger.debug(
                    f"Возможность больше не выгодна: {opportunity.symbol} - "
                    f"прибыль: {updated_opportunity.profit_margin_pct:.2%}"
                )
                return False

        except Exception as e:
            logger.warning(
                f"Ошибка при проверке возможности {opportunity.symbol}: {str(e)}"
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
            Словарь с рассчитанными размерами сделок или пустой словарь, если балансы недостаточны
        """
        try:
            # Получаем символы базовой и котируемой валюты
            symbol_parts = opportunity.symbol.split("/")
            if len(symbol_parts) != 2:
                logger.warning(f"Некорректный формат символа: {opportunity.symbol}")
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
            if buy_quote_balance < min_trade_amount:
                logger.debug(
                    f"Недостаточно {quote_currency} на {opportunity.buy_exchange}: "
                    f"{buy_quote_balance} < {min_trade_amount}"
                )
                return {}

            if sell_base_balance <= 0:
                logger.debug(
                    f"Недостаточно {base_currency} на {opportunity.sell_exchange}: "
                    f"{sell_base_balance} <= 0"
                )
                return {}

            # Рассчитываем максимальный объем для торговли
            max_buy_amount = buy_quote_balance / opportunity.buy_price

            # Ограничиваем объем торговли
            trade_amount = min(max_buy_amount, sell_base_balance)

            # Если объем слишком мал, не торгуем
            if trade_amount * opportunity.buy_price < min_trade_amount:
                logger.debug(
                    f"Торговый объем слишком мал: "
                    f"{trade_amount * opportunity.buy_price} < {min_trade_amount} USD"
                )
                return {}

            # Рассчитываем объемы для покупки и продажи
            buy_amount = trade_amount
            sell_amount = (
                trade_amount * 0.998
            )  # Небольшой запас для округления и комиссий

            # Возвращаем объемы для торговли
            trade_sizes = {
                "buy_amount": buy_amount,
                "sell_amount": sell_amount,
                "buy_cost": buy_amount * opportunity.buy_price,
                "sell_proceeds": sell_amount * opportunity.sell_price,
            }

            logger.debug(
                f"Рассчитаны объемы для торговли {opportunity.symbol}: "
                f"покупка {buy_amount} за {buy_amount * opportunity.buy_price} {quote_currency}, "
                f"продажа {sell_amount} за {sell_amount * opportunity.sell_price} {quote_currency}"
            )

            # Сохраняем объемы в возможности
            opportunity.trade_sizes = trade_sizes

            return trade_sizes

        except Exception as e:
            logger.error(
                f"Ошибка при проверке балансов для {opportunity.symbol}: {str(e)}"
            )
            return {}

    @async_handle_error
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Выполняет арбитражную сделку.

        Args:
            opportunity: Объект арбитражной возможности

        Returns:
            True, если сделка выполнена успешно, иначе False
        """
        # Проверяем актуальность возможности
        is_valid = await self.verify_opportunity(opportunity)
        if not is_valid:
            logger.warning(f"Возможность больше не актуальна: {opportunity.symbol}")
            opportunity.status = "expired"
            return False

        # Проверяем балансы и рассчитываем объемы
        trade_sizes = await self.check_balances(opportunity)
        if not trade_sizes:
            logger.warning(f"Недостаточно балансов для арбитража {opportunity.symbol}")
            opportunity.status = "failed"
            return False

        # Генерируем уникальный ID для арбитража
        arbitrage_id = f"arb_{int(time.time())}_{opportunity.symbol.replace('/', '_')}"

        try:
            # Выполняем покупку
            buy_order = await create_order(
                exchange_id=opportunity.buy_exchange,
                symbol=opportunity.symbol,
                order_type="market",
                side="buy",
                amount=trade_sizes["buy_amount"],
            )

            if not buy_order or not buy_order.get("id"):
                logger.error(
                    f"Не удалось создать ордер на покупку для {opportunity.symbol} на {opportunity.buy_exchange}"
                )
                opportunity.status = "failed"
                return False

            # Добавляем статус выполнения
            self.status_updates.append(
                {
                    "arbitrage_id": arbitrage_id,
                    "step": "buy",
                    "exchange": opportunity.buy_exchange,
                    "symbol": opportunity.symbol,
                    "order_id": buy_order.get("id"),
                    "amount": trade_sizes["buy_amount"],
                    "price": opportunity.buy_price,
                    "timestamp": time.time(),
                }
            )

            # Небольшая задержка перед продажей
            await asyncio.sleep(0.5)

            # Выполняем продажу
            sell_order = await create_order(
                exchange_id=opportunity.sell_exchange,
                symbol=opportunity.symbol,
                order_type="market",
                side="sell",
                amount=trade_sizes["sell_amount"],
            )

            if not sell_order or not sell_order.get("id"):
                logger.error(
                    f"Не удалось создать ордер на продажу для {opportunity.symbol} на {opportunity.sell_exchange}"
                )
                opportunity.status = "failed"
                return False

            # Добавляем статус выполнения
            self.status_updates.append(
                {
                    "arbitrage_id": arbitrage_id,
                    "step": "sell",
                    "exchange": opportunity.sell_exchange,
                    "symbol": opportunity.symbol,
                    "order_id": sell_order.get("id"),
                    "amount": trade_sizes["sell_amount"],
                    "price": opportunity.sell_price,
                    "timestamp": time.time(),
                }
            )

            # Арбитраж выполнен успешно
            opportunity.status = "executed"
            self.active_arbitrages[arbitrage_id] = opportunity

            # Отправляем уведомление
            await send_trading_signal(
                f"Арбитраж выполнен: {opportunity.symbol}\n"
                f"Покупка на {opportunity.buy_exchange} по {opportunity.buy_price:.8f}\n"
                f"Продажа на {opportunity.sell_exchange} по {opportunity.sell_price:.8f}\n"
                f"Прибыль: {opportunity.profit_margin_pct:.2%}"
            )

            logger.info(
                f"Арбитраж выполнен успешно: {opportunity.symbol} - "
                f"купить на {opportunity.buy_exchange} за {opportunity.buy_price:.8f}, "
                f"продать на {opportunity.sell_exchange} за {opportunity.sell_price:.8f}, "
                f"прибыль: {opportunity.profit_margin_pct:.2%}"
            )

            return True

        except Exception as e:
            logger.error(
                f"Ошибка при выполнении арбитража для {opportunity.symbol}: {str(e)}"
            )
            opportunity.status = "failed"
            return False

    @async_handle_error
    async def find_triangular_arbitrage(
        self, exchange_id: str, base_currencies: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Ищет возможности для треугольного арбитража на одной бирже.

        Args:
            exchange_id: Идентификатор биржи
            base_currencies: Список базовых валют (None для ["USDT", "BTC", "ETH"])

        Returns:
            Список возможностей треугольного арбитража
        """
        if exchange_id not in self.supported_exchanges:
            logger.warning(f"Биржа {exchange_id} не поддерживается")
            return []

        if base_currencies is None:
            base_currencies = ["USDT", "BTC", "ETH"]

        # Получаем все тикеры для биржи
        try:
            # В реальной реализации здесь был бы вызов API для получения всех тикеров
            # Для простоты симулируем получение данных

            all_tickers = await self.market_data.get_all_tickers(exchange_id)

            if not all_tickers:
                logger.warning(f"Не удалось получить тикеры для {exchange_id}")
                return []

            opportunities = []

            # Для каждой базовой валюты
            for base in base_currencies:
                # Находим все пары с данной базовой валютой
                base_pairs = {}

                for symbol, ticker in all_tickers.items():
                    if symbol.endswith(f"/{base}"):
                        coin = symbol.split("/")[0]
                        price = ticker.get("last") or ticker.get("close")
                        if price:
                            base_pairs[coin] = price

                # Если найдено не менее двух пар с этой базой
                if len(base_pairs) >= 2:
                    # Перебираем все возможные треугольники
                    for coin_a, price_a in base_pairs.items():
                        for coin_b, price_b in base_pairs.items():
                            if coin_a == coin_b:
                                continue

                            # Проверяем наличие прямой пары между монетами
                            direct_pair = f"{coin_a}/{coin_b}"
                            reverse_pair = f"{coin_b}/{coin_a}"

                            direct_price = None
                            for symbol, ticker in all_tickers.items():
                                if symbol == direct_pair or symbol == reverse_pair:
                                    direct_price = ticker.get("last") or ticker.get(
                                        "close"
                                    )
                                    direct_symbol = symbol
                                    break

                            if direct_price:
                                # Рассчитываем результат треугольного арбитража
                                # base -> coin_a -> coin_b -> base

                                # Определяем, прямая или обратная пара
                                if direct_symbol == direct_pair:
                                    # coin_a/coin_b - продаем coin_a за coin_b
                                    step1_result = 1 / price_a  # base -> coin_a
                                    step2_result = (
                                        step1_result * direct_price
                                    )  # coin_a -> coin_b
                                    step3_result = (
                                        step2_result * price_b
                                    )  # coin_b -> base
                                else:
                                    # coin_b/coin_a - продаем coin_b за coin_a
                                    step1_result = 1 / price_a  # base -> coin_a
                                    step2_result = (
                                        step1_result / direct_price
                                    )  # coin_a -> coin_b
                                    step3_result = (
                                        step2_result * price_b
                                    )  # coin_b -> base

                                # Учитываем комиссии (3 сделки)
                                fee = self.fee_rates.get(exchange_id, 0.002)
                                net_result = (
                                    step3_result * (1 - fee) * (1 - fee) * (1 - fee)
                                )
                                profit_pct = net_result - 1

                                # Если есть прибыль
                                if profit_pct > 0.001:  # минимальный порог 0.1%
                                    opportunity = {
                                        "exchange": exchange_id,
                                        "base": base,
                                        "step1": f"Купить {coin_a} за {base}",
                                        "step2": f"{'Продать' if direct_symbol == direct_pair else 'Купить'} "
                                        f"{coin_a} за {coin_b}",
                                        "step3": f"Продать {coin_b} за {base}",
                                        "profit_pct": profit_pct,
                                        "gross_result": step3_result - 1,
                                        "net_result": net_result - 1,
                                        "timestamp": time.time(),
                                    }

                                    opportunities.append(opportunity)

                                    logger.debug(
                                        f"Найдена возможность треугольного арбитража на {exchange_id}: "
                                        f"{base} -> {coin_a} -> {coin_b} -> {base}, "
                                        f"прибыль: {profit_pct:.2%}"
                                    )

            # Сортируем возможности по убыванию прибыли
            opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)

            return opportunities

        except Exception as e:
            logger.error(
                f"Ошибка при поиске треугольного арбитража на {exchange_id}: {str(e)}"
            )
            return []

    @async_handle_error
    async def execute_triangular_arbitrage(self, opportunity: Dict[str, Any]) -> bool:
        """
        Выполняет треугольный арбитраж на одной бирже.

        Args:
            opportunity: Словарь с данными о возможности треугольного арбитража

        Returns:
            True, если арбитраж выполнен успешно, иначе False
        """
        # В реальной реализации здесь был бы код для выполнения трех сделок
        # Для простоты только логируем выполнение

        logger.info(f"Выполнение треугольного арбитража на {opportunity['exchange']}:")
        logger.info(f"  Шаг 1: {opportunity['step1']}")
        logger.info(f"  Шаг 2: {opportunity['step2']}")
        logger.info(f"  Шаг 3: {opportunity['step3']}")
        logger.info(f"  Ожидаемая прибыль: {opportunity['profit_pct']:.2%}")

        # Отправляем уведомление
        await send_trading_signal(
            f"Треугольный арбитраж на {opportunity['exchange']}:\n"
            f"Шаг 1: {opportunity['step1']}\n"
            f"Шаг 2: {opportunity['step2']}\n"
            f"Шаг 3: {opportunity['step3']}\n"
            f"Прибыль: {opportunity['profit_pct']:.2%}"
        )

        # Симулируем успешное выполнение
        return True
