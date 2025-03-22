"""
Модуль multi_exchange.
Реализует кросс‑биржевой арбитраж: синхронизация балансов,
поиск спредов между биржами, оценка прибыльности с учетом комиссий
и исполнение арбитражных сделок.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Tuple, Optional, Set
from decimal import Decimal, ROUND_DOWN
import pandas as pd
import numpy as np
from ..utils.ccxt_exchanges import ExchangeManager
from project.config import load_config
from project.risk_management.risk_manager import RiskManager

config = load_config()

# Настройка логирования
logger = logging.getLogger("MultiExchangeArb")

class ArbitrageOpportunity:
    """Класс для представления арбитражной возможности."""
    
    def __init__(
        self, 
        buy_exchange: str, 
        sell_exchange: str, 
        symbol: str, 
        raw_spread: float,
        buy_price: float,
        sell_price: float,
        buy_volume: float,
        sell_volume: float,
        buy_fee: float,
        sell_fee: float
    ):
        """
        Инициализация арбитражной возможности.
        
        Args:
            buy_exchange: Биржа для покупки
            sell_exchange: Биржа для продажи
            symbol: Торговая пара
            raw_spread: Сырой спред (разница цен)
            buy_price: Цена покупки
            sell_price: Цена продажи
            buy_volume: Доступный объем для покупки
            sell_volume: Доступный объем для продажи
            buy_fee: Комиссия на бирже покупки (в %)
            sell_fee: Комиссия на бирже продажи (в %)
        """
        self.buy_exchange = buy_exchange
        self.sell_exchange = sell_exchange
        self.symbol = symbol
        self.raw_spread = raw_spread
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee
        
        # Расчет спреда с учетом комиссий
        self.net_spread = self._calculate_net_spread()
        
        # Рассчитываем максимальный возможный объем для арбитража
        self.max_trade_volume = min(buy_volume, sell_volume)
        
        # Расчет ожидаемой прибыли
        self.expected_profit = self._calculate_expected_profit()
        
        # Временная метка создания возможности
        self.timestamp = time.time()
    
    def _calculate_net_spread(self) -> float:
        """
        Рассчитывает чистый спред с учетом комиссий.
        
        Returns:
            float: Чистый спред в процентах
        """
        # Учитываем комиссии при покупке и продаже
        effective_buy_price = self.buy_price * (1 + self.buy_fee / 100)
        effective_sell_price = self.sell_price * (1 - self.sell_fee / 100)
        
        # Расчет чистого спреда в процентах
        net_spread_pct = (effective_sell_price / effective_buy_price - 1) * 100
        
        return net_spread_pct
    
    def _calculate_expected_profit(self) -> float:
        """
        Рассчитывает ожидаемую прибыль от арбитража.
        
        Returns:
            float: Ожидаемая прибыль в базовой валюте
        """
        # Предполагаем покупку на buy_exchange и продажу на sell_exchange
        buy_cost = self.max_trade_volume * self.buy_price * (1 + self.buy_fee / 100)
        sell_revenue = self.max_trade_volume * self.sell_price * (1 - self.sell_fee / 100)
        
        return sell_revenue - buy_cost
    
    def is_profitable(self, min_profit_threshold: float = 0.0) -> bool:
        """
        Проверяет, является ли возможность прибыльной.
        
        Args:
            min_profit_threshold: Минимальный порог прибыли
            
        Returns:
            bool: True, если возможность прибыльна
        """
        return self.net_spread > 0 and self.expected_profit > min_profit_threshold
    
    def is_expired(self, max_age_seconds: float = 5.0) -> bool:
        """
        Проверяет, не устарела ли возможность.
        
        Args:
            max_age_seconds: Максимальный возраст возможности в секундах
            
        Returns:
            bool: True, если возможность устарела
        """
        return (time.time() - self.timestamp) > max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует возможность в словарь.
        
        Returns:
            Dict[str, Any]: Словарь с данными о возможности
        """
        return {
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "symbol": self.symbol,
            "raw_spread": self.raw_spread,
            "net_spread": self.net_spread,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "max_volume": self.max_trade_volume,
            "expected_profit": self.expected_profit,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """Строковое представление арбитражной возможности."""
        return (
            f"Арбитраж {self.symbol}: Купить на {self.buy_exchange} по {self.buy_price}, "
            f"Продать на {self.sell_exchange} по {self.sell_price}, "
            f"Чистый спред: {self.net_spread:.2f}%, "
            f"Макс. объем: {self.max_trade_volume}, "
            f"Ожидаемая прибыль: {self.expected_profit:.6f}"
        )


class MultiExchangeArb:
    """
    Класс для кросс‑биржевого арбитража.

    Обеспечивает синхронизацию балансов, поиск выгодных спредов
    и исполнение арбитражных сделок.
    """

    def __init__(self, risk_manager: Optional[RiskManager] = None) -> None:
        """
        Инициализация менеджера кросс-биржевого арбитража.
        
        Args:
            risk_manager: Менеджер рисков для контроля торговых операций
        """
        self.ex_manager = ExchangeManager(config)
        self.risk_manager = risk_manager
        
        # Загрузка конфигурации
        self.arb_config = config.get("arbitrage", {})
        self.min_profit_threshold = self.arb_config.get("min_profit_threshold", 0.5)
        self.max_opportunity_age = self.arb_config.get("max_opportunity_age", 5.0)
        self.default_order_size = self.arb_config.get("default_order_size", 0.001)
        self.max_slippage_percent = self.arb_config.get("max_slippage_percent", 0.5)
        self.retry_attempts = self.arb_config.get("retry_attempts", 3)
        self.retry_delay = self.arb_config.get("retry_delay", 1.0)
        
        # Кэширование данных о комиссиях
        self.fee_cache: Dict[str, Dict[str, float]] = {}
        
        # Статистика
        self.stats = {
            "opportunities_found": 0,
            "profitable_opportunities": 0,
            "executed_trades": 0,
            "failed_trades": 0,
            "total_profit": 0.0,
            "start_time": time.time()
        }
        
        # Список отслеживаемых символов
        self.tracked_symbols = self.arb_config.get("tracked_symbols", [])
        
        logger.info(f"MultiExchangeArb инициализирован. Отслеживаемые символы: {self.tracked_symbols}")

    async def sync_balances(self) -> Dict[str, Any]:
        """
        Получает балансы со всех бирж.

        Returns:
            dict: Балансы по биржам.
        """
        try:
            balances = await self.ex_manager.multi_exchange_balance_sync()
            
            # Дополнительная обработка для удобного формата
            formatted_balances = {}
            for exchange, data in balances.items():
                if "total" in data:
                    # Фильтруем только непустые балансы
                    non_zero = {
                        currency: amount 
                        for currency, amount in data["total"].items() 
                        if amount > 0
                    }
                    formatted_balances[exchange] = non_zero
            
            return formatted_balances
        except Exception as e:
            logger.error(f"Ошибка при синхронизации балансов: {e}")
            return {}

    async def get_exchange_fee(self, exchange_id: str, symbol: str) -> Tuple[float, float]:
        """
        Получает информацию о комиссиях на бирже для указанного символа.
        
        Args:
            exchange_id: Идентификатор биржи
            symbol: Торговая пара
            
        Returns:
            Tuple[float, float]: Кортеж (maker_fee, taker_fee) в процентах
        """
        # Проверяем кэш
        if exchange_id in self.fee_cache and symbol in self.fee_cache[exchange_id]:
            return self.fee_cache[exchange_id].get("maker", 0.1), self.fee_cache[exchange_id].get("taker", 0.1)
        
        try:
            exchange = self.ex_manager.exchanges.get(exchange_id)
            if not exchange:
                logger.warning(f"Биржа {exchange_id} не найдена")
                return 0.1, 0.1  # Возвращаем стандартные значения по умолчанию
            
            # Получение информации о рынке
            markets = await exchange.fetch_markets()
            market = next((m for m in markets if m['symbol'] == symbol), None)
            
            if market and 'fee' in market:
                maker_fee = market['fee'].get('maker', 0.1)
                taker_fee = market['fee'].get('taker', 0.1)
            else:
                # Если информация о комиссиях недоступна, используем стандартные значения
                # или пытаемся получить из общих настроек биржи
                try:
                    trading_fees = await exchange.fetch_trading_fees()
                    if symbol in trading_fees:
                        maker_fee = trading_fees[symbol].get('maker', 0.1)
                        taker_fee = trading_fees[symbol].get('taker', 0.1)
                    else:
                        maker_fee = 0.1
                        taker_fee = 0.1
                except Exception:
                    maker_fee = 0.1
                    taker_fee = 0.1
            
            # Кэшируем результат
            if exchange_id not in self.fee_cache:
                self.fee_cache[exchange_id] = {}
            self.fee_cache[exchange_id][symbol] = {"maker": maker_fee, "taker": taker_fee}
            
            return maker_fee, taker_fee
            
        except Exception as e:
            logger.error(f"Ошибка при получении комиссий для {exchange_id}/{symbol}: {e}")
            return 0.1, 0.1  # Возвращаем стандартные значения по умолчанию в случае ошибки

    async def fetch_orderbook_with_retry(
        self, 
        exchange, 
        exchange_id: str, 
        symbol: str,
        attempts: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Получает книгу ордеров с повторными попытками в случае ошибки.
        
        Args:
            exchange: Объект биржи
            exchange_id: Идентификатор биржи для логирования
            symbol: Торговая пара
            attempts: Количество попыток
            
        Returns:
            Optional[Dict[str, Any]]: Книга ордеров или None в случае ошибки
        """
        for attempt in range(attempts):
            try:
                orderbook = await exchange.fetch_order_book(symbol)
                return orderbook
            except Exception as e:
                logger.warning(f"Попытка {attempt+1}/{attempts} получения OB для {exchange_id}/{symbol} не удалась: {e}")
                if attempt < attempts - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Экспоненциальная задержка
                else:
                    logger.error(f"Не удалось получить OB для {exchange_id}/{symbol} после {attempts} попыток")
                    return None

    async def find_multi_ex_arb(
        self, 
        symbol: str, 
        min_spread: float = 0.0,
        min_volume: float = 0.0,
        depth: int = 1,
        check_fees: bool = True
    ) -> List[ArbitrageOpportunity]:
        """
        Ищет арбитражные возможности между биржами с учетом глубины стакана и комиссий.

        Args:
            symbol: Торговая пара
            min_spread: Минимальный сырой спред в процентах
            min_volume: Минимальный объем для торговли
            depth: Глубина проверки ордербука (количество уровней)
            check_fees: Учитывать ли комиссии при расчете

        Returns:
            List[ArbitrageOpportunity]: Список арбитражных возможностей
        """
        orderbooks = {}
        tasks = []
        exchanges = self.ex_manager.exchanges
        
        # Запрашиваем ордербуки со всех бирж параллельно
        for eid, exch in exchanges.items():
            tasks.append(self.fetch_orderbook_with_retry(exch, eid, symbol))
        
        results = await asyncio.gather(*tasks)
        
        # Обрабатываем результаты
        for idx, eid in enumerate(exchanges.keys()):
            res = results[idx]
            if res is not None:
                orderbooks[eid] = res
        
        if len(orderbooks) < 2:
            logger.warning(f"Недостаточно данных для арбитража {symbol}. Получено ордербуков: {len(orderbooks)}")
            return []
        
        opportunities = []
        keys = list(orderbooks.keys())
        
        # Получаем комиссии для всех бирж
        fee_tasks = []
        for eid in keys:
            fee_tasks.append(self.get_exchange_fee(eid, symbol))
        
        fee_results = await asyncio.gather(*fee_tasks, return_exceptions=True)
        fees = {}
        
        for idx, eid in enumerate(keys):
            if not isinstance(fee_results[idx], Exception):
                maker_fee, taker_fee = fee_results[idx]
                fees[eid] = {"maker": maker_fee, "taker": taker_fee}
            else:
                logger.warning(f"Не удалось получить комиссии для {eid}: {fee_results[idx]}")
                fees[eid] = {"maker": 0.1, "taker": 0.1}  # Значения по умолчанию
        
        # Сравниваем каждую пару бирж
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                try:
                    # Получаем лучшие предложения с учетом глубины
                    exchange_i = keys[i]
                    exchange_j = keys[j]
                    
                    # Проверяем первые `depth` уровней ордербука
                    for d in range(min(depth, len(orderbooks[exchange_i]["bids"]), len(orderbooks[exchange_j]["asks"]))):
                        bid_i = orderbooks[exchange_i]["bids"][d][0]  # Цена покупки на бирже i
                        bid_vol_i = orderbooks[exchange_i]["bids"][d][1]  # Объем покупки на бирже i
                        
                        ask_j = orderbooks[exchange_j]["asks"][d][0]  # Цена продажи на бирже j
                        ask_vol_j = orderbooks[exchange_j]["asks"][d][1]  # Объем продажи на бирже j
                        
                        # Рассчитываем сырой спред (процент)
                        raw_spread_pct_i_j = (bid_i / ask_j - 1) * 100
                        
                        # Проверяем, есть ли возможность арбитража (i -> j)
                        if raw_spread_pct_i_j > min_spread and bid_vol_i >= min_volume and ask_vol_j >= min_volume:
                            # Создаем объект арбитражной возможности
                            opportunity = ArbitrageOpportunity(
                                buy_exchange=exchange_j,
                                sell_exchange=exchange_i,
                                symbol=symbol,
                                raw_spread=raw_spread_pct_i_j,
                                buy_price=ask_j,
                                sell_price=bid_i,
                                buy_volume=ask_vol_j,
                                sell_volume=bid_vol_i,
                                buy_fee=fees[exchange_j]["taker"],
                                sell_fee=fees[exchange_i]["maker"]
                            )
                            
                            # Добавляем только если спред с учетом комиссий положительный
                            if not check_fees or opportunity.net_spread > 0:
                                opportunities.append(opportunity)
                                self.stats["opportunities_found"] += 1
                    
                    # Проверяем обратное направление (j -> i)
                    for d in range(min(depth, len(orderbooks[exchange_j]["bids"]), len(orderbooks[exchange_i]["asks"]))):
                        bid_j = orderbooks[exchange_j]["bids"][d][0]
                        bid_vol_j = orderbooks[exchange_j]["bids"][d][1]
                        
                        ask_i = orderbooks[exchange_i]["asks"][d][0]
                        ask_vol_i = orderbooks[exchange_i]["asks"][d][1]
                        
                        # Рассчитываем сырой спред (процент)
                        raw_spread_pct_j_i = (bid_j / ask_i - 1) * 100
                        
                        # Проверяем, есть ли возможность арбитража (j -> i)
                        if raw_spread_pct_j_i > min_spread and bid_vol_j >= min_volume and ask_vol_i >= min_volume:
                            # Создаем объект арбитражной возможности
                            opportunity = ArbitrageOpportunity(
                                buy_exchange=exchange_i,
                                sell_exchange=exchange_j,
                                symbol=symbol,
                                raw_spread=raw_spread_pct_j_i,
                                buy_price=ask_i,
                                sell_price=bid_j,
                                buy_volume=ask_vol_i,
                                sell_volume=bid_vol_j,
                                buy_fee=fees[exchange_i]["taker"],
                                sell_fee=fees[exchange_j]["maker"]
                            )
                            
                            # Добавляем только если спред с учетом комиссий положительный
                            if not check_fees or opportunity.net_spread > 0:
                                opportunities.append(opportunity)
                                self.stats["opportunities_found"] += 1
                
                except Exception as e:
                    logger.error(
                        f"Ошибка при вычислении спреда для {keys[i]}, {keys[j]}, символ {symbol}: {e}"
                    )
        
        # Сортируем возможности по чистому спреду (от наибольшего к наименьшему)
        opportunities.sort(key=lambda x: x.net_spread, reverse=True)
        
        # Обновляем статистику прибыльных возможностей
        profitable_opps = [opp for opp in opportunities if opp.is_profitable(self.min_profit_threshold)]
        self.stats["profitable_opportunities"] += len(profitable_opps)
        
        return opportunities

    async def execute_arbitrage(
        self, 
        opportunity: ArbitrageOpportunity,
        trade_size: Optional[float] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Исполняет арбитражную возможность.
        
        Args:
            opportunity: Объект арбитражной возможности
            trade_size: Размер сделки (если None, используется меньший из доступных объемов или настройка по умолчанию)
            dry_run: Если True, только симулирует исполнение без реальных сделок
            
        Returns:
            Dict[str, Any]: Результат исполнения арбитража
        """
        # Определение размера сделки
        if trade_size is None:
            # Ограничиваем размер меньшим из доступных объемов, но не больше настройки по умолчанию
            trade_size = min(opportunity.max_trade_volume, self.default_order_size)
        else:
            # Убеждаемся, что запрошенный размер не превышает доступный объем
            trade_size = min(trade_size, opportunity.max_trade_volume)
        
        logger.info(
            f"Исполнение арбитража: {opportunity.symbol}, "
            f"Покупка на {opportunity.buy_exchange} по {opportunity.buy_price}, "
            f"Продажа на {opportunity.sell_exchange} по {opportunity.sell_price}, "
            f"Объем: {trade_size}, Ожидаемая прибыль: {opportunity.expected_profit:.6f}"
        )
        
        # В режиме dry_run только возвращаем предполагаемый результат
        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "symbol": opportunity.symbol,
                "trade_size": trade_size,
                "expected_profit": opportunity.expected_profit,
                "execution_time": 0,
                "raw_spread": opportunity.raw_spread,
                "net_spread": opportunity.net_spread
            }
        
        try:
            # Получаем объекты бирж
            buy_exchange = self.ex_manager.exchanges.get(opportunity.buy_exchange)
            sell_exchange = self.ex_manager.exchanges.get(opportunity.sell_exchange)
            
            if not buy_exchange or not sell_exchange:
                raise ValueError(f"Не удалось получить объекты бирж: {opportunity.buy_exchange}, {opportunity.sell_exchange}")
            
            start_time = time.time()
            
            # Проверка через менеджер рисков, если он доступен
            if self.risk_manager:
                risk_allowed = await self.risk_manager.check_trade_risk(
                    symbol=opportunity.symbol,
                    order_type="market",
                    side="buy",
                    amount=trade_size,
                    price=opportunity.buy_price
                )
                
                if not risk_allowed:
                    logger.warning(f"Менеджер рисков отклонил сделку для {opportunity.symbol}")
                    return {
                        "success": False,
                        "reason": "risk_manager_rejected",
                        "buy_exchange": opportunity.buy_exchange,
                        "sell_exchange": opportunity.sell_exchange,
                        "symbol": opportunity.symbol
                    }
            
            # Округление объема в меньшую сторону для избежания ошибок при торговле
            # Получаем точность символа, если можем
            try:
                buy_markets = await buy_exchange.fetch_markets()
                buy_market = next((m for m in buy_markets if m['symbol'] == opportunity.symbol), None)
                precision = buy_market.get('precision', {}).get('amount', 8) if buy_market else 8
                
                # Округляем объем с учетом точности
                rounded_size = round(trade_size - 1e-10, precision)
                
                # Если точность задана в виде дробных разрядов,
                # то нужно округлить до нужного количества знаков после запятой
                trade_size = float(Decimal(str(rounded_size)).quantize(
                    Decimal('0.' + '0' * precision), rounding=ROUND_DOWN))
            except Exception as e:
                logger.warning(f"Не удалось получить точность для {opportunity.symbol}: {e}")
                # В случае ошибки, округляем до 8 знаков после запятой
                trade_size = float(Decimal(str(trade_size)).quantize(
                    Decimal('0.00000001'), rounding=ROUND_DOWN))
            
            # Параллельное выполнение сделок
            buy_task = buy_exchange.create_market_buy_order(
                opportunity.symbol, trade_size)
            
            sell_task = sell_exchange.create_market_sell_order(
                opportunity.symbol, trade_size)
            
            buy_order, sell_order = await asyncio.gather(buy_task, sell_task)
            
            execution_time = time.time() - start_time
            
            # Расчет фактической прибыли (в общем случае требуется более сложная логика)
            try:
                buy_cost = float(buy_order.get('cost', trade_size * opportunity.buy_price))
                sell_value = float(sell_order.get('cost', trade_size * opportunity.sell_price))
                actual_profit = sell_value - buy_cost
                
                # Обновление статистики
                self.stats["executed_trades"] += 1
                self.stats["total_profit"] += actual_profit
                
                return {
                    "success": True,
                    "buy_exchange": opportunity.buy_exchange,
                    "sell_exchange": opportunity.sell_exchange,
                    "symbol": opportunity.symbol,
                    "trade_size": trade_size,
                    "expected_profit": opportunity.expected_profit,
                    "actual_profit": actual_profit,
                    "execution_time": execution_time,
                    "buy_order": buy_order,
                    "sell_order": sell_order,
                    "raw_spread": opportunity.raw_spread,
                    "net_spread": opportunity.net_spread
                }
            except Exception as e:
                logger.error(f"Ошибка при расчете фактической прибыли: {e}")
                
                return {
                    "success": True,
                    "buy_exchange": opportunity.buy_exchange,
                    "sell_exchange": opportunity.sell_exchange,
                    "symbol": opportunity.symbol,
                    "trade_size": trade_size,
                    "expected_profit": opportunity.expected_profit,
                    "execution_time": execution_time,
                    "buy_order": buy_order,
                    "sell_order": sell_order,
                    "raw_spread": opportunity.raw_spread,
                    "net_spread": opportunity.net_spread,
                    "profit_calc_error": str(e)
                }
        
        except Exception as e:
            logger.error(f"Ошибка при исполнении арбитража: {e}")
            self.stats["failed_trades"] += 1
            
            return {
                "success": False,
                "buy_exchange": opportunity.buy_exchange,
                "sell_exchange": opportunity.sell_exchange,
                "symbol": opportunity.symbol,
                "error": str(e)
            }

    async def scan_all_symbols(
        self, 
        symbols: Optional[List[str]] = None,
        min_spread: float = 0.5,
        min_volume: float = 0.001,
        check_fees: bool = True
    ) -> Dict[str, List[ArbitrageOpportunity]]:
        """
        Сканирует все указанные символы в поисках арбитражных возможностей.
        
        Args:
            symbols: Список символов для сканирования (если None, используются символы из конфигурации)
            min_spread: Минимальный спред в процентах
            min_volume: Минимальный объем для торговли
            check_fees: Учитывать ли комиссии при расчете
            
        Returns:
            Dict[str, List[ArbitrageOpportunity]]: Словарь возможностей по символам
        """
        if symbols is None:
            if not self.tracked_symbols:
                logger.warning("Нет отслеживаемых символов. Использую список по умолчанию.")
                symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            else:
                symbols = self.tracked_symbols
        
        all_opportunities = {}
        tasks = []
        
        # Создаем задачи для каждого символа
        for symbol in symbols:
            tasks.append(self.find_multi_ex_arb(
                symbol=symbol,
                min_spread=min_spread,
                min_volume=min_volume,
                check_fees=check_fees
            ))
        
        # Выполняем задачи параллельно
        results = await asyncio.gather(*tasks)
        
        # Обрабатываем результаты
        for idx, symbol in enumerate(symbols):
            opportunities = results[idx]
            
            # Отфильтровываем только прибыльные возможности
            profitable = [opp for opp in opportunities if opp.is_profitable(self.min_profit_threshold)]
            
            if profitable:
                all_opportunities[symbol] = profitable
        
        return all_opportunities
    
    async def run_continuous_arbitrage(
        self,
        interval: float = 1.0,
        max_concurrent_trades: int = 1,
        dry_run: bool = True,
        max_iterations: Optional[int] = None
    ) -> None:
        """
        Запускает непрерывное сканирование и исполнение арбитражных возможностей.
        
        Args:
            interval: Интервал между сканированиями в секундах
            max_concurrent_trades: Максимальное количество одновременных сделок
            dry_run: Если True, только симулирует исполнение без реальных сделок
            max_iterations: Максимальное количество итераций (None для бесконечного выполнения)
        """
        iteration = 0
        active_trades = set()
        
        logger.info(
            f"Запуск непрерывного арбитража. "
            f"Интервал: {interval}с, Макс. сделок: {max_concurrent_trades}, "
            f"Dry run: {dry_run}, Макс. итераций: {max_iterations if max_iterations else 'бесконечно'}"
        )
        
        while max_iterations is None or iteration < max_iterations:
            try:
                iteration += 1
                
                # Сканируем все символы
                opportunities_by_symbol = await self.scan_all_symbols()
                
                # Общее количество найденных возможностей
                total_opportunities = sum(len(opps) for opps in opportunities_by_symbol.values())
                
                if total_opportunities == 0:
                    logger.debug(f"Итерация {iteration}: Арбитражных возможностей не найдено")
                else:
                    logger.info(f"Итерация {iteration}: Найдено {total_opportunities} арбитражных возможностей")
                
                # Если есть возможности и свободные слоты для сделок
                if total_opportunities > 0 and len(active_trades) < max_concurrent_trades:
                    # Отбираем лучшие возможности для исполнения
                    all_opportunities = []
                    for symbol_opps in opportunities_by_symbol.values():
                        all_opportunities.extend(symbol_opps)
                    
                    # Сортируем по ожидаемой прибыли (от большей к меньшей)
                    all_opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
                    
                    # Исполняем лучшие возможности в пределах доступных слотов
                    available_slots = max_concurrent_trades - len(active_trades)
                    
                    for i in range(min(available_slots, len(all_opportunities))):
                        opportunity = all_opportunities[i]
                        
                        # Пропускаем символы, которые уже торгуются
                        opportunity_key = f"{opportunity.symbol}_{opportunity.buy_exchange}_{opportunity.sell_exchange}"
                        if opportunity_key in active_trades:
                            continue
                        
                        # Добавляем в активные сделки перед исполнением
                        active_trades.add(opportunity_key)
                        
                        # Выполняем арбитраж
                        try:
                            result = await self.execute_arbitrage(opportunity, dry_run=dry_run)
                            
                            if result["success"]:
                                profit = result.get("actual_profit", opportunity.expected_profit)
                                logger.info(
                                    f"Успешное исполнение арбитража: {opportunity.symbol}, "
                                    f"Прибыль: {profit:.6f}, "
                                    f"Время исполнения: {result.get('execution_time', 0):.3f}с"
                                )
                            else:
                                logger.warning(
                                    f"Неудачное исполнение арбитража: {opportunity.symbol}, "
                                    f"Причина: {result.get('reason', 'unknown')}"
                                )
                        
                        except Exception as e:
                            logger.error(f"Ошибка при исполнении арбитража: {e}")
                        
                        finally:
                            # Удаляем из активных сделок
                            active_trades.remove(opportunity_key)
                
                # Выводим текущую статистику каждые 10 итераций
                if iteration % 10 == 0:
                    runtime = time.time() - self.stats["start_time"]
                    logger.info(
                        f"Статистика арбитража за {runtime:.1f}с: "
                        f"Найдено: {self.stats['opportunities_found']}, "
                        f"Прибыльных: {self.stats['profitable_opportunities']}, "
                        f"Исполнено: {self.stats['executed_trades']}, "
                        f"Неудачных: {self.stats['failed_trades']}, "
                        f"Общая прибыль: {self.stats['total_profit']:.6f}"
                    )
                
                # Пауза между итерациями
                await asyncio.sleep(interval)
            
            except Exception as e:
                logger.error(f"Ошибка в цикле непрерывного арбитража: {e}")
                await asyncio.sleep(interval)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику работы арбитража.
        
        Returns:
            Dict[str, Any]: Статистика арбитража
        """
        runtime = time.time() - self.stats["start_time"]
        
        return {
            **self.stats,
            "runtime_seconds": runtime,
            "runtime_formatted": f"{int(runtime // 3600)}ч {int((runtime % 3600) // 60)}м {int(runtime % 60)}с",
            "profit_per_hour": (self.stats["total_profit"] / runtime) * 3600 if runtime > 0 else 0,
            "success_rate": (self.stats["executed_trades"] / (self.stats["executed_trades"] + self.stats["failed_trades"])) * 100 
                if (self.stats["executed_trades"] + self.stats["failed_trades"]) > 0 else 0,
            "opportunity_rate": self.stats["opportunities_found"] / runtime if runtime > 0 else 0,
            "time": time.time()
        }


async def main():
    """Функция для тестирования функциональности."""
    # Инициализация арбитражного класса
    arb = MultiExchangeArb()
    
    # Получение балансов
    balances = await arb.sync_balances()
    print("Балансы на биржах:")
    for exchange, currencies in balances.items():
        print(f"{exchange}: {currencies}")
    
    # Поиск арбитражных возможностей
    symbol = "BTC/USDT"
    print(f"\nПоиск арбитражных возможностей для {symbol}...")
    opportunities = await arb.find_multi_ex_arb(symbol, min_spread=0.1, min_volume=0.001)
    
    if opportunities:
        print(f"Найдено {len(opportunities)} возможностей:")
        for idx, opp in enumerate(opportunities):
            print(f"{idx+1}. {opp}")
        
        # Выполнение первой возможности в режиме dry_run
        if opportunities:
            print("\nСимуляция исполнения первой возможности:")
            result = await arb.execute_arbitrage(opportunities[0], dry_run=True)
            print(f"Результат: {result}")
    else:
        print("Арбитражных возможностей не найдено")
    
    # Демонстрация непрерывного сканирования
    print("\nЗапуск непрерывного сканирования (3 итерации)...")
    await arb.run_continuous_arbitrage(interval=2.0, max_iterations=3, dry_run=True)
    
    # Вывод статистики
    print("\nСтатистика арбитража:")
    stats = arb.get_statistics()
    for key, value in stats.items():
        if key != "time":
            print(f"{key}: {value}")


if __name__ == "__main__":
    # Для ручного запуска и тестирования
    asyncio.run(main())