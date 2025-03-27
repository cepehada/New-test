"""
Стратегия кросс-торговли для работы с несколькими биржами.
Использует арбитраж и разницу в ценах для получения прибыли.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from project.bots.strategies.base_strategy import BaseStrategy, StrategyStatus
from project.config import get_config
from project.data.market_data import MarketData
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CrossStrategy(BaseStrategy):
    """
    Стратегия кросс-торговли между разными биржами.
    """

    def __init__(self, name: str = "CrossStrategy", exchange_id: str = "binance",
                symbols: List[str] = None, timeframes: List[str] = None,
                config: Dict[str, Any] = None):
        """
        Инициализирует стратегию кросс-торговли.

        Args:
            name: Имя стратегии
            exchange_id: Идентификатор основной биржи
            symbols: Список символов для торговли
            timeframes: Список таймфреймов для анализа
            config: Конфигурация стратегии
        """
        # Устанавливаем значения по умолчанию
        config = config or {}
        default_config = {
            "secondary_exchanges": ["kucoin", "huobi", "okex"],  # дополнительные биржи
            "min_price_difference": 0.01,  # минимальная разница в цене (1%)
            "min_volume": 10000,  # минимальный объем для торговли
            "max_position_size_pct": 0.1,  # максимальный размер позиции (10% от баланса)
            "check_interval": 60,  # интервал проверки в секундах
            # минимальный баланс для торговли (10% от максимального размера позиции)
            "balance_threshold": 0.1,
            "fee_margin": 0.002,  # маржа для комиссий (0.2%)
            "max_open_positions": 3,  # максимальное количество открытых позиций
            "use_triangular_arbitrage": False,  # использовать треугольный арбитраж
            "base_currencies": ["USDT", "BTC", "ETH"],  # базовые валюты для треугольного арбитража
            "arbitrage_threshold": 0.005  # порог для арбитража (0.5%)
        }

        # Объединяем с пользовательской конфигурацией
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        # Устанавливаем базовые значения
        symbols = symbols or ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT"]
        timeframes = timeframes or ["5m", "15m"]

        super().__init__(name, exchange_id, symbols, timeframes, config)

        # Дополнительные параметры
        self.exchange_prices: Dict[str, Dict[str, float]] = {}  # exchange -> symbol -> price
        self.exchange_volumes: Dict[str, Dict[str, float]] = {}  # exchange -> symbol -> volume
        self.arbitrage_opportunities: List[Dict[str, Any]] = []  # список возможностей для арбитража
        self.last_check_time = 0  # время последней проверки

        logger.debug("Создана стратегия кросс-торговли {self.name}" %)

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """
        # Обновляем числовые параметры
        for param in [
            "min_price_difference", "max_position_size_pct", "balance_threshold",
            "fee_margin", "arbitrage_threshold"
        ]:
            if param in config:
                self.strategy_config[param] = float(config[param])

        for param in ["min_volume", "check_interval", "max_open_positions"]:
            if param in config:
                self.strategy_config[param] = int(config[param])

        # Обновляем булевы параметры
        if "use_triangular_arbitrage" in config:
            self.strategy_config["use_triangular_arbitrage"] = bool(
                config["use_triangular_arbitrage"])

        # Обновляем списки
        if "secondary_exchanges" in config:
            self.strategy_config["secondary_exchanges"] = config["secondary_exchanges"]

        if "base_currencies" in config:
            self.strategy_config["base_currencies"] = config["base_currencies"]

    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        """
        # Инициализируем словари для хранения цен и объемов
        exchanges = [self.exchange_id] + self.strategy_config["secondary_exchanges"]

        for exchange in exchanges:
            self.exchange_prices[exchange] = {}
            self.exchange_volumes[exchange] = {}


        # Устанавливаем интервал обновления
        self.update_interval = self.strategy_config["check_interval"]

        # Загружаем начальные данные
        await self._update_cross_market_data()

    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        """
        # Нет специфических ресурсов для очистки
    
    @async_handle_error
    async def _update_cross_market_data(self) -> None:
        """
        Обновляет данные по ценам и объемам на разных биржах.
        """
        exchanges = [self.exchange_id] + self.strategy_config["secondary_exchanges"]
        
        for exchange in exchanges:
            for symbol in self.symbols:
                try:
                    # Получаем тикер
                    ticker = await self.market_data.get_ticker(exchange, symbol)
                    
                    if ticker:
                        # Сохраняем цену и объем
                        price = ticker.get("last", 0)
                        volume = ticker.get("quoteVolume", 0) or ticker.get("volume", 0)
                        
                        if price > 0:
                            self.exchange_prices[exchange][symbol] = price
                        
                        if volume > 0:
                            self.exchange_volumes[exchange][symbol] = volume
                
                except Exception as e:
                    logger.warning("Ошибка при получении данных для {symbol} на {exchange}: {str(e)}" %)
        
        self.last_check_time = time.time()
    
    @async_handle_error
    async def _find_arbitrage_opportunities(self) -> None:
        """
        Ищет возможности для арбитража между биржами.
        """
        self.arbitrage_opportunities = []
        
        # Получаем списки бирж и символов
        exchanges = [self.exchange_id] + self.strategy_config["secondary_exchanges"]
        
        for symbol in self.symbols:
            # Собираем цены по всем биржам
            symbol_prices = {}
            symbol_volumes = {}
            
            for exchange in exchanges:
                price = self.exchange_prices.get(exchange, {}).get(symbol, 0)
                volume = self.exchange_volumes.get(exchange, {}).get(symbol, 0)
                
                if price > 0 и volume >= self.strategy_config["min_volume"]:
                    symbol_prices[exchange] = price
                    symbol_volumes[exchange] = volume
            
            # Если цены есть как минимум на двух биржах
            if len(symbol_prices) >= 2:
                # Находим биржу с минимальной и максимальной ценой
                min_exchange = min(symbol_prices, key=symbol_prices.get)
                max_exchange = max(symbol_prices, key=symbol_prices.get)
                
                min_price = symbol_prices[min_exchange]
                max_price = symbol_prices[max_exchange]
                
                # Рассчитываем разницу в процентах
                price_diff_pct = (max_price / min_price - 1)
                
                # Учитываем комиссии
                net_profit_pct = price_diff_pct - self.strategy_config["fee_margin"]
                
                # Если разница больше минимального порога
                if net_profit_pct > self.strategy_config["min_price_difference"]:
                    # Добавляем возможность арбитража
                    opportunity = {
                        "symbol": symbol,
                        "buy_exchange": min_exchange,
                        "sell_exchange": max_exchange,
                        "buy_price": min_price,
                        "sell_price": max_price,
                        "price_diff_pct": price_diff_pct,
                        "net_profit_pct": net_profit_pct,
                        "buy_volume": symbol_volumes[min_exchange],
                        "sell_volume": symbol_volumes[max_exchange],
                        "timestamp": time.time()
                    }
                    
                    self.arbitrage_opportunities.append(opportunity)
                    
                    logger.debug("Найдена возможность арбитража для {symbol}: " %
                                f"купить на {min_exchange} по {min_price:.8f}, "
                                f"продать на {max_exchange} по {max_price:.8f}, "
                                f"прибыль: {net_profit_pct:.2%}")
        
        # Сортируем возможности по прибыли
        self.arbitrage_opportunities.sort(key=lambda x: x["net_profit_pct"], reverse=True)
    
    @async_handle_error
    async def _find_triangular_arbitrage(self) -> None:
        """
        Ищет возможности для треугольного арбитража на одной бирже.
        """
        if not self.strategy_config["use_triangular_arbitrage"]:
            return
        
        for exchange in [self.exchange_id] + self.strategy_config["secondary_exchanges"]:
            try:
                # Получаем все цены для этой биржи
                prices = self.exchange_prices.get(exchange, {})
                
                if not prices:
                    continue
                
                # Проходим по всем базовым валютам
                for base in self.strategy_config["base_currencies"]:
                    # Ищем все пары с этой базовой валютой
                    base_pairs = {}
                    
                    for symbol, price in prices.items():
                        if symbol.endswith(f"/{base}"):
                            coin = symbol.split('/')[0]
                            base_pairs[coin] = price
                    
                    # Если найдено как минимум 2 пары
                    if len(base_pairs) >= 2:
                        # Ищем все возможные треугольники
                        for coin_a, price_a in base_pairs.items():
                            for coin_b, price_b in base_pairs.items():
                                if coin_a == coin_b:
                                    continue
                                
                                # Проверяем, есть ли прямая пара между монетами
                                direct_pair = f"{coin_a}/{coin_b}"
                                reverse_pair = f"{coin_b}/{coin_a}"
                                
                                direct_price = prices.get(direct_pair, 0)
                                reverse_price = prices.get(reverse_pair, 0)
                                
                                if direct_price > 0:
                                    # Треугольник: base -> coin_a -> coin_b -> base
                                    # 1. Купить coin_a за base: 1/price_a coin_a
                                    # 2. Купить coin_b за coin_a: 1/price_a * direct_price coin_b
                                    # 3. Продать coin_b за base: 1/price_a * direct_price * price_b base
                                    
                                    # Рассчитываем чистый результат
                                    result = (1 / price_a) * direct_price * price_b
                                    profit_pct = result - 1 - self.strategy_config["fee_margin"]
                                    
                                    if profit_pct > self.strategy_config["arbitrage_threshold"]:
                                        opportunity = {
                                            "type": "triangular",
                                            "exchange": exchange,
                                            "base": base,
                                            "step1": f"Купить {coin_a} за {base}",
                                            "step2": f"Купить {coin_b} за {coin_a}",
                                            "step3": f"Продать {coin_b} за {base}",
                                            "profit_pct": profit_pct,
                                            "timestamp": time.time()
                                        }
                                        
                                        self.arbitrage_opportunities.append(opportunity)
                                        
                                        logger.debug("Найдена возможность треугольного арбитража на {exchange}: " %
                                                    f"{base} -> {coin_a} -> {coin_b} -> {base}, "
                                                    f"прибыль: {profit_pct:.2%}")
                                
                                elif reverse_price > 0:
                                    # Треугольник: base -> coin_a -> coin_b -> base
                                    # 1. Купить coin_a за base: 1/price_a coin_a
                                    # 2. Продать coin_a за coin_b: 1/price_a * 1/reverse_price coin_b
                                    # 3. Продать coin_b за base: 1/price_a * 1/reverse_price * price_b base
                                    
                                    # Рассчитываем чистый результат
                                    result = (1 / price_a) * (1 / reverse_price) * price_b
                                    profit_pct = result - 1 - self.strategy_config["fee_margin"]
                                    
                                    if profit_pct > self.strategy_config["arbitrage_threshold"]:
                                        opportunity = {
                                            "type": "triangular",
                                            "exchange": exchange,
                                            "base": base,
                                            "step1": f"Купить {coin_a} за {base}",
                                            "step2": f"Продать {coin_a} за {coin_b}",
                                            "step3": f"Продать {coin_b} за {base}",
                                            "profit_pct": profit_pct,
                                            "timestamp": time.time()
                                        }
                                        
                                        self.arbitrage_opportunities.append(opportunity)
                                        
                                        logger.debug("Найдена возможность треугольного арбитража на {exchange}: " %
                                                    f"{base} -> {coin_a} -> {coin_b} -> {base}, "
                                                    f"прибыль: {profit_pct:.2%}")
            
            except Exception as e:
                logger.error("Ошибка при поиске треугольного арбитража на {exchange}: {str(e)}" %)
    
    @async_handle_error
    async def _check_balances(self, opportunity: Dict[str, Any]) -> bool:
        """
        Проверяет наличие достаточного баланса для выполнения арбитража.
        
        Args:
            opportunity: Словарь с данными о возможности арбитража
            
        Returns:
            True, если баланс достаточен, иначе False
        """
        if "type" in opportunity и opportunity["type"] == "triangular":
            # Треугольный арбитраж
            exchange = opportunity["exchange"]
            base = opportunity["base"]
            
            try:
                # Получаем баланс
                balance = await self.market_data.get_balance(exchange)
                if not balance:
                    return False
                
                # Проверяем наличие достаточного баланса базовой валюты
                available = balance.get("free", {}).get(base, 0)
                
                # Рассчитываем требуемый объем
                required = self.strategy_config["max_position_size_pct"] * self.strategy_config.get("account_balance", 10000)
                minimum = required * self.strategy_config["balance_threshold"]
                
                return available >= minimum
                
            except Exception as e:
                logger.error("Ошибка при проверке баланса для треугольного арбитража: {str(e)}" %)
                return False
        
        else:
            # Обычный арбитраж между биржами
            buy_exchange = opportunity["buy_exchange"]
            sell_exchange = opportunity["sell_exchange"]
            symbol = opportunity["symbol"]
            
            try:
                # Получаем базовую и котируемую валюты
                base, quote = symbol.split('/')
                
                # Получаем балансы на обеих биржах
                buy_balance = await self.market_data.get_balance(buy_exchange)
                sell_balance = await self.market_data.get_balance(sell_exchange)
                
                if not buy_balance or not sell_balance:
                    return False
                
                # Проверяем наличие достаточного баланса котируемой валюты на бирже для покупки
                buy_available = buy_balance.get("free", {}).get(quote, 0)
                
                # Проверяем наличие достаточного баланса базовой валюты на бирже для продажи
                sell_available = sell_balance.get("free", {}).get(base, 0)
                
                # Рассчитываем требуемый объем
                required_quote = self.strategy_config["max_position_size_pct"] * self.strategy_config.get("account_balance", 10000)
                required_base = required_quote / opportunity["buy_price"]
                
                # Устанавливаем минимальные пороги
                min_quote = required_quote * self.strategy_config["balance_threshold"]
                min_base = required_base * self.strategy_config["balance_threshold"]
                
                # Проверяем достаточность балансов
                return buy_available >= min_quote и sell_available >= min_base
                
            except Exception as e:
                logger.error("Ошибка при проверке балансов для арбитража: {str(e)}" %)
                return False
    
    @async_handle_error
    async def _execute_arbitrage(self, opportunity: Dict[str, Any]) -> bool:
        """
        Выполняет арбитражную операцию.
        
        Args:
            opportunity: Словарь с данными о возможности арбитража
            
        Returns:
            True, если операция выполнена успешно, иначе False
        """
        # В реальном проекте здесь должен быть код для выполнения арбитражной операции
        # Поскольку это требует интеграции с биржами и управления реальными средствами,
        # мы просто имитируем выполнение и возвращаем успех
        
        if "type" в opportunity и opportunity["type"] == "triangular":
            # Треугольный арбитраж
            logger.info("Выполняем треугольный арбитраж на {opportunity['exchange']}: " %
                       f"{opportunity['step1']}, {opportunity['step2']}, {opportunity['step3']}, "
                       f"ожидаемая прибыль: {opportunity['profit_pct']:.2%}")
        else:
            # Обычный арбитраж между биржами
            logger.info("Выполняем арбитраж для {opportunity['symbol']}: " %
                       f"покупка на {opportunity['buy_exchange']} по {opportunity['buy_price']:.8f}, "
                       f"продажа на {opportunity['sell_exchange']} по {opportunity['sell_price']:.8f}, "
                       f"ожидаемая прибыль: {opportunity['net_profit_pct']:.2%}")
        
        # Имитируем успешное выполнение
        return True
    
    @async_handle_error
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы на основе арбитражных возможностей.
        
        Returns:
            Словарь с сигналами для каждого символа
        """
        # Проверяем, прошло ли достаточно времени с последней проверки
        current_time = time.time()
        if current_time - self.last_check_time < self.strategy_config["check_interval"]:
            return {}
        
        # Обновляем данные по ценам и объемам
        await self._update_cross_market_data()
        
        # Ищем возможности для арбитража
        await self._find_arbitrage_opportunities()
        
        # Если включен треугольный арбитраж, ищем такие возможности
        if self.strategy_config["use_triangular_arbitrage"]:
            await self._find_triangular_arbitrage()
        
        # Если нет возможностей, возвращаем пустой словарь
        if not self.arbitrage_opportunities:
            return {}
        
        signals = {}
        
        # Проверяем каждую возможность
        for opportunity в self.arbitrage_opportunities:
            # Проверяем, не превышено ли максимальное количество открытых позиций
            if len(self.open_positions) >= self.strategy_config["max_open_positions"]:
                break
            
            # Проверяем балансы
            has_balance = await self._check_balances(opportunity)
            if not has_balance:
                continue
            
            # Выполняем арбитраж
            success = await self._execute_arbitrage(opportunity)
            
            if success:
                # Формируем сигнал на основе успешной операции
                if "type" в opportunity и opportunity["type"] == "triangular":
                    # Треугольный арбитраж не создает позицию в обычном смысле
                    continue
                
                else:
                    # Создаем сигнал для обычного арбитража
                    symbol = opportunity["symbol"]
                    
                    signals[symbol] = {
                        "symbol": symbol,
                        "action": "arbitrage",
                        "buy_exchange": opportunity["buy_exchange"],
                        "sell_exchange": opportunity["sell_exchange"],
                        "buy_price": opportunity["buy_price"],
                        "sell_price": opportunity["sell_price"],
                        "price_diff_pct": opportunity["price_diff_pct"],
                        "timestamp": time.time()
                    }
                    
                    # В реальной стратегии здесь должен быть код для отслеживания арбитражной позиции
                    # Например, создание записи в self.open_positions
        
        return signals

    def calculate_signals(self, data):
        """Рассчитывает торговые сигналы на основе пересечения индикаторов"""
        # ...existing code...
        
        # Исправление отступа на строке 123
        for i in range(len(data)):
            # Логика расчета сигналов
            pass
            
        # ...existing code...

    def _get_signal(self, fast_line, slow_line, previous_fast, previous_slow):
        """Определяет сигнал на основе пересечения линий"""
        if fast_line > slow_line and previous_fast <= previous_slow:
            return "buy"
        elif fast_line < slow_line and previous_fast >= previous_slow:
            return "sell"
        else:
            return "neutral"

    def calculate_signal(self):
        # Make sure this is properly indented
        if self.data is None or len(self.data) < self.fast_ma_period + 10:
            return "neutral"
        
        # Rest of the method with proper indentation
        # ...
