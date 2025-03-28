import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from project.utils.logging_utils import setup_logger
from project.config import get_config
from project.data.market_data import MarketDataProvider
from project.exchange.exchange_manager import ExchangeManager
from project.trade_executor.advanced_order_manager import (
    AdvancedOrderManager,
    OrderType,
    OrderSide,
)
from project.trade_executor.capital_manager import CapitalManager
from project.trade_executor.strategy_manager import StrategyManager
from project.trade_executor.symbols_updater import SymbolsUpdater
from project.trade_executor.dynamic_sl_tp import DynamicSLTPManager
from project.utils.notify import send_notification

logger = setup_logger("autonomous_bot")


class AutonomousBot:
    """Полностью автономный торговый бот с динамическим управлением"""

    def __init__(self, config_path: str = None):
        self.config = get_config()
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)

        # Инициализируем основные компоненты
        self.exchange_manager = ExchangeManager(self.config)
        self.market_data = MarketDataProvider()
        self.order_manager = AdvancedOrderManager(self.config)
        self.capital_manager = CapitalManager(self.config)
        self.strategy_manager = StrategyManager(self.config)
        self.symbols_updater = SymbolsUpdater(self.config)

        # Инициализируем менеджер динамических SL/TP
        self.sl_tp_manager = DynamicSLTPManager(self.order_manager, self.config)

        # Настройки бота
        self.trading_enabled = self.config.get("trading_enabled", False)
        self.update_interval = self.config.get("update_interval", 60)  # секунд
        self.max_open_positions = self.config.get("max_open_positions", 5)
        self.max_risk_per_trade = self.config.get(
            "max_risk_per_trade", 0.02
        )  # 2% риска
        self.timeframes = self.config.get("timeframes", ["1h", "4h", "1d"])

        # Состояние бота
        self.open_positions = {}
        self._stop_requested = False
        self._main_task = None
        self.last_analyzed = {}  # Время последнего анализа по каждому символу
        self.blacklisted_symbols = set()  # Символы, которые следует игнорировать

        logger.info("Автономный бот инициализирован")

    def _load_config(self, config_path: str):
        """Загружает конфигурацию из файла"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            self.config.update(config)
            logger.info(f"Конфигурация загружена из {config_path}")
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации: {str(e)}")

    async def start(self):
        """Запускает бота"""
        if self._main_task is not None:
            logger.warning("Бот уже запущен")
            return

        # Запускаем все компоненты
        await self.exchange_manager.start()
        await self.order_manager.start()
        await self.symbols_updater.start()
        await self.sl_tp_manager.start()

        # Запускаем основной цикл
        self._stop_requested = False
        self._main_task = asyncio.create_task(self._main_loop())

        logger.info("Автономный бот запущен")

    async def stop(self):
        """Останавливает бота"""
        if self._main_task is None:
            logger.warning("Бот не запущен")
            return

        self._stop_requested = True

        # Останавливаем основной цикл
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
            self._main_task = None

        # Останавливаем все компоненты
        await self.sl_tp_manager.stop()
        await self.symbols_updater.stop()
        await self.order_manager.stop()
        await self.exchange_manager.stop()

        logger.info("Автономный бот остановлен")

    async def _main_loop(self):
        """Основной цикл работы бота"""
        try:
            # Первый запуск - обновляем список символов
            await self.symbols_updater.update_symbols()

            # Основной цикл
            while not self._stop_requested:
                try:
                    # Получаем рекомендуемые символы
                    symbols = self.symbols_updater.get_recommended_symbols()

                    if not symbols:
                        logger.warning("Нет рекомендуемых символов для торговли")
                        await asyncio.sleep(60)
                        continue

                    # Анализируем каждый символ
                    for symbol in symbols:
                        if symbol in self.blacklisted_symbols:
                            continue

                        # Проверяем, не анализировали ли мы этот символ недавно
                        if symbol in self.last_analyzed:
                            time_since_last = (
                                datetime.now() - self.last_analyzed[symbol]
                            )
                            if time_since_last < timedelta(
                                minutes=15
                            ):  # Не анализируем чаще, чем раз в 15 минут
                                continue

                        # Анализируем символ и принимаем решения
                        await self._analyze_and_trade(symbol)

                        # Небольшая пауза между символами, чтобы не перегрузить API
                        await asyncio.sleep(2)

                    # Проверяем и обновляем открытые позиции
                    await self._check_open_positions()

                    # Делаем паузу перед следующей итерацией
                    await asyncio.sleep(self.update_interval)

                except asyncio.CancelledError:
                    logger.info("Основной цикл бота отменен")
                    break
                except Exception as e:
                    logger.error(f"Ошибка в основном цикле бота: {str(e)}")
                    await asyncio.sleep(30)  # Делаем паузу перед повторной попыткой

        finally:
            logger.info("Основной цикл бота завершен")

    async def _analyze_and_trade(self, symbol: str):
        """
        Анализирует рынок для указанного символа и принимает решение о торговле

        Args:
            symbol: Торговая пара для анализа
        """
        try:
            # Отмечаем время анализа
            self.last_analyzed[symbol] = datetime.now()

            # Получаем и анализируем рыночные данные
            market_data = await self._get_market_analysis(symbol)

            # Проверяем, достаточно ли данных для анализа
            if not market_data or not market_data.get("last_price"):
                logger.warning(f"Недостаточно данных для анализа {symbol}")
                return

            # Проверяем, не превышено ли максимальное количество открытых позиций
            if len(self.open_positions) >= self.max_open_positions:
                logger.debug(
                    f"Достигнуто максимальное количество открытых позиций ({
                        self.max_open_positions})"
                )
                return

            # Выбираем наиболее подходящую стратегию
            strategy = self.strategy_manager.select_best_strategy(market_data)

            # Получаем сигнал от стратегии
            signal = await self._get_strategy_signal(strategy, symbol, market_data)

            if not signal:
                return

            action = signal.get("action")
            if action not in ["buy", "sell"]:
                return

            # Рассчитываем сумму для торговли
            balance = await self._get_balance()
            allocation = self.capital_manager.calculate_allocation(
                balance=balance,
                volatility=market_data.get("volatility", 0.01),
                slippage=market_data.get("slippage", 0.001),
            )

            if allocation <= 0:
                return

            # Рассчитываем количество на основе цены
            price = market_data.get("last_price")
            amount = allocation / price

            # Округляем количество с учетом минимального шага
            amount = self._round_amount(amount, symbol)

            if amount <= 0:
                return

            # Выбираем тип ордера
            order_type = (
                OrderType.MARKET
                if signal.get("confidence", 0) > 0.8
                else OrderType.LIMIT
            )

            # Определяем цену для лимитного ордера
            limit_price = None
            if order_type == OrderType.LIMIT:
                # Для покупки ставим цену чуть ниже текущей, для продажи - чуть выше
                if action == "buy":
                    limit_price = price * 0.995  # На 0.5% ниже
                else:
                    limit_price = price * 1.005  # На 0.5% выше

            # Выбираем биржу
            exchange_id = await self._select_best_exchange(symbol)

            # Создаем ордер только если торговля включена
            if self.trading_enabled:
                side = OrderSide.BUY if action == "buy" else OrderSide.SELL

                # Создаем ордер
                order = await self.order_manager.create_order(
                    symbol=symbol,
                    order_type=order_type,
                    side=side,
                    amount=amount,
                    price=limit_price,
                    exchange_id=exchange_id,
                    params={"autoSLTP": True},
                )

                logger.info(
                    f"Создан ордер {order.order_id} для {symbol} на {exchange_id}: "
                    f"{action} {amount} @ {limit_price or 'рынок'} "
                    f"(стратегия: {strategy}, confidence: {signal.get('confidence', 0):.2f})"
                )

                # Если ордер сразу выполнен, добавляем в открытые позиции
                if order.status == OrderStatus.FILLED:
                    self.open_positions[order.order_id] = order

                    # Создаем динамические SL/TP
                    sl_tp = await self.sl_tp_manager.create_dynamic_sl_tp(
                        order, market_data
                    )
                    logger.info(
                        f"Созданы SL/TP для ордера {order.order_id}: "
                        f"SL={sl_tp.get('sl', {}).get('order_id')}, "
                        f"TP={sl_tp.get('tp', {}).get('order_id')}"
                    )
            else:
                logger.info(
                    f"Получен сигнал для {symbol}: {action} (стратегия: {strategy}, "
                    f"confidence: {signal.get('confidence', 0):.2f}), но торговля отключена"
                )

        except Exception as e:
            logger.error(f"Ошибка при анализе и торговле {symbol}: {str(e)}")

    async def _get_market_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Получает и анализирует рыночные данные для символа

        Args:
            symbol: Торговая пара для анализа

        Returns:
            Dict: Результаты анализа рынка
        """
        result = {"symbol": symbol, "timestamp": datetime.now().timestamp()}

        try:
            # Получаем данные OHLCV для разных таймфреймов
            ohlcv_data = {}

            for tf in self.timeframes:
                data = await self.market_data.get_ohlcv(symbol, timeframe=tf, limit=100)
                if data is not None:
                    ohlcv_data[tf] = data

            if not ohlcv_data:
                logger.warning(f"Не удалось получить OHLCV данные для {symbol}")
                return result

            # Получаем последнюю цену
            ticker = await self.market_data.get_ticker(symbol)
            if ticker:
                result["last_price"] = ticker.get("last", 0)
                result["bid"] = ticker.get("bid", 0)
                result["ask"] = ticker.get("ask", 0)

            # Рассчитываем волатильность (на основе ATR)
            daily_data = ohlcv_data.get("1d", None)
            if daily_data is not None:
                # ATR (Average True Range)
                highs = daily_data["high"]
                lows = daily_data["low"]
                closes = daily_data["close"]

                # Рассчитываем True Range
                tr = []
                for i in range(1, len(closes)):
                    tr.append(
                        max(
                            highs[i] - lows[i],  # Текущий диапазон
                            abs(
                                highs[i] - closes[i - 1]
                            ),  # Текущий максимум - предыдущее закрытие
                            abs(
                                lows[i] - closes[i - 1]
                            ),  # Текущий минимум - предыдущее закрытие
                        )
                    )

                # Средний True Range за 14 дней
                atr = sum(tr[-14:]) / min(14, len(tr)) if tr else 0
                result["atr"] = atr

                # Волатильность как отношение ATR к цене
                if result.get("last_price", 0) > 0:
                    result["volatility"] = atr / result["last_price"]
                else:
                    result["volatility"] = 0

            # Оценка тренда (на основе EMA)
            if "1d" in ohlcv_data:
                closes = ohlcv_data["1d"]["close"]

                # EMA 20 и 50
                ema20 = self._calculate_ema(closes, 20)
                ema50 = self._calculate_ema(closes, 50)

                if ema20 > ema50:
                    result["trend"] = "uptrend"
                elif ema20 < ema50:
                    result["trend"] = "downtrend"
                else:
                    result["trend"] = "sideways"

                # Сила тренда (отношение EMA20 к EMA50)
                result["trend_strength"] = abs(ema20 / ema50 - 1) if ema50 > 0 else 0

            # Оценка слиппейджа на основе книги ордеров
            order_book = await self.market_data.get_order_book(symbol)
            if order_book:
                # Оценка ликвидности - сумма объемов в стакане
                bid_volume = sum(
                    amount for price, amount in order_book.get("bids", [])[:5]
                )
                ask_volume = sum(
                    amount for price, amount in order_book.get("asks", [])[:5]
                )

                # Средний объем
                avg_volume = (
                    (bid_volume + ask_volume) / 2 if bid_volume and ask_volume else 0
                )

                # Спред
                best_bid = order_book["bids"][0][0] if order_book.get("bids") else 0
                best_ask = order_book["asks"][0][0] if order_book.get("asks") else 0

                if best_bid > 0 and best_ask > 0:
                    spread = (best_ask - best_bid) / best_bid
                    result["spread"] = spread

                    # Оценка слиппейджа на основе спреда и ликвидности
                    # Чем больше спред и меньше ликвидность, тем больше слиппейдж
                    result["slippage"] = spread * (1 + (1 / (avg_volume + 1)))

            return result

        except Exception as e:
            logger.error(f"Ошибка при анализе рынка для {symbol}: {str(e)}")
            return result

    def _calculate_ema(self, prices, period):
        """Рассчитывает Exponential Moving Average"""
        if len(prices) < period:
            return 0

        # Коэффициент сглаживания
        multiplier = 2 / (period + 1)

        # Инициализируем EMA как SMA за первый период
        ema = sum(prices[:period]) / period

        # Рассчитываем EMA
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    async def _get_strategy_signal(
        self, strategy: str, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Получает торговый сигнал от выбранной стратегии

        Args:
            strategy: Название стратегии
            symbol: Торговая пара
            market_data: Данные о рынке

        Returns:
            Dict: Сигнал стратегии
        """
        # Здесь нужно реализовать логику выбора сигналов для разных стратегий
        if strategy == "scalping":
            return await self._get_scalping_signal(symbol, market_data)
        elif strategy == "trend_following":
            return await self._get_trend_following_signal(symbol, market_data)
        elif strategy == "arbitrage":
            return await self._get_arbitrage_signal(symbol, market_data)
        else:
            return None

    async def _get_scalping_signal(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Получает сигнал для скальпинга"""
        # Скальпинг работает хорошо на высоковолатильных рынках с маленьким спредом
        volatility = market_data.get("volatility", 0)
        spread = market_data.get("spread", 1)

        # Для скальпинга нужна высокая волатильность и низкий спред
        if volatility < 0.01 or spread > 0.002:
            return None

        # Используем ATR для определения направления
        atr = market_data.get("atr", 0)
        last_price = market_data.get("last_price", 0)

        # Простой алгоритм: если цена выше скользящей средней на 0.5*ATR - сигнал на покупку
        # Если цена ниже скользящей средней на 0.5*ATR - сигнал на продажу
        if "1h" in market_data:
            closes = market_data["1h"]["close"]
            ma = sum(closes[-20:]) / min(20, len(closes))

            if last_price > ma + 0.5 * atr:
                return {
                    "action": "buy",
                    "confidence": min(0.6, volatility * 10),
                    "price": last_price,
                }
            elif last_price < ma - 0.5 * atr:
                return {
                    "action": "sell",
                    "confidence": min(0.6, volatility * 10),
                    "price": last_price,
                }

        return None

    async def _get_trend_following_signal(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Получает сигнал для трендовой стратегии"""
        trend = market_data.get("trend")
        trend_strength = market_data.get("trend_strength", 0)

        if not trend or trend_strength < 0.02:
            return None

        # В трендовой стратегии торгуем только по направлению тренда
        if trend == "uptrend":
            return {
                "action": "buy",
                "confidence": min(0.8, trend_strength * 10),
                "price": market_data.get("last_price"),
            }
        elif trend == "downtrend":
            return {
                "action": "sell",
                "confidence": min(0.8, trend_strength * 10),
                "price": market_data.get("last_price"),
            }

        return None

    async def _get_arbitrage_signal(
        self, symbol: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Получает сигнал для арбитражной стратегии"""
        # Упрощенная версия арбитража - смотрим разницу между биржами
        # В полноценной версии нужно сравнивать цены на разных биржах
        return None

    async def _select_best_exchange(self, symbol: str) -> str:
        """
        Выбирает лучшую биржу для выполнения ордера

        Args:
            symbol: Торговая пара

        Returns:
            str: ID биржи
        """
        exchanges = self.config.get("exchanges", ["binance", "bybit", "kucoin"])

        # Проверяем, поддерживается ли символ на каждой бирже
        supported_exchanges = []
        for exchange_id in exchanges:
            if await self.exchange_manager.has_symbol(exchange_id, symbol):
                supported_exchanges.append(exchange_id)

        if not supported_exchanges:
            return exchanges[0] if exchanges else "binance"

        # В будущем здесь можно добавить более сложную логику выбора биржи
        # Например, на основе комиссий, ликвидности и т.д.

        return supported_exchanges[0]

    async def _get_balance(self) -> float:
        """
        Получает баланс с биржи

        Returns:
            float: Доступный баланс
        """
        try:
            # Получаем баланс с основной биржи
            main_exchange = self.config.get("main_exchange", "binance")

            # Получаем информацию о балансе
            balance_info = await self.exchange_manager.get_balance(main_exchange)

            # Получаем баланс конкретной валюты (например, USDT)
            quote_currency = self.config.get("quote_currency", "USDT")

            if quote_currency in balance_info:
                return balance_info[quote_currency].get("free", 0)

            return 0

        except Exception as e:
            logger.error(f"Ошибка при получении баланса: {str(e)}")
            return 0

    def _round_amount(self, amount: float, symbol: str) -> float:
        """
        Округляет количество с учетом минимального шага

        Args:
            amount: Количество
            symbol: Торговая пара

        Returns:
            float: Округленное количество
        """
        try:
            # Здесь нужно получить информацию о минимальном шаге для символа
            # и округлить соответственно
            # В упрощенной версии просто округляем до 4 знаков
            return round(amount, 4)
        except Exception as e:
            logger.error(f"Ошибка при округлении количества: {str(e)}")
            return amount

    async def _check_open_positions(self):
        """Проверяет и обновляет открытые позиции"""
        try:
            # Копируем словарь, чтобы избежать изменения во время итерации
            positions_to_check = dict(self.open_positions)

            for order_id, order in positions_to_check.items():
                try:
                    # Получаем актуальный статус ордера
                    updated_order = await self.order_manager.get_order(order_id)

                    if not updated_order:
                        continue

                    # Если ордер больше не активен, удаляем из открытых позиций
                    if not updated_order.is_active():
                        if order_id in self.open_positions:
                            del self.open_positions[order_id]

                    # Обновляем информацию о позиции
                    self.open_positions[order_id] = updated_order

                except Exception as e:
                    logger.error(f"Ошибка при проверке позиции {order_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Ошибка при проверке открытых позиций: {str(e)}")

    def get_status(self) -> Dict[str, Any]:
        """
        Получает текущий статус бота

        Returns:
            Dict: Информация о состоянии бота
        """
        return {
            "trading_enabled": self.trading_enabled,
            "open_positions": len(self.open_positions),
            "analyzed_symbols": len(self.last_analyzed),
            "last_update": datetime.now().isoformat(),
            "positions": [order.to_dict() for order in self.open_positions.values()],
        }

    def toggle_trading(self, enabled: bool):
        """Включает или выключает торговлю"""
        self.trading_enabled = enabled
        logger.info(f"Торговля {'включена' if enabled else 'выключена'}")
