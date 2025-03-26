"""
Стратегия для торговли на фьючерсах.
Использует особенности фьючерсного рынка, включая маржинальную торговлю и плечо.
"""

import asyncio
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Set

from project.config import get_config
from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error
from project.data.market_data import MarketData
from project.technicals.indicators import Indicators
from project.bots.strategies.base_strategy import BaseStrategy, StrategyStatus

logger = get_logger(__name__)


class FuturesStrategy(BaseStrategy):
    """
    Стратегия для торговли на фьючерсном рынке.
    """

    def __init__(
        self,
        name: str = "FuturesStrategy",
        exchange_id: str = "binance",
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Инициализирует стратегию для фьючерсов.

        Args:
            name: Имя стратегии
            exchange_id: Идентификатор биржи
            symbols: Список символов для торговли
            timeframes: Список таймфреймов для анализа
            config: Конфигурация стратегии
        """
        # Устанавливаем значения по умолчанию
        config = config or {}
        default_config = {
            "leverage": 3,  # размер плеча
            "max_position_size_usd": 1000,  # максимальный размер позиции в USD
            "risk_per_trade_pct": 0.01,  # риск на сделку (1% от баланса)
            "take_profit_atr_multiplier": 2.0,  # множитель ATR для тейк-профита
            "stop_loss_atr_multiplier": 1.0,  # множитель ATR для стоп-лосса
            "atr_period": 14,  # период ATR
            "rsi_period": 14,  # период RSI
            "rsi_overbought": 70,  # порог перекупленности
            "rsi_oversold": 30,  # порог перепроданности
            "ema_fast_period": 12,  # период быстрой EMA
            "ema_slow_period": 26,  # период медленной EMA
            "funding_rate_threshold": 0.001,  # порог ставки финансирования
            "use_funding_rate": True,  # использовать ставку финансирования
            "trend_confirmation": True,  # требовать подтверждения тренда
            "use_order_book": True,  # использовать данные ордербука
            "order_book_depth": 10,  # глубина ордербука для анализа
            "max_open_positions": 5,  # максимальное количество открытых позиций
            "hedge_mode": False,  # режим хеджирования (long и short одновременно)
        }

        # Объединяем с пользовательской конфигурацией
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        # Устанавливаем базовые значения
        # По умолчанию используем фьючерсные контракты USDT-M
        symbols = symbols or [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
            "ADA/USDT:USDT",
            "BNB/USDT:USDT",
        ]
        timeframes = timeframes or ["15m", "1h", "4h"]

        # Используем binance-futures, если не указано иначе
        if exchange_id == "binance":
            exchange_id = "binance-futures"

        super().__init__(name, exchange_id, symbols, timeframes, config)

        # Дополнительные параметры
        self.funding_rates: Dict[str, float] = {}  # symbol -> ставка финансирования
        self.order_book_data: Dict[str, Dict[str, Any]] = (
            {}
        )  # symbol -> данные ордербука
        self.leverage_set: Set[str] = (
            set()
        )  # символы, для которых уже установлено плечо
        self.liquidation_prices: Dict[str, float] = {}  # symbol -> цена ликвидации

        logger.debug(f"Создана стратегия для фьючерсов {self.name}")

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """
        # Обновляем числовые параметры
        for param in [
            "leverage",
            "max_position_size_usd",
            "risk_per_trade_pct",
            "take_profit_atr_multiplier",
            "stop_loss_atr_multiplier",
            "funding_rate_threshold",
        ]:
            if param in config:
                self.strategy_config[param] = float(config[param])

        for param in [
            "atr_period",
            "rsi_period",
            "rsi_overbought",
            "rsi_oversold",
            "ema_fast_period",
            "ema_slow_period",
            "order_book_depth",
            "max_open_positions",
        ]:
            if param in config:
                self.strategy_config[param] = int(config[param])

        # Обновляем булевы параметры
        for param in [
            "use_funding_rate",
            "trend_confirmation",
            "use_order_book",
            "hedge_mode",
        ]:
            if param in config:
                self.strategy_config[param] = bool(config[param])

    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        """
        # Устанавливаем плечо для каждого символа
        for symbol in self.symbols:
            await self._set_leverage(symbol)

        # Загружаем начальные данные
        await self._load_futures_data()

    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        """
        # Нет специфических ресурсов для очистки
        pass

    @async_handle_error
    async def _set_leverage(self, symbol: str) -> bool:
        """
        Устанавливает плечо для торговли.

        Args:
            symbol: Символ для установки плеча

        Returns:
            True в случае успеха, иначе False
        """
        # Если плечо уже установлено, пропускаем
        if symbol in self.leverage_set:
            return True

        try:
            # В реальной реализации здесь должен быть код для установки плеча
            # на бирже через API

            # Имитируем успешную установку плеча
            leverage = self.strategy_config["leverage"]
            logger.info(
                f"Установлено плечо {leverage}x для {symbol} на {self.exchange_id}"
            )

            self.leverage_set.add(symbol)
            return True

        except Exception as e:
            logger.error(f"Ошибка при установке плеча для {symbol}: {str(e)}")
            return False

    @async_handle_error
    async def _load_futures_data(self) -> None:
        """
        Загружает специфические данные для фьючерсного рынка.
        """
        for symbol in self.symbols:
            try:
                # Загружаем ставки финансирования
                if self.strategy_config["use_funding_rate"]:
                    funding_rate = await self._get_funding_rate(symbol)
                    if funding_rate is not None:
                        self.funding_rates[symbol] = funding_rate

                # Загружаем данные ордербука
                if self.strategy_config["use_order_book"]:
                    await self._update_order_book(symbol)

            except Exception as e:
                logger.error(f"Ошибка при загрузке данных для {symbol}: {str(e)}")

    @async_handle_error
    async def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Получает текущую ставку финансирования для символа.

        Args:
            symbol: Символ фьючерсного контракта

        Returns:
            Ставка финансирования или None в случае ошибки
        """
        try:
            # В реальной реализации здесь должен быть код для получения
            # ставки финансирования с биржи

            # Имитируем получение ставки финансирования
            # (случайное значение в диапазоне [-0.001, 0.001])
            funding_rate = (np.random.random() * 0.002) - 0.001

            logger.debug(f"Ставка финансирования для {symbol}: {funding_rate:.6f}")
            return funding_rate

        except Exception as e:
            logger.error(
                f"Ошибка при получении ставки финансирования для {symbol}: {str(e)}"
            )
            return None

    @async_handle_error
    async def _update_order_book(self, symbol: str) -> None:
        """
        Обновляет данные ордербука для символа.

        Args:
            symbol: Символ для обновления ордербука
        """
        try:
            depth = self.strategy_config["order_book_depth"]
            order_book = await self.market_data.get_orderbook(
                self.exchange_id, symbol, limit=depth
            )

            if order_book and "bids" in order_book and "asks" in order_book:
                self.order_book_data[symbol] = order_book
                logger.debug(f"Обновлен ордербук для {symbol}")

        except Exception as e:
            logger.error(f"Ошибка при обновлении ордербука для {symbol}: {str(e)}")

    @async_handle_error
    async def _calculate_liquidation_price(
        self, symbol: str, side: str, entry_price: float, position_size: float
    ) -> float:
        """
        Рассчитывает цену ликвидации для позиции.

        Args:
            symbol: Символ контракта
            side: Сторона позиции (long или short)
            entry_price: Цена входа
            position_size: Размер позиции в контрактах

        Returns:
            Цена ликвидации
        """
        # Получаем настройки контракта
        leverage = self.strategy_config["leverage"]

        # Рассчитываем уровень ликвидации
        # Это упрощенный расчет, в реальности формула зависит от биржи и типа контракта
        if side == "long":
            # Для длинной позиции ликвидация наступает, когда цена падает
            # Формула: entry_price * (1 - 1/leverage) * safety_factor
            liquidation_price = entry_price * (1 - 1 / leverage) * 0.98
        else:  # short
            # Для короткой позиции ликвидация наступает, когда цена растет
            # Формула: entry_price * (1 + 1/leverage) * safety_factor
            liquidation_price = entry_price * (1 + 1 / leverage) * 1.02

        logger.debug(
            f"Рассчитана цена ликвидации для {symbol} {side}: {liquidation_price:.8f}"
        )
        return liquidation_price

    @async_handle_error
    async def _analyze_funding_rate(self, symbol: str) -> str:
        """
        Анализирует ставку финансирования и рекомендует сторону позиции.

        Args:
            symbol: Символ контракта

        Returns:
            Рекомендуемая сторона (long, short, neutral)
        """
        if not self.strategy_config["use_funding_rate"]:
            return "neutral"

        funding_rate = self.funding_rates.get(symbol)
        if funding_rate is None:
            return "neutral"

        threshold = self.strategy_config["funding_rate_threshold"]

        # Отрицательная ставка -> длинная позиция (получаем платеж)
        if funding_rate < -threshold:
            return "long"

        # Положительная ставка -> короткая позиция (получаем платеж)
        elif funding_rate > threshold:
            return "short"

        # Ставка близка к нулю -> нейтрально
        return "neutral"

    @async_handle_error
    async def _analyze_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        Анализирует ордербук и выявляет дисбалансы спроса/предложения.

        Args:
            symbol: Символ контракта

        Returns:
            Словарь с результатами анализа
        """
        if not self.strategy_config["use_order_book"]:
            return {"signal": "neutral", "imbalance": 0.0}

        order_book = self.order_book_data.get(symbol)
        if not order_book or "bids" not in order_book or "asks" not in order_book:
            return {"signal": "neutral", "imbalance": 0.0}

        # Получаем биды и аски
        bids = order_book["bids"]
        asks = order_book["asks"]

        if not bids or not asks:
            return {"signal": "neutral", "imbalance": 0.0}

        # Суммируем объемы первых N уровней
        depth = min(self.strategy_config["order_book_depth"], len(bids), len(asks))
        bid_volume = sum(bid[1] for bid in bids[:depth] if len(bid) > 1)
        ask_volume = sum(ask[1] for ask in asks[:depth] if len(ask) > 1)

        if bid_volume <= 0 or ask_volume <= 0:
            return {"signal": "neutral", "imbalance": 0.0}

        # Рассчитываем отношение объемов
        volume_ratio = bid_volume / ask_volume

        # Рассчитываем имбаланс (от -1 до 1, где 0 - баланс)
        imbalance = (volume_ratio - 1) / (volume_ratio + 1)

        # Определяем сигнал
        signal = "neutral"
        if imbalance > 0.2:
            signal = "long"  # больше покупателей
        elif imbalance < -0.2:
            signal = "short"  # больше продавцов

        return {
            "signal": signal,
            "imbalance": imbalance,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "volume_ratio": volume_ratio,
        }

    @async_handle_error
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы для фьючерсного рынка.

        Returns:
            Словарь с сигналами для каждого символа
        """
        signals = {}

        # Обновляем данные ставок финансирования и ордербука
        await self._load_futures_data()

        for symbol in self.symbols:
            try:
                # Получаем тикер
                ticker = await self.market_data.get_ticker(self.exchange_id, symbol)
                if not ticker:
                    continue

                # Получаем текущую цену
                current_price = ticker.get("last", 0)
                if current_price <= 0:
                    continue

                # Получаем данные OHLCV для основного таймфрейма
                timeframe = (
                    self.timeframes[1]
                    if len(self.timeframes) > 1
                    else self.timeframes[0]
                )
                ohlcv = await self.market_data.get_ohlcv(
                    self.exchange_id, symbol, timeframe, limit=100
                )

                if ohlcv.empty:
                    continue

                # Рассчитываем ATR
                atr = Indicators.average_true_range(
                    ohlcv, self.strategy_config["atr_period"]
                )
                current_atr = atr.iloc[-1] if not atr.empty else 0

                # Рассчитываем RSI
                rsi = Indicators.relative_strength_index(
                    ohlcv, self.strategy_config["rsi_period"]
                )
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50

                # Рассчитываем EMA
                ema_fast = Indicators.exponential_moving_average(
                    ohlcv, self.strategy_config["ema_fast_period"]
                )
                ema_slow = Indicators.exponential_moving_average(
                    ohlcv, self.strategy_config["ema_slow_period"]
                )

                last_ema_fast = ema_fast.iloc[-1] if not ema_fast.empty else 0
                last_ema_slow = ema_slow.iloc[-1] if not ema_slow.empty else 0

                # Анализируем ставку финансирования
                funding_signal = await self._analyze_funding_rate(symbol)

                # Анализируем ордербук
                order_book_analysis = await self._analyze_order_book(symbol)
                order_book_signal = order_book_analysis["signal"]

                # Генерируем технический сигнал
                technical_signal = "neutral"

                # EMA Cross сигнал
                if last_ema_fast > last_ema_slow:
                    technical_signal = "long"
                elif last_ema_fast < last_ema_slow:
                    technical_signal = "short"

                # RSI сигнал (перевешивает EMA в случае экстремальных значений)
                if current_rsi <= self.strategy_config["rsi_oversold"]:
                    technical_signal = "long"
                elif current_rsi >= self.strategy_config["rsi_overbought"]:
                    technical_signal = "short"

                # Комбинируем все сигналы
                final_signal = self._combine_signals(
                    technical_signal, funding_signal, order_book_signal
                )

                # Если у нас уже есть открытая позиция по этому символу, проверяем условия выхода
                if symbol in self.open_positions:
                    position = self.open_positions[symbol]
                    position_side = position["side"]

                    # Если сигнал противоположен стороне позиции, закрываем позицию
                    if (position_side == "long" and final_signal == "short") or (
                        position_side == "short" and final_signal == "long"
                    ):
                        signals[symbol] = {
                            "symbol": symbol,
                            "action": "exit",
                            "price": current_price,
                            "reason": "opposite_signal",
                            "timestamp": time.time(),
                        }
                    # Если сигнал совпадает со стороной позиции или "neutral", ничего не делаем
                    else:
                        continue

                # Если нет открытой позиции, проверяем возможность входа
                elif final_signal in ["long", "short"]:
                    # Проверяем, не превышено ли максимальное количество открытых позиций
                    if (
                        not self.strategy_config["hedge_mode"]
                        and len(self.open_positions)
                        >= self.strategy_config["max_open_positions"]
                    ):
                        continue

                    # Проверяем тренд, если требуется подтверждение
                    if self.strategy_config["trend_confirmation"]:
                        # Получаем данные для старшего таймфрейма
                        higher_tf = self.timeframes[-1]
                        higher_ohlcv = await self.market_data.get_ohlcv(
                            self.exchange_id, symbol, higher_tf, limit=50
                        )

                        if not higher_ohlcv.empty:
                            # Рассчитываем EMA для старшего таймфрейма
                            higher_ema_fast = Indicators.exponential_moving_average(
                                higher_ohlcv, self.strategy_config["ema_fast_period"]
                            )
                            higher_ema_slow = Indicators.exponential_moving_average(
                                higher_ohlcv, self.strategy_config["ema_slow_period"]
                            )

                            higher_fast = (
                                higher_ema_fast.iloc[-1]
                                if not higher_ema_fast.empty
                                else 0
                            )
                            higher_slow = (
                                higher_ema_slow.iloc[-1]
                                if not higher_ema_slow.empty
                                else 0
                            )

                            # Определяем тренд старшего таймфрейма
                            higher_trend = "neutral"
                            if higher_fast > higher_slow:
                                higher_trend = "long"
                            elif higher_fast < higher_slow:
                                higher_trend = "short"

                            # Если тренд не совпадает с сигналом, пропускаем
                            if (
                                higher_trend != final_signal
                                and higher_trend != "neutral"
                            ):
                                continue

                    # Рассчитываем размер позиции
                    position_size = self._calculate_position_size(
                        symbol, final_signal, current_price, current_atr
                    )

                    # Рассчитываем уровни тейк-профита и стоп-лосса
                    if final_signal == "long":
                        take_profit = current_price + (
                            current_atr
                            * self.strategy_config["take_profit_atr_multiplier"]
                        )
                        stop_loss = current_price - (
                            current_atr
                            * self.strategy_config["stop_loss_atr_multiplier"]
                        )
                    else:  # short
                        take_profit = current_price - (
                            current_atr
                            * self.strategy_config["take_profit_atr_multiplier"]
                        )
                        stop_loss = current_price + (
                            current_atr
                            * self.strategy_config["stop_loss_atr_multiplier"]
                        )

                    # Рассчитываем цену ликвидации
                    liquidation_price = await self._calculate_liquidation_price(
                        symbol, final_signal, current_price, position_size
                    )

                    # Проверяем, что стоп-лосс не близко к ликвидации
                    if final_signal == "long" and stop_loss <= liquidation_price * 1.05:
                        # Корректируем стоп-лосс
                        stop_loss = liquidation_price * 1.1
                    elif (
                        final_signal == "short"
                        and stop_loss >= liquidation_price * 0.95
                    ):
                        # Корректируем стоп-лосс
                        stop_loss = liquidation_price * 0.9

                    # Формируем сигнал
                    signals[symbol] = {
                        "symbol": symbol,
                        "action": final_signal,
                        "price": current_price,
                        "position_size": position_size,
                        "take_profit": take_profit,
                        "stop_loss": stop_loss,
                        "liquidation_price": liquidation_price,
                        "leverage": self.strategy_config["leverage"],
                        "technical_signal": technical_signal,
                        "funding_signal": funding_signal,
                        "order_book_signal": order_book_signal,
                        "atr": current_atr,
                        "rsi": current_rsi,
                        "timestamp": time.time(),
                    }

            except Exception as e:
                logger.error(f"Ошибка при генерации сигналов для {symbol}: {str(e)}")

        return signals

    def _combine_signals(
        self, technical_signal: str, funding_signal: str, order_book_signal: str
    ) -> str:
        """
        Комбинирует различные сигналы в один финальный.

        Args:
            technical_signal: Сигнал от технического анализа
            funding_signal: Сигнал от ставки финансирования
            order_book_signal: Сигнал от ордербука

        Returns:
            Финальный сигнал (long, short, neutral)
        """
        # Подсчитываем голоса для каждого сигнала
        long_votes = 0
        short_votes = 0

        # Технический сигнал имеет наибольший вес
        if technical_signal == "long":
            long_votes += 3
        elif technical_signal == "short":
            short_votes += 3

        # Сигнал от ставки финансирования
        if funding_signal == "long":
            long_votes += 1
        elif funding_signal == "short":
            short_votes += 1

        # Сигнал от ордербука
        if order_book_signal == "long":
            long_votes += 2
        elif order_book_signal == "short":
            short_votes += 2

        # Определяем итоговый сигнал
        if long_votes > short_votes and long_votes >= 3:
            return "long"
        elif short_votes > long_votes and short_votes >= 3:
            return "short"
        else:
            return "neutral"

    def _calculate_position_size(
        self, symbol: str, side: str, current_price: float, atr: float
    ) -> float:
        """
        Рассчитывает размер позиции на основе риска.

        Args:
            symbol: Символ контракта
            side: Сторона позиции
            current_price: Текущая цена
            atr: Значение ATR

        Returns:
            Размер позиции в единицах контракта
        """
        # Получаем параметры для расчета
        account_balance = self.strategy_config.get("account_balance", 10000.0)
        risk_pct = self.strategy_config["risk_per_trade_pct"]
        max_position_size_usd = self.strategy_config["max_position_size_usd"]
        leverage = self.strategy_config["leverage"]

        # Рассчитываем размер риска в USD
        risk_amount = account_balance * risk_pct

        # Рассчитываем стоп-лосс
        stop_loss_multiplier = self.strategy_config["stop_loss_atr_multiplier"]
        if side == "long":
            stop_loss_price = current_price - (atr * stop_loss_multiplier)
            price_risk = current_price - stop_loss_price
        else:  # short
            stop_loss_price = current_price + (atr * stop_loss_multiplier)
            price_risk = stop_loss_price - current_price

        # Рассчитываем размер позиции с учетом плеча
        if price_risk <= 0:
            # Используем процент от цены, если ATR слишком мал
            price_risk = current_price * 0.01

        position_size_usd = (risk_amount / price_risk) * leverage

        # Ограничиваем максимальный размер позиции
        position_size_usd = min(position_size_usd, max_position_size_usd)

        # Конвертируем размер позиции из USD в количество контрактов
        position_size = position_size_usd / current_price

        # Округляем до 5 знаков после запятой
        position_size = round(position_size, 5)

        logger.debug(
            f"Рассчитан размер позиции для {symbol} {side}: {position_size} "
            f"(USD: {position_size_usd:.2f}, риск: {risk_amount:.2f}, "
            f"стоп-лосс: {stop_loss_price:.8f})"
        )

        return position_size
