"""
Стратегия, основанная на волатильности.
Торгует на основе индикаторов волатильности и технических уровней.
"""

import time
from typing import Any, Dict, List, Optional

import pandas as pd
from project.bots.strategies.base_strategy import BaseStrategy
from project.bots.strategies.modifications import StrategyModifications
from project.technicals.indicators import Indicators
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VolatilityStrategy(BaseStrategy):
    """
    Стратегия, основанная на волатильности.
    """

    def __init__(
        self,
        name: str = "VolatilityStrategy",
        exchange_id: str = "binance",
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Инициализирует стратегию волатильности.

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
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "min_volatility": 0.01,  # Минимальная волатильность для торговли (1%)
            "max_volatility": 0.05,  # Максимальная волатильность для торговли (5%)
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.01,
            "volatility_ranking_enabled": True,  # Ранжирование символов по волатильности
            "volatility_ranking_period": 24,  # Период для ранжирования
            "volatility_ranking_top": 5,  # Количество верхних символов для торговли
            "volatility_breakout_enabled": True,  # Торговля на пробоях волатильности
            "volatility_contraction_enabled": True,  # Торговля на сжатии волатильности
        }

        # Объединяем с пользовательской конфигурацией
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        # Устанавливаем базовые значения
        symbols = symbols or ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        timeframes = timeframes or ["15m", "1h", "4h"]

        super().__init__(name, exchange_id, symbols, timeframes, config)

        # Дополнительные параметры
        self.volatility_data: Dict[str, Dict[str, Any]] = (
            {}
        )  # symbol -> данные волатильности
        self.ranked_symbols: List[str] = []  # символы, отсортированные по волатильности
        self.highest_prices: Dict[str, float] = (
            {}
        )  # symbol -> максимальная цена для трейлинг-стопа
        self.lowest_prices: Dict[str, float] = (
            {}
        )  # symbol -> минимальная цена для трейлинг-стопа

        logger.debug("Создана стратегия волатильности {self.name}" %)

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """
        # Обновляем параметры индикаторов
        if "atr_period" in config:
            self.strategy_config["atr_period"] = int(config["atr_period"])

        if "atr_multiplier" in config:
            self.strategy_config["atr_multiplier"] = float(config["atr_multiplier"])

        if "bollinger_period" in config:
            self.strategy_config["bollinger_period"] = int(config["bollinger_period"])

        if "bollinger_std" in config:
            self.strategy_config["bollinger_std"] = float(config["bollinger_std"])

        if "min_volatility" in config:
            self.strategy_config["min_volatility"] = float(config["min_volatility"])

        if "max_volatility" in config:
            self.strategy_config["max_volatility"] = float(config["max_volatility"])

        if "use_trailing_stop" in config:
            self.strategy_config["use_trailing_stop"] = bool(
                config["use_trailing_stop"]
            )

        if "trailing_stop_pct" in config:
            self.strategy_config["trailing_stop_pct"] = float(
                config["trailing_stop_pct"]
            )

        if "volatility_ranking_enabled" in config:
            self.strategy_config["volatility_ranking_enabled"] = bool(
                config["volatility_ranking_enabled"]
            )

        if "volatility_ranking_period" in config:
            self.strategy_config["volatility_ranking_period"] = int(
                config["volatility_ranking_period"]
            )

        if "volatility_ranking_top" in config:
            self.strategy_config["volatility_ranking_top"] = int(
                config["volatility_ranking_top"]
            )

        if "volatility_breakout_enabled" in config:
            self.strategy_config["volatility_breakout_enabled"] = bool(
                config["volatility_breakout_enabled"]
            )

        if "volatility_contraction_enabled" in config:
            self.strategy_config["volatility_contraction_enabled"] = bool(
                config["volatility_contraction_enabled"]
            )

    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        """
        # Инициализируем данные волатильности
        for symbol in self.symbols:
            self.volatility_data[symbol] = {
                "current_atr": 0.0,
                "relative_volatility": 0.0,
                "bollinger_width": 0.0,
                "breakout_level": 0.0,
                "contraction_level": 0.0,
                "last_volatility_direction": "neutral",
            }

            self.highest_prices[symbol] = 0.0
            self.lowest_prices[symbol] = float("inf")

        # Рассчитываем волатильность и ранжируем символы
        await self._calculate_volatility()

        if self.strategy_config["volatility_ranking_enabled"]:
            self._rank_symbols_by_volatility()

    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        """
        # Нет специфических ресурсов для очистки

    @async_handle_error
    async def _calculate_volatility(self) -> None:
        """
        Рассчитывает волатильность для всех символов.
        """
        for symbol in self.symbols:
            try:
                # Определяем основной таймфрейм для расчета волатильности
                timeframe = self.timeframes[-1]  # Используем самый старший таймфрейм

                # Получаем данные OHLCV
                ohlcv = await self.market_data.get_ohlcv(
                    self.exchange_id, symbol, timeframe, limit=100
                )

                if ohlcv.empty:
                    logger.warning(
                        f"Нет данных OHLCV для {symbol} на таймфрейме {timeframe}"
                    )
                    continue

                # Рассчитываем ATR
                atr = Indicators.average_true_range(
                    ohlcv, self.strategy_config["atr_period"]
                )
                current_atr = atr.iloc[-1] if not atr.empty else 0

                # Рассчитываем среднюю цену
                average_price = ohlcv["close"].mean()
                current_price = ohlcv["close"].iloc[-1]

                # Рассчитываем относительную волатильность (ATR/Цена)
                relative_volatility = (
                    current_atr / average_price if average_price > 0 else 0
                )

                # Рассчитываем полосы Боллинджера
                bb_data = Indicators.bollinger_bands(
                    ohlcv,
                    self.strategy_config["bollinger_period"],
                    self.strategy_config["bollinger_std"],
                )

                upper = bb_data["upper"].iloc[-1] if not bb_data["upper"].empty else 0
                lower = bb_data["lower"].iloc[-1] if not bb_data["lower"].empty else 0

                # Рассчитываем ширину полос Боллинджера
                bollinger_width = (
                    (upper - lower) / current_price if current_price > 0 else 0
                )

                # Рассчитываем уровень пробоя (ATR * множитель)
                breakout_level = current_atr * self.strategy_config["atr_multiplier"]

                # Рассчитываем уровень сжатия (минимальная ширина полос Боллинджера за последние N периодов)
                bollinger_width_history = []
                for i in range(min(20, len(ohlcv))):
                    idx = -i - 1
                    if idx < -len(ohlcv):
                        break

                    bb_upper = (
                        bb_data["upper"].iloc[idx] if not bb_data["upper"].empty else 0
                    )
                    bb_lower = (
                        bb_data["lower"].iloc[idx] if not bb_data["lower"].empty else 0
                    )
                    bb_price = ohlcv["close"].iloc[idx]

                    if bb_price > 0:
                        bb_width = (bb_upper - bb_lower) / bb_price
                        bollinger_width_history.append(bb_width)

                contraction_level = (
                    min(bollinger_width_history) if bollinger_width_history else 0
                )

                # Определяем направление изменения волатильности
                prev_volatility = self.volatility_data[symbol]["relative_volatility"]

                if relative_volatility > prev_volatility * 1.1:
                    volatility_direction = "increasing"
                elif relative_volatility < prev_volatility * 0.9:
                    volatility_direction = "decreasing"
                else:
                    volatility_direction = "neutral"

                # Обновляем данные волатильности
                self.volatility_data[symbol] = {
                    "current_atr": current_atr,
                    "relative_volatility": relative_volatility,
                    "bollinger_width": bollinger_width,
                    "breakout_level": breakout_level,
                    "contraction_level": contraction_level,
                    "last_volatility_direction": volatility_direction,
                }

                # Обновляем максимальные и минимальные цены для трейлинг-стопа
                if symbol in self.open_positions:
                    position = self.open_positions[symbol]

                    if position["side"] == "long":
                        self.highest_prices[symbol] = max(
                            self.highest_prices[symbol], current_price
                        )
                    else:  # short
                        self.lowest_prices[symbol] = min(
                            self.lowest_prices[symbol], current_price
                        )

            except Exception as e:
                logger.error("Ошибка при расчете волатильности для {symbol}: {str(e)}" %)

    def _rank_symbols_by_volatility(self) -> None:
        """
        Ранжирует символы по волатильности.
        """
        try:
            # Создаем список пар (символ, волатильность)
            volatility_ranking = []

            for symbol, data in self.volatility_data.items():
                volatility_ranking.append((symbol, data["relative_volatility"]))

            # Сортируем по убыванию волатильности
            volatility_ranking.sort(key=lambda x: x[1], reverse=True)

            # Выбираем top_n символов
            top_n = self.strategy_config["volatility_ranking_top"]
            top_symbols = [item[0] for item in volatility_ranking[:top_n]]

            # Обновляем список ранжированных символов
            self.ranked_symbols = top_symbols
            logger.debug(
                f"Символы, ранжированные по волатильности: {self.ranked_symbols}"
            )

        except Exception as e:
            logger.error("Ошибка при ранжировании символов по волатильности: {str(e)}" %)

    @async_handle_error
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы на основе волатильности.

        Returns:
            Словарь с сигналами для каждого символа
        """
        # Рассчитываем волатильность
        await self._calculate_volatility()

        # Ранжируем символы по волатильности
        if self.strategy_config["volatility_ranking_enabled"]:
            self._rank_symbols_by_volatility()

        signals = {}

        # Определяем символы для торговли
        trading_symbols = (
            self.ranked_symbols
            if self.strategy_config["volatility_ranking_enabled"]
            else self.symbols
        )

        for symbol in trading_symbols:
            try:
                # Получаем тикер
                ticker = await self.market_data.get_ticker(self.exchange_id, symbol)
                if not ticker:
                    continue

                # Получаем текущую цену
                current_price = ticker.get("last", 0)
                if current_price <= 0:
                    continue

                # Получаем данные волатильности
                volatility_data = self.volatility_data.get(symbol, {})
                if not volatility_data:
                    continue

                # Проверяем, находится ли волатильность в допустимом диапазоне
                relative_volatility = volatility_data["relative_volatility"]

                if relative_volatility < self.strategy_config["min_volatility"]:
                    logger.debug(
                        f"Волатильность {symbol} слишком низкая: {relative_volatility}"
                    )
                    continue

                if relative_volatility > self.strategy_config["max_volatility"]:
                    logger.debug(
                        f"Волатильность {symbol} слишком высокая: {relative_volatility}"
                    )
                    continue

                # Получаем данные OHLCV для основного таймфрейма
                timeframe = self.timeframes[-1]
                ohlcv = await self.market_data.get_ohlcv(
                    self.exchange_id, symbol, timeframe, limit=100
                )

                if ohlcv.empty:
                    continue

                # Генерируем сигналы в зависимости от настроек
                signal = None

                # Сигналы на основе пробоев волатильности
                if self.strategy_config["volatility_breakout_enabled"]:
                    signal = self._generate_breakout_signal(
                        symbol, current_price, ohlcv, volatility_data
                    )

                    if signal and signal["action"] != "hold":
                        signals[symbol] = signal
                        continue

                # Сигналы на основе сжатия волатильности
                if self.strategy_config["volatility_contraction_enabled"]:
                    signal = self._generate_contraction_signal(
                        symbol, current_price, ohlcv, volatility_data
                    )

                    if signal and signal["action"] != "hold":
                        signals[symbol] = signal
                        continue
                # Проверяем, нужно ли обновить стоп-лосс
                if self.strategy_config["use_trailing_stop"]:
                    await self._update_trailing_stops(symbol, current_price)

            except Exception as e:
                logger.error("Ошибка при генерации сигналов для {symbol}: {str(e)}" %)

        return signals

    def _generate_breakout_signal(
        self,
        symbol: str,
        current_price: float,
        ohlcv: pd.DataFrame,
        volatility_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Генерирует сигнал на основе пробоя волатильности.

        Args:
            symbol: Символ для торговли
            current_price: Текущая цена
            ohlcv: DataFrame с данными OHLCV
            volatility_data: Данные волатильности

        Returns:
            Словарь с сигналом или None
        """
        # Получаем уровень пробоя
        breakout_level = volatility_data["breakout_level"]

        if breakout_level <= 0:
            return None

        # Рассчитываем опорную цену (предыдущее закрытие)
        reference_price = ohlcv["close"].iloc[-2] if len(ohlcv) > 1 else current_price

        # Проверяем пробой вверх
        if current_price > reference_price + breakout_level:
            # Проверяем, что цена не слишком далеко от опорной
            if current_price < reference_price * 1.05:  # Не более 5% от опорной
                return {
                    "symbol": symbol,
                    "action": "buy",
                    "price": current_price,
                    "reason": "volatility_breakout_up",
                    "reference_price": reference_price,
                    "breakout_level": breakout_level,
                    "timestamp": time.time(),
                }

        # Проверяем пробой вниз
        if current_price < reference_price - breakout_level:
            # Проверяем, что цена не слишком далеко от опорной
            if current_price > reference_price * 0.95:  # Не менее 95% от опорной
                return {
                    "symbol": symbol,
                    "action": "sell",
                    "price": current_price,
                    "reason": "volatility_breakout_down",
                    "reference_price": reference_price,
                    "breakout_level": breakout_level,
                    "timestamp": time.time(),
                }

        return None

    def _generate_contraction_signal(
        self,
        symbol: str,
        current_price: float,
        ohlcv: pd.DataFrame,
        volatility_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Генерирует сигнал на основе сжатия волатильности.

        Args:
            symbol: Символ для торговли
            current_price: Текущая цена
            ohlcv: DataFrame с данными OHLCV
            volatility_data: Данные волатильности

        Returns:
            Словарь с сигналом или None
        """
        # Получаем ширину полос Боллинджера и уровень сжатия
        bollinger_width = volatility_data["bollinger_width"]
        contraction_level = volatility_data["contraction_level"]
        last_volatility_direction = volatility_data["last_volatility_direction"]

        if bollinger_width <= 0 or contraction_level <= 0:
            return None

        # Проверяем сжатие волатильности
        if bollinger_width < contraction_level * 1.2:  # Ширина близка к минимальной
            # Ищем направление пробоя
            if last_volatility_direction == "increasing":
                # Определяем направление тренда по EMA
                ema_short = Indicators.exponential_moving_average(ohlcv, 9)
                ema_long = Indicators.exponential_moving_average(ohlcv, 21)

                if len(ema_short) > 0 and len(ema_long) > 0:
                    last_short = ema_short.iloc[-1]
                    last_long = ema_long.iloc[-1]

                    if last_short > last_long:
                        # Восходящий тренд
                        return {
                            "symbol": symbol,
                            "action": "buy",
                            "price": current_price,
                            "reason": "volatility_contraction",
                            "bollinger_width": bollinger_width,
                            "contraction_level": contraction_level,
                            "timestamp": time.time(),
                        }
                    elif last_short < last_long:
                        # Нисходящий тренд
                        return {
                            "symbol": symbol,
                            "action": "sell",
                            "price": current_price,
                            "reason": "volatility_contraction",
                            "bollinger_width": bollinger_width,
                            "contraction_level": contraction_level,
                            "timestamp": time.time(),
                        }

        return None

    @async_handle_error
    async def _update_trailing_stops(self, symbol: str, current_price: float) -> None:
        """
        Обновляет трейлинг-стопы для открытых позиций.

        Args:
            symbol: Символ для торговли
            current_price: Текущая цена
        """
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        side = position["side"]
        entry_price = position["entry_price"]
        current_stop = position.get("stop_loss")

        if current_stop is None:
            return

        # Обновляем максимальные/минимальные цены
        if side == "long":
            self.highest_prices[symbol] = max(
                self.highest_prices[symbol], current_price
            )
            highest_price = self.highest_prices[symbol]

            # Рассчитываем новый трейлинг-стоп
            new_stop = StrategyModifications.add_trailing_stop(
                current_price=current_price,
                entry_price=entry_price,
                highest_price=highest_price,
                side=side,
                initial_stop_pct=self.stop_loss_pct,
                trailing_pct=self.strategy_config["trailing_stop_pct"],
            )

            # Обновляем стоп-лосс, только если он выше текущего
            if new_stop > current_stop:
                self.open_positions[symbol]["stop_loss"] = new_stop
                logger.debug(
                    f"Обновлен трейлинг-стоп для {symbol} (long): {current_stop} -> {new_stop}"
                )

        elif side == "short":
            self.lowest_prices[symbol] = min(self.lowest_prices[symbol], current_price)
            lowest_price = self.lowest_prices[symbol]

            # Рассчитываем новый трейлинг-стоп
            new_stop = StrategyModifications.add_trailing_stop(
                current_price=current_price,
                entry_price=entry_price,
                highest_price=lowest_price,
                side=side,
                initial_stop_pct=self.stop_loss_pct,
                trailing_pct=self.strategy_config["trailing_stop_pct"],
            )

            # Обновляем стоп-лосс, только если он ниже текущего
            if new_stop < current_stop:
                self.open_positions[symbol]["stop_loss"] = new_stop
                logger.debug(
                    f"Обновлен трейлинг-стоп для {symbol} (short): {current_stop} -> {new_stop}"
                )
