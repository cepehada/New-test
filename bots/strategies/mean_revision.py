"""
Стратегия возврата к среднему (Mean Reversion).
Торгует на отклонениях цены от среднего значения, ожидая возврата.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from project.bots.strategies.base_strategy import BaseStrategy
from project.technicals.indicators import Indicators
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Стратегия возврата к среднему для торговли на отклонениях цены.
    """

    def __init__(
        self,
        name: str = "MeanReversionStrategy",
        exchange_id: str = "binance",
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Инициализирует стратегию возврата к среднему.

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
            "ma_type": "sma",  # тип скользящей средней (sma, ema, wma)
            "ma_period": 20,  # период скользящей средней
            "bb_period": 20,  # период для полос Боллинджера
            "bb_std": 2.0,  # число стандартных отклонений для полос Боллинджера
            "rsi_period": 14,  # период RSI
            "rsi_overbought": 70,  # уровень перекупленности
            "rsi_oversold": 30,  # уровень перепроданности
            "deviation_threshold": 0.02,  # порог отклонения от среднего (2%)
            "exit_threshold": 0.005,  # порог для выхода (0.5%)
            "stop_loss_pct": 0.03,  # стоп-лосс (3%)
            "max_positions": 5,  # максимальное количество позиций
            "position_size_pct": 0.1,  # размер позиции (10% от доступного капитала)
            "use_bollinger": True,  # использовать полосы Боллинджера
            "use_rsi": True,  # использовать RSI
            "use_volume_filter": True,  # фильтровать по объему
            "min_volume_multiplier": 1.5,  # множитель минимального объема
            "volume_lookback": 10,  # период для расчета среднего объема
            "use_correlation_filter": False,  # фильтровать по корреляции
            "correlation_threshold": 0.8,  # порог корреляции
            "correlation_lookback": 50,  # период для расчета корреляции
        }

        # Объединяем с пользовательской конфигурацией
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        # Устанавливаем базовые значения
        symbols = symbols or [
            "BTC/USDT",
            "ETH/USDT",
            "XRP/USDT",
            "SOL/USDT",
            "LINK/USDT",
        ]
        timeframes = timeframes or ["15m", "1h", "4h"]

        super().__init__(name, exchange_id, symbols, timeframes, config)

        # Дополнительные параметры
        self.deviation_data: Dict[str, Dict[str, float]] = (
            {}
        )  # symbol -> timeframe -> отклонение от среднего
        self.recent_highs: Dict[str, float] = {}  # symbol -> недавний максимум
        self.recent_lows: Dict[str, float] = {}  # symbol -> недавний минимум
        self.volume_data: Dict[str, List[float]] = {}  # symbol -> история объемов

        logger.debug(f"Создана стратегия возврата к среднему {self.name}" )

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """
        # Обновляем строковые параметры
        if "ma_type" in config:
            self.strategy_config["ma_type"] = config["ma_type"]

        # Обновляем числовые параметры
        for param in [
            "ma_period",
            "bb_period",
            "rsi_period",
            "rsi_overbought",
            "rsi_oversold",
            "max_positions",
            "volume_lookback",
            "correlation_lookback",
        ]:
            if param in config:
                self.strategy_config[param] = int(config[param])

        for param in [
            "bb_std",
            "deviation_threshold",
            "exit_threshold",
            "stop_loss_pct",
            "position_size_pct",
            "min_volume_multiplier",
            "correlation_threshold",
        ]:
            if param in config:
                self.strategy_config[param] = float(config[param])

        # Обновляем булевы параметры
        for param in [
            "use_bollinger",
            "use_rsi",
            "use_volume_filter",
            "use_correlation_filter",
        ]:
            if param in config:
                self.strategy_config[param] = bool(config[param])

    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        """
        # Инициализируем структуры данных
        for symbol in self.symbols:
            self.deviation_data[symbol] = {}
            self.recent_highs[symbol] = 0.0
            self.recent_lows[symbol] = float("inf")
            self.volume_data[symbol] = []

        # Загружаем начальные данные
        await self._load_historical_data()

    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        """
        # Нет специфических ресурсов для очистки

    @async_handle_error
    async def _load_historical_data(self) -> None:
        """
        Загружает исторические данные для расчета индикаторов.
        """
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    # Загружаем данные OHLCV
                    ohlcv = await self.market_data.get_ohlcv(
                        self.exchange_id, symbol, timeframe, limit=100
                    )

                    if ohlcv.empty:
                        logger.warning(
                            f"Нет данных OHLCV для {symbol} на таймфрейме {timeframe}"
                        )
                        continue

                    # Сохраняем недавние максимумы и минимумы
                    recent_high = ohlcv["high"].max()
                    recent_low = ohlcv["low"].min()

                    self.recent_highs[symbol] = max(
                        self.recent_highs[symbol], recent_high
                    )
                    self.recent_lows[symbol] = min(self.recent_lows[symbol], recent_low)

                    # Сохраняем историю объемов
                    if (
                        timeframe == self.timeframes[0]
                    ):  # используем только основной таймфрейм
                        self.volume_data[symbol] = ohlcv["volume"].tolist()

                    # Рассчитываем отклонение от среднего
                    self.deviation_data[symbol][timeframe] = self._calculate_deviation(
                        ohlcv
                    )

                    logger.debug(
                        f"Загружены данные для {symbol} на таймфрейме {timeframe}: "
                        f"отклонение={self.deviation_data[symbol][timeframe]:.2%}"
                    )

                except Exception as e:
                    logger.error(
                        f"Ошибка при загрузке данных для {symbol} на {timeframe}: {str(e)}"
                    )

    def _calculate_deviation(self, ohlcv: pd.DataFrame) -> float:
        """
        Рассчитывает отклонение текущей цены от скользящей средней.

        Args:
            ohlcv: DataFrame с данными OHLCV

        Returns:
            Отклонение от среднего (в процентах)
        """
        if ohlcv.empty:
            return 0.0

        # Получаем текущую цену (последнее закрытие)
        current_price = ohlcv["close"].iloc[-1]

        # Рассчитываем скользящую среднюю
        ma_period = self.strategy_config["ma_period"]
        ma_type = self.strategy_config["ma_type"]

        if ma_type == "sma":
            ma = ohlcv["close"].rolling(window=ma_period).mean()
        elif ma_type == "ema":
            ma = ohlcv["close"].ewm(span=ma_period, adjust=False).mean()
        elif ma_type == "wma":
            weights = np.arange(1, ma_period + 1)
            ma = (
                ohlcv["close"]
                .rolling(window=ma_period)
                .apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
            )
        else:
            # По умолчанию используем SMA
            ma = ohlcv["close"].rolling(window=ma_period).mean()

        # Получаем последнее значение скользящей средней
        last_ma = ma.iloc[-1]

        # Рассчитываем отклонение
        if last_ma > 0:
            deviation = (current_price / last_ma) - 1
        else:
            deviation = 0.0

        return deviation

    @async_handle_error
    async def _analyze_volume(self, symbol: str) -> bool:
        """
        Анализирует объем и проверяет, достаточен ли он для торговли.

        Args:
            symbol: Символ для анализа

        Returns:
            True, если объем достаточен, иначе False
        """
        if not self.strategy_config["use_volume_filter"]:
            return True

        volumes = self.volume_data.get(symbol, [])

        if not volumes:
            return False

        # Получаем текущий объем
        current_volume = volumes[-1] if volumes else 0

        # Рассчитываем средний объем за предыдущие периоды
        lookback = min(self.strategy_config["volume_lookback"], len(volumes) - 1)
        if lookback <= 0:
            return False

        avg_volume = sum(volumes[-lookback - 1: -1]) / lookback

        # Проверяем, что текущий объем выше среднего
        min_volume = avg_volume * self.strategy_config["min_volume_multiplier"]

        return current_volume >= min_volume

    @async_handle_error
    async def _check_correlation(self, symbol: str) -> bool:
        """
        Проверяет корреляцию с другими активами для диверсификации.

        Args:
            symbol: Символ для проверки

        Returns:
            True, если корреляция приемлема, иначе False
        """
        if not self.strategy_config["use_correlation_filter"]:
            return True

        # Если у нас нет открытых позиций, корреляция не важна
        if not self.open_positions:
            return True

        # Получаем данные для расчета корреляции
        timeframe = self.timeframes[0]
        symbol_ohlcv = await self.market_data.get_ohlcv(
            self.exchange_id,
            symbol,
            timeframe,
            limit=self.strategy_config["correlation_lookback"],
        )

        if symbol_ohlcv.empty:
            return False

        # Проверяем корреляцию с каждой открытой позицией
        for position_symbol in self.open_positions:
            if position_symbol == symbol:
                continue

            position_ohlcv = await self.market_data.get_ohlcv(
                self.exchange_id,
                position_symbol,
                timeframe,
                limit=self.strategy_config["correlation_lookback"],
            )

            if position_ohlcv.empty:
                continue

            # Находим общий период
            min_length = min(len(symbol_ohlcv), len(position_ohlcv))
            if min_length < 10:  # нужно хотя бы 10 точек для корреляции
                continue

            # Рассчитываем корреляцию
            symbol_returns = symbol_ohlcv["close"].pct_change().iloc[-min_length:]
            position_returns = position_ohlcv["close"].pct_change().iloc[-min_length:]

            correlation = symbol_returns.corr(position_returns)

            # Проверяем, что корреляция не выше порога
            if abs(correlation) > self.strategy_config["correlation_threshold"]:
                logger.debug(
                    f"Слишком высокая корреляция между {symbol} и {position_symbol}: {
                        correlation:.2f}")
                return False

        return True

    @async_handle_error
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы для стратегии возврата к среднему.

        Returns:
            Словарь с сигналами для каждого символа
        """
        signals = {}

        # Обновляем исторические данные
        await self._load_historical_data()

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

                # Проверяем объем
                volume_ok = await self._analyze_volume(symbol)
                if not volume_ok:
                    logger.debug(f"Недостаточный объем для {symbol}" )
                    continue

                # Проверяем корреляцию
                correlation_ok = await self._check_correlation(symbol)
                if not correlation_ok:
                    logger.debug(
                        f"Высокая корреляция с существующими позициями для {symbol}"
                    )
                    continue

                # Получаем данные OHLCV для основного таймфрейма
                timeframe = self.timeframes[0]
                ohlcv = await self.market_data.get_ohlcv(
                    self.exchange_id, symbol, timeframe, limit=100
                )

                if ohlcv.empty:
                    continue

                # Рассчитываем индикаторы
                # - Отклонение от скользящей средней
                deviation = self.deviation_data.get(symbol, {}).get(timeframe, 0.0)

                # - Полосы Боллинджера
                bb_signal = None
                if self.strategy_config["use_bollinger"]:
                    bb_data = Indicators.bollinger_bands(
                        ohlcv,
                        self.strategy_config["bb_period"],
                        self.strategy_config["bb_std"],
                    )

                    if not bb_data["upper"].empty and not bb_data["lower"].empty:
                        upper = bb_data["upper"].iloc[-1]
                        lower = bb_data["lower"].iloc[-1]

                        if current_price <= lower:
                            bb_signal = "buy"
                        elif current_price >= upper:
                            bb_signal = "sell"

                # - RSI
                rsi_signal = None
                if self.strategy_config["use_rsi"]:
                    rsi = Indicators.relative_strength_index(
                        ohlcv, self.strategy_config["rsi_period"]
                    )

                    if not rsi.empty:
                        current_rsi = rsi.iloc[-1]

                        if current_rsi <= self.strategy_config["rsi_oversold"]:
                            rsi_signal = "buy"
                        elif current_rsi >= self.strategy_config["rsi_overbought"]:
                            rsi_signal = "sell"

                # Генерируем сигнал
                signal = self._generate_mean_reversion_signal(
                    symbol, current_price, deviation, bb_signal, rsi_signal
                )

                if signal:
                    signals[symbol] = signal

            except Exception as e:
                logger.error(f"Ошибка при генерации сигналов для {symbol}: {str(e)}" )

        return signals

    def _generate_mean_reversion_signal(
        self,
        symbol: str,
        current_price: float,
        deviation: float,
        bb_signal: Optional[str],
        rsi_signal: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Генерирует сигнал для стратегии возврата к среднему.

        Args:
            symbol: Символ для торговли
            current_price: Текущая цена
            deviation: Отклонение от среднего
            bb_signal: Сигнал от полос Боллинджера
            rsi_signal: Сигнал от RSI

        Returns:
            Словарь с сигналом или None
        """
        # Проверяем, есть ли уже открытая позиция по этому символу
        if symbol in self.open_positions:
            position = self.open_positions[symbol]
            entry_price = position["entry_price"]
            side = position["side"]

            # Рассчитываем текущее отклонение от цены входа
            price_change = (current_price / entry_price - 1) * (
                1 if side == "long" else -1
            )

            # Проверяем условия выхода
            exit_threshold = self.strategy_config["exit_threshold"]

            if (
                side == "long"
                and ((price_change >= exit_threshold) or (deviation >= 0))
            ) or (
                side == "short"
                and ((price_change >= exit_threshold) or (deviation <= 0))
            ):
                # Формируем сигнал выхода
                return {
                    "symbol": symbol,
                    "action": "exit",
                    "price": current_price,
                    "reason": (
                        "target_reached"
                        if price_change >= exit_threshold
                        else "mean_reversion"
                    ),
                    "price_change": price_change,
                    "deviation": deviation,
                    "timestamp": time.time(),
                }

            # Иначе продолжаем держать позицию
            return None

        # Если нет открытой позиции, проверяем возможность входа

        # Проверяем количество открытых позиций
        if len(self.open_positions) >= self.strategy_config["max_positions"]:
            return None

        # Определяем направление сигнала (по отклонению от среднего)
        signal_direction = None

        # Проверяем отклонение от среднего
        threshold = self.strategy_config["deviation_threshold"]

        if abs(deviation) >= threshold:
            if deviation < 0:
                # Цена ниже средней - сигнал на покупку
                signal_direction = "buy"
            else:
                # Цена выше средней - сигнал на продажу
                signal_direction = "sell"

        # Проверяем согласованность сигналов
        if signal_direction and bb_signal and rsi_signal:
            if signal_direction == bb_signal == rsi_signal:
                # Все сигналы согласованы - сильный сигнал
                pass
            elif (signal_direction == bb_signal) or (signal_direction == rsi_signal):
                # Два из трех сигналов согласованы - умеренный сигнал
                pass
            else:
                # Сигналы противоречат друг другу
                signal_direction = None

        # Если нет четкого направления, нет сигнала
        if not signal_direction:
            return None

        # Формируем сигнал входа
        side = "long" if signal_direction == "buy" else "short"

        # Рассчитываем стоп-лосс
        stop_loss_pct = self.strategy_config["stop_loss_pct"]
        stop_loss = (
            current_price * (1 - stop_loss_pct)
            if side == "long"
            else current_price * (1 + stop_loss_pct)
        )

        # Рассчитываем целевую цену для выхода
        target_deviation = (
            -deviation
            * self.strategy_config["exit_threshold"]
            / self.strategy_config["deviation_threshold"]
        )
        target_price = current_price * (1 + target_deviation)

        return {
            "symbol": symbol,
            "action": signal_direction,
            "side": side,
            "price": current_price,
            "stop_loss": stop_loss,
            "target_price": target_price,
            "deviation": deviation,
            "bb_signal": bb_signal,
            "rsi_signal": rsi_signal,
            "timestamp": time.time(),
        }
