"""
Стратегия для торговли мемкоинами.
Фокусируется на выявлении и торговле монетами с высокой волатильностью и объемом.
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np
from project.bots.strategies.base_strategy import BaseStrategy
from project.technicals.indicators import Indicators
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MemcoinStrategy(BaseStrategy):
    """
    Стратегия для торговли мемкоинами с высокой волатильностью.
    """

    def __init__(
        self,
        name: str = "MemcoinStrategy",
        exchange_id: str = "binance",
        symbols: List[str] = None,
        timeframes: List[str] = None,
        config: Dict[str, Any] = None,
    ):
        """
        Инициализирует стратегию для мемкоинов.

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
            "volume_increase_threshold": 3.0,  # порог увеличения объема (3x от среднего)
            "price_increase_threshold": 0.05,  # порог увеличения цены (5%)
            "rsi_threshold": 75,  # порог RSI для определения перекупленности
            "take_profit_pct": 0.1,  # тейк-профит (10%)
            "stop_loss_pct": 0.05,  # стоп-лосс (5%)
            "max_coins": 5,  # максимальное количество монет для одновременной торговли
            "max_position_size_pct": 0.02,  # максимальный размер позиции (2% от счета)
            "check_social_media": True,  # проверять активность в социальных сетях
            "social_media_sources": ["twitter", "reddit"],  # источники социальных медиа
            "hold_time": 24,  # время удержания позиции в часах
            "trailing_stop_enabled": True,  # использовать трейлинг-стоп
            "trailing_stop_pct": 0.03,  # процент трейлинг-стопа (3%)
            "auto_scan_mode": False,  # режим автоматического сканирования новых монет
        }

        # Объединяем с пользовательской конфигурацией
        for key, value in default_config.items():
            if key not in config:
                config[key] = value

        # Устанавливаем базовые значения
        # Если символы не указаны, используем популярные мемкоины
        symbols = symbols or ["DOGE/USDT", "SHIB/USDT", "FLOKI/USDT", "PEPE/USDT"]
        timeframes = timeframes or ["15m", "1h", "4h"]

        super().__init__(name, exchange_id, symbols, timeframes, config)

        # Дополнительные параметры
        self.volume_data: Dict[str, List[float]] = {}  # symbol -> история объемов
        self.price_data: Dict[str, List[float]] = {}  # symbol -> история цен
        self.social_metrics: Dict[str, Dict[str, float]] = (
            {}
        )  # symbol -> метрики активности
        self.last_scan_time = 0  # время последнего сканирования новых монет
        self.potential_coins: List[Dict[str, Any]] = (
            []
        )  # список потенциальных монет для торговли

        logger.debug(f"Создана стратегия для мемкоинов {self.name}" )

    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Обновляет специфические параметры конфигурации.

        Args:
            config: Словарь с новыми параметрами конфигурации
        """
        # Обновляем числовые параметры
        for param in [
            "volume_increase_threshold",
            "price_increase_threshold",
            "rsi_threshold",
            "take_profit_pct",
            "stop_loss_pct",
            "max_position_size_pct",
            "trailing_stop_pct",
        ]:
            if param in config:
                self.strategy_config[param] = float(config[param])

        for param in ["max_coins", "hold_time"]:
            if param in config:
                self.strategy_config[param] = int(config[param])

        # Обновляем булевы параметры
        for param in ["check_social_media", "trailing_stop_enabled", "auto_scan_mode"]:
            if param in config:
                self.strategy_config[param] = bool(config[param])

        # Обновляем списки
        if "social_media_sources" in config:
            self.strategy_config["social_media_sources"] = config[
                "social_media_sources"
            ]

    async def _strategy_initialize(self) -> None:
        """
        Выполняет дополнительную инициализацию стратегии.
        """
        # Инициализируем структуры данных для хранения истории
        for symbol in self.symbols:
            self.volume_data[symbol] = []
            self.price_data[symbol] = []
            self.social_metrics[symbol] = {
                "twitter_mentions": 0,
                "reddit_mentions": 0,
                "social_score": 0,
            }

        # Загружаем историю цен и объемов
        await self._load_historical_data()

        # Если включен режим автоматического сканирования, запускаем сканирование
        if self.strategy_config["auto_scan_mode"]:
            await self._scan_for_new_coins()

    async def _strategy_cleanup(self) -> None:
        """
        Выполняет дополнительную очистку ресурсов стратегии.
        """
        # Нет специфических ресурсов для очистки

    @async_handle_error
    async def _load_historical_data(self) -> None:
        """
        Загружает историю цен и объемов для анализа.
        """
        for symbol in self.symbols:
            try:
                # Загружаем данные OHLCV для часового таймфрейма
                ohlcv = await self.market_data.get_ohlcv(
                    self.exchange_id, symbol, "1h", limit=48
                )

                if not ohlcv.empty:
                    # Сохраняем историю цен (закрытия)
                    self.price_data[symbol] = ohlcv["close"].tolist()

                    # Сохраняем историю объемов
                    self.volume_data[symbol] = ohlcv["volume"].tolist()

                    logger.debug(f"Загружена история для {symbol}: {len(ohlcv)} свечей" )

            except Exception as e:
                logger.error(f"Ошибка при загрузке истории для {symbol}: {str(e)}" )

    @async_handle_error
    async def _scan_for_new_coins(self) -> None:
        """
        Сканирует рынок на предмет новых перспективных мемкоинов.
        """
        # Проверяем, не слишком ли часто выполняем сканирование
        current_time = time.time()
        if current_time - self.last_scan_time < 3600:  # не чаще раза в час
            return

        self.last_scan_time = current_time

        try:
            logger.info("Сканирование рынка на предмет новых мемкоинов...")

            # Получаем список всех доступных монет с USDT парой
            # В реальной реализации здесь должен быть код для получения всех торговых пар
            # и фильтрации их на основе объема, рыночной капитализации и т.д.

            # Для примера, имитируем получение списка монет
            potential_coins = [
                "DOGE/USDT",
                "SHIB/USDT",
                "FLOKI/USDT",
                "PEPE/USDT",
                "BONK/USDT",
                "MEME/USDT",
                "WIF/USDT",
                "BABYDOGE/USDT",
            ]

            # Проверяем каждую монету на соответствие нашим критериям
            for coin in potential_coins:
                if coin in self.symbols:
                    continue  # пропускаем уже отслеживаемые монеты

                # Получаем данные OHLCV
                ohlcv = await self.market_data.get_ohlcv(
                    self.exchange_id, coin, "1h", limit=48
                )

                if ohlcv.empty:
                    continue

                # Проверяем объем
                recent_volume = ohlcv["volume"].iloc[-1]
                avg_volume = ohlcv["volume"].iloc[-24:].mean()

                # Проверяем изменение цены
                recent_price = ohlcv["close"].iloc[-1]
                day_ago_price = (
                    ohlcv["close"].iloc[-24]
                    if len(ohlcv) >= 24
                    else ohlcv["close"].iloc[0]
                )

                price_change_pct = (
                    (recent_price / day_ago_price - 1) if day_ago_price > 0 else 0
                )

                # Рассчитываем RSI
                rsi = Indicators.relative_strength_index(ohlcv, 14)
                recent_rsi = rsi.iloc[-1] if not rsi.empty else 50

                # Оцениваем монету
                if (
                    recent_volume
                    > avg_volume * self.strategy_config["volume_increase_threshold"]
                    and price_change_pct
                    > self.strategy_config["price_increase_threshold"]
                ):

                    # Добавляем монету в список потенциальных
                    self.potential_coins.append(
                        {
                            "symbol": coin,
                            "volume": recent_volume,
                            "volume_increase": recent_volume / avg_volume,
                            "price_change_pct": price_change_pct,
                            "rsi": recent_rsi,
                            "discovered_at": current_time,
                        }
                    )

                    logger.info(
                        f"Обнаружена потенциальная монета: {coin}, "
                        f"увеличение объема: {recent_volume / avg_volume:.2f}x, "
                        f"изменение цены: {price_change_pct:.2f}%"
                    )

            # Сортируем потенциальные монеты по изменению объема
            self.potential_coins.sort(key=lambda x: x["volume_increase"], reverse=True)

            # Добавляем лучшие монеты в список отслеживаемых
            coins_to_add = min(
                3, len(self.potential_coins)
            )  # не более 3 новых монет за раз

            for i in range(coins_to_add):
                if i < len(self.potential_coins):
                    coin = self.potential_coins[i]["symbol"]
                    if coin not in self.symbols:
                        self.symbols.append(coin)
                        self.volume_data[coin] = []
                        self.price_data[coin] = []
                        self.social_metrics[coin] = {
                            "twitter_mentions": 0,
                            "reddit_mentions": 0,
                            "social_score": 0,
                        }

                        logger.info(f"Добавлена новая монета для отслеживания: {coin}" )

        except Exception as e:
            logger.error(f"Ошибка при сканировании новых монет: {str(e)}" )

    @async_handle_error
    async def _check_social_media(self, symbol: str) -> None:
        """
        Проверяет активность в социальных сетях для указанной монеты.

        Args:
            symbol: Символ для проверки
        """
        if not self.strategy_config["check_social_media"]:
            return

        # Извлекаем базовую монету из символа (например, "DOGE" из "DOGE/USDT")
        base_coin = symbol.split("/")[0]

        try:
            # В реальной реализации здесь должен быть код для получения данных
            # из социальных сетей (Twitter, Reddit и т.д.)

            # Имитируем получение данных
            if "twitter" in self.strategy_config["social_media_sources"]:
                # Имитируем количество упоминаний в Twitter
                twitter_mentions = np.random.randint(10, 1000)
                self.social_metrics[symbol]["twitter_mentions"] = twitter_mentions

            if "reddit" in self.strategy_config["social_media_sources"]:
                # Имитируем количество упоминаний в Reddit
                reddit_mentions = np.random.randint(5, 500)
                self.social_metrics[symbol]["reddit_mentions"] = reddit_mentions

            # Рассчитываем общий социальный балл
            social_score = (
                self.social_metrics[symbol]["twitter_mentions"] * 0.6
                + self.social_metrics[symbol]["reddit_mentions"] * 0.4
            )

            self.social_metrics[symbol]["social_score"] = social_score

            logger.debug(
                f"Социальная активность для {symbol}: Twitter={
                    self.social_metrics[symbol]['twitter_mentions']}, " f"Reddit={
                    self.social_metrics[symbol]['reddit_mentions']}, Score={
                    social_score:.2f}")

        except Exception as e:
            logger.error(f"Ошибка при проверке социальных медиа для {symbol}: {str(e)}" )

    @async_handle_error
    async def _generate_trading_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Генерирует торговые сигналы для мемкоинов.

        Returns:
            Словарь с сигналами для каждого символа
        """
        signals = {}

        # Если включен режим автоматического сканирования, проверяем новые монеты
        if self.strategy_config["auto_scan_mode"]:
            await self._scan_for_new_coins()

        # Обновляем историю цен и объемов
        await self._load_historical_data()

        for symbol in self.symbols:
            try:
                # Получаем тикер
                ticker = await self.market_data.get_ticker(self.exchange_id, symbol)
                if not ticker:
                    continue

                # Получаем текущую цену и объем
                current_price = ticker.get("last", 0)
                current_volume = ticker.get("quoteVolume", 0) or ticker.get("volume", 0)

                if current_price <= 0 or current_volume <= 0:
                    continue

                # Проверяем социальные медиа
                await self._check_social_media(symbol)

                # Обновляем историю
                self.price_data[symbol].append(current_price)
                self.volume_data[symbol].append(current_volume)

                # Ограничиваем размер истории
                if len(self.price_data[symbol]) > 48:
                    self.price_data[symbol] = self.price_data[symbol][-48:]
                if len(self.volume_data[symbol]) > 48:
                    self.volume_data[symbol] = self.volume_data[symbol][-48:]

                # Рассчитываем метрики
                avg_volume = (
                    np.mean(self.volume_data[symbol][-24:])
                    if len(self.volume_data[symbol]) >= 24
                    else current_volume
                )
                volume_increase = current_volume / avg_volume if avg_volume > 0 else 1

                prev_price = (
                    self.price_data[symbol][-2]
                    if len(self.price_data[symbol]) >= 2
                    else current_price
                )
                price_change_1h = (
                    (current_price / prev_price - 1) if prev_price > 0 else 0
                )

                day_ago_price = (
                    self.price_data[symbol][-24]
                    if len(self.price_data[symbol]) >= 24
                    else (
                        self.price_data[symbol][0]
                        if self.price_data[symbol]
                        else current_price
                    )
                )
                price_change_24h = (
                    (current_price / day_ago_price - 1) if day_ago_price > 0 else 0
                )

                # Получаем RSI
                ohlcv = await self.market_data.get_ohlcv(
                    self.exchange_id, symbol, "1h", limit=30
                )
                rsi = Indicators.relative_strength_index(ohlcv, 14)
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50

                # Социальная активность
                social_score = self.social_metrics[symbol].get("social_score", 0)

                # Генерируем сигнал
                signal = self._generate_memcoin_signal(
                    symbol,
                    current_price,
                    volume_increase,
                    price_change_1h,
                    price_change_24h,
                    current_rsi,
                    social_score,
                )

                if signal:
                    signals[symbol] = signal

                # Проверяем открытые позиции
                if symbol in self.open_positions:
                    await self._check_memcoin_position(symbol, current_price)

            except Exception as e:
                logger.error(f"Ошибка при генерации сигналов для {symbol}: {str(e)}" )

        return signals

    def _generate_memcoin_signal(
        self,
        symbol: str,
        current_price: float,
        volume_increase: float,
        price_change_1h: float,
        price_change_24h: float,
        current_rsi: float,
        social_score: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Генерирует сигнал для мемкоина на основе различных метрик.

        Args:
            symbol: Символ монеты
            current_price: Текущая цена
            volume_increase: Увеличение объема
            price_change_1h: Изменение цены за 1 час
            price_change_24h: Изменение цены за 24 часа
            current_rsi: Текущее значение RSI
            social_score: Социальный балл

        Returns:
            Словарь с сигналом или None
        """
        # Проверяем, есть ли уже открытая позиция
        if symbol in self.open_positions:
            return None

        # Проверяем, не превышено ли максимальное количество открытых позиций
        if len(self.open_positions) >= self.strategy_config["max_coins"]:
            return None

        # Рассчитываем совокупный балл
        score = 0.0

        # Учитываем увеличение объема
        if volume_increase > self.strategy_config["volume_increase_threshold"]:
            volume_score = min(
                (volume_increase / self.strategy_config["volume_increase_threshold"])
                * 2,
                10,
            )
            score += volume_score

        # Учитываем изменение цены
        if price_change_1h > 0:
            price_score = min(
                (price_change_1h / self.strategy_config["price_increase_threshold"])
                * 3,
                15,
            )
            score += price_score

        # Учитываем RSI (избегаем перекупленности)
        if current_rsi < self.strategy_config["rsi_threshold"]:
            rsi_score = (self.strategy_config["rsi_threshold"] - current_rsi) / 10
            score += rsi_score
        else:
            # Если RSI высокий, снижаем балл
            score -= (current_rsi - self.strategy_config["rsi_threshold"]) / 5

        # Учитываем социальную активность
        if social_score > 100:
            social_score_norm = min(social_score / 100, 5)
            score += social_score_norm

        # Генерируем сигнал, только если балл достаточно высок
        if score > 10:  # пороговое значение
            return {
                "symbol": symbol,
                "action": "buy",
                "price": current_price,
                "score": score,
                "volume_increase": volume_increase,
                "price_change_1h": price_change_1h,
                "price_change_24h": price_change_24h,
                "rsi": current_rsi,
                "social_score": social_score,
                "timestamp": time.time(),
            }

        return None

    @async_handle_error
    async def _check_memcoin_position(self, symbol: str, current_price: float) -> None:
        """
        Проверяет состояние открытой позиции мемкоина.

        Args:
            symbol: Символ монеты
            current_price: Текущая цена
        """
        if symbol not in self.open_positions:
            return

        position = self.open_positions[symbol]
        entry_price = position["entry_price"]
        entry_time = position["entry_time"]

        # Рассчитываем процентное изменение
        percent_change = (current_price / entry_price - 1) * 100

        # Проверяем условия выхода
        exit_reason = None

        # Проверяем тейк-профит
        take_profit_pct = self.strategy_config["take_profit_pct"] * 100
        if percent_change >= take_profit_pct:
            exit_reason = "take_profit"

        # Проверяем стоп-лосс
        stop_loss_pct = self.strategy_config["stop_loss_pct"] * 100
        if percent_change <= -stop_loss_pct:
            exit_reason = "stop_loss"

        # Проверяем время удержания
        hold_time_seconds = time.time() - entry_time
        hold_time_hours = hold_time_seconds / 3600

        if hold_time_hours >= self.strategy_config["hold_time"]:
            exit_reason = "time_limit"

        # Если есть причина выхода, закрываем позицию
        if exit_reason:
            logger.info(
                f"Закрываем позицию по {symbol} по причине {exit_reason}. "
                f"Изменение цены: {percent_change:.2f}%, "
                f"Время удержания: {hold_time_hours:.2f} часов"
            )

            await self._close_position(symbol, exit_reason)

        # Обновляем трейлинг-стоп, если включен
        elif self.strategy_config["trailing_stop_enabled"] and percent_change > 0:
            # Рассчитываем новый уровень стоп-лосса
            trailing_stop_pct = self.strategy_config["trailing_stop_pct"] * 100
            new_stop_level = current_price * (1 - trailing_stop_pct / 100)

            # Получаем текущий стоп-лосс
            current_stop = position.get("stop_loss")

            # Обновляем стоп-лосс, только если новый выше текущего
            if current_stop is None or new_stop_level > current_stop:
                self.open_positions[symbol]["stop_loss"] = new_stop_level
                logger.debug(
                    f"Обновлен трейлинг-стоп для {symbol}: {new_stop_level:.8f}"
                )
