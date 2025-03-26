"""
Модуль для получения и хранения рыночных данных.
Предоставляет интерфейс для работы с котировками, ордербуками и другими данными.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error, async_with_retry
from project.utils.cache_utils import async_cache
from project.utils.ccxt_exchanges import (
    connect_exchange,
    fetch_ticker,
    fetch_ohlcv,
    fetch_order_book,
)

logger = get_logger(__name__)


class MarketData:
    """
    Класс для получения и хранения рыночных данных.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "MarketData":
        """
        Получает экземпляр класса MarketData (Singleton).

        Returns:
            Экземпляр класса MarketData
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Инициализирует объект для работы с рыночными данными.
        """
        # Кэш для данных
        self.ticker_cache: Dict[str, Dict[str, Any]] = {}
        self.ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self.orderbook_cache: Dict[str, Dict[str, Any]] = {}

        # Метаданные для кэша
        self.ticker_timestamps: Dict[str, float] = {}
        self.ohlcv_timestamps: Dict[str, float] = {}
        self.orderbook_timestamps: Dict[str, float] = {}

        # Настройки кэширования
        self.ticker_ttl = 5.0  # время жизни тикера в секундах
        self.ohlcv_ttl = 60.0  # время жизни OHLCV в секундах
        self.orderbook_ttl = 2.0  # время жизни ордербука в секундах

        # Задачи для автоматического обновления данных
        self.update_tasks = []

        logger.debug("Создан экземпляр MarketData")

    @async_cache(ttl=5.0)
    async def get_ticker(self, exchange_id: str, symbol: str) -> Dict[str, Any]:
        """
        Получает данные тикера для указанного символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Данные тикера
        """
        # Генерируем ключ для кэша
        cache_key = f"{exchange_id}:{symbol}:ticker"

        # Проверяем наличие данных в кэше и их актуальность
        if (
            cache_key in self.ticker_cache
            and time.time() - self.ticker_timestamps.get(cache_key, 0) < self.ticker_ttl
        ):
            logger.debug(
                f"Получены данные тикера из кэша для {symbol} на {exchange_id}"
            )
            return self.ticker_cache[cache_key]

        # Получаем данные с биржи
        try:
            ticker = await fetch_ticker(exchange_id, symbol)

            # Кэшируем данные
            self.ticker_cache[cache_key] = ticker
            self.ticker_timestamps[cache_key] = time.time()

            logger.debug(
                f"Получены данные тикера с биржи для {symbol} на {exchange_id}"
            )
            return ticker
        except Exception as e:
            logger.error(
                f"Ошибка при получении тикера для {symbol} на {exchange_id}: {str(e)}"
            )

            # Возвращаем кэшированные данные, если они есть
            if cache_key in self.ticker_cache:
                logger.warning(
                    f"Возвращены устаревшие данные тикера для {symbol} на {exchange_id}"
                )
                return self.ticker_cache[cache_key]

            raise

    @async_cache(ttl=60.0)
    async def get_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Получает данные OHLCV для указанного символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары
            timeframe: Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d, ...)
            limit: Ограничение на количество свечей
            since: Временная метка начала в миллисекундах

        Returns:
            DataFrame с данными OHLCV
        """
        # Генерируем ключ для кэша
        cache_key = f"{exchange_id}:{symbol}:{timeframe}:{limit}:{since}:ohlcv"

        # Проверяем наличие данных в кэше и их актуальность
        if (
            cache_key in self.ohlcv_cache
            and time.time() - self.ohlcv_timestamps.get(cache_key, 0) < self.ohlcv_ttl
        ):
            logger.debug(f"Получены данные OHLCV из кэша для {symbol} на {exchange_id}")
            return self.ohlcv_cache[cache_key]

        # Получаем данные с биржи
        try:
            ohlcv_data = await fetch_ohlcv(exchange_id, symbol, timeframe, since, limit)

            # Преобразуем данные в DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Преобразуем timestamp в datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            # Кэшируем данные
            self.ohlcv_cache[cache_key] = df
            self.ohlcv_timestamps[cache_key] = time.time()

            logger.debug(f"Получены данные OHLCV с биржи для {symbol} на {exchange_id}")
            return df
        except Exception as e:
            logger.error(
                f"Ошибка при получении OHLCV для {symbol} на {exchange_id}: {str(e)}"
            )

            # Возвращаем кэшированные данные, если они есть
            if cache_key in self.ohlcv_cache:
                logger.warning(
                    f"Возвращены устаревшие данные OHLCV для {symbol} на {exchange_id}"
                )
                return self.ohlcv_cache[cache_key]

            raise

    @async_cache(ttl=2.0)
    async def get_orderbook(
        self, exchange_id: str, symbol: str, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Получает данные ордербука для указанного символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

            limit: Ограничение на количество уровней

        Returns:
            Данные ордербука
        """
        # Генерируем ключ для кэша
        cache_key = f"{exchange_id}:{symbol}:{limit}:orderbook"

        # Проверяем наличие данных в кэше и их актуальность
        if (
            cache_key in self.orderbook_cache
            and time.time() - self.orderbook_timestamps.get(cache_key, 0)
            < self.orderbook_ttl
        ):
            logger.debug(
                f"Получены данные ордербука из кэша для {symbol} на {exchange_id}"
            )
            return self.orderbook_cache[cache_key]

        # Получаем данные с биржи
        try:
            orderbook = await fetch_order_book(exchange_id, symbol, limit)

            # Кэшируем данные
            self.orderbook_cache[cache_key] = orderbook
            self.orderbook_timestamps[cache_key] = time.time()

            logger.debug(
                f"Получены данные ордербука с биржи для {symbol} на {exchange_id}"
            )
            return orderbook
        except Exception as e:
            logger.error(
                f"Ошибка при получении ордербука для {symbol} на {exchange_id}: {str(e)}"
            )

            # Возвращаем кэшированные данные, если они есть
            if cache_key in self.orderbook_cache:
                logger.warning(
                    f"Возвращены устаревшие данные ордербука для {symbol} на {exchange_id}"
                )
                return self.orderbook_cache[cache_key]

            raise

    async def start_automatic_updates(
        self,
        exchange_id: str,
        symbols: List[str],
        update_tickers: bool = True,
        update_ohlcv: bool = True,
        update_orderbooks: bool = False,
        ticker_interval: float = 10.0,
        ohlcv_interval: float = 60.0,
        orderbook_interval: float = 5.0,
    ) -> None:
        """
        Запускает автоматическое обновление данных.

        Args:
            exchange_id: Идентификатор биржи
            symbols: Список символов для обновления
            update_tickers: Обновлять тикеры
            update_ohlcv: Обновлять OHLCV
            update_orderbooks: Обновлять ордербуки
            ticker_interval: Интервал обновления тикеров в секундах
            ohlcv_interval: Интервал обновления OHLCV в секундах
            orderbook_interval: Интервал обновления ордербуков в секундах
        """
        # Останавливаем существующие задачи
        await self.stop_automatic_updates()

        # Создаем и запускаем новые задачи
        if update_tickers:
            task = asyncio.create_task(
                self._update_tickers_periodically(exchange_id, symbols, ticker_interval)
            )
            self.update_tasks.append(task)
            logger.info(
                f"Запущено автоматическое обновление тикеров для {len(symbols)} символов на {exchange_id}"
            )

        if update_ohlcv:
            task = asyncio.create_task(
                self._update_ohlcv_periodically(exchange_id, symbols, ohlcv_interval)
            )
            self.update_tasks.append(task)
            logger.info(
                f"Запущено автоматическое обновление OHLCV для {len(symbols)} символов на {exchange_id}"
            )

        if update_orderbooks:
            task = asyncio.create_task(
                self._update_orderbooks_periodically(
                    exchange_id, symbols, orderbook_interval
                )
            )
            self.update_tasks.append(task)
            logger.info(
                f"Запущено автоматическое обновление ордербуков для {len(symbols)} символов на {exchange_id}"
            )

    async def stop_automatic_updates(self) -> None:
        """
        Останавливает автоматическое обновление данных.
        """
        for task in self.update_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.update_tasks = []
        logger.info("Автоматическое обновление данных остановлено")

    async def _update_tickers_periodically(
        self, exchange_id: str, symbols: List[str], interval: float
    ) -> None:
        """
        Периодически обновляет данные тикеров.

        Args:
            exchange_id: Идентификатор биржи
            symbols: Список символов для обновления
            interval: Интервал обновления в секундах
        """
        try:
            while True:
                for symbol in symbols:
                    try:
                        await self.get_ticker(exchange_id, symbol)
                    except Exception as e:
                        logger.error(
                            f"Ошибка при обновлении тикера для {symbol} на {exchange_id}: {str(e)}"
                        )

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Задача обновления тикеров отменена")
            raise

    async def _update_ohlcv_periodically(
        self, exchange_id: str, symbols: List[str], interval: float
    ) -> None:
        """
        Периодически обновляет данные OHLCV.

        Args:
            exchange_id: Идентификатор биржи
            symbols: Список символов для обновления
            interval: Интервал обновления в секундах
        """
        try:
            while True:
                for symbol in symbols:
                    try:
                        # Обновляем данные для разных таймфреймов
                        for timeframe in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                            await self.get_ohlcv(exchange_id, symbol, timeframe)
                    except Exception as e:
                        logger.error(
                            f"Ошибка при обновлении OHLCV для {symbol} на {exchange_id}: {str(e)}"
                        )

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Задача обновления OHLCV отменена")
            raise

    async def _update_orderbooks_periodically(
        self, exchange_id: str, symbols: List[str], interval: float
    ) -> None:
        """
        Периодически обновляет данные ордербуков.

        Args:
            exchange_id: Идентификатор биржи
            symbols: Список символов для обновления
            interval: Интервал обновления в секундах
        """
        try:
            while True:
                for symbol in symbols:
                    try:
                        await self.get_orderbook(exchange_id, symbol)
                    except Exception as e:
                        logger.error(
                            f"Ошибка при обновлении ордербука для {symbol} на {exchange_id}: {str(e)}"
                        )

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.debug("Задача обновления ордербуков отменена")
            raise

    def get_current_price(self, exchange_id: str, symbol: str) -> float:
        """
        Получает текущую цену для указанного символа.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Текущая цена
        """
        cache_key = f"{exchange_id}:{symbol}:ticker"

        if cache_key in self.ticker_cache:
            ticker = self.ticker_cache[cache_key]
            return float(ticker.get("last", 0))

        raise ValueError(f"Нет данных тикера для {symbol} на {exchange_id}")

    def calculate_market_metrics(
        self, exchange_id: str, symbol: str
    ) -> Dict[str, float]:
        """
        Рассчитывает метрики рынка на основе имеющихся данных.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Словарь с метриками рынка
        """
        metrics = {}

        try:
            # Получаем тикер из кэша
            ticker_key = f"{exchange_id}:{symbol}:ticker"
            if ticker_key in self.ticker_cache:
                ticker = self.ticker_cache[ticker_key]

                metrics["price"] = float(ticker.get("last", 0))
                metrics["bid"] = float(ticker.get("bid", 0))
                metrics["ask"] = float(ticker.get("ask", 0))
                metrics["volume"] = float(ticker.get("volume", 0))
                metrics["change"] = float(ticker.get("percentage", 0))
                metrics["spread"] = (
                    (metrics["ask"] - metrics["bid"]) / metrics["bid"] * 100
                    if metrics["bid"]
                    else 0
                )

            # Получаем OHLCV из кэша для 1h
            ohlcv_key = f"{exchange_id}:{symbol}:1h:100:None:ohlcv"
            if ohlcv_key in self.ohlcv_cache:
                df = self.ohlcv_cache[ohlcv_key]

                # Рассчитываем метрики на основе OHLCV
                if not df.empty:
                    metrics["1h_change"] = (
                        df["close"].iloc[-1] / df["close"].iloc[-2] - 1
                    ) * 100
                    metrics["1h_volatility"] = (
                        df["high"].iloc[-1] / df["low"].iloc[-1] - 1
                    )
                    metrics["24h_high"] = df["high"].iloc[-24:].max()
                    metrics["24h_low"] = df["low"].iloc[-24:].min()
                    metrics["24h_volume"] = df["volume"].iloc[-24:].sum()

            # Получаем ордербук из кэша
            orderbook_key = f"{exchange_id}:{symbol}:None:orderbook"
            if orderbook_key in self.orderbook_cache:
                orderbook = self.orderbook_cache[orderbook_key]

                # Рассчитываем метрики на основе ордербука
                if "bids" in orderbook and "asks" in orderbook:
                    bids = orderbook["bids"]
                    asks = orderbook["asks"]

                    if bids and asks:
                        metrics["orderbook_spread"] = (
                            (asks[0][0] - bids[0][0]) / bids[0][0] * 100
                        )

                        # Рассчитываем глубину рынка (нарастающий объем до 2% от цены)
                        price = metrics.get("price", bids[0][0])
                        buy_depth = sum(
                            amount
                            for bid_price, amount in bids
                            if bid_price >= price * 0.98
                        )
                        sell_depth = sum(
                            amount
                            for ask_price, amount in asks
                            if ask_price <= price * 1.02
                        )

                        metrics["buy_depth"] = buy_depth
                        metrics["sell_depth"] = sell_depth
                        metrics["buy_sell_ratio"] = (
                            buy_depth / sell_depth if sell_depth else float("inf")
                        )

        except Exception as e:
            logger.error(
                f"Ошибка при расчете метрик рынка для {symbol} на {exchange_id}: {str(e)}"
            )

        return metrics
