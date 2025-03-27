"""
Бот для торговли на основе новостей и информационных событий.
Мониторит новостные источники и генерирует торговые сигналы.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

from project.bots.base_bot import BaseBot, BotStatus
from project.bots.news.parsers.bitcoinmag_parser import BitcoinMagParser
from project.bots.news.parsers.coindesk_parser import CoindeskParser
from project.config import get_config
from project.utils.ccxt_exchanges import fetch_ticker
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)


class NewsBot(BaseBot):
    """
    Бот для торговли на основе новостей и информационных событий.
    """

    def __init__(
        self,
        name: str = "NewsBot",
        exchange_id: str = "binance",
        symbols: List[str] = None,
        news_sources: List[str] = None,
    ):
        """
        Инициализирует новостной бот.

        Args:
            name: Имя бота
            exchange_id: Идентификатор биржи
            symbols: Список символов для торговли
            news_sources: Список источников новостей
        """
        symbols = symbols or [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
        ]

        super().__init__(name, exchange_id, symbols)

        self.config = get_config()
        self.news_sources = news_sources or [
            "coindesk",
            "bitcoinmagazine",
            "cointelegraph",
        ]
        self.parsers = {}

        # Инициализируем парсеры
        self._initialize_parsers()

        # Ключевые слова для мониторинга
        self.positive_keywords = {
            "BTC": [
                "bitcoin adoption",
                "bitcoin bull",
                "btc rally",
                "institutional adoption",
                "bitcoin etf approval",
            ],
            "ETH": [
                "ethereum upgrade",
                "eth 2.0",
                "eth scaling",
                "ethereum adoption",
                "defi growth",
            ],
            "SOL": [
                "solana upgrade",
                "sol partnership",
                "solana adoption",
                "solana scaling",
                "solana development",
            ],
            "XRP": [
                "ripple win",
                "xrp lawsuit",
                "ripple partnership",
                "xrp adoption",
                "ripple success",
            ],
            "ADA": [
                "cardano upgrade",
                "ada staking",
                "cardano partnership",
                "smart contracts",
                "cardano adoption",
            ],
        }

        self.negative_keywords = {
            "BTC": [
                "bitcoin ban",
                "btc crash",
                "crypto regulation",
                "btc bearish",
                "bitcoin sell-off",
            ],
            "ETH": [
                "ethereum hack",
                "eth vulnerability",
                "ethereum delay",
                "eth bearish",
                "ethereum criticized",
            ],
            "SOL": [
                "solana outage",
                "sol hack",
                "solana vulnerability",
                "solana criticized",
                "sol bearish",
            ],
            "XRP": [
                "ripple lawsuit",
                "xrp penalty",
                "ripple regulation",
                "xrp delisting",
                "ripple sec",
            ],
            "ADA": [
                "cardano delay",
                "ada vulnerability",
                "cardano criticized",
                "cardano bearish",
                "cardano hack",
            ],
        }

        # Настройки торговли
        self.position_size_pct = 0.02  # 2% от баланса
        self.stop_loss_pct = 0.03  # 3% стоп-лосс
        self.take_profit_pct = 0.05  # 5% тейк-профит
        self.max_positions = 3  # максимум 3 одновременные позиции
        self.news_relevance_time = 6  # время актуальности новости (часов)

        # Хранение данных
        self.processed_news = {}  # url -> данные новости
        self.active_news_signals = {}  # symbol -> данные сигнала
        self.pending_signals = []  # ожидающие сигналы

        # Настройки обновления
        self.news_check_interval = 300  # проверять новости каждые 5 минут
        self.update_interval = 60  # основной цикл каждую минуту

        # Статистика
        self.stats.update(
            {
                "news_processed": 0,
                "positive_signals": 0,
                "negative_signals": 0,
                "signals_executed": 0,
            }
        )

        # Добавляем необходимый атрибут
        self.open_positions = {}

        logger.debug("Создан новостной бот {self.name}" %)

    def _initialize_parsers(self) -> None:
        """
        Инициализирует парсеры для различных новостных источников.
        """
        if "coindesk" in self.news_sources:
            self.parsers["coindesk"] = CoindeskParser()

        if "bitcoinmagazine" in self.news_sources:
            self.parsers["bitcoinmagazine"] = BitcoinMagParser()

        # Для других источников можно добавить соответствующие парсеры
        # Например:
        # if "cointelegraph" in self.news_sources:
        #     self.parsers["cointelegraph"] = CointelegraphParser()

        logger.debug(
            f"Инициализированы парсеры для источников: {list(self.parsers.keys())}"
        )

    async def _initialize(self) -> None:
        """
        Инициализирует бота перед запуском.
        """
        await super()._initialize()

        # Дополнительная инициализация для новостного бота
        logger.info("Инициализация новостного бота {self.name}" %)

        # Запускаем отдельную задачу для проверки новостей
        self.news_task = asyncio.create_task(self._check_news_periodically())

    async def _cleanup(self) -> None:
        """
        Выполняет очистку ресурсов при остановке бота.
        """
        # Отменяем задачу проверки новостей
        if hasattr(self, "news_task") and not self.news_task.done():
            self.news_task.cancel()
            try:
                await self.news_task
            except asyncio.CancelledError:
                pass

        # Вызываем базовый метод очистки
        await super()._cleanup()

    @async_handle_error
    async def _check_news_periodically(self) -> None:
        """
        Периодически проверяет новости из различных источников.
        """
        try:
            logger.info("Запущена задача проверки новостей для {self.name}" %)

            while True:
                if self.status == BotStatus.RUNNING:
                    await self._fetch_and_process_news()

                # Ждем до следующей проверки новостей
                await asyncio.sleep(self.news_check_interval)

        except asyncio.CancelledError:
            logger.info("Задача проверки новостей для {self.name} отменена" %)
            raise
        except Exception as e:
            logger.error("Ошибка в задаче проверки новостей для {self.name}: {str(e)}" %)

    @async_handle_error
    async def _fetch_and_process_news(self) -> None:
        """
        Получает новости из всех источников и обрабатывает их.
        """
        for source, parser in self.parsers.items():
            try:
                logger.debug("Получение новостей из {source}" %)

                # Получаем последние новости
                news = await parser.fetch_latest_news()

                if not news:
                    logger.debug("Нет новых новостей из {source}" %)
                    continue

                logger.debug("Получено {len(news)} новостей из {source}" %)

                # Обрабатываем каждую новость
                for article in news:
                    await self._process_news_article(article, source)

            except Exception as e:
                logger.error("Ошибка при получении новостей из {source}: {str(e)}" %)

    @async_handle_error
    async def _process_news_article(self, article: Dict[str, Any], source: str) -> None:
        """
        Обрабатывает отдельную новостную статью.

        Args:
            article: Словарь с данными статьи
            source: Источник новости
        """
        # Проверяем, обрабатывали ли мы уже эту новость
        url = article.get("url", "")
        if url in self.processed_news:
            return

        # Извлекаем данные из статьи
        title = article.get("title", "")
        content = article.get("content", "")
        published_at = article.get("published_at")

        # Проверяем свежесть новости
        if published_at:
            try:
                # Преобразуем строку в datetime
                if isinstance(published_at, str):
                    published_at = datetime.fromisoformat(
                        published_at.replace("Z", "+00:00")
                    )

                # Проверяем, не старше ли новость заданного времени
                if datetime.now(published_at.tzinfo) - published_at > timedelta(
                    hours=self.news_relevance_time
                ):
                    logger.debug("Пропуск устаревшей новости: {title}" %)
                    return
            except Exception as e:
                logger.warning("Ошибка при проверке даты публикации: {str(e)}" %)

        # Анализируем новость для каждого символа
        affected_symbols = []
        signals = []

        for symbol in self.symbols:
            # Извлекаем базовую валюту из символа (например, BTC из BTC/USDT)
            base_currency = symbol.split("/")[0]

            # Проверяем наличие ключевых слов
            positive_score = self._check_keywords(
                title, content, self.positive_keywords.get(base_currency, [])
            )
            negative_score = self._check_keywords(
                title, content, self.negative_keywords.get(base_currency, [])
            )

            # Если есть совпадения ключевых слов
            if positive_score > 0 or negative_score > 0:
                affected_symbols.append(symbol)

                # Определяем тип сигнала
                signal_type = "buy" if positive_score > negative_score else "sell"
                confidence = max(positive_score, negative_score)

                signals.append(
                    {
                        "symbol": symbol,
                        "signal_type": signal_type,
                        "confidence": confidence,
                        "source": source,
                        "title": title,
                        "url": url,
                        "published_at": published_at,
                        "timestamp": time.time(),
                    }
                )

                # Обновляем статистику
                if signal_type == "buy":
                    self.stats["positive_signals"] += 1
                else:
                    self.stats["negative_signals"] += 1

        # Если новость затрагивает какие-либо символы, сохраняем ее
        if affected_symbols:
            logger.info("Обнаружена значимая новость: {title}" %)
            logger.info("Затронутые символы: {affected_symbols}" %)

            # Сохраняем новость
            self.processed_news[url] = {
                "title": title,
                "content": content,
                "source": source,
                "url": url,
                "published_at": published_at,
                "affected_symbols": affected_symbols,
                "signals": signals,
                "processed_at": time.time(),
            }

            # Добавляем сигналы в очередь
            for signal in signals:
                self.pending_signals.append(signal)

            # Обновляем статистику
            self.stats["news_processed"] += 1

            # Отправляем уведомление
            await self._send_news_notification(title, affected_symbols, source, url)

    def _check_keywords(self, title: str, content: str, keywords: List[str]) -> int:
        """
        Проверяет наличие ключевых слов в тексте.

        Args:
            title: Заголовок статьи
            content: Содержимое статьи
            keywords: Список ключевых слов для проверки

        Returns:
            Оценка релевантности (количество найденных ключевых слов)
        """
        if not keywords:
            return 0

        # Приводим к нижнему регистру
        title_lower = title.lower()
        content_lower = content.lower()

        # Проверяем наличие ключевых слов
        score = 0

        for keyword in keywords:
            # Проверяем в заголовке (с большим весом)
            if keyword.lower() in title_lower:
                score += 2
            # Проверяем в содержимом
            elif keyword.lower() in content_lower:
                score += 1

        return score

    @async_handle_error
    async def _send_news_notification(
        self, title: str, symbols: List[str], source: str, url: str
    ) -> None:
        """
        Отправляет уведомление о найденной новости.

        Args:
            title: Заголовок новости
            symbols: Список затронутых символов
            source: Источник новости
            url: URL новости
        """
        message = (
            f"📰 Новая важная новость:\n\n"
            f"{title}\n\n"
            f"Источник: {source}\n"
            f"Затронутые символы: {', '.join(symbols)}\n"
            f"URL: {url}"
        )

        await send_trading_signal(message)

    @async_handle_error
    async def _process_news_signals(self) -> None:
        """
        Обрабатывает накопленные новостные сигналы.
        """
        if not self.pending_signals:
            return

        # Копируем список сигналов, чтобы можно было удалять из оригинала
        signals_to_process = self.pending_signals.copy()

        for signal in signals_to_process:
            # Проверяем, можно ли выполнить сигнал
            if await self._can_execute_signal(signal):
                # Выполняем сигнал
                if await self._execute_news_signal(signal):
                    # Удаляем из списка ожидающих
                    if signal in self.pending_signals:
                        self.pending_signals.remove(signal)

                    # Добавляем в активные сигналы
                    self.active_news_signals[signal["symbol"]] = signal

                    # Обновляем статистику
                    self.stats["signals_executed"] += 1
            else:
                # Проверяем, не устарел ли сигнал
                if time.time() - signal["timestamp"] > self.news_relevance_time * 3600:
                    # Удаляем устаревший сигнал
                    if signal in self.pending_signals:
                        self.pending_signals.remove(signal)

    @async_handle_error
    async def _can_execute_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Проверяет, можно ли выполнить новостной сигнал.

        Args:
            signal: Словарь с данными сигнала

        Returns:
            True, если сигнал можно выполнить, иначе False
        """
        symbol = signal["symbol"]

        # Проверяем, есть ли уже открытая позиция по этому символу
        if symbol in self.open_positions:
            return False

        # Проверяем, есть ли уже активный сигнал по этому символу
        if symbol in self.active_news_signals:
            return False

        # Проверяем количество открытых позиций
        if len(self.open_positions) >= self.max_positions:
            return False

        # Проверяем уровень уверенности
        if signal["confidence"] < 2:  # Минимальный порог уверенности
            return False

        return True

    @async_handle_error
    async def _execute_news_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Выполняет новостной торговый сигнал.

        Args:
            signal: Словарь с данными сигнала

        Returns:
            True, если сигнал выполнен успешно, иначе False
        """
        symbol = signal["symbol"]
        signal_type = signal["signal_type"]

        try:
            # Получаем текущую цену
            ticker = await fetch_ticker(self.exchange_id, symbol)
            if not ticker:
                logger.warning("Не удалось получить тикер для {symbol}" %)
                return False

            current_price = ticker.get("last", 0)
            if current_price <= 0:
                logger.warning("Некорректная цена для {symbol}: {current_price}" %)
                return False

            # Определяем сторону ордера
            side = "buy" if signal_type == "buy" else "sell"

            # Рассчитываем стоп-лосс и тейк-профит
            if side == "buy":
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
            else:  # sell
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)

            # Рассчитываем размер позиции
            account_balance = self.config.get(
                "account_balance", 10000.0
            )  # По умолчанию 10000
            position_size = account_balance * self.position_size_pct
            quantity = position_size / current_price

            # Выполняем ордер
            order_result = await self.order_executor.execute_order(
                symbol=symbol,
                side=side,
                amount=quantity,
                order_type="market",
                exchange_id=self.exchange_id,
            )

            if not order_result.success:
                logger.error(
                    f"Ошибка при выполнении ордера для {symbol}: {order_result.error}"
                )
                return False

            # Создаем запись о позиции
            self.open_positions[symbol] = {
                "symbol": symbol,
                "side": "long" if side == "buy" else "short",
                "entry_price": current_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": time.time(),
                "signal": signal,
            }

            logger.info(
                f"Выполнен новостной сигнал для {symbol}: {side} по цене {current_price}"
            )

            # Отправляем уведомление
            message = (
                f"🔔 Новостной сигнал выполнен:\n\n"
                f"Символ: {symbol}\n"
                f"Действие: {'Покупка' if side == 'buy' else 'Продажа'}\n"
                f"Цена: {current_price}\n"
                f"Стоп-лосс: {stop_loss}\n"
                f"Тейк-профит: {take_profit}\n\n"
                f"Новость: {signal['title']}"
            )

            await send_trading_signal(message)

            return True

        except Exception as e:
            logger.error(
                f"Ошибка при выполнении новостного сигнала для {symbol}: {str(e)}"
            )
            return False

    @async_handle_error
    async def _check_news_positions(self) -> None:
        """
        Проверяет состояние открытых позиций, основанных на новостных сигналах.
        """
        # Копируем ключи, чтобы можно было удалять элементы во время итерации
        symbols = list(self.open_positions.keys())

        for symbol in symbols:
            if symbol not in self.open_positions:
                continue

            position = self.open_positions[symbol]

            # Проверяем, была ли позиция открыта на основе новостного сигнала
            if "signal" not in position:
                continue

            # Получаем текущую цену
            ticker = await fetch_ticker(self.exchange_id, symbol)
            if not ticker:
                continue

            current_price = ticker.get("last", 0)
            if current_price <= 0:
                continue

            # Получаем детали позиции
            entry_price = position["entry_price"]
            side = position["side"]
            stop_loss = position["stop_loss"]
            take_profit = position["take_profit"]

            # Проверяем условия выхода из позиции
            exit_reason = None

            if side == "long":
                # Проверяем стоп-лосс
                if current_price <= stop_loss:
                    exit_reason = "stop_loss"
                # Проверяем тейк-профит
                elif current_price >= take_profit:
                    exit_reason = "take_profit"
            else:  # short
                # Проверяем стоп-лосс
                if current_price >= stop_loss:
                    exit_reason = "stop_loss"
                # Проверяем тейк-профит
                elif current_price <= take_profit:
                    exit_reason = "take_profit"

            # Проверяем время жизни новостного сигнала
            signal_age = time.time() - position["entry_time"]
            max_signal_age = self.news_relevance_time * 3600  # в секундах

            if signal_age > max_signal_age:
                # Если прошло больше заданного времени, закрываем позицию
                exit_reason = "time_expired"

            # Если есть причина для выхода, закрываем позицию
            if exit_reason:
                logger.info("Закрытие позиции по {symbol} (причина: {exit_reason})" %)

                # Определяем сторону для закрытия (противоположную открытию)
                close_side = "sell" if side == "long" else "buy"

                # Выполняем ордер закрытия
                order_result = await self.order_executor.execute_order(
                    symbol=symbol,
                    side=close_side,
                    amount=position["quantity"],
                    order_type="market",
                    exchange_id=self.exchange_id,
                )

                if order_result.success:
                    # Рассчитываем прибыль/убыток
                    if side == "long":
                        pnl = (current_price - entry_price) * position["quantity"]
                        pnl_pct = (current_price / entry_price - 1) * 100
                    else:  # short
                        pnl = (entry_price - current_price) * position["quantity"]
                        pnl_pct = (entry_price / current_price - 1) * 100

                    # Записываем результат сделки
                    trade_result = {
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": current_price,
                        "quantity": position["quantity"],
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "entry_time": position["entry_time"],
                        "exit_time": time.time(),
                        "news_signal": position["signal"],
                    }

                    # Обновляем статистику
                    self._update_stats(trade_result)

                    # Удаляем позицию и связанный сигнал
                    del self.open_positions[symbol]

                    if symbol in self.active_news_signals:
                        del self.active_news_signals[symbol]

                    # Отправляем уведомление
                    message = (
                        f"🔔 Позиция закрыта:\n\n"
                        f"Символ: {symbol}\n"
                        f"Действие: {'Продажа' if close_side == 'sell' else 'Покупка'}\n"
                        f"Цена входа: {entry_price}\n"
                        f"Цена выхода: {current_price}\n"
                        f"P&L: {pnl:.2f} ({pnl_pct:.2f}%)\n"
                        f"Причина: {exit_reason}"
                    )

                    await send_trading_signal(message)

                    logger.info(
                        f"Позиция по {symbol} закрыта: "
                        f"P&L={pnl:.2f} ({pnl_pct:.2f}%), причина={exit_reason}"
                    )
                else:
                    logger.error(
                        f"Ошибка при закрытии позиции по {symbol}: {order_result.error}"
                    )

    async def _execute_bot_step(self) -> None:
        """
        Выполняет один шаг работы бота.
        """
        # Обрабатываем накопленные новостные сигналы
        await self._process_news_signals()

        # Проверяем состояние открытых позиций
        await self._check_news_positions()

    def _update_stats(self, trade_result: Dict[str, Any]) -> None:
        """
        Обновляет статистику бота.

        Args:
            trade_result: Результат сделки
        """
        super()._update_stats(trade_result["pnl"], trade_result["pnl"] > 0)
