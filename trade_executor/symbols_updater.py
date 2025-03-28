import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

from project.utils.logging_utils import setup_logger
from project.utils.ccxt_exchanges import get_exchange_instance
from project.data.market_data import MarketDataProvider
from project.config import get_config

logger = setup_logger("symbols_updater")


class SymbolsUpdater:
    """Модуль для автоматического обновления торговых пар"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        self.market_data = MarketDataProvider()
        self.update_interval = self.config.get(
            "symbols_update_interval", 3600
        )  # 1 час по умолчанию
        self.min_volume = self.config.get(
            "min_daily_volume", 1000000
        )  # Минимальный суточный объем
        self.top_pairs_count = self.config.get(
            "top_pairs_count", 20
        )  # Количество выбираемых пар
        self.last_update = None
        self._symbols_cache = []
        self._update_task = None
        self._stop_requested = False

    async def start(self):
        """Запускает автоматическое обновление пар"""
        if self._update_task is not None:
            logger.warning("Обновление пар уже запущено")
            return

        self._stop_requested = False
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("Запущено автоматическое обновление пар")

    async def stop(self):
        """Останавливает автоматическое обновление пар"""
        if self._update_task is None:
            logger.warning("Обновление пар не запущено")
            return

        self._stop_requested = True
        self._update_task.cancel()
        try:
            await self._update_task
        except asyncio.CancelledError:
            pass
        self._update_task = None
        logger.info("Остановлено автоматическое обновление пар")

    async def _update_loop(self):
        """Фоновая задача для обновления пар"""
        while not self._stop_requested:
            try:
                await self.update_symbols()
                self.last_update = datetime.now()
                logger.info("Обновление пар выполнено успешно")

                # Ждем до следующего обновления
                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                logger.info("Задача обновления пар отменена")
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле обновления пар: {str(e)}")
                await asyncio.sleep(60)  # Подождем 1 минуту перед повторной попыткой

    async def update_symbols(self) -> List[str]:
        """Обновляет список торговых пар на основе ликвидности и объема"""
        # Получаем все доступные пары на биржах
        all_pairs = []

        exchanges = self.config.get("exchanges", ["binance", "bybit", "kucoin"])

        for exchange_id in exchanges:
            try:
                logger.info(f"Получение данных с биржи {exchange_id}...")
                exchange = await get_exchange_instance(exchange_id)
                markets = await exchange.fetch_markets()

                # Фильтруем только спотовые рынки или фьючерсы
                filtered_markets = []
                for market in markets:
                    if self.config.get("include_futures", True) and market.get(
                        "future", False
                    ):
                        filtered_markets.append(market)
                    elif self.config.get("include_spot", True) and not market.get(
                        "future", False
                    ):
                        filtered_markets.append(market)

                # Собираем информацию о парах
                symbols_processed = 0
                for market in filtered_markets:
                    try:
                        symbol = market["symbol"]

                        # Пропускаем символы, не содержащие основные стейблкоины
                        stablecoins = ["USDT", "BUSD", "USDC"]
                        base_included = False
                        for stable in stablecoins:
                            if stable in symbol:
                                base_included = True
                                break

                        if not base_included:
                            continue

                        # Получаем тикер для оценки объема
                        ticker = await exchange.fetch_ticker(symbol)

                        if ticker:
                            volume_24h = ticker.get("quoteVolume", 0)
                            price_change = ticker.get("percentage", 0) or 0

                            all_pairs.append(
                                {
                                    "symbol": symbol,
                                    "exchange": exchange_id,
                                    "volume_24h": volume_24h,
                                    "price_change": price_change,
                                    "last_price": ticker.get("last", 0),
                                    "timestamp": datetime.now().timestamp(),
                                }
                            )

                            symbols_processed += 1

                            # Делаем небольшую паузу, чтобы не перегрузить API
                            if symbols_processed % 10 == 0:
                                await asyncio.sleep(0.5)

                    except Exception as e:
                        logger.warning(
                            f"Ошибка при обработке символа {market.get('symbol')}: {str(e)}"
                        )

                logger.info(
                    f"Обработано {symbols_processed} символов на бирже {exchange_id}"
                )

                # Закрываем соединение с биржей
                await exchange.close()

            except Exception as e:
                logger.error(
                    f"Ошибка при получении данных с биржи {exchange_id}: {str(e)}"
                )

        # Сортируем пары по объему (от большего к меньшему)
        all_pairs.sort(key=lambda x: x["volume_24h"], reverse=True)

        # Фильтруем пары с минимальным объемом
        filtered_pairs = [p for p in all_pairs if p["volume_24h"] >= self.min_volume]

        # Берем топ-N пар
        top_pairs = filtered_pairs[: self.top_pairs_count]

        # Обновляем кеш
        self._symbols_cache = top_pairs

        # Сохраняем в файл для анализа
        self._save_to_file(top_pairs)

        # Возвращаем список символов
        symbols = [p["symbol"] for p in top_pairs]
        logger.info(f"Выбрано {len(symbols)} торговых пар: {symbols}")
        return symbols

    def get_recommended_symbols(self) -> List[str]:
        """Возвращает рекомендуемые торговые пары из кеша"""
        if not self._symbols_cache:
            return []
        return [p["symbol"] for p in self._symbols_cache]

    def _save_to_file(self, pairs: List[Dict[str, Any]]):
        """Сохраняет информацию о парах в файл"""
        try:
            with open("data/recommended_pairs.json", "w") as f:
                json.dump(pairs, f, indent=4)
        except Exception as e:
            logger.error(f"Ошибка при сохранении пар в файл: {str(e)}")
