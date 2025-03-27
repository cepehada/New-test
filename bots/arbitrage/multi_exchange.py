"""
Модуль для арбитража между несколькими биржами.
Реализует стратегию межбиржевого арбитража.
"""

# Стандартные импорты
import time
import asyncio
from typing import Dict, List, Any

# Внутренние импорты
from project.config import get_config
from project.bots.base_bot import BaseBot
from project.data.market_data import MarketData
from project.utils.logging_utils import get_logger
from project.utils.error_handler import async_handle_error
from project.utils.ccxt_exchanges import fetch_balance
from project.bots.arbitrage.core import ArbitrageCore, ArbitrageOpportunity
from project.bots.arbitrage.utils import calculate_max_trade_sizes

logger = get_logger(__name__)


class MultiExchangeArbitrage(BaseBot):
    """
    Бот для межбиржевого арбитража.
    Отслеживает ценовые разницы между биржами и выполняет арбитражные сделки.
    """

    def __init__(self, config=None, name="arbitrage_multi_exchange"):
        """
        Инициализирует арбитражного бота.

        Args:
            config (Dict, optional): Конфигурация бота. По умолчанию None.
            name (str, optional): Имя бота. По умолчанию "arbitrage_multi_exchange".
        """
        super().__init__(config=config, name=name)

        # Арбитражное ядро и данные рынка
        self.arbitrage_core = ArbitrageCore(self.config)
        self.market_data = MarketData.get_instance()

        # Настройки арбитража
        arb_config = self.config.get("arbitrage", {})
        self.min_profit_pct = arb_config.get("min_profit_pct", 0.8)
        self.min_volume_usd = arb_config.get("min_volume_usd", 10.0)
        self.check_interval = arb_config.get("check_interval", 5.0)
        self.max_active_opportunities = arb_config.get("max_active_opportunities", 3)

        # Биржи и торговые пары
        self.exchanges = arb_config.get("exchanges", ["binance", "kucoin", "okx"])
        self.symbols = arb_config.get("symbols", [])

        # Данные о возможностях и сделках
        self.current_opportunities = {}
        self.completed_arbitrages = []
        self.active_monitors = {}
        self.running = False

    @async_handle_error
    async def start(self):
        """
        Запускает арбитражного бота.
        """
        if self.running:
            logger.info("Арбитражный бот уже запущен")
            return

        self.running = True
        logger.info(
            "Запуск арбитражного бота между %s для %d символов",
            ", ".join(self.exchanges),
            len(self.symbols),
        )

        # Начинаем главный цикл
        asyncio.create_task(self._main_loop())

    @async_handle_error
    async def stop(self):
        """
        Останавливает арбитражного бота.
        """
        if not self.running:
            logger.info("Арбитражный бот уже остановлен")
            return

        logger.info("Остановка арбитражного бота")
        self.running = False

        # Ждем завершения всех задач
        for symbol, task in self.active_monitors.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Арбитражный бот остановлен")

    @async_handle_error
    async def _main_loop(self):
        """
        Основной цикл работы арбитражного бота.
        """
        try:
            logger.info("Начало основного цикла арбитражного бота")

            while self.running:
                # Поиск новых возможностей
                opportunities = await self.arbitrage_core.check_arbitrage_opportunities(
                    symbols=self.symbols,
                    exchanges=self.exchanges,
                    min_profit_pct=self.min_profit_pct,
                    min_volume_usd=self.min_volume_usd,
                )

                # Обрабатываем найденные возможности
                for opportunity in opportunities:
                    symbol = opportunity.symbol

                    # Пропускаем, если возможность уже отслеживается
                    if symbol in self.current_opportunities:
                        continue

                    # Проверяем возможность
                    is_valid = await self.arbitrage_core.verify_opportunity(
                        opportunity=opportunity,
                        min_volume=self.min_volume_usd / opportunity.buy_price,
                        max_age_seconds=10.0,
                    )

                    if is_valid:
                        logger.info(
                            "Найдена арбитражная возможность для %s: "
                            "Покупка на %s по %.6f, продажа на %s по %.6f, "
                            "прибыль: %.2f%%",
                            symbol,
                            opportunity.buy_exchange,
                            opportunity.buy_price,
                            opportunity.sell_exchange,
                            opportunity.sell_price,
                            opportunity.profit_margin_pct,
                        )

                        # Проверяем балансы и размеры сделок
                        trade_sizes = await self.arbitrage_core.check_balances(
                            opportunity
                        )

                        if trade_sizes:
                            # Добавляем возможность и запускаем монитор
                            self.current_opportunities[symbol] = opportunity

                            # Ограничиваем количество активных мониторов
                            if (
                                len(self.active_monitors)
                                < self.max_active_opportunities
                            ):
                                self.active_monitors[symbol] = asyncio.create_task(
                                    self._monitor_opportunity(symbol, opportunity)
                                )

                # Ждем следующей итерации
                await asyncio.sleep(self.check_interval)

        except Exception as e:
            logger.error("Ошибка в основном цикле арбитражного бота: %s", str(e))

    @async_handle_error
    async def _monitor_opportunity(
        self, symbol: str, opportunity: ArbitrageOpportunity
    ):
        """
        Мониторит и исполняет конкретную арбитражную возможность.

        Args:
            symbol: Символ торговой пары
            opportunity: Объект арбитражной возможности
        """
        try:
            logger.info("Начало мониторинга арбитражной возможности для %s", symbol)

            monitor_interval = 1.0  # Интервал проверки в секундах
            max_monitoring_time = 60.0  # Максимальное время мониторинга в секундах
            start_time = time.time()

            while self.running and (time.time() - start_time) < max_monitoring_time:
                # Проверяем актуальность возможности
                is_valid = await self.arbitrage_core.verify_opportunity(
                    opportunity=opportunity,
                    min_volume=self.min_volume_usd / opportunity.buy_price,
                )

                if not is_valid:
                    logger.info(
                        "Арбитражная возможность для %s больше не актуальна", symbol
                    )
                    break

                # Пытаемся выполнить арбитражные сделки
                executed = await self._execute_if_possible(opportunity)

                if executed:
                    logger.info("Арбитражные сделки для %s успешно выполнены", symbol)
                    break

                # Ждем следующей итерации
                await asyncio.sleep(monitor_interval)

            # Удаляем возможность из текущих, если она еще там
            if symbol in self.current_opportunities:
                del self.current_opportunities[symbol]

            logger.info("Завершение мониторинга арбитражной возможности для %s", symbol)

        except Exception as e:
            logger.error("Ошибка при мониторинге арбитража для %s: %s", symbol, str(e))

            # Удаляем возможность при ошибке
            if symbol in self.current_opportunities:
                del self.current_opportunities[symbol]

    @async_handle_error
    async def _execute_if_possible(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Выполняет арбитражные сделки, если это возможно.

        Args:
            opportunity: Арбитражная возможность

        Returns:
            True, если сделки выполнены успешно, иначе False
        """
        try:
            # Проверяем, достаточно ли у нас балансов
            trade_sizes = await self.arbitrage_core.check_balances(opportunity)

            if not trade_sizes:
                return False

            # Подготавливаем параметры сделки
            buy_amount = trade_sizes["buy_amount"]
            sell_amount = trade_sizes["sell_amount"]

            logger.info(
                "Выполнение арбитража для %s: покупка %.6f на %s по %.6f, "
                "продажа %.6f на %s по %.6f",
                opportunity.symbol,
                buy_amount,
                opportunity.buy_exchange,
                opportunity.buy_price,
                sell_amount,
                opportunity.sell_exchange,
                opportunity.sell_price,
            )

            # Получаем ордер-исполнитель
            from project.trade_executor.order_executor import OrderExecutor

            order_executor = OrderExecutor.get_instance()

            # Выполняем сделки одновременно
            buy_result, sell_result = await asyncio.gather(
                order_executor.execute_order(
                    symbol=opportunity.symbol,
                    side="buy",
                    amount=buy_amount,
                    price=opportunity.buy_price,
                    exchange_id=opportunity.buy_exchange,
                    order_type="limit",
                ),
                order_executor.execute_order(
                    symbol=opportunity.symbol,
                    side="sell",
                    amount=sell_amount,
                    price=opportunity.sell_price,
                    exchange_id=opportunity.sell_exchange,
                    order_type="limit",
                ),
            )

            # Проверяем результаты
            success = buy_result.success and sell_result.success

            if success:
                self.completed_arbitrages.append(
                    {
                        "symbol": opportunity.symbol,
                        "buy_exchange": opportunity.buy_exchange,
                        "sell_exchange": opportunity.sell_exchange,
                        "buy_price": opportunity.buy_price,
                        "sell_price": opportunity.sell_price,
                        "buy_amount": trade_sizes["buy_amount"],
                        "sell_amount": trade_sizes["sell_amount"],
                        "buy_cost": trade_sizes["buy_cost"],
                        "sell_proceeds": trade_sizes["sell_proceeds"],
                        "profit": trade_sizes["sell_proceeds"]
                        - trade_sizes["buy_cost"],
                        "profit_pct": opportunity.profit_margin_pct,
                        "timestamp": time.time(),
                    }
                )

                # Удаляем из текущих возможностей
                if opportunity.symbol in self.current_opportunities:
                    del self.current_opportunities[opportunity.symbol]

                logger.info(
                    "Успешно выполнен арбитраж для %s: прибыль %.2f USD",
                    opportunity.symbol,
                    trade_sizes["sell_proceeds"] - trade_sizes["buy_cost"],
                )

                return True

            return False

        except Exception as e:
            logger.error(
                "Ошибка при выполнении арбитража для %s: %s", opportunity.symbol, str(e)
            )
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """
        Получает статистику арбитражной системы.

        Returns:
            Словарь со статистикой
        """
        stats = dict(self.stats)

        # Добавляем дополнительную информацию
        stats["running"] = self.running
        stats["monitored_symbols"] = len(self.active_monitors)
        stats["active_opportunities"] = len(self.current_opportunities)
        stats["uptime"] = (
            time.time() - self.stats["start_time"]
            if self.stats["start_time"] > 0
            else 0
        )

        if stats["total_arbitrages_executed"] > 0:
            stats["avg_profit"] = (
                stats["total_profit"] / stats["total_arbitrages_executed"]
            )
            stats["avg_volume"] = (
                stats["total_volume"] / stats["total_arbitrages_executed"]
            )
        else:
            stats["avg_profit"] = 0
            stats["avg_volume"] = 0

        return stats

    @async_handle_error
    async def _check_exchanges_are_ready(
        self, opportunity: ArbitrageOpportunity
    ) -> bool:
        """
        Проверяет готовность бирж для выполнения арбитража.

        Args:
            opportunity: Объект арбитражной возможности

        Returns:
            True если биржи готовы, иначе False
        """
        try:
            # Проверяем доступность бирж
            exchanges_to_check = [opportunity.buy_exchange, opportunity.sell_exchange]

            for exchange in exchanges_to_check:
                # Проверяем наличие доступного баланса
                balance = await fetch_balance(exchange)

                if not balance:
                    logger.info("Не удалось получить баланс на бирже %s", exchange)
                    return False

            return True

        except Exception as e:
            logger.error(
                "Ошибка при проверке готовности бирж для %s: %s",
                opportunity.symbol,
                str(e),
            )
            return False

    @async_handle_error
    async def _get_market_data(self, exchange_id: str, symbol: str) -> Dict[str, Any]:
        """
        Получает данные о рынке с указанной биржи.

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ торговой пары

        Returns:
            Словарь с данными рынка
        """
        try:
            # Получаем текущий тикер
            ticker = await self.market_data.get_ticker(exchange_id, symbol)

            # Получаем стакан заказов
            orderbook = await self.market_data.get_orderbook(exchange_id, symbol)

            # Получаем последние сделки
            trades = await self.market_data.get_recent_trades(exchange_id, symbol)

            # Получаем баланс
            balance = await fetch_balance(exchange_id)

            return {
                "ticker": ticker,
                "orderbook": orderbook,
                "trades": trades,
                "balance": balance,
            }

        except Exception as e:
            logger.error(
                "Ошибка при получении рыночных данных для %s на %s: %s",
                symbol,
                exchange_id,
                str(e),
            )
            return {}

    @async_handle_error
    async def _execute_arbitrage(
        self, opportunity: ArbitrageOpportunity, trade_sizes: Dict[str, float]
    ) -> bool:
        """
        Выполняет арбитражные сделки.

        Args:
            opportunity: Арбитражная возможность
            trade_sizes: Размеры сделок

        Returns:
            True, если сделки выполнены успешно, иначе False
        """
        try:
            # Логика исполнения арбитражных сделок
            logger.info(
                "Выполнение арбитражных сделок для %s: "
                "Покупка %.6f на %s по %.6f, "
                "Продажа %.6f на %s по %.6f",
                opportunity.symbol,
                trade_sizes["buy_amount"],
                opportunity.buy_exchange,
                opportunity.buy_price,
                trade_sizes["sell_amount"],
                opportunity.sell_exchange,
                opportunity.sell_price,
            )

            # Получаем ордер-исполнитель
            from project.trade_executor.order_executor import OrderExecutor

            order_executor = OrderExecutor.get_instance()

            # Выполняем покупку
            buy_result = await order_executor.market_buy(
                symbol=opportunity.symbol,
                amount=trade_sizes["buy_amount"],
                exchange_id=opportunity.buy_exchange,
            )

            if not buy_result.success:
                logger.error(
                    "Ошибка при выполнении покупки для %s: %s",
                    opportunity.symbol,
                    buy_result.error,
                )
                return False

            # Выполняем продажу
            sell_result = await order_executor.market_sell(
                symbol=opportunity.symbol,
                amount=trade_sizes["sell_amount"],
                exchange_id=opportunity.sell_exchange,
            )

            if not sell_result.success:
                logger.error(
                    "Ошибка при выполнении продажи для %s: %s",
                    opportunity.symbol,
                    sell_result.error,
                )
                return False

            # Обновляем статистику
            self.stats["total_arbitrages_executed"] += 1
            self.stats["total_profit"] += (
                trade_sizes["sell_proceeds"] - trade_sizes["buy_cost"]
            )
            self.stats["total_volume"] += trade_sizes["buy_cost"]

            return True

        except Exception as e:
            logger.error(
                "Ошибка при выполнении арбитража для %s: %s", opportunity.symbol, str(e)
            )
            return False
