"""
Модуль для мультибиржевого арбитража.
Предоставляет функциональность для отслеживания и исполнения арбитражных возможностей
между различными криптовалютными биржами.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from project.bots.arbitrage.core import ArbitrageCore, ArbitrageOpportunity
from project.config import get_config
from project.data.market_data import MarketData
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class MultiExchangeArbitrage:
    """
    Класс для выполнения мультибиржевого арбитража.
    """

    def __init__(
        self,
        exchanges: List[str] = None,
        symbols: List[str] = None,
        min_profit_pct: float = 0.01,
        max_trade_size: float = 100.0,
    ):
        """
        Инициализирует систему мультибиржевого арбитража.

        Args:
            exchanges: Список бирж для мониторинга
            symbols: Список символов для мониторинга
            min_profit_pct: Минимальный процент прибыли для выполнения арбитража
            max_trade_size: Максимальный размер сделки в базовой валюте
        """
        self.config = get_config()
        self.arbitrage_core = ArbitrageCore.get_instance()
        self.market_data = MarketData.get_instance()

        self.exchanges = exchanges or ["binance", "kucoin", "huobi", "okex"]
        self.symbols = symbols or [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "XRP/USDT",
            "ADA/USDT",
        ]
        self.min_profit_pct = min_profit_pct
        self.max_trade_size = max_trade_size

        # Внутренние состояния
        self.active_monitors = {}  # symbol -> task
        self.current_opportunities = {}  # symbol -> ArbitrageOpportunity
        self.completed_arbitrages = []  # список завершенных арбитражей
        self.running = False

        # Статистика
        self.stats = {
            "total_opportunities_found": 0,
            "total_arbitrages_executed": 0,
            "total_profit": 0.0,
            "total_volume": 0.0,
            "start_time": 0,
            "avg_opportunity_lifetime": 0.0,
        }

        logger.debug("Создан экземпляр MultiExchangeArbitrage")

    async def start(self) -> bool:
        """
        Запускает систему мультибиржевого арбитража.

        Returns:
            True, если запуск успешен, иначе False
        """
        if self.running:
            logger.warning("Система арбитража уже запущена")
            return False

        try:
            logger.info("Запуск системы мультибиржевого арбитража")

            # Обновляем время запуска
            self.stats["start_time"] = time.time()

            # Запускаем мониторы для всех символов
            for symbol in self.symbols:
                self._start_monitor(symbol)

            self.running = True

            logger.info(
                f"Система мультибиржевого арбитража запущена: "
                f"{len(self.symbols)} символов на {len(self.exchanges)} биржах"
            )

            return True

        except Exception as e:
            logger.error("Ошибка при запуске системы арбитража: {str(e)}" %)
            return False

    async def stop(self) -> bool:
        """
        Останавливает систему мультибиржевого арбитража.

        Returns:
            True, если остановка успешна, иначе False
        """
        if not self.running:
            logger.warning("Система арбитража не запущена")
            return False

        try:
            logger.info("Остановка системы мультибиржевого арбитража")

            # Отменяем все активные мониторы
            for symbol, task in self.active_monitors.items():
                if not task.done():
                    task.cancel()

            self.active_monitors = {}
            self.running = False

            logger.info("Система мультибиржевого арбитража остановлена")

            return True

        except Exception as e:
            logger.error("Ошибка при остановке системы арбитража: {str(e)}" %)
            return False

    def _start_monitor(self, symbol: str) -> None:
        """
        Запускает задачу для мониторинга арбитражных возможностей для символа.

        Args:
            symbol: Символ для мониторинга
        """
        if symbol in self.active_monitors and not self.active_monitors[symbol].done():
            logger.debug("Монитор для {symbol} уже запущен" %)
            return

        task = asyncio.create_task(self._monitor_symbol(symbol))
        self.active_monitors[symbol] = task
        logger.debug("Запущен монитор для {symbol}" %)

    @async_handle_error
    async def _monitor_symbol(self, symbol: str) -> None:
        """
        Мониторит арбитражные возможности для символа.

        Args:
            symbol: Символ для мониторинга
        """
        try:
            logger.debug("Начат мониторинг {symbol}" %)

            while True:
                if not self.running:
                    logger.debug("Мониторинг {symbol} остановлен" %)
                    break

                # Сканируем возможности
                opportunities = await self.arbitrage_core.scan_opportunities(
                    [symbol], self.exchanges
                )

                # Если найдены возможности
                if opportunities:
                    opportunity = opportunities[0]  # берем лучшую возможность

                    # Проверяем, достаточна ли прибыль
                    if opportunity.profit_margin_pct >= self.min_profit_pct:
                        self.stats["total_opportunities_found"] += 1

                        logger.info(
                            f"Найдена арбитражная возможность для {symbol}: "
                            f"купить на {opportunity.buy_exchange} по {opportunity.buy_price:.8f}, "
                            f"продать на {opportunity.sell_exchange} по {opportunity.sell_price:.8f}, "
                            f"прибыль: {opportunity.profit_margin_pct:.2%}"
                        )

                        # Сохраняем возможность
                        self.current_opportunities[symbol] = opportunity

                        # Проверяем возможность исполнения
                        await self._execute_if_possible(opportunity)

                # Ждем перед следующей проверкой
                await asyncio.sleep(5)  # проверяем каждые 5 секунд

        except asyncio.CancelledError:
            logger.debug("Мониторинг {symbol} отменен" %)
            raise
        except Exception as e:
            logger.error("Ошибка при мониторинге {symbol}: {str(e)}" %)

    @async_handle_error
    async def _execute_if_possible(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Проверяет и выполняет арбитражную возможность, если она возможна.

        Args:
            opportunity: Арбитражная возможность

        Returns:
            True, если арбитраж выполнен, иначе False
        """
        try:
            # Проверяем актуальность возможности
            is_valid = await self.arbitrage_core.verify_opportunity(opportunity)
            if not is_valid:
                logger.debug(
                    f"Возможность для {opportunity.symbol} больше не актуальна"
                )
                return False

            # Проверяем балансы
            trade_sizes = await self.arbitrage_core.check_balances(
                opportunity, min_trade_amount=10.0
            )
            if not trade_sizes:
                logger.debug(
                    f"Недостаточно балансов для арбитража {opportunity.symbol}"
                )
                return False

            # Ограничиваем размер сделки
            if trade_sizes["buy_cost"] > self.max_trade_size:
                # Корректируем размеры
                ratio = self.max_trade_size / trade_sizes["buy_cost"]
                trade_sizes["buy_amount"] *= ratio
                trade_sizes["sell_amount"] *= ratio
                trade_sizes["buy_cost"] *= ratio
                trade_sizes["sell_proceeds"] *= ratio

                opportunity.trade_sizes = trade_sizes

            # Выполняем арбитраж
            success = await self.arbitrage_core.execute_arbitrage(opportunity)

            if success:
                # Обновляем статистику
                self.stats["total_arbitrages_executed"] += 1
                self.stats["total_volume"] += trade_sizes["buy_cost"]
                self.stats["total_profit"] += (
                    trade_sizes["sell_proceeds"] - trade_sizes["buy_cost"]
                )

                # Добавляем в историю
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
                    f"Успешно выполнен арбитраж для {opportunity.symbol}: "
                    f"прибыль {(trade_sizes['sell_proceeds'] - trade_sizes['buy_cost']):.2f} USD"
                )

                return True

            return False

        except Exception as e:
            logger.error(
                f"Ошибка при выполнении арбитража для {opportunity.symbol}: {str(e)}"
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

        # Добавляем текущие возможности
        stats["current_opportunities"] = []
        for symbol, opportunity in self.current_opportunities.items():
            stats["current_opportunities"].append(
                {
                    "symbol": opportunity.symbol,
                    "buy_exchange": opportunity.buy_exchange,
                    "sell_exchange": opportunity.sell_exchange,
                    "buy_price": opportunity.buy_price,
                    "sell_price": opportunity.sell_price,
                    "profit_pct": opportunity.profit_margin_pct,
                    "age": time.time() - opportunity.timestamp,
                }
            )

        # Добавляем последние выполненные арбитражи (максимум 10)
        stats["recent_arbitrages"] = (
            self.completed_arbitrages[-10:] if self.completed_arbitrages else []
        )

        return stats

    async def add_symbol(self, symbol: str) -> bool:
        """
        Добавляет символ для мониторинга.

        Args:
            symbol: Символ для добавления

        Returns:
            True, если символ добавлен, иначе False
        """
        if symbol in self.symbols:
            logger.warning("Символ {symbol} уже отслеживается" %)
            return False

        try:
            # Добавляем символ в список
            self.symbols.append(symbol)

            # Если система запущена, запускаем монитор для нового символа
            if self.running:
                self._start_monitor(symbol)

            logger.info("Добавлен символ {symbol} для мониторинга" %)

            return True

        except Exception as e:
            logger.error("Ошибка при добавлении символа {symbol}: {str(e)}" %)
            return False

    async def remove_symbol(self, symbol: str) -> bool:
        """
        Удаляет символ из мониторинга.

        Args:
            symbol: Символ для удаления

        Returns:
            True, если символ удален, иначе False
        """
        if symbol not in self.symbols:
            logger.warning("Символ {symbol} не отслеживается" %)
            return False

        try:
            # Удаляем символ из списка
            self.symbols.remove(symbol)

            # Если есть активный монитор, отменяем его
            if symbol in self.active_monitors:
                task = self.active_monitors[symbol]
                if not task.done():
                    task.cancel()
                del self.active_monitors[symbol]

            # Удаляем из текущих возможностей
            if symbol in self.current_opportunities:
                del self.current_opportunities[symbol]

            logger.info("Удален символ {symbol} из мониторинга" %)

            return True

        except Exception as e:
            logger.error("Ошибка при удалении символа {symbol}: {str(e)}" %)
            return False

    async def add_exchange(self, exchange: str) -> bool:
        """
        Добавляет биржу для мониторинга.

        Args:
            exchange: Биржа для добавления

        Returns:
            True, если биржа добавлена, иначе False
        """
        if exchange in self.exchanges:
            logger.warning("Биржа {exchange} уже отслеживается" %)
            return False

        if exchange not in self.arbitrage_core.supported_exchanges:
            logger.warning("Биржа {exchange} не поддерживается" %)
            return False

        try:
            # Добавляем биржу в список
            self.exchanges.append(exchange)

            logger.info("Добавлена биржа {exchange} для мониторинга" %)

            return True

        except Exception as e:
            logger.error("Ошибка при добавлении биржи {exchange}: {str(e)}" %)
            return False

    async def remove_exchange(self, exchange: str) -> bool:
        """
        Удаляет биржу из мониторинга.

        Args:
            exchange: Биржа для удаления

        Returns:
            True, если биржа удалена, иначе False
        """
        if exchange not in self.exchanges:
            logger.warning("Биржа {exchange} не отслеживается" %)
            return False

        try:
            # Удаляем биржу из списка
            self.exchanges.remove(exchange)

            # Удаляем возможности, связанные с этой биржей
            for symbol, opportunity in list(self.current_opportunities.items()):
                if (
                    opportunity.buy_exchange == exchange
                    or opportunity.sell_exchange == exchange
                ):
                    del self.current_opportunities[symbol]

            logger.info("Удалена биржа {exchange} из мониторинга" %)

            return True

        except Exception as e:
            logger.error("Ошибка при удалении биржи {exchange}: {str(e)}" %)
            return False

    async def update_settings(
        self,
        min_profit_pct: Optional[float] = None,
        max_trade_size: Optional[float] = None,
    ) -> bool:
        """
        Обновляет настройки арбитражной системы.

        Args:
            min_profit_pct: Минимальный процент прибыли для арбитража
            max_trade_size: Максимальный размер сделки

        Returns:
            True, если настройки обновлены, иначе False
        """
        try:
            if min_profit_pct is not None:
                if min_profit_pct <= 0:
                    logger.warning(
                        f"Некорректное значение min_profit_pct: {min_profit_pct}"
                    )
                else:
                    self.min_profit_pct = min_profit_pct
                    logger.info("Обновлен min_profit_pct: {min_profit_pct}" %)

            if max_trade_size is not None:
                if max_trade_size <= 0:
                    logger.warning(
                        f"Некорректное значение max_trade_size: {max_trade_size}"
                    )
                else:
                    self.max_trade_size = max_trade_size
                    logger.info("Обновлен max_trade_size: {max_trade_size}" %)

            return True

        except Exception as e:
            logger.error("Ошибка при обновлении настроек: {str(e)}" %)
            return False
