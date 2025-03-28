#!/usr/bin/env python
"""
Консольный интерфейс для управления торговым ботом.
Предоставляет команды для запуска, остановки, мониторинга и настройки бота.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

# Добавляем директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from main.application import Application
from trading.trading_bot import TradingBot
from utils.logging_utils import setup_logger

logger = setup_logger("console")


class ConsoleInterface:
    """Консольный интерфейс для управления торговым ботом"""

    def __init__(self):
        """Инициализирует консольный интерфейс"""
        self.config = get_config()
        self.app: Optional[Application] = None
        self.bots: Dict[str, TradingBot] = {}

    async def initialize(self):
        """Инициализирует приложение"""
        self.app = Application()
        await self.app.initialize()

    async def shutdown(self):
        """Останавливает приложение"""
        if self.app:
            await self.app.shutdown()

    async def start_bot(self, symbol: str, strategy_id: str = None):
        """
        Запускает торгового бота для указанного символа и стратегии

        Args:
            symbol: Торговая пара (например, BTC/USDT)
            strategy_id: ID стратегии
        """
        if not self.app:
            await self.initialize()

        # Получаем конфигурацию для бота
        bot_config = self.config.get_bot_config(symbol, strategy_id)

        # Создаем и запускаем бота
        bot = await self.app.create_bot(bot_config)
        await bot.start()

        # Сохраняем ссылку на бота
        self.bots[bot.bot_id] = bot

        logger.info(f"Bot started for {symbol} with strategy {strategy_id}")
        return bot

    async def stop_bot(self, bot_id: str):
        """
        Останавливает торгового бота

        Args:
            bot_id: ID бота
        """
        if bot_id in self.bots:
            await self.bots[bot_id].stop()
            logger.info(f"Bot {bot_id} stopped")
        else:
            logger.error(f"Bot {bot_id} not found")

    async def list_bots(self):
        """
        Выводит список запущенных ботов и их состояние

        Returns:
            List[Dict]: Список ботов с их параметрами
        """
        result = []
        for bot_id, bot in self.bots.items():
            info = bot.get_info()
            result.append({
                "bot_id": bot_id,
                "symbol": info["symbol"],
                "exchange": info["exchange_id"],
                "state": info["state"],
                "strategy": info["strategy_id"],
                "pnl": info["stats"]["total_pnl"],
                "trades": info["stats"]["trades_count"]
            })
        return result

    async def show_performance(self, bot_id: str):
        """
        Выводит информацию о производительности бота

        Args:
            bot_id: ID бота

        Returns:
            Dict: Статистика производительности
        """
        if bot_id in self.bots:
            info = self.bots[bot_id].get_info()
            return info["stats"]
        else:
            logger.error(f"Bot {bot_id} not found")
            return None

    async def backtest(self, symbol: str, strategy_id: str, start_date: str, end_date: str):
        """
        Запускает бэктестирование для указанного символа и стратегии

        Args:
            symbol: Торговая пара
            strategy_id: ID стратегии
            start_date: Начальная дата в формате YYYY-MM-DD
            end_date: Конечная дата в формате YYYY-MM-DD

        Returns:
            Dict: Результаты бэктестирования
        """
        if not self.app:
            await self.initialize()

        result = await self.app.run_backtest(symbol, strategy_id, start_date, end_date)
        return result

    async def optimize(self, symbol: str, strategy_id: str, start_date: str, end_date: str):
        """
        Оптимизирует параметры стратегии

        Args:
            symbol: Торговая пара
            strategy_id: ID стратегии
            start_date: Начальная дата в формате YYYY-MM-DD
            end_date: Конечная дата в формате YYYY-MM-DD

        Returns:
            Dict: Результаты оптимизации
        """
        if not self.app:
            await self.initialize()

        result = await self.app.run_optimization(symbol, strategy_id, start_date, end_date)
        return result


async def main():
    """Основная функция консольного интерфейса"""
    parser = argparse.ArgumentParser(description="Trading Bot Console Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Команда start
    start_parser = subparsers.add_parser("start", help="Start a trading bot")
    start_parser.add_argument("symbol", type=str, help="Trading symbol (e.g., BTC/USDT)")
    start_parser.add_argument("--strategy", "-s", type=str, help="Strategy ID")

    # Команда stop
    stop_parser = subparsers.add_parser("stop", help="Stop a trading bot")
    stop_parser.add_argument("bot_id", type=str, help="Bot ID to stop")

    # Команда list
    subparsers.add_parser("list", help="List running bots")

    # Команда info
    info_parser = subparsers.add_parser("info", help="Show bot performance")
    info_parser.add_argument("bot_id", type=str, help="Bot ID to show info for")

    # Команда backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.add_argument("symbol", type=str, help="Trading symbol (e.g., BTC/USDT)")
    backtest_parser.add_argument("strategy", type=str, help="Strategy ID")
    backtest_parser.add_argument("--start", "-s", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", "-e", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")

    # Команда optimize
    optimize_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters")
    optimize_parser.add_argument("symbol", type=str, help="Trading symbol (e.g., BTC/USDT)")
    optimize_parser.add_argument("strategy", type=str, help="Strategy ID")
    optimize_parser.add_argument("--start", "-s", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    optimize_parser.add_argument("--end", "-e", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    console = ConsoleInterface()

    try:
        if args.command == "start":
            bot = await console.start_bot(args.symbol, args.strategy)
            print(f"Bot started: {bot.bot_id}")

        elif args.command == "stop":
            await console.stop_bot(args.bot_id)
            print(f"Bot {args.bot_id} stopped")

        elif args.command == "list":
            bots = await console.list_bots()
            if not bots:
                print("No running bots")
            else:
                print(f"{'BOT ID':<10} {'SYMBOL':<10} {'EXCHANGE':<10} {'STATE':<10} {'STRATEGY':<15} {'PNL':<10} {'TRADES':<6}")
                print("-" * 80)
                for bot in bots:
                    print(f"{bot['bot_id']:<10} {bot['symbol']:<10} {bot['exchange']:<10} {bot['state']:<10} {bot['strategy']:<15} {bot['pnl']:<10.2f} {bot['trades']:<6}")

        elif args.command == "info":
            stats = await console.show_performance(args.bot_id)
            if stats:
                print(f"Performance for Bot {args.bot_id}:")
                print(f"Total PnL: {stats['total_pnl']:.2f}")
                print(f"Win Rate: {stats['win_rate'] * 100:.2f}%")
                print(f"Profit Factor: {stats['profit_factor']:.2f}")
                print(f"Max Drawdown: {stats['max_drawdown_pct']:.2f}%")
                print(f"Trades: {stats['trades_count']} (Win: {stats['winning_trades']}, Loss: {stats['losing_trades']})")
                print(f"Average Win: {stats['average_win']:.2f}")
                print(f"Average Loss: {stats['average_loss']:.2f}")

        elif args.command == "backtest":
            result = await console.backtest(args.symbol, args.strategy, args.start, args.end)
            if result:
                print(f"Backtest Results for {args.symbol} with {args.strategy}:")
                print(f"Net Profit: {result['net_profit']:.2f}")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Win Rate: {result['win_rate'] * 100:.2f}%")
                print(f"Profit Factor: {result['profit_factor']:.2f}")
                print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
                print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")

        elif args.command == "optimize":
            result = await console.optimize(args.symbol, args.strategy, args.start, args.end)
            if result and 'best_params' in result:
                print(f"Optimization Results for {args.symbol} with {args.strategy}:")
                print("Best Parameters:")
                for param, value in result['best_params'].items():
                    print(f"  {param}: {value}")
                print(f"Fitness: {result['best_fitness']:.4f}")
                print(f"Net Profit: {result['best_result']['net_profit']:.2f}")
                print(f"Win Rate: {result['best_result']['win_rate'] * 100:.2f}%")
                print(f"Profit Factor: {result['best_result']['profit_factor']:.2f}")
        else:
            parser.print_help()

    finally:
        await console.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
