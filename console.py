"""
Консольный интерфейс для управления торговым ботом.
"""
# Стандартные библиотеки
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Библиотеки проекта
from project.config import get_config
from project.utils.error_handler import handle_error

# Настройка логгера
from project.utils.logging_utils import get_logger
logger = get_logger(__name__)

# Глобальные константы (должны быть в верхнем регистре)
MARKET_DATA = None
ORDER_EXECUTOR = None
STRATEGY_MANAGER = None
ARBITRAGE_CORE = None
API_RUNNER = None

# Получаем конфигурацию
config = get_config()

def initialize_components():
    """Инициализация основных компонентов системы"""
    try:
        # Импорт внутри функции, чтобы избежать циклических импортов
        from project.data.market_data import MarketData
        from project.trade_executor.order_executor import OrderExecutor
        from project.bots.strategies.strategy_manager import StrategyManager
        from project.bots.arbitrage.core import ArbitrageCore

        global MARKET_DATA, ORDER_EXECUTOR, STRATEGY_MANAGER, ARBITRAGE_CORE
        
        MARKET_DATA = MarketData()
        ORDER_EXECUTOR = OrderExecutor()
        STRATEGY_MANAGER = StrategyManager()
        ARBITRAGE_CORE = ArbitrageCore()
        
        logger.info("Компоненты системы инициализированы")
    except Exception as e:
        logger.error("Ошибка инициализации компонентов: %s", str(e))
        raise

def show_help():
    """Показывает справку по командам"""
    # Просто заглушка, никаких действий не требуется
    pass

def handle_command(command, arg=None):
    """
    Обработчик команд консоли
    
    Args:
        command: Команда для выполнения
        arg: Аргументы команды
    """
    logger.info("Обработка команды: %s", command)
    # Здесь должна быть реализация обработки команд
    # ...existing code...

# Другие функции консоли
# ...existing code...

import asyncio
import logging
import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime
import cmd
import shlex

from project.config import get_config
from project.utils.logging_utils import get_logger, setup_logging
from project.utils.error_handler import async_handle_error
from project.data.market_data import MarketData
from project.trade_executor.order_executor import OrderExecutor
from project.bots.bot_manager import BotManager
from project.bots.strategies.strategy_manager import StrategyManager
from project.bots.arbitrage.core import ArbitrageCore
from project.bots.arbitrage.multi_exchange import MultiExchangeArbitrage
from project.api.rest_api import start_api_server, stop_api_server
from project.utils.notify import send_trading_signal

# Настройка логирования
logger = get_logger(__name__)

# Глобальные объекты
config = get_config()
market_data = MarketData.get_instance()
order_executor = OrderExecutor.get_instance()
bot_manager = BotManager.get_instance()
strategy_manager = StrategyManager.get_instance()
arbitrage_core = ArbitrageCore.get_instance()
api_runner = None


# Класс консольного интерфейса
class TradingBotConsole(cmd.Cmd):
    """
    Класс интерактивного консольного интерфейса.
    """

    intro = """
    ========================================
    Trading Bot System Console
    Type 'help' or '?' to list commands.
    Type 'exit' or 'quit' to exit.
    ========================================
    """
    prompt = "trading> "

    def __init__(self):
        """
        Инициализирует консольный интерфейс.
        """
        super().__init__()
        self.loop = asyncio.get_event_loop()
        self.multi_exchange_arbitrage = None

    def default(self, line):
        """
        Обрабатывает неизвестные команды.
        """
        print(f"Unknown command: {line}")
        return False

    def emptyline(self):
        """
        Обрабатывает пустые строки.
        """
        pass

    def do_exit(self, arg):
        """
        Выход из консоли.

        Использование: exit
        """
        return self.do_quit(arg)

    def do_quit(self, arg):
        """
        Выход из консоли.

        Использование: quit
        """
        print("Exiting...")
        return True

    def do_market(self, arg):
        """
        Получает рыночные данные.

        Использование: market [тип_данных] [символ] [биржа]
        Типы данных: ticker, orderbook, ohlcv

        Примеры:
          market ticker BTC/USDT binance
          market orderbook ETH/USDT
          market ohlcv SOL/USDT binance 1h 10
        """
        args = shlex.split(arg)

        if not args:
            print("Please specify data type (ticker, orderbook, ohlcv)")
            return

        data_type = args[0].lower()

        # Получаем символ (со значением по умолчанию)
        symbol = args[1] if len(args) > 1 else "BTC/USDT"

        # Получаем биржу (со значением по умолчанию)
        exchange = args[2] if len(args) > 2 else "binance"

        # Обрабатываем разные типы данных
        if data_type == "ticker":
            result = self.loop.run_until_complete(
                market_data.get_ticker(exchange, symbol)
            )
            if result:
                print(f"Ticker for {symbol} on {exchange}:")
                print(json.dumps(result, indent=2))
            else:
                print(f"Failed to get ticker for {symbol} on {exchange}")

        elif data_type == "orderbook":
            # Получаем глубину (со значением по умолчанию)
            depth = int(args[3]) if len(args) > 3 else 10

            result = self.loop.run_until_complete(
                market_data.get_orderbook(exchange, symbol, limit=depth)
            )
            if result:
                print(f"Order book for {symbol} on {exchange} (depth: {depth}):")

                # Форматируем вывод ордербука для удобства
                if "bids" in result and "asks" in result:
                    print("Bids:")
                    for i, bid in enumerate(result["bids"][:5]):
                        print(f"  {i+1}. Price: {bid[0]}, Amount: {bid[1]}")

                    print("Asks:")
                    for i, ask in enumerate(result["asks"][:5]):
                        print(f"  {i+1}. Price: {ask[0]}, Amount: {ask[1]}")
                else:
                    print(json.dumps(result, indent=2))
            else:
                print(f"Failed to get order book for {symbol} on {exchange}")

        elif data_type == "ohlcv":
            # Получаем таймфрейм (со значением по умолчанию)
            timeframe = args[3] if len(args) > 3 else "1h"

            # Получаем лимит (со значением по умолчанию)
            limit = int(args[4]) if len(args) > 4 else 10

            result = self.loop.run_until_complete(
                market_data.get_ohlcv(exchange, symbol, timeframe, limit=limit)
            )
            if result is not None and not result.empty:
                print(
                    f"OHLCV for {symbol} on {exchange} ({timeframe}, {limit} candles):"
                )

                # Форматируем вывод OHLCV для удобства
                for index, row in result.iterrows():
                    time_str = index.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"  {time_str} | O: {row['open']:.8f} | H: {row['high']:.8f} | L: {row['low']:.8f} | C: {row['close']:.8f} | V: {row['volume']:.8f}"
                    )
            else:
                print(f"Failed to get OHLCV data for {symbol} on {exchange}")

        else:
            print(f"Unknown data type: {data_type}")
            print("Available types: ticker, orderbook, ohlcv")

    def do_order(self, arg):
        """
        Создает торговый ордер.

        Использование: order [действие] [символ] [количество] [тип] [цена] [биржа]

        Примеры:
          order buy BTC/USDT 0.001 market
          order sell ETH/USDT 0.1 limit 3000 binance
        """
        args = shlex.split(arg)

        if len(args) < 3:
            print("Usage: order [action] [symbol] [amount] [type] [price] [exchange]")
            return

        # Получаем основные параметры
        action = args[0].lower()
        symbol = args[1]
        amount = float(args[2])

        # Проверяем действие
        if action not in ["buy", "sell"]:
            print(f"Unknown action: {action}")
            print("Available actions: buy, sell")
            return

        # Получаем опциональные параметры
        order_type = args[3].lower() if len(args) > 3 else "market"
        price = float(args[4]) if len(args) > 4 and order_type == "limit" else None
        exchange = args[5] if len(args) > 5 else "binance"

        # Создаем ордер
        result = self.loop.run_until_complete(
            order_executor.execute_order(
                symbol=symbol,
                side=action,
                amount=amount,
                order_type=order_type,
                price=price,
                exchange_id=exchange,
            )
        )

        if result.success:
            print(f"Order created successfully:")
            print(f"  Exchange: {exchange}")
            print(f"  Symbol: {symbol}")
            print(f"  Action: {action}")
            print(f"  Amount: {amount}")
            print(f"  Type: {order_type}")
            if price:
                print(f"  Price: {price}")
            print(f"  Order ID: {result.order_id}")
            print(f"  Status: {result.status}")
        else:
            print(f"Failed to create order: {result.error}")

    def do_bot(self, arg):
        """
        Управляет торговыми ботами.

        Использование: bot [действие] [параметры]
        Действия: list, start, stop, state

        Примеры:
          bot list
          bot start basic TestBot binance BTC/USDT,ETH/USDT
          bot start news NewsBot binance BTC/USDT,ETH/USDT
          bot stop [id_бота]
          bot state [id_бота]
        """
        args = shlex.split(arg)

        if not args:
            print("Please specify action (list, start, stop, state)")
            return

        action = args[0].lower()

        if action == "list":
            # Получаем список ботов
            bots = bot_manager.get_bots()

            if not bots:
                print("No running bots")
                return

            print(f"Running bots ({len(bots)}):")
            for bot_id, bot in bots.items():
                print(f"  ID: {bot_id}")
                print(f"  Name: {bot.name}")
                print(f"  Type: {bot.__class__.__name__}")
                print(f"  Exchange: {bot.exchange_id}")
                print(f"  Symbols: {', '.join(bot.symbols)}")
                print(f"  Status: {bot.get_status()}")
                print("")

        elif action == "start":
            # Запускаем нового бота
            if len(args) < 4:
                print("Usage: bot start [type] [name] [exchange] [symbols]")
                return

            bot_type = args[1]
            bot_name = args[2]
            exchange = args[3]
            symbols = args[4].split(",") if len(args) > 4 else ["BTC/USDT"]

            # Проверяем тип бота
            if bot_type not in ["basic", "news", "signal", "grid"]:
                print(f"Unknown bot type: {bot_type}")
                print("Available types: basic, news, signal, grid")
                return

            # Запускаем бота
            bot_id = self.loop.run_until_complete(
                bot_manager.start_bot(
                    bot_type=bot_type,
                    name=bot_name,
                    exchange_id=exchange,
                    symbols=symbols,
                )
            )

            if bot_id:
                print(f"Bot started successfully:")
                print(f"  ID: {bot_id}")
                print(f"  Name: {bot_name}")
                print(f"  Type: {bot_type}")
                print(f"  Exchange: {exchange}")
                print(f"  Symbols: {', '.join(symbols)}")
            else:
                print("Failed to start bot")

        elif action == "stop":
            # Останавливаем бота
            if len(args) < 2:
                print("Usage: bot stop [bot_id]")
                return

            bot_id = args[1]

            # Останавливаем бота
            success = self.loop.run_until_complete(bot_manager.stop_bot(bot_id))

            if success:
                print(f"Bot {bot_id} stopped successfully")
            else:
                print(f"Failed to stop bot {bot_id}")

        elif action == "state":
            # Получаем состояние бота
            if len(args) < 2:
                print("Usage: bot state [bot_id]")
                return

            bot_id = args[1]

            # Получаем состояние бота
            state = self.loop.run_until_complete(bot_manager.get_bot_state(bot_id))

            if state:
                print(f"State of bot {bot_id}:")
                print(json.dumps(state, indent=2))
            else:
                print(f"Failed to get state of bot {bot_id}")

        else:
            print(f"Unknown action: {action}")
            print("Available actions: list, start, stop, state")

    def do_strategy(self, arg):
        """
        Управляет торговыми стратегиями.

        Использование: strategy [действие] [параметры]
        Действия: list, start, stop, state

        Примеры:
          strategy list
          strategy start main binance BTC/USDT,ETH/USDT 15m,1h
          strategy stop [id_стратегии]
          strategy state [id_стратегии]
        """
        args = shlex.split(arg)

        if not args:
            print("Please specify action (list, start, stop, state)")
            return

        action = args[0].lower()

        if action == "list":
            # Получаем список стратегий
            available_strategies = strategy_manager.get_available_strategies()
            running_strategies = strategy_manager.get_running_strategies()

            print(f"Available strategies ({len(available_strategies)}):")
            for strategy in available_strategies:
                print(f"  {strategy}")

            print(f"\nRunning strategies ({len(running_strategies)}):")
            for strategy in running_strategies:
                print(f"  ID: {strategy['id']}")
                print(f"  Name: {strategy['name']}")
                print(f"  Class: {strategy['class']}")
                print(f"  Exchange: {strategy['exchange']}")
                print(f"  Symbols: {', '.join(strategy['symbols'])}")
                print(f"  Timeframes: {', '.join(strategy['timeframes'])}")
                print(f"  Status: {strategy['status']}")
                print("")

        elif action == "start":
            # Запускаем новую стратегию
            if len(args) < 2:
                print(
                    "Usage: strategy start [strategy_name] [exchange] [symbols] [timeframes]"
                )
                return

            strategy_name = args[1]
            exchange = args[2] if len(args) > 2 else "binance"
            symbols = args[3].split(",") if len(args) > 3 else ["BTC/USDT"]
            timeframes = args[4].split(",") if len(args) > 4 else ["1h"]

            # Запускаем стратегию
            strategy_id = self.loop.run_until_complete(
                strategy_manager.start_strategy(
                    strategy_name=strategy_name,
                    exchange_id=exchange,
                    symbols=symbols,
                    timeframes=timeframes,
                )
            )

            if strategy_id:
                print(f"Strategy started successfully:")
                print(f"  ID: {strategy_id}")
                print(f"  Name: {strategy_name}")
                print(f"  Exchange: {exchange}")
                print(f"  Symbols: {', '.join(symbols)}")
                print(f"  Timeframes: {', '.join(timeframes)}")
            else:
                print("Failed to start strategy")

        elif action == "stop":
            # Останавливаем стратегию
            if len(args) < 2:
                print("Usage: strategy stop [strategy_id]")
                return

            strategy_id = args[1]

            # Останавливаем стратегию
            success = self.loop.run_until_complete(
                strategy_manager.stop_strategy(strategy_id)
            )

            if success:
                print(f"Strategy {strategy_id} stopped successfully")
            else:
                print(f"Failed to stop strategy {strategy_id}")

        elif action == "state":
            # Получаем состояние стратегии
            if len(args) < 2:
                print("Usage: strategy state [strategy_id]")
                return

            strategy_id = args[1]

            # Получаем состояние стратегии
            state = self.loop.run_until_complete(
                strategy_manager.get_strategy_state(strategy_id)
            )

            if state:
                print(f"State of strategy {strategy_id}:")
                print(json.dumps(state, indent=2))
            else:
                print(f"Failed to get state of strategy {strategy_id}")

        else:
            print(f"Unknown action: {action}")
            print("Available actions: list, start, stop, state")

    def do_arbitrage(self, arg):
        """
        Управляет арбитражной торговлей.

        Использование: arbitrage [действие] [параметры]
        Действия: scan, start, stop, stats

        Примеры:
          arbitrage scan BTC/USDT,ETH/USDT binance,kucoin
          arbitrage start
          arbitrage stop
          arbitrage stats
        """
        args = shlex.split(arg)

        if not args:
            print("Please specify action (scan, start, stop, stats)")
            return

        action = args[0].lower()

        if action == "scan":
            # Сканируем арбитражные возможности
            symbols = args[1].split(",") if len(args) > 1 else ["BTC/USDT", "ETH/USDT"]
            exchanges = (
                args[2].split(",") if len(args) > 2 else ["binance", "kucoin", "huobi"]
            )

            print(
                f"Scanning arbitrage opportunities for {', '.join(symbols)} on {', '.join(exchanges)}..."
            )

            # Сканируем возможности
            opportunities = self.loop.run_until_complete(
                arbitrage_core.scan_opportunities(symbols, exchanges)
            )

            if opportunities:
                print(f"Found {len(opportunities)} arbitrage opportunities:")
                for i, opp in enumerate(opportunities):
                    print(f"  {i+1}. {opp.symbol}:")
                    print(f"     Buy on: {opp.buy_exchange} at {opp.buy_price:.8f}")
                    print(f"     Sell on: {opp.sell_exchange} at {opp.sell_price:.8f}")
                    print(f"     Price diff: {opp.price_diff_pct:.2%}")
                    print(f"     Profit after fees: {opp.profit_margin_pct:.2%}")
                    print("")
            else:
                print("No arbitrage opportunities found")

        elif action == "start":
            # Запускаем систему мультибиржевого арбитража
            if self.multi_exchange_arbitrage is not None:
                print("Arbitrage system is already running")
                return

            # Создаем и запускаем систему арбитража
            exchanges = (
                args[1].split(",") if len(args) > 1 else ["binance", "kucoin", "huobi"]
            )
            symbols = (
                args[2].split(",")
                if len(args) > 2
                else ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
            )
            min_profit = float(args[3]) if len(args) > 3 else 0.01

            self.multi_exchange_arbitrage = MultiExchangeArbitrage(
                exchanges=exchanges, symbols=symbols, min_profit_pct=min_profit
            )

            success = self.loop.run_until_complete(
                self.multi_exchange_arbitrage.start()
            )

            if success:
                print(f"Arbitrage system started successfully:")
                print(f"  Exchanges: {', '.join(exchanges)}")
                print(f"  Symbols: {', '.join(symbols)}")
                print(f"  Min profit: {min_profit:.2%}")
            else:
                print("Failed to start arbitrage system")
                self.multi_exchange_arbitrage = None

        elif action == "stop":
            # Останавливаем систему мультибиржевого арбитража
            if self.multi_exchange_arbitrage is None:
                print("Arbitrage system is not running")
                return

            success = self.loop.run_until_complete(self.multi_exchange_arbitrage.stop())

            if success:
                print("Arbitrage system stopped successfully")
                self.multi_exchange_arbitrage = None
            else:
                print("Failed to stop arbitrage system")

        elif action == "stats":
            # Получаем статистику системы мультибиржевого арбитража
            if self.multi_exchange_arbitrage is None:
                print("Arbitrage system is not running")
                return

            stats = self.loop.run_until_complete(
                self.multi_exchange_arbitrage.get_stats()
            )

            print("Arbitrage system statistics:")
            print(f"  Running: {stats['running']}")
            print(f"  Monitored symbols: {stats['monitored_symbols']}")
            print(f"  Active opportunities: {stats['active_opportunities']}")
            print(f"  Total opportunities found: {stats['total_opportunities_found']}")
            print(f"  Total arbitrages executed: {stats['total_arbitrages_executed']}")
            print(f"  Total profit: {stats['total_profit']:.8f} USD")
            print(f"  Total volume: {stats['total_volume']:.8f} USD")
            print(f"  Uptime: {stats['uptime']:.2f} seconds")

            if stats["current_opportunities"]:
                print("\nCurrent opportunities:")
                for i, opp in enumerate(stats["current_opportunities"]):
                    print(f"  {i+1}. {opp['symbol']}:")
                    print(
                        f"     Buy on: {opp['buy_exchange']} at {opp['buy_price']:.8f}"
                    )
                    print(
                        f"     Sell on: {opp['sell_exchange']} at {opp['sell_price']:.8f}"
                    )
                    print(f"     Profit: {opp['profit_pct']:.2%}")
                    print(f"     Age: {opp['age']:.2f} seconds")

        else:
            print(f"Unknown action: {action}")
            print("Available actions: scan, start, stop, stats")

    def do_api(self, arg):
        """
        Управляет API-сервером.

        Использование: api [действие] [порт]
        Действия: start, stop, status

        Примеры:
          api start 8080
          api stop
          api status
        """
        global api_runner

        args = shlex.split(arg)

        if not args:
            print("Please specify action (start, stop, status)")
            return

        action = args[0].lower()

        if action == "start":
            # Запускаем API-сервер
            if api_runner is not None:
                print("API server is already running")
                return

            port = int(args[1]) if len(args) > 1 else 8080

            print(f"Starting API server on port {port}...")

            # Запускаем сервер
            api_runner, site = self.loop.run_until_complete(start_api_server(port=port))

            print(f"API server started at http://0.0.0.0:{port}")

        elif action == "stop":
            # Останавливаем API-сервер
            if api_runner is None:
                print("API server is not running")
                return

            self.loop.run_until_complete(stop_api_server(api_runner))
            api_runner = None

            print("API server stopped")

        elif action == "status":
            # Проверяем статус API-сервера
            if api_runner is not None:
                print("API server is running")
            else:
                print("API server is not running")

        else:
            print(f"Unknown action: {action}")
            print("Available actions: start, stop, status")

    def do_notify(self, arg):
        """
        Отправляет уведомление.

        Использование: notify [сообщение]

        Примеры:
          notify Тестовое уведомление
        """
        if not arg:
            print("Please specify message")
            return

        success = self.loop.run_until_complete(send_trading_signal(arg))

        if success:
            print("Notification sent successfully")
        else:
            print("Failed to send notification")


# Точка входа
def main():
    """
    Основная функция консольного интерфейса.
    """
    # Настройка логирования
    setup_logging()

    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Trading Bot System Console")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--api", action="store_true", help="Start API server automatically"
    )
    parser.add_argument(
        "--api-port", type=int, default=8080, help="Port for API server"
    )

    # Парсим аргументы
    args = parser.parse_args()

    # Устанавливаем уровень логирования
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Создаем консольный интерфейс
    console = TradingBotConsole()

    # Если указан флаг --api, запускаем API-сервер
    if args.api:
        console.loop.run_until_complete(console.do_api(f"start {args.api_port}"))

    try:
        # Запускаем консоль
        console.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Останавливаем API-сервер, если он запущен
        if api_runner is not None:
            console.loop.run_until_complete(stop_api_server(api_runner))

        # Останавливаем систему арбитража, если она запущена
        if console.multi_exchange_arbitrage is not None:
            console.loop.run_until_complete(console.multi_exchange_arbitrage.stop())

        # Останавливаем всех ботов
        console.loop.run_until_complete(bot_manager.stop_all_bots())

        # Останавливаем все стратегии
        console.loop.run_until_complete(strategy_manager.shutdown())


if __name__ == "__main__":
    main()
