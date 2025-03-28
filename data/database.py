import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import aiosqlite
import pandas as pd
from project.utils.logging_utils import setup_logger

logger = setup_logger("database")


class Database:
    """Класс для работы с базой данных"""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует соединение с базой данных

        Args:
            config: Конфигурация базы данных
        """
        self.config = config
        self.db_path = config.get(
            "db_path", os.environ.get("DB_PATH", "data/trading.db")
        )
        self._connection = None
        self._lock = asyncio.Lock()

        # Создаем директорию для БД, если её нет
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        logger.info(f"Database initialized: {self.db_path}")

    async def connect(self):
        """Устанавливает соединение с базой данных"""
        if self._connection is None:
            try:
                self._connection = await aiosqlite.connect(self.db_path)
                # Включаем поддержку внешних ключей
                await self._connection.execute("PRAGMA foreign_keys = ON")
                # Установка режима записи
                await self._connection.execute("PRAGMA journal_mode = WAL")
                # Включаем кэширование
                await self._connection.execute("PRAGMA cache_size = 10000")
                await self._connection.commit()

                logger.info(f"Connected to database: {self.db_path}")

                # Инициализируем таблицы
                await self._init_tables()

            except Exception as e:
                logger.error(f"Error connecting to database: {str(e)}")
                if self._connection:
                    await self._connection.close()
                    self._connection = None
                raise

    async def disconnect(self):
        """Закрывает соединение с базой данных"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Disconnected from database")

    async def _init_tables(self):
        """Инициализирует таблицы в базе данных"""
        # Создаем таблицы, если их нет
        async with self._lock:
            try:
                # Таблица для исторических данных OHLCV
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    UNIQUE(symbol, exchange, timestamp, timeframe)
                )
                """
                )

                # Таблица для сделок
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    cost REAL NOT NULL,
                    fee_cost REAL,
                    fee_currency TEXT,
                    order_id TEXT,
                    type TEXT,
                    takerOrMaker TEXT,
                    UNIQUE(trade_id, exchange)
                )
                """
                )

                # Таблица для ордеров
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL,
                    amount REAL NOT NULL,
                    filled REAL,
                    remaining REAL,
                    cost REAL,
                    average REAL,
                    status TEXT NOT NULL,
                    fee_cost REAL,
                    fee_currency TEXT,
                    params TEXT,
                    is_closed BOOLEAN NOT NULL,
                    client_order_id TEXT,
                    strategy_id TEXT,
                    UNIQUE(order_id, exchange)
                )
                """
                )

                # Таблица для сигналов
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    strength REAL NOT NULL,
                    price REAL,
                    timestamp INTEGER NOT NULL,
                    expiration INTEGER,
                    params TEXT,
                    strategy_id TEXT,
                    is_executed BOOLEAN DEFAULT FALSE,
                    execution_timestamp INTEGER,
                    order_id TEXT,
                    UNIQUE(signal_id)
                )
                """
                )

                # Таблица для позиций
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    amount REAL NOT NULL,
                    open_timestamp INTEGER NOT NULL,
                    close_timestamp INTEGER,
                    close_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    fees REAL,
                    params TEXT,
                    strategy_id TEXT,
                    UNIQUE(position_id)
                )
                """
                )

                # Таблица для оптимизации стратегий
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    best_parameters TEXT NOT NULL,
                    best_fitness REAL NOT NULL,
                    best_metrics TEXT NOT NULL,
                    optimization_stats TEXT NOT NULL,
                    parameter_ranges TEXT NOT NULL,
                    UNIQUE(optimization_id)
                )
                """
                )

                # Таблица для параметров стратегий
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS strategy_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    description TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(strategy_id, timestamp)
                )
                """
                )

                # Таблица для бэктестов
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_timestamp INTEGER NOT NULL,
                    end_timestamp INTEGER NOT NULL,
                    parameters TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    trades TEXT NOT NULL,
                    equity_curve TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    UNIQUE(backtest_id)
                )
                """
                )

                # Таблица для состояния ботов
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS bot_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    config TEXT NOT NULL,
                    last_signal_timestamp INTEGER,
                    last_trade_timestamp INTEGER,
                    is_active BOOLEAN DEFAULT TRUE,
                    UNIQUE(bot_id)
                )
                """
                )

                # Таблица для событий
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    data TEXT NOT NULL,
                    source TEXT,
                    UNIQUE(event_id)
                )
                """
                )

                # Таблица для логов
                await self._connection.execute(
                    """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    level TEXT NOT NULL,
                    logger TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT
                )
                """
                )

                # Индексы для ускорения запросов
                await self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv (symbol, timeframe)"
                )
                await self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades (symbol)"
                )
                await self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol)"
                )
                await self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol)"
                )
                await self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions (symbol)"
                )

                await self._connection.commit()

                logger.info("Database tables initialized")

            except Exception as e:
                logger.error(f"Error initializing database tables: {str(e)}")
                raise

    async def save_order(self, order_data: Dict[str, Any]) -> bool:
        """
        Сохраняет информацию о ордере в базу данных

        Args:
            order_data: Данные ордера для сохранения

        Returns:
            bool: True если успешно, иначе False
        """
        try:
            # Проверяем, что соединение установлено
            if not self._connection:
                await self.connect()

            # Экранируем JSON параметры
            params_json = "{}"
            if "params" in order_data:
                params_json = json.dumps(order_data["params"])

            # Создаем заготовку SQL-запроса
            async with self._lock:
                await self._connection.execute(
                    """
                    INSERT INTO orders (
                        order_id, symbol, exchange, timestamp, type, side,
                        price, amount, filled, remaining, cost, average,
                        status, fee_cost, fee_currency, params, is_closed, client_order_id, strategy_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(order_id, exchange) DO UPDATE SET
                        status = excluded.status,
                        filled = excluded.filled,
                        remaining = excluded.remaining,
                        average = excluded.average,
                        cost = excluded.cost,
                        fee_cost = excluded.fee_cost,
                        is_closed = excluded.is_closed
                    """,
                    (
                        order_data.get("order_id", ""),
                        order_data.get("symbol", ""),
                        order_data.get("exchange", ""),
                        order_data.get("timestamp", int(time.time() * 1000)),
                        order_data.get("type", ""),
                        order_data.get("side", ""),
                        order_data.get("price", 0.0),
                        order_data.get("amount", 0.0),
                        order_data.get("filled", 0.0),
                        order_data.get("remaining", 0.0),
                        order_data.get("cost", 0.0),
                        order_data.get("average", 0.0),
                        order_data.get("status", "open"),
                        order_data.get("fee_cost", 0.0),
                        order_data.get("fee_currency", ""),
                        params_json,
                        order_data.get("is_closed", False),
                        order_data.get("client_order_id", ""),
                        order_data.get("strategy_id", ""),
                    ),
                )
                await self._connection.commit()

            logger.debug(f"Ордер сохранен/обновлен: {order_data.get('order_id')}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при сохранении ордера: {str(e)}")
            return False
