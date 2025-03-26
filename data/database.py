import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import aiosqlite
import pandas as pd
import numpy as np
from pathlib import Path

from project.utils.logging_utils import setup_logger

logger = setup_logger("database")


class Database:
    """Класс для работы с базой данных SQLite"""

    def __init__(self, db_path: str = None):
        """
        Инициализирует соединение с базой данных

        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path or os.environ.get("DB_PATH", "data/trading.db")
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

    async def save_ohlcv(
        self, symbol: str, exchange: str, timeframe: str, data: List[List]
    ):
        """
        Сохраняет данные OHLCV в базу данных

        Args:
            symbol: Торговая пара
            exchange: Биржа
            timeframe: Временной интервал
            data: Данные OHLCV в формате [[timestamp, open, high, low, close, volume], ...]
        """
        await self.connect()

        async with self._lock:
            try:
                # Начинаем транзакцию
                await self._connection.execute("BEGIN TRANSACTION")

                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO ohlcv 
                (symbol, exchange, timestamp, timeframe, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Подготавливаем данные
                params = [
                    (
                        symbol,
                        exchange,
                        int(candle[0]),
                        timeframe,
                        float(candle[1]),
                        float(candle[2]),
                        float(candle[3]),
                        float(candle[4]),
                        float(candle[5]),
                    )
                    for candle in data
                ]

                # Выполняем запрос
                await self._connection.executemany(query, params)

                # Фиксируем транзакцию
                await self._connection.commit()

                logger.debug(
                    f"Saved {len(data)} OHLCV records for {symbol} {timeframe} on {exchange}"
                )

            except Exception as e:
                # Отменяем транзакцию в случае ошибки
                await self._connection.execute("ROLLBACK")
                logger.error(f"Error saving OHLCV data: {str(e)}")
                raise

    async def get_ohlcv(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Получает данные OHLCV из базы данных

        Args:
            symbol: Торговая пара
            exchange: Биржа
            timeframe: Временной интервал
            start_time: Начальное время (UNIX timestamp)
            end_time: Конечное время (UNIX timestamp)
            limit: Максимальное количество записей

        Returns:
            pd.DataFrame: DataFrame с данными OHLCV
        """
        await self.connect()

        try:
            # Формируем запрос
            query = """
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv 
            WHERE symbol = ? AND exchange = ? AND timeframe = ?
            """

            params = [symbol, exchange, timeframe]

            # Добавляем фильтрацию по времени
            if start_time:
                query += " AND timestamp >= ?"
                params.append(int(start_time))

            if end_time:
                query += " AND timestamp <= ?"
                params.append(int(end_time))

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp ASC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в DataFrame
            if rows:
                df = pd.DataFrame(
                    rows,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )

                # Преобразуем timestamp в datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                # Устанавливаем timestamp в качестве индекса
                df.set_index("timestamp", inplace=True)

                return df
            else:
                # Возвращаем пустой DataFrame с нужными колонками
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

        except Exception as e:
            logger.error(f"Error getting OHLCV data: {str(e)}")
            raise

    async def save_trade(self, trade: Dict):
        """
        Сохраняет сделку в базу данных

        Args:
            trade: Информация о сделке
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO trades 
                (trade_id, symbol, exchange, timestamp, side, price, amount, cost, 
                fee_cost, fee_currency, order_id, type, takerOrMaker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Извлекаем комиссию
                fee_cost = None
                fee_currency = None
                if "fee" in trade and trade["fee"] is not None:
                    fee_cost = trade["fee"].get("cost")
                    fee_currency = trade["fee"].get("currency")

                # Подготавливаем параметры
                params = (
                    trade.get("id"),
                    trade.get("symbol"),
                    trade.get("exchange", "unknown"),
                    int(trade.get("timestamp", 0)),
                    trade.get("side"),
                    float(trade.get("price", 0)),
                    float(trade.get("amount", 0)),
                    float(trade.get("cost", 0)),
                    fee_cost,
                    fee_currency,
                    trade.get("order"),
                    trade.get("type"),
                    trade.get("takerOrMaker"),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(f"Saved trade {trade.get('id')} for {trade.get('symbol')}")

            except Exception as e:
                logger.error(f"Error saving trade: {str(e)}")
                raise

    async def get_trades(
        self,
        symbol: str = None,
        exchange: str = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ) -> List[Dict]:
        """
        Получает сделки из базы данных

        Args:
            symbol: Торговая пара
            exchange: Биржа
            start_time: Начальное время (UNIX timestamp)
            end_time: Конечное время (UNIX timestamp)
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список сделок
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM trades"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)

            if exchange:
                conditions.append("exchange = ?")
                params.append(exchange)

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(int(start_time))

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(int(end_time))

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    trade_dict = dict(zip(columns, row))

                    # Преобразуем комиссию в словарь
                    if trade_dict.get("fee_cost") is not None and trade_dict.get(
                        "fee_currency"
                    ):
                        trade_dict["fee"] = {
                            "cost": trade_dict.pop("fee_cost"),
                            "currency": trade_dict.pop("fee_currency"),
                        }
                    else:
                        trade_dict.pop("fee_cost", None)
                        trade_dict.pop("fee_currency", None)
                        trade_dict["fee"] = None

                    result.append(trade_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting trades: {str(e)}")
            raise

    async def save_order(self, order: Dict):
        """
        Сохраняет ордер в базу данных

        Args:
            order: Информация об ордере
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO orders 
                (order_id, symbol, exchange, timestamp, type, side, price, amount, filled, remaining, 
                cost, average, status, fee_cost, fee_currency, params, is_closed, client_order_id, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Извлекаем комиссию
                fee_cost = None
                fee_currency = None
                if "fee" in order and order["fee"] is not None:
                    fee_cost = order["fee"].get("cost")
                    fee_currency = order["fee"].get("currency")

                # Сериализуем параметры
                params_json = (
                    json.dumps(order.get("params", {})) if "params" in order else None
                )

                # Подготавливаем параметры
                params = (
                    order.get("id", order.get("order_id")),
                    order.get("symbol"),
                    order.get("exchange", "unknown"),
                    int(
                        order.get(
                            "timestamp",
                            order.get("datetime", datetime.now().timestamp() * 1000),
                        )
                    ),
                    order.get("type"),
                    order.get("side"),
                    (
                        float(order.get("price", 0))
                        if order.get("price") is not None
                        else None
                    ),
                    float(order.get("amount", 0)),
                    float(order.get("filled", 0)),
                    float(order.get("remaining", 0)),
                    (
                        float(order.get("cost", 0))
                        if order.get("cost") is not None
                        else None
                    ),
                    (
                        float(order.get("average", 0))
                        if order.get("average") is not None
                        else None
                    ),
                    order.get("status"),
                    fee_cost,
                    fee_currency,
                    params_json,
                    bool(order.get("is_closed", False)),
                    order.get("clientOrderId", order.get("client_order_id")),
                    order.get("strategy_id"),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(
                    f"Saved order {order.get('id', order.get('order_id'))} for {order.get('symbol')}"
                )

            except Exception as e:
                logger.error(f"Error saving order: {str(e)}")
                raise

    async def get_orders(
        self,
        symbol: str = None,
        exchange: str = None,
        status: str = None,
        strategy_id: str = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ) -> List[Dict]:
        """
        Получает ордеры из базы данных

        Args:
            symbol: Торговая пара
            exchange: Биржа
            status: Статус ордера
            strategy_id: ID стратегии
            start_time: Начальное время (UNIX timestamp)
            end_time: Конечное время (UNIX timestamp)
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список ордеров
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM orders"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)

            if exchange:
                conditions.append("exchange = ?")
                params.append(exchange)

            if status:
                conditions.append("status = ?")
                params.append(status)

            if strategy_id:
                conditions.append("strategy_id = ?")
                params.append(strategy_id)

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(int(start_time))

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(int(end_time))

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    order_dict = dict(zip(columns, row))

                    # Преобразуем комиссию в словарь
                    if order_dict.get("fee_cost") is not None and order_dict.get(
                        "fee_currency"
                    ):
                        order_dict["fee"] = {
                            "cost": order_dict.pop("fee_cost"),
                            "currency": order_dict.pop("fee_currency"),
                        }
                    else:
                        order_dict.pop("fee_cost", None)
                        order_dict.pop("fee_currency", None)
                        order_dict["fee"] = None

                    # Десериализуем параметры
                    if order_dict.get("params"):
                        try:
                            order_dict["params"] = json.loads(order_dict["params"])
                        except json.JSONDecodeError:
                            order_dict["params"] = {}

                    result.append(order_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            raise

    async def save_signal(self, signal: Dict):
        """
        Сохраняет сигнал в базу данных

        Args:
            signal: Информация о сигнале
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO signals 
                (signal_id, symbol, direction, strength, price, timestamp, expiration, 
                params, strategy_id, is_executed, execution_timestamp, order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Сериализуем параметры
                params_json = (
                    json.dumps(signal.get("params", {})) if "params" in signal else None
                )

                # Получаем timestamp
                timestamp = signal.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = int(
                        datetime.fromisoformat(timestamp).timestamp() * 1000
                    )
                elif isinstance(timestamp, datetime):
                    timestamp = int(timestamp.timestamp() * 1000)

                # Получаем expiration
                expiration = signal.get("expiration")
                if expiration:
                    if isinstance(expiration, str):
                        expiration = int(
                            datetime.fromisoformat(expiration).timestamp() * 1000
                        )
                    elif isinstance(expiration, datetime):
                        expiration = int(expiration.timestamp() * 1000)

                # Подготавливаем параметры
                params = (
                    signal.get("id", signal.get("signal_id")),
                    signal.get("symbol"),
                    signal.get("direction"),
                    float(signal.get("strength", 1.0)),
                    (
                        float(signal.get("price", 0))
                        if signal.get("price") is not None
                        else None
                    ),
                    timestamp,
                    expiration,
                    params_json,
                    signal.get("strategy_id"),
                    bool(signal.get("is_executed", False)),
                    signal.get("execution_timestamp"),
                    signal.get("order_id"),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(
                    f"Saved signal {signal.get('id', signal.get('signal_id'))} for {signal.get('symbol')}"
                )

            except Exception as e:
                logger.error(f"Error saving signal: {str(e)}")
                raise

    async def get_signals(
        self,
        symbol: str = None,
        direction: str = None,
        strategy_id: str = None,
        is_executed: bool = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ) -> List[Dict]:
        """
        Получает сигналы из базы данных

        Args:
            symbol: Торговая пара
            direction: Направление (buy, sell, close)
            strategy_id: ID стратегии
            is_executed: Флаг исполнения
            start_time: Начальное время (UNIX timestamp)
            end_time: Конечное время (UNIX timestamp)
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список сигналов
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM signals"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)

            if direction:
                conditions.append("direction = ?")
                params.append(direction)

            if strategy_id:
                conditions.append("strategy_id = ?")
                params.append(strategy_id)

            if is_executed is not None:
                conditions.append("is_executed = ?")
                params.append(int(is_executed))

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(int(start_time))

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(int(end_time))

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    signal_dict = dict(zip(columns, row))

                    # Десериализуем параметры
                    if signal_dict.get("params"):
                        try:
                            signal_dict["params"] = json.loads(signal_dict["params"])
                        except json.JSONDecodeError:
                            signal_dict["params"] = {}

                    # Преобразуем timestamp в строку ISO
                    if signal_dict.get("timestamp"):
                        signal_dict["timestamp"] = datetime.fromtimestamp(
                            signal_dict["timestamp"] / 1000
                        ).isoformat()

                    # Преобразуем expiration в строку ISO
                    if signal_dict.get("expiration"):
                        signal_dict["expiration"] = datetime.fromtimestamp(
                            signal_dict["expiration"] / 1000
                        ).isoformat()

                    result.append(signal_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting signals: {str(e)}")
            raise

    async def save_position(self, position: Dict):
        """
        Сохраняет позицию в базу данных

        Args:
            position: Информация о позиции
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO positions 
                (position_id, symbol, exchange, direction, entry_price, amount, open_timestamp, 
                close_timestamp, close_price, stop_loss, take_profit, unrealized_pnl, realized_pnl, 
                fees, params, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Сериализуем параметры
                params_json = (
                    json.dumps(position.get("params", {}))
                    if "params" in position
                    else None
                )

                # Получаем timestamp
                open_timestamp = position.get(
                    "open_time", position.get("open_timestamp")
                )
                if isinstance(open_timestamp, str):
                    open_timestamp = int(
                        datetime.fromisoformat(open_timestamp).timestamp() * 1000
                    )
                elif isinstance(open_timestamp, datetime):
                    open_timestamp = int(open_timestamp.timestamp() * 1000)

                # Получаем close_timestamp
                close_timestamp = position.get(
                    "close_time", position.get("close_timestamp")
                )
                if close_timestamp:
                    if isinstance(close_timestamp, str):
                        close_timestamp = int(
                            datetime.fromisoformat(close_timestamp).timestamp() * 1000
                        )
                    elif isinstance(close_timestamp, datetime):
                        close_timestamp = int(close_timestamp.timestamp() * 1000)

                # Подготавливаем параметры
                params = (
                    position.get("id", position.get("position_id")),
                    position.get("symbol"),
                    position.get("exchange", "unknown"),
                    position.get("direction"),
                    float(position.get("entry_price", 0)),
                    float(position.get("amount", 0)),
                    open_timestamp,
                    close_timestamp,
                    (
                        float(position.get("close_price", 0))
                        if position.get("close_price") is not None
                        else None
                    ),
                    (
                        float(position.get("stop_loss", 0))
                        if position.get("stop_loss") is not None
                        else None
                    ),
                    (
                        float(position.get("take_profit", 0))
                        if position.get("take_profit") is not None
                        else None
                    ),
                    float(position.get("unrealized_pnl", 0)),
                    float(position.get("realized_pnl", 0)),
                    float(position.get("fees", 0)),
                    params_json,
                    position.get("strategy_id"),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(
                    f"Saved position {position.get('id', position.get('position_id'))} for {position.get('symbol')}"
                )

            except Exception as e:
                logger.error(f"Error saving position: {str(e)}")
                raise

    async def get_positions(
        self,
        symbol: str = None,
        exchange: str = None,
        direction: str = None,
        strategy_id: str = None,
        is_open: bool = None,
        limit: int = None,
    ) -> List[Dict]:
        """
        Получает позиции из базы данных

        Args:
            symbol: Торговая пара
            exchange: Биржа
            direction: Направление (long, short)
            strategy_id: ID стратегии
            is_open: Флаг открытой позиции
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список позиций
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM positions"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)

            if exchange:
                conditions.append("exchange = ?")
                params.append(exchange)

            if direction:
                conditions.append("direction = ?")
                params.append(direction)

            if strategy_id:
                conditions.append("strategy_id = ?")
                params.append(strategy_id)

            if is_open is not None:
                if is_open:
                    conditions.append("close_timestamp IS NULL")
                else:
                    conditions.append("close_timestamp IS NOT NULL")

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY open_timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    position_dict = dict(zip(columns, row))

                    # Десериализуем параметры
                    if position_dict.get("params"):
                        try:
                            position_dict["params"] = json.loads(
                                position_dict["params"]
                            )
                        except json.JSONDecodeError:
                            position_dict["params"] = {}

                    # Преобразуем timestamp в строку ISO
                    if position_dict.get("open_timestamp"):
                        position_dict["open_time"] = datetime.fromtimestamp(
                            position_dict.pop("open_timestamp") / 1000
                        ).isoformat()

                    # Преобразуем close_timestamp в строку ISO
                    if position_dict.get("close_timestamp"):
                        position_dict["close_time"] = datetime.fromtimestamp(
                            position_dict.pop("close_timestamp") / 1000
                        ).isoformat()
                    else:
                        position_dict.pop("close_timestamp", None)

                    result.append(position_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise

    async def save_optimization_result(self, result: Dict):
        """
        Сохраняет результат оптимизации стратегии

        Args:
            result: Результат оптимизации
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO optimization_results 
                (optimization_id, strategy_id, timestamp, best_parameters, best_fitness, 
                best_metrics, optimization_stats, parameter_ranges)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Генерируем ID оптимизации, если его нет
                optimization_id = result.get("optimization_id")
                if not optimization_id:
                    timestamp = int(time.time())
                    optimization_id = f"{result.get('strategy_id')}_{timestamp}"

                # Сериализуем параметры и метрики
                best_parameters = json.dumps(result.get("best_parameters", {}))
                best_metrics = json.dumps(result.get("best_metrics", {}))
                optimization_stats = json.dumps(result.get("optimization_stats", {}))
                parameter_ranges = json.dumps(result.get("parameter_ranges", {}))

                # Подготавливаем параметры
                params = (
                    optimization_id,
                    result.get("strategy_id"),
                    int(time.time() * 1000),
                    best_parameters,
                    float(result.get("best_fitness", 0.0)),
                    best_metrics,
                    optimization_stats,
                    parameter_ranges,
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(
                    f"Saved optimization result for {result.get('strategy_id')}"
                )

            except Exception as e:
                logger.error(f"Error saving optimization result: {str(e)}")
                raise

    async def get_optimization_results(
        self, strategy_id: str = None, limit: int = None
    ) -> List[Dict]:
        """
        Получает результаты оптимизации

        Args:
            strategy_id: ID стратегии
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список результатов оптимизации
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM optimization_results"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if strategy_id:
                conditions.append("strategy_id = ?")
                params.append(strategy_id)

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    optimization_dict = dict(zip(columns, row))

                    # Десериализуем параметры и метрики
                    if optimization_dict.get("best_parameters"):
                        try:
                            optimization_dict["best_parameters"] = json.loads(
                                optimization_dict["best_parameters"]
                            )
                        except json.JSONDecodeError:
                            optimization_dict["best_parameters"] = {}

                    if optimization_dict.get("best_metrics"):
                        try:
                            optimization_dict["best_metrics"] = json.loads(
                                optimization_dict["best_metrics"]
                            )
                        except json.JSONDecodeError:
                            optimization_dict["best_metrics"] = {}

                    if optimization_dict.get("optimization_stats"):
                        try:
                            optimization_dict["optimization_stats"] = json.loads(
                                optimization_dict["optimization_stats"]
                            )
                        except json.JSONDecodeError:
                            optimization_dict["optimization_stats"] = {}

                    if optimization_dict.get("parameter_ranges"):
                        try:
                            optimization_dict["parameter_ranges"] = json.loads(
                                optimization_dict["parameter_ranges"]
                            )
                        except json.JSONDecodeError:
                            optimization_dict["parameter_ranges"] = {}

                    # Преобразуем timestamp в строку ISO
                    if optimization_dict.get("timestamp"):
                        optimization_dict["timestamp"] = datetime.fromtimestamp(
                            optimization_dict["timestamp"] / 1000
                        ).isoformat()

                    result.append(optimization_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting optimization results: {str(e)}")
            raise

    async def save_strategy_parameters(
        self,
        strategy_id: str,
        parameters: Dict,
        description: str = None,
        is_active: bool = True,
    ):
        """
        Сохраняет параметры стратегии

        Args:
            strategy_id: ID стратегии
            parameters: Параметры
            description: Описание
            is_active: Флаг активности
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT INTO strategy_parameters 
                (strategy_id, parameters, timestamp, description, is_active)
                VALUES (?, ?, ?, ?, ?)
                """

                # Сериализуем параметры
                parameters_json = json.dumps(parameters)

                # Подготавливаем параметры
                params = (
                    strategy_id,
                    parameters_json,
                    int(time.time() * 1000),
                    description,
                    int(is_active),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)

                # Если параметры активны, деактивируем предыдущие
                if is_active:
                    await self._connection.execute(
                        "UPDATE strategy_parameters SET is_active = 0 WHERE strategy_id = ? AND timestamp != ?",
                        (strategy_id, params[2]),
                    )

                await self._connection.commit()

                logger.debug(f"Saved parameters for strategy {strategy_id}")

            except Exception as e:
                logger.error(f"Error saving strategy parameters: {str(e)}")
                raise

    async def get_strategy_parameters(
        self, strategy_id: str, active_only: bool = False
    ) -> List[Dict]:
        """
        Получает параметры стратегии

        Args:
            strategy_id: ID стратегии
            active_only: Только активные параметры

        Returns:
            List[Dict]: Список параметров
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM strategy_parameters WHERE strategy_id = ?"

            params = [strategy_id]

            # Добавляем фильтрацию по активности
            if active_only:
                query += " AND is_active = 1"

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    param_dict = dict(zip(columns, row))

                    # Десериализуем параметры
                    if param_dict.get("parameters"):
                        try:
                            param_dict["parameters"] = json.loads(
                                param_dict["parameters"]
                            )
                        except json.JSONDecodeError:
                            param_dict["parameters"] = {}

                    # Преобразуем timestamp в строку ISO
                    if param_dict.get("timestamp"):
                        param_dict["timestamp"] = datetime.fromtimestamp(
                            param_dict["timestamp"] / 1000
                        ).isoformat()

                    # Преобразуем is_active в boolean
                    if "is_active" in param_dict:
                        param_dict["is_active"] = bool(param_dict["is_active"])

                    result.append(param_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting strategy parameters: {str(e)}")
            raise

    async def save_backtest(self, backtest: Dict):
        """
        Сохраняет результат бэктеста

        Args:
            backtest: Результат бэктеста
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO backtests 
                (backtest_id, strategy_id, symbol, exchange, timeframe, start_timestamp, 
                end_timestamp, parameters, metrics, trades, equity_curve, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Сериализуем параметры и результаты
                parameters_json = json.dumps(backtest.get("parameters", {}))
                metrics_json = json.dumps(backtest.get("metrics", {}))
                trades_json = json.dumps(backtest.get("trades", []))
                equity_curve_json = json.dumps(backtest.get("equity_curve", []))

                # Переводим временные метки в миллисекунды
                start_timestamp = backtest.get(
                    "start_timestamp", backtest.get("start_time")
                )
                if isinstance(start_timestamp, str):
                    start_timestamp = int(
                        datetime.fromisoformat(start_timestamp).timestamp() * 1000
                    )
                elif isinstance(start_timestamp, datetime):
                    start_timestamp = int(start_timestamp.timestamp() * 1000)

                end_timestamp = backtest.get("end_timestamp", backtest.get("end_time"))
                if isinstance(end_timestamp, str):
                    end_timestamp = int(
                        datetime.fromisoformat(end_timestamp).timestamp() * 1000
                    )
                elif isinstance(end_timestamp, datetime):
                    end_timestamp = int(end_timestamp.timestamp() * 1000)

                # Подготавливаем параметры
                params = (
                    backtest.get("backtest_id"),
                    backtest.get("strategy_id"),
                    backtest.get("symbol"),
                    backtest.get("exchange"),
                    backtest.get("timeframe"),
                    start_timestamp,
                    end_timestamp,
                    parameters_json,
                    metrics_json,
                    trades_json,
                    equity_curve_json,
                    int(time.time() * 1000),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(
                    f"Saved backtest {backtest.get('backtest_id')} for {backtest.get('strategy_id')}"
                )

            except Exception as e:
                logger.error(f"Error saving backtest: {str(e)}")
                raise

    async def get_backtests(
        self, strategy_id: str = None, symbol: str = None, limit: int = None
    ) -> List[Dict]:
        """
        Получает результаты бэктестов

        Args:
            strategy_id: ID стратегии
            symbol: Торговая пара
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список результатов бэктестов
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM backtests"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if strategy_id:
                conditions.append("strategy_id = ?")
                params.append(strategy_id)

            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    backtest_dict = dict(zip(columns, row))

                    # Десериализуем параметры и результаты
                    for field in ["parameters", "metrics", "trades", "equity_curve"]:
                        if backtest_dict.get(field):
                            try:
                                backtest_dict[field] = json.loads(backtest_dict[field])
                            except json.JSONDecodeError:
                                backtest_dict[field] = (
                                    {} if field in ["parameters", "metrics"] else []
                                )

                    # Преобразуем временные метки в строки ISO
                    for field in ["timestamp", "start_timestamp", "end_timestamp"]:
                        if backtest_dict.get(field):
                            backtest_dict[field.replace("_timestamp", "_time")] = (
                                datetime.fromtimestamp(
                                    backtest_dict.pop(field) / 1000
                                ).isoformat()
                            )

                    result.append(backtest_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting backtests: {str(e)}")
            raise

    async def save_bot_state(self, bot_state: Dict):
        """
        Сохраняет состояние бота

        Args:
            bot_state: Состояние бота
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT OR REPLACE INTO bot_states 
                (bot_id, symbol, exchange, strategy_id, state, timestamp, config, 
                last_signal_timestamp, last_trade_timestamp, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                # Сериализуем состояние и конфигурацию
                state_json = json.dumps(bot_state.get("state", {}))
                config_json = json.dumps(bot_state.get("config", {}))

                # Подготавливаем параметры
                params = (
                    bot_state.get("bot_id"),
                    bot_state.get("symbol"),
                    bot_state.get("exchange"),
                    bot_state.get("strategy_id"),
                    state_json,
                    int(time.time() * 1000),
                    config_json,
                    bot_state.get("last_signal_timestamp"),
                    bot_state.get("last_trade_timestamp"),
                    int(bot_state.get("is_active", True)),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(f"Saved state for bot {bot_state.get('bot_id')}")

            except Exception as e:
                logger.error(f"Error saving bot state: {str(e)}")
                raise

    async def get_bot_states(self, is_active: bool = None) -> List[Dict]:
        """
        Получает состояния ботов

        Args:
            is_active: Фильтр по активности

        Returns:
            List[Dict]: Список состояний ботов
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM bot_states"

            params = []
            conditions = []

            # Добавляем фильтрацию по активности
            if is_active is not None:
                conditions.append("is_active = ?")
                params.append(int(is_active))

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    state_dict = dict(zip(columns, row))

                    # Десериализуем состояние и конфигурацию
                    for field in ["state", "config"]:
                        if state_dict.get(field):
                            try:
                                state_dict[field] = json.loads(state_dict[field])
                            except json.JSONDecodeError:
                                state_dict[field] = {}

                    # Преобразуем временные метки в строки ISO
                    for field in [
                        "timestamp",
                        "last_signal_timestamp",
                        "last_trade_timestamp",
                    ]:
                        if state_dict.get(field):
                            field_name = (
                                field
                                if field == "timestamp"
                                else field.replace("_timestamp", "_time")
                            )
                            state_dict[field_name] = datetime.fromtimestamp(
                                state_dict.pop(field) / 1000
                            ).isoformat()

                    # Преобразуем is_active в boolean
                    if "is_active" in state_dict:
                        state_dict["is_active"] = bool(state_dict["is_active"])

                    result.append(state_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting bot states: {str(e)}")
            raise

    async def save_event(self, event: Dict):
        """
        Сохраняет событие

        Args:
            event: Событие
        """
        await self.connect()

        async with self._lock:
            try:
                # Подготавливаем запрос
                query = """
                INSERT INTO events 
                (event_id, event_type, timestamp, data, source)
                VALUES (?, ?, ?, ?, ?)
                """

                # Генерируем ID события, если его нет
                event_id = event.get("event_id")
                if not event_id:
                    event_id = f"{int(time.time())}_{event.get('event_type')}_{event.get('source', 'system')}"

                # Сериализуем данные
                data_json = json.dumps(event.get("data", {}))

                # Подготавливаем параметры
                params = (
                    event_id,
                    event.get("event_type"),
                    int(time.time() * 1000),
                    data_json,
                    event.get("source"),
                )

                # Выполняем запрос
                await self._connection.execute(query, params)
                await self._connection.commit()

                logger.debug(
                    f"Saved event {event_id} of type {event.get('event_type')}"
                )

            except Exception as e:
                logger.error(f"Error saving event: {str(e)}")
                raise

    async def get_events(
        self,
        event_type: str = None,
        source: str = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ) -> List[Dict]:
        """
        Получает события

        Args:
            event_type: Тип события
            source: Источник события
            start_time: Начальное время (UNIX timestamp)
            end_time: Конечное время (UNIX timestamp)
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список событий
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM events"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if event_type:
                conditions.append("event_type = ?")
                params.append(event_type)

            if source:
                conditions.append("source = ?")
                params.append(source)

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(int(start_time))

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(int(end_time))

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    event_dict = dict(zip(columns, row))

                    # Десериализуем данные
                    if event_dict.get("data"):
                        try:
                            event_dict["data"] = json.loads(event_dict["data"])
                        except json.JSONDecodeError:
                            event_dict["data"] = {}

                    # Преобразуем timestamp в строку ISO
                    if event_dict.get("timestamp"):
                        event_dict["timestamp"] = datetime.fromtimestamp(
                            event_dict["timestamp"] / 1000
                        ).isoformat()

                    result.append(event_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting events: {str(e)}")
            raise

    async def save_log(self, log: Dict):
        """
        Сохраняет лог

        Args:
            log: Запись лога
        """
        await self.connect()

        try:
            # Подготавливаем запрос
            query = """
            INSERT INTO logs 
            (timestamp, level, logger, message, data)
            VALUES (?, ?, ?, ?, ?)
            """

            # Сериализуем данные
            data_json = json.dumps(log.get("data", {})) if log.get("data") else None

            # Подготавливаем параметры
            params = (
                int(time.time() * 1000),
                log.get("level"),
                log.get("logger"),
                log.get("message"),
                data_json,
            )

            # Выполняем запрос
            await self._connection.execute(query, params)
            await self._connection.commit()

        except Exception as e:
            # Здесь не используем логирование, чтобы избежать рекурсии
            print(f"Error saving log: {str(e)}")

    async def get_logs(
        self,
        level: str = None,
        logger: str = None,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ) -> List[Dict]:
        """
        Получает логи

        Args:
            level: Уровень логирования
            logger: Название логгера
            start_time: Начальное время (UNIX timestamp)
            end_time: Конечное время (UNIX timestamp)
            limit: Максимальное количество записей

        Returns:
            List[Dict]: Список логов
        """
        await self.connect()

        try:
            # Формируем запрос
            query = "SELECT * FROM logs"

            params = []
            conditions = []

            # Добавляем фильтрацию
            if level:
                conditions.append("level = ?")
                params.append(level)

            if logger:
                conditions.append("logger = ?")
                params.append(logger)

            if start_time:
                conditions.append("timestamp >= ?")
                params.append(int(start_time))

            if end_time:
                conditions.append("timestamp <= ?")
                params.append(int(end_time))

            # Добавляем условия, если они есть
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Добавляем сортировку по времени
            query += " ORDER BY timestamp DESC"

            # Добавляем ограничение на количество записей
            if limit:
                query += " LIMIT ?"
                params.append(int(limit))

            # Выполняем запрос
            async with self._connection.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    log_dict = dict(zip(columns, row))

                    # Десериализуем данные
                    if log_dict.get("data"):
                        try:
                            log_dict["data"] = json.loads(log_dict["data"])
                        except json.JSONDecodeError:
                            log_dict["data"] = {}

                    # Преобразуем timestamp в строку ISO
                    if log_dict.get("timestamp"):
                        log_dict["timestamp"] = datetime.fromtimestamp(
                            log_dict["timestamp"] / 1000
                        ).isoformat()

                    result.append(log_dict)

            return result

        except Exception as e:
            logger.error(f"Error getting logs: {str(e)}")
            raise

    async def execute_query(self, query: str, params: List = None) -> List[Dict]:
        """
        Выполняет произвольный SQL-запрос

        Args:
            query: SQL-запрос
            params: Параметры запроса

        Returns:
            List[Dict]: Результат запроса
        """
        await self.connect()

        try:
            # Выполняем запрос
            async with self._connection.execute(query, params or []) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    result.append(dict(zip(columns, row)))

            return result

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise

    async def execute_update(self, query: str, params: List = None) -> int:
        """
        Выполняет SQL-запрос на обновление данных

        Args:
            query: SQL-запрос
            params: Параметры запроса

        Returns:
            int: Количество измененных строк
        """
        await self.connect()

        async with self._lock:
            try:
                # Выполняем запрос
                cursor = await self._connection.execute(query, params or [])
                await self._connection.commit()

                # Возвращаем количество измененных строк
                return cursor.rowcount

            except Exception as e:
                logger.error(f"Error executing update: {str(e)}")
                raise

    async def get_table_schema(self, table_name: str) -> List[Dict]:
        """
        Получает схему таблицы

        Args:
            table_name: Название таблицы

        Returns:
            List[Dict]: Схема таблицы
        """
        await self.connect()

        try:
            # Формируем запрос
            query = f"PRAGMA table_info({table_name})"

            # Выполняем запрос
            async with self._connection.execute(query) as cursor:
                rows = await cursor.fetchall()

            # Преобразуем результат в список словарей
            result = []
            if rows:
                columns = [desc[0] for desc in cursor.description]
                for row in rows:
                    result.append(dict(zip(columns, row)))

            return result

        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise

    async def backup_database(self, backup_path: str) -> bool:
        """
        Создает резервную копию базы данных

        Args:
            backup_path: Путь для сохранения резервной копии

        Returns:
            bool: True, если резервная копия создана успешно, иначе False
        """
        await self.connect()

        try:
            # Создаем директорию, если её нет
            os.makedirs(os.path.dirname(os.path.abspath(backup_path)), exist_ok=True)

            # Получаем текущую дату и время
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Если путь - это директория, создаем имя файла с временной меткой
            if os.path.isdir(backup_path):
                backup_path = os.path.join(backup_path, f"backup_{timestamp}.db")

            # Создаем резервную копию
            await self._connection.execute("BEGIN IMMEDIATE")

            # Используем выполнение в отдельном процессе, чтобы не блокировать основной поток
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._backup_db_sync(backup_path)
            )

            logger.info(f"Database backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error creating database backup: {str(e)}")
            return False

    def _backup_db_sync(self, backup_path: str):
        """
        Синхронно создает резервную копию базы данных

        Args:
            backup_path: Путь для сохранения резервной копии
        """
        import sqlite3

        # Открываем соединение с БД
        source = sqlite3.connect(self.db_path)

        # Создаем резервную копию
        backup = sqlite3.connect(backup_path)
        source.backup(backup)

        # Закрываем соединения
        backup.close()
        source.close()

    async def restore_database(self, backup_path: str) -> bool:
        """
        Восстанавливает базу данных из резервной копии

        Args:
            backup_path: Путь к резервной копии

        Returns:
            bool: True, если восстановление успешно, иначе False
        """
        try:
            # Проверяем, существует ли резервная копия
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Закрываем текущее соединение
            await self.disconnect()

            # Используем выполнение в отдельном процессе, чтобы не блокировать основной поток
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._restore_db_sync(backup_path)
            )

            # Повторно подключаемся к БД
            await self.connect()

            logger.info(f"Database restored from: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Error restoring database: {str(e)}")
            return False

    def _restore_db_sync(self, backup_path: str):
        """
        Синхронно восстанавливает базу данных из резервной копии

        Args:
            backup_path: Путь к резервной копии
        """
        import sqlite3
        import shutil

        # Создаем резервную копию текущей БД
        current_backup = self.db_path + ".bak"
        shutil.copy2(self.db_path, current_backup)

        try:
            # Копируем резервную копию в основной файл БД
            shutil.copy2(backup_path, self.db_path)
        except Exception as e:
            # В случае ошибки восстанавливаем предыдущую копию
            shutil.copy2(current_backup, self.db_path)
            raise e

    async def get_database_info(self) -> Dict:
        """
        Получает информацию о базе данных

        Returns:
            Dict: Информация о базе данных
        """
        await self.connect()

        try:
            # Получаем список таблиц
            async with self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                tables = await cursor.fetchall()
                table_names = [row[0] for row in tables]

            # Получаем размер БД
            db_size = os.path.getsize(self.db_path)

            # Получаем количество записей в каждой таблице
            table_counts = {}
            for table in table_names:
                async with self._connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ) as cursor:
                    count = await cursor.fetchone()
                    table_counts[table] = count[0]

            # Формируем информацию
            info = {
                "database_path": self.db_path,
                "database_size": db_size,
                "database_size_mb": round(db_size / (1024 * 1024), 2),
                "tables": table_names,
                "table_counts": table_counts,
                "last_modified": datetime.fromtimestamp(
                    os.path.getmtime(self.db_path)
                ).isoformat(),
            }

            return info

        except Exception as e:
            logger.error(f"Error getting database info: {str(e)}")
            raise

    async def optimize_database(self) -> bool:
        """
        Оптимизирует базу данных

        Returns:
            bool: True, если оптимизация успешна, иначе False
        """
        await self.connect()

        async with self._lock:
            try:
                # Выполняем VACUUM для оптимизации
                await self._connection.execute("VACUUM")

                # Анализируем таблицы
                await self._connection.execute("ANALYZE")

                # Оптимизируем индексы
                await self._connection.execute("PRAGMA optimize")

                await self._connection.commit()

                logger.info("Database optimized")
                return True

            except Exception as e:
                logger.error(f"Error optimizing database: {str(e)}")
                return False

    async def get_individual(self, individual_id: str) -> Dict:
        """
        Получает информацию об индивидууме

        Args:
            individual_id: ID индивидуума

        Returns:
            Dict: Информация об индивидууме
        """
        # В этой функции мы можем использовать различные запросы для получения
        # всей необходимой информации об индивидууме (например, для генетического алгоритма)
        # Здесь приведен упрощенный пример

        await self.connect()

        try:
            # Проверяем, возможно это ID оптимизации
            async with self._connection.execute(
                "SELECT * FROM optimization_results WHERE optimization_id = ?",
                [individual_id],
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Получаем информацию из результатов оптимизации
                columns = [desc[0] for desc in cursor.description]
                optimization = dict(zip(columns, row))

                # Десериализуем параметры и метрики
                for field in [
                    "best_parameters",
                    "best_metrics",
                    "optimization_stats",
                    "parameter_ranges",
                ]:
                    if optimization.get(field):
                        try:
                            optimization[field] = json.loads(optimization[field])
                        except json.JSONDecodeError:
                            optimization[field] = (
                                {}
                                if field
                                in [
                                    "best_parameters",
                                    "best_metrics",
                                    "parameter_ranges",
                                ]
                                else []
                            )

                # Преобразуем timestamp в строку ISO
                if optimization.get("timestamp"):
                    optimization["timestamp"] = datetime.fromtimestamp(
                        optimization["timestamp"] / 1000
                    ).isoformat()

                return {
                    "individual": {
                        "id": individual_id,
                        "parameters": optimization.get("best_parameters", {}),
                        "fitness": optimization.get("best_fitness"),
                        "metrics": optimization.get("best_metrics", {}),
                        "timestamp": optimization.get("timestamp"),
                    },
                    "optimization": optimization,
                }

            # Если это не ID оптимизации, возможно это другой тип индивидуума
            # Здесь можно добавить дополнительные запросы к другим таблицам

            return None

        except Exception as e:
            logger.error(f"Error getting individual: {str(e)}")
            raise

    async def clean_old_data(self, older_than_days: int = 30) -> Dict:
        """
        Удаляет старые данные из базы данных

        Args:
            older_than_days: Удалить данные старше указанного количества дней

        Returns:
            Dict: Результаты удаления
        """
        await self.connect()

        async with self._lock:
            try:
                # Вычисляем временную метку для фильтрации
                timestamp = int(
                    (datetime.now() - timedelta(days=older_than_days)).timestamp()
                    * 1000
                )

                # Удаляем старые данные из разных таблиц
                results = {}

                # Удаляем старые логи
                cursor = await self._connection.execute(
                    "DELETE FROM logs WHERE timestamp < ?", [timestamp]
                )
                results["logs"] = cursor.rowcount

                # Удаляем старые события
                cursor = await self._connection.execute(
                    "DELETE FROM events WHERE timestamp < ?", [timestamp]
                )
                results["events"] = cursor.rowcount

                # Фиксируем изменения
                await self._connection.commit()

                logger.info(f"Cleaned old data: {results}")
                return results

            except Exception as e:
                logger.error(f"Error cleaning old data: {str(e)}")
                raise
