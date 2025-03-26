import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import uuid
import json
import traceback

from project.utils.logging_utils import setup_logger
from project.utils.math_utils import calculate_statistics
from project.backtesting.portfolio import Portfolio
from project.trading.strategy_base import Strategy, Signal, Position

logger = setup_logger("backtester")


@dataclass
class BacktestSettings:
    """Настройки бэктеста"""

    initial_balance: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0001
    position_size_pct: float = 1.0
    max_positions: int = 1
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.02
    use_take_profit: bool = False
    take_profit_pct: float = 0.05
    enable_fractional: bool = True
    enable_shorting: bool = False
    trade_on_close: bool = True
    trade_at_day_end: bool = False
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    log_trades: bool = True
    verbose: bool = False
    warmup_period: int = 0
    enable_compounding: bool = True
    order_types: List[str] = None
    custom_settings: Dict = None


class BacktestResult:
    """Результаты бэктеста"""

    def __init__(self):
        """Инициализирует результаты бэктеста"""
        # Общие метрики
        self.metrics = {}

        # Список сделок
        self.trades = []

        # График капитала
        self.equity_curve = []

        # Список сигналов
        self.signals = []

        # Позиции
        self.positions = []

        # Ошибки и предупреждения
        self.errors = []
        self.warnings = []

        # Время выполнения
        self.execution_time = 0.0

        # Дополнительные метаданные
        self.metadata = {}

    def to_dict(self) -> Dict:
        """
        Преобразует результаты в словарь

        Returns:
            Dict: Результаты бэктеста в виде словаря
        """
        return {
            "metrics": self.metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "signals": [
                s.to_dict() if hasattr(s, "to_dict") else s for s in self.signals
            ],
            "positions": [
                p.to_dict() if hasattr(p, "to_dict") else p for p in self.positions
            ],
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BacktestResult":
        """
        Создает результаты из словаря

        Args:
            data: Словарь с результатами

        Returns:
            BacktestResult: Результаты бэктеста
        """
        result = cls()
        result.metrics = data.get("metrics", {})
        result.trades = data.get("trades", [])
        result.equity_curve = data.get("equity_curve", [])
        result.signals = data.get("signals", [])
        result.positions = data.get("positions", [])
        result.errors = data.get("errors", [])
        result.warnings = data.get("warnings", [])
        result.execution_time = data.get("execution_time", 0.0)
        result.metadata = data.get("metadata", {})
        return result


class Backtester:
    """Класс для бэктестинга торговых стратегий"""

    def __init__(self, settings: Union[Dict, BacktestSettings] = None):
        """
        Инициализирует бэктестер

        Args:
            settings: Настройки бэктеста
        """
        if settings is None:
            self.settings = BacktestSettings()
        elif isinstance(settings, dict):
            self.settings = BacktestSettings(**settings)
        else:
            self.settings = settings

        # Флаг отмены бэктеста
        self._cancel_requested = False

        # Текущий результат
        self.current_result = None

        # Функция обратного вызова для обновления прогресса
        self.progress_callback = None

        # Статистика
        self.backtest_count = 0
        self.total_execution_time = 0.0

        logger.info("Backtester initialized")

    async def backtest(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        settings: Union[Dict, BacktestSettings] = None,
    ) -> Dict:
        """
        Выполняет бэктестинг стратегии

        Args:
            strategy: Экземпляр стратегии
            data: Исторические данные
            settings: Настройки бэктеста (переопределяют настройки по умолчанию)

        Returns:
            Dict: Результаты бэктеста
        """
        # Сбрасываем флаг отмены
        self._cancel_requested = False

        # Применяем настройки
        if settings is not None:
            if isinstance(settings, dict):
                # Создаем копию текущих настроек и обновляем их
                current_settings = self.settings.__dict__.copy()
                current_settings.update(settings)
                backtest_settings = BacktestSettings(**current_settings)
            else:
                backtest_settings = settings
        else:
            backtest_settings = self.settings

        # Инициализируем результат
        result = BacktestResult()
        self.current_result = result

        # Добавляем метаданные
        result.metadata = {
            "strategy_name": strategy.name,
            "backtest_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "settings": {k: v for k, v in backtest_settings.__dict__.items()},
            "data_info": {
                "start_date": data.index[0].isoformat() if len(data) > 0 else None,
                "end_date": data.index[-1].isoformat() if len(data) > 0 else None,
                "timeframe": data.attrs.get("timeframe", "unknown"),
                "symbol": data.attrs.get("symbol", "unknown"),
                "exchange": data.attrs.get("exchange", "unknown"),
                "rows": len(data),
            },
            "strategy_parameters": strategy.parameters,
        }

        try:
            # Засекаем время выполнения
            start_time = time.time()

            # Проверяем наличие данных
            if len(data) == 0:
                result.errors.append("Empty data provided")
                result.execution_time = time.time() - start_time
                return result.to_dict()

            # Проверяем наличие необходимых колонок
            required_columns = ["open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in data.columns:
                    result.errors.append(f"Missing required column: {col}")
                    result.execution_time = time.time() - start_time
                    return result.to_dict()

            # Фильтруем данные по дате, если указана
            if backtest_settings.start_date:
                start_date = pd.to_datetime(backtest_settings.start_date)
                data = data[data.index >= start_date]

            if backtest_settings.end_date:
                end_date = pd.to_datetime(backtest_settings.end_date)
                data = data[data.index <= end_date]

            # Проверяем, остались ли данные после фильтрации
            if len(data) == 0:
                result.errors.append("No data after date filtering")
                result.execution_time = time.time() - start_time
                return result.to_dict()

            # Инициализируем портфель
            portfolio = Portfolio(
                initial_balance=backtest_settings.initial_balance,
                commission=backtest_settings.commission,
                slippage=backtest_settings.slippage,
                position_size_pct=backtest_settings.position_size_pct,
                max_positions=backtest_settings.max_positions,
                enable_fractional=backtest_settings.enable_fractional,
                enable_shorting=backtest_settings.enable_shorting,
                enable_compounding=backtest_settings.enable_compounding,
            )

            # Устанавливаем обработчики событий стратегии
            strategy.on_signal = None  # Сбрасываем обработчик сигналов

            # Определяем период прогрева
            warmup_bars = backtest_settings.warmup_period

            # Выполняем бэктест
            await self._run_backtest(
                strategy, data, portfolio, result, warmup_bars, backtest_settings
            )

            # Проверяем, был ли запрос на отмену
            if self._cancel_requested:
                result.warnings.append("Backtest cancelled by user")

            # Рассчитываем метрики
            result.metrics = calculate_statistics(result)

            # Завершаем
            result.execution_time = time.time() - start_time

            # Обновляем статистику
            self.backtest_count += 1
            self.total_execution_time += result.execution_time

            logger.info(f"Backtest completed in {result.execution_time:.2f} seconds")

            return result.to_dict()

        except Exception as e:
            # Логируем ошибку
            error_msg = f"Error during backtest: {str(e)}"
            error_traceback = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_traceback}")

            # Добавляем ошибку в результат
            result.errors.append(error_msg)
            result.execution_time = time.time() - start_time

            return result.to_dict()

    async def _run_backtest(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        portfolio: Portfolio,
        result: BacktestResult,
        warmup_bars: int,
        settings: BacktestSettings,
    ):
        """
        Выполняет основной цикл бэктеста

        Args:
            strategy: Экземпляр стратегии
            data: Исторические данные
            portfolio: Портфель
            result: Результат бэктеста
            warmup_bars: Количество баров для прогрева
            settings: Настройки бэктеста
        """
        # Копируем данные, чтобы избежать изменения оригинального DataFrame
        df = data.copy()

        # Получаем символ из атрибутов данных или используем имя стратегии
        symbol = df.attrs.get("symbol", strategy.name)

        # Инициализируем индекс текущего бара
        current_bar = warmup_bars

        # Общее количество баров
        total_bars = len(df)

        # Список для сбора сигналов
        signals = []

        # Список для сбора котировок по времени
        equity_curve = []

        # Сохраняем начальный баланс
        initial_equity = portfolio.get_equity()

        # Добавляем начальную точку в график капитала
        equity_curve.append(
            {
                "timestamp": df.index[0].isoformat() if current_bar < len(df) else None,
                "equity": initial_equity,
                "balance": portfolio.balance,
                "position_value": 0.0,
                "drawdown_pct": 0.0,
                "drawdown": 0.0,
            }
        )

        # Максимальный капитал для расчета просадки
        max_equity = initial_equity

        # Перебираем все бары
        while current_bar < total_bars:
            # Проверяем, был ли запрос на отмену
            if self._cancel_requested:
                break

            # Получаем текущий бар
            current_data = df.iloc[: current_bar + 1].copy()
            current_row = current_data.iloc[-1]
            current_timestamp = current_data.index[-1]

            # Получаем цены
            current_open = current_row["open"]
            current_high = current_row["high"]
            current_low = current_row["low"]
            current_close = current_row["close"]

            # Цена для торговли
            trade_price = current_close if settings.trade_on_close else current_open

            # Обновляем стратегию
            new_signals = await strategy.update(current_data)

            # Обрабатываем сигналы
            if new_signals:
                for signal in new_signals:
                    # Проверяем, что сигнал для нужного символа
                    if signal.symbol != symbol and signal.symbol is not None:
                        continue

                    # Добавляем сигнал в список
                    signals.append(signal)

                    # Обрабатываем сигнал
                    await self._process_signal(
                        signal, portfolio, trade_price, current_timestamp, settings
                    )

            # Обновляем позиции
            await self._update_positions(
                portfolio,
                current_high,
                current_low,
                current_close,
                current_timestamp,
                settings,
            )

            # Получаем текущий капитал
            current_equity = portfolio.get_equity(current_close)

            # Обновляем максимальный капитал
            max_equity = max(max_equity, current_equity)

            # Рассчитываем просадку
            drawdown = max_equity - current_equity
            drawdown_pct = drawdown / max_equity if max_equity > 0 else 0.0

            # Добавляем точку в график капитала
            equity_curve.append(
                {
                    "timestamp": current_timestamp.isoformat(),
                    "equity": current_equity,
                    "balance": portfolio.balance,
                    "position_value": portfolio.get_position_value(current_close),
                    "drawdown": drawdown,
                    "drawdown_pct": drawdown_pct,
                }
            )

            # Обновляем прогресс, если указан обработчик
            if self.progress_callback:
                progress = (current_bar + 1) / total_bars
                try:
                    await self.progress_callback(
                        progress,
                        {
                            "bar": current_bar + 1,
                            "total_bars": total_bars,
                            "timestamp": current_timestamp.isoformat(),
                            "equity": current_equity,
                            "drawdown_pct": drawdown_pct,
                        },
                    )
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")

            # Увеличиваем индекс текущего бара
            current_bar += 1

        # Закрываем все открытые позиции
        if current_bar > 0:
            last_bar = df.iloc[current_bar - 1]
            close_price = last_bar["close"]
            await self._close_all_positions(
                portfolio, close_price, df.index[current_bar - 1]
            )

        # Сохраняем результаты
        result.equity_curve = equity_curve
        result.signals = signals
        result.trades = portfolio.trades
        result.positions = portfolio.positions

    async def _process_signal(
        self,
        signal: Signal,
        portfolio: Portfolio,
        price: float,
        timestamp: datetime,
        settings: BacktestSettings,
    ):
        """
        Обрабатывает торговый сигнал

        Args:
            signal: Торговый сигнал
            portfolio: Портфель
            price: Текущая цена
            timestamp: Временная метка
            settings: Настройки бэктеста
        """
        # Проверяем направление сигнала
        if signal.direction == "buy":
            # Рассчитываем размер позиции
            position_size = portfolio.calculate_position_size(
                price, settings.position_size_pct
            )

            # Проверяем, можно ли открыть позицию
            if position_size > 0 and portfolio.can_open_position():
                # Рассчитываем уровни стоп-лосса и тейк-профита
                stop_loss = (
                    price * (1 - settings.stop_loss_pct)
                    if settings.use_stop_loss
                    else None
                )
                take_profit = (
                    price * (1 + settings.take_profit_pct)
                    if settings.use_take_profit
                    else None
                )

                # Открываем позицию
                await portfolio.open_position(
                    "long", price, position_size, timestamp, stop_loss, take_profit
                )

                # Логируем сделку, если включено
                if settings.log_trades:
                    logger.debug(f"Opened LONG position at {price} ({timestamp})")

        elif signal.direction == "sell":
            if settings.enable_shorting:
                # Рассчитываем размер позиции
                position_size = portfolio.calculate_position_size(
                    price, settings.position_size_pct
                )

                # Проверяем, можно ли открыть позицию
                if position_size > 0 and portfolio.can_open_position():
                    # Рассчитываем уровни стоп-лосса и тейк-профита
                    stop_loss = (
                        price * (1 + settings.stop_loss_pct)
                        if settings.use_stop_loss
                        else None
                    )
                    take_profit = (
                        price * (1 - settings.take_profit_pct)
                        if settings.use_take_profit
                        else None
                    )

                    # Открываем короткую позицию
                    await portfolio.open_position(
                        "short", price, position_size, timestamp, stop_loss, take_profit
                    )

                    # Логируем сделку, если включено
                    if settings.log_trades:
                        logger.debug(f"Opened SHORT position at {price} ({timestamp})")

        elif signal.direction == "close":
            # Закрываем все позиции
            await self._close_all_positions(portfolio, price, timestamp)

    async def _update_positions(
        self,
        portfolio: Portfolio,
        high: float,
        low: float,
        close: float,
        timestamp: datetime,
        settings: BacktestSettings,
    ):
        """
        Обновляет позиции (проверяет стоп-лоссы и тейк-профиты)

        Args:
            portfolio: Портфель
            high: Максимальная цена
            low: Минимальная цена
            close: Цена закрытия
            timestamp: Временная метка
            settings: Настройки бэктеста
        """
        # Обновляем каждую открытую позицию
        for position in list(portfolio.positions.values()):
            if not position.is_open():
                continue

            # Проверяем стоп-лосс и тейк-профит для длинной позиции
            if position.direction == "long":
                # Проверяем, был ли достигнут стоп-лосс
                if (
                    settings.use_stop_loss
                    and position.stop_loss is not None
                    and low <= position.stop_loss
                ):
                    # Закрываем позицию по стоп-лоссу
                    await portfolio.close_position(
                        position.id, position.stop_loss, timestamp
                    )

                    # Логируем сделку, если включено
                    if settings.log_trades:
                        logger.debug(
                            f"Closed LONG position at {position.stop_loss} (STOP-LOSS) ({timestamp})"
                        )

                # Проверяем, был ли достигнут тейк-профит
                elif (
                    settings.use_take_profit
                    and position.take_profit is not None
                    and high >= position.take_profit
                ):
                    # Закрываем позицию по тейк-профиту
                    await portfolio.close_position(
                        position.id, position.take_profit, timestamp
                    )

                    # Логируем сделку, если включено
                    if settings.log_trades:
                        logger.debug(
                            f"Closed LONG position at {position.take_profit} (TAKE-PROFIT) ({timestamp})"
                        )

            # Проверяем стоп-лосс и тейк-профит для короткой позиции
            elif position.direction == "short":
                # Проверяем, был ли достигнут стоп-лосс
                if (
                    settings.use_stop_loss
                    and position.stop_loss is not None
                    and high >= position.stop_loss
                ):
                    # Закрываем позицию по стоп-лоссу
                    await portfolio.close_position(
                        position.id, position.stop_loss, timestamp
                    )

                    # Логируем сделку, если включено
                    if settings.log_trades:
                        logger.debug(
                            f"Closed SHORT position at {position.stop_loss} (STOP-LOSS) ({timestamp})"
                        )

                # Проверяем, был ли достигнут тейк-профит
                elif (
                    settings.use_take_profit
                    and position.take_profit is not None
                    and low <= position.take_profit
                ):
                    # Закрываем позицию по тейк-профиту
                    await portfolio.close_position(
                        position.id, position.take_profit, timestamp
                    )

                    # Логируем сделку, если включено
                    if settings.log_trades:
                        logger.debug(
                            f"Closed SHORT position at {position.take_profit} (TAKE-PROFIT) ({timestamp})"
                        )

    async def _close_all_positions(
        self, portfolio: Portfolio, price: float, timestamp: datetime
    ):
        """
        Закрывает все открытые позиции

        Args:
            portfolio: Портфель
            price: Текущая цена
            timestamp: Временная метка
        """
        # Получаем список открытых позиций
        for position in list(portfolio.positions.values()):
            if position.is_open():
                # Закрываем позицию
                await portfolio.close_position(position.id, price, timestamp)

    async def cancel(self):
        """Отменяет выполнение текущего бэктеста"""
        self._cancel_requested = True
        logger.info("Backtest cancellation requested")

    async def optimize(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        parameter_ranges: Dict,
        fitness_function: Callable = None,
        optimization_method: str = "grid",
        max_iterations: int = 100,
        population_size: int = 20,
        settings: Union[Dict, BacktestSettings] = None,
    ) -> Dict:
        """
        Оптимизирует параметры стратегии

        Args:
            strategy_class: Класс стратегии
            data: Исторические данные
            parameter_ranges: Диапазоны параметров для оптимизации
            fitness_function: Функция оценки качества стратегии
            optimization_method: Метод оптимизации ('grid', 'random', 'genetic')
            max_iterations: Максимальное количество итераций
            population_size: Размер популяции для генетического алгоритма
            settings: Настройки бэктеста

        Returns:
            Dict: Результаты оптимизации
        """
        # Проверяем метод оптимизации
        if optimization_method not in ["grid", "random", "genetic"]:
            raise ValueError(f"Unknown optimization method: {optimization_method}")

        # Сбрасываем флаг отмены
        self._cancel_requested = False

        # Засекаем время выполнения
        start_time = time.time()

        # Инициализируем результаты
        results = {
            "best_parameters": None,
            "best_result": None,
            "best_metrics": None,
            "all_results": [],
            "execution_time": 0.0,
            "iterations": 0,
            "canceled": False,
        }

        try:
            # Выбираем метод оптимизации
            if optimization_method == "grid":
                await self._optimize_grid(
                    strategy_class,
                    data,
                    parameter_ranges,
                    fitness_function,
                    results,
                    settings,
                )
            elif optimization_method == "random":
                await self._optimize_random(
                    strategy_class,
                    data,
                    parameter_ranges,
                    fitness_function,
                    results,
                    settings,
                    max_iterations,
                )
            elif optimization_method == "genetic":
                # Для генетического алгоритма используем внешний модуль
                from project.optimization.genetic_optimizer import GeneticOptimizer

                # Настройки генетического алгоритма
                genetic_config = {
                    "population_size": population_size,
                    "generations": max_iterations // population_size,
                    "fitness_metrics": {
                        "total_return": 1.0,
                        "sharpe_ratio": 1.0,
                        "max_drawdown": -0.5,
                        "win_rate": 0.3,
                    },
                }

                # Создаем оптимизатор
                optimizer = GeneticOptimizer(genetic_config)

                # Определяем функцию обратного вызова для обновления прогресса
                async def progress_callback(progress, individual):
                    if self.progress_callback:
                        await self.progress_callback(
                            progress,
                            {
                                "current": optimizer.optimization_stats.get(
                                    "total_evaluations", 0
                                ),
                                "total": max_iterations,
                                "best_fitness": optimizer.optimization_stats.get(
                                    "best_fitness", None
                                ),
                            },
                        )

                # Запускаем оптимизацию
                optimization_result = await optimizer.optimize(
                    strategy_class,
                    parameter_ranges,
                    data,
                    initial_population=None,
                    progress_callback=progress_callback,
                )

                # Обновляем результаты
                results["best_parameters"] = optimization_result.get("best_parameters")
                results["best_metrics"] = optimization_result.get("best_metrics")
                results["best_result"] = (
                    None  # Результат бэктеста не сохраняется генетическим алгоритмом
                )
                results["iterations"] = optimizer.optimization_stats.get(
                    "total_evaluations", 0
                )
                results["optimization_stats"] = optimization_result.get(
                    "optimization_stats"
                )

            # Проверяем, был ли запрос на отмену
            if self._cancel_requested:
                results["canceled"] = True

            # Завершаем
            results["execution_time"] = time.time() - start_time

            return results

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            traceback_str = traceback.format_exc()
            logger.error(traceback_str)

            # Добавляем ошибку в результаты
            results["error"] = str(e)
            results["execution_time"] = time.time() - start_time

            return results

    async def _optimize_grid(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        parameter_ranges: Dict,
        fitness_function: Callable,
        results: Dict,
        settings: Union[Dict, BacktestSettings] = None,
    ):
        """
        Оптимизирует параметры методом перебора

        Args:
            strategy_class: Класс стратегии
            data: Исторические данные
            parameter_ranges: Диапазоны параметров для оптимизации
            fitness_function: Функция оценки качества стратегии
            results: Словарь для сохранения результатов
            settings: Настройки бэктеста
        """
        # Генерируем все комбинации параметров
        parameter_combinations = self._generate_parameter_combinations(parameter_ranges)

        # Общее количество комбинаций
        total_combinations = len(parameter_combinations)

        # Счетчик итераций
        iteration = 0

        # Лучший результат
        best_fitness = float("-inf")
        best_parameters = None
        best_result = None
        best_metrics = None

        # Перебираем все комбинации
        for parameters in parameter_combinations:
            # Проверяем, был ли запрос на отмену
            if self._cancel_requested:
                break

            # Создаем экземпляр стратегии с текущими параметрами
            strategy = strategy_class(parameters=parameters)

            # Выполняем бэктест
            result_dict = await self.backtest(strategy, data, settings)

            # Преобразуем результаты в экземпляр BacktestResult
            result = BacktestResult.from_dict(result_dict)

            # Вычисляем значение функции приспособленности
            fitness = self._calculate_fitness(result, fitness_function)

            # Сохраняем результат
            results["all_results"].append(
                {
                    "parameters": parameters,
                    "fitness": fitness,
                    "metrics": result.metrics,
                }
            )

            # Обновляем лучший результат
            if fitness > best_fitness:
                best_fitness = fitness
                best_parameters = parameters
                best_result = result
                best_metrics = result.metrics

            # Увеличиваем счетчик итераций
            iteration += 1

            # Обновляем прогресс, если указан обработчик
            if self.progress_callback:
                progress = iteration / total_combinations
                try:
                    await self.progress_callback(
                        progress,
                        {
                            "current": iteration,
                            "total": total_combinations,
                            "best_fitness": best_fitness,
                        },
                    )
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")

        # Сохраняем лучший результат
        results["best_parameters"] = best_parameters
        results["best_result"] = best_result.to_dict() if best_result else None
        results["best_metrics"] = best_metrics
        results["iterations"] = iteration

    async def _optimize_random(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        parameter_ranges: Dict,
        fitness_function: Callable,
        results: Dict,
        settings: Union[Dict, BacktestSettings] = None,
        max_iterations: int = 100,
    ):
        """
        Оптимизирует параметры методом случайного поиска

        Args:
            strategy_class: Класс стратегии
            data: Исторические данные
            parameter_ranges: Диапазоны параметров для оптимизации
            fitness_function: Функция оценки качества стратегии
            results: Словарь для сохранения результатов
            settings: Настройки бэктеста
            max_iterations: Максимальное количество итераций
        """
        # Счетчик итераций
        iteration = 0

        # Лучший результат
        best_fitness = float("-inf")
        best_parameters = None
        best_result = None
        best_metrics = None

        # Выполняем заданное количество итераций
        while iteration < max_iterations:
            # Проверяем, был ли запрос на отмену
            if self._cancel_requested:
                break

            # Генерируем случайные параметры
            parameters = self._generate_random_parameters(parameter_ranges)

            # Создаем экземпляр стратегии с текущими параметрами
            strategy = strategy_class(parameters=parameters)

            # Выполняем бэктест
            result_dict = await self.backtest(strategy, data, settings)

            # Преобразуем результаты в экземпляр BacktestResult
            result = BacktestResult.from_dict(result_dict)

            # Вычисляем значение функции приспособленности
            fitness = self._calculate_fitness(result, fitness_function)

            # Сохраняем результат
            results["all_results"].append(
                {
                    "parameters": parameters,
                    "fitness": fitness,
                    "metrics": result.metrics,
                }
            )

            # Обновляем лучший результат
            if fitness > best_fitness:
                best_fitness = fitness
                best_parameters = parameters
                best_result = result
                best_metrics = result.metrics

            # Увеличиваем счетчик итераций
            iteration += 1

            # Обновляем прогресс, если указан обработчик
            if self.progress_callback:
                progress = iteration / max_iterations
                try:
                    await self.progress_callback(
                        progress,
                        {
                            "current": iteration,
                            "total": max_iterations,
                            "best_fitness": best_fitness,
                        },
                    )
                except Exception as e:
                    logger.error(f"Error in progress callback: {str(e)}")

        # Сохраняем лучший результат
        results["best_parameters"] = best_parameters
        results["best_result"] = best_result.to_dict() if best_result else None
        results["best_metrics"] = best_metrics
        results["iterations"] = iteration

    def _generate_parameter_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """
        Генерирует все комбинации параметров

        Args:
            parameter_ranges: Диапазоны параметров для оптимизации

        Returns:
            List[Dict]: Список комбинаций параметров
        """
        # Результат
        combinations = [{}]

        # Для каждого параметра
        for param_name, param_range in parameter_ranges.items():
            # Новый список комбинаций
            new_combinations = []

            # Определяем возможные значения
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Диапазон значений
                start, end = param_range

                # Определяем шаг
                if isinstance(start, int) and isinstance(end, int):
                    # Целочисленный шаг
                    step = 1
                    values = list(range(start, end + 1, step))
                else:
                    # Определяем количество точек для вещественных чисел
                    num_points = 5
                    step = (end - start) / (num_points - 1)
                    values = [start + i * step for i in range(num_points)]
            elif isinstance(param_range, list):
                # Список значений
                values = param_range
            elif isinstance(param_range, bool) or param_range in [True, False]:
                # Булевы значения
                values = [True, False]
            else:
                # Одно значение
                values = [param_range]

            # Для каждой существующей комбинации
            for combo in combinations:
                # Для каждого возможного значения параметра
                for value in values:
                    # Создаем новую комбинацию
                    new_combo = combo.copy()
                    new_combo[param_name] = value
                    new_combinations.append(new_combo)

            # Обновляем список комбинаций
            combinations = new_combinations

        return combinations

    def _generate_random_parameters(self, parameter_ranges: Dict) -> Dict:
        """
        Генерирует случайные параметры в заданных диапазонах

        Args:
            parameter_ranges: Диапазоны параметров для оптимизации

        Returns:
            Dict: Словарь случайных параметров
        """
        import random

        # Результат
        parameters = {}

        # Для каждого параметра
        for param_name, param_range in parameter_ranges.items():
            # Определяем случайное значение
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Диапазон значений
                start, end = param_range

                # Генерируем случайное значение
                if isinstance(start, int) and isinstance(end, int):
                    # Целочисленное значение
                    parameters[param_name] = random.randint(start, end)
                else:
                    # Вещественное значение
                    parameters[param_name] = random.uniform(start, end)
            elif isinstance(param_range, list):
                # Список значений
                parameters[param_name] = random.choice(param_range)
            elif isinstance(param_range, bool) or param_range in [True, False]:
                # Булево значение
                parameters[param_name] = random.choice([True, False])
            else:
                # Одно значение
                parameters[param_name] = param_range

        return parameters

    def _calculate_fitness(
        self, result: BacktestResult, fitness_function: Callable = None
    ) -> float:
        """
        Вычисляет значение функции приспособленности

        Args:
            result: Результат бэктеста
            fitness_function: Функция оценки качества стратегии

        Returns:
            float: Значение функции приспособленности
        """
        # Если указана пользовательская функция приспособленности, используем её
        if fitness_function:
            return fitness_function(result)

        # Иначе используем встроенную функцию
        metrics = result.metrics

        # Если метрик нет, возвращаем минимальное значение
        if not metrics:
            return float("-inf")

        # Базовая функция приспособленности
        fitness = 0.0

        # Учитываем общую доходность
        if "total_return" in metrics:
            fitness += metrics["total_return"] * 1.0

        # Учитываем коэффициент Шарпа
        if "sharpe_ratio" in metrics:
            fitness += metrics["sharpe_ratio"] * 1.0

        # Учитываем максимальную просадку (с отрицательным весом)
        if "max_drawdown" in metrics:
            fitness -= metrics["max_drawdown"] * 0.5

        # Учитываем процент выигрышных сделок
        if "win_rate" in metrics:
            fitness += metrics["win_rate"] * 0.3

        return fitness

    def get_stats(self) -> Dict:
        """
        Возвращает статистику бэктестера

        Returns:
            Dict: Статистика бэктестера
        """
        return {
            "backtest_count": self.backtest_count,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": (
                self.total_execution_time / self.backtest_count
                if self.backtest_count > 0
                else 0.0
            ),
        }


# Функция для проведения бэктеста без создания экземпляра класса
async def backtest_strategy(
    strategy: Strategy,
    data: pd.DataFrame,
    portfolio: Portfolio = None,
    settings: Union[Dict, BacktestSettings] = None,
) -> Dict:
    """
    Проводит бэктест стратегии

    Args:
        strategy: Экземпляр стратегии
        data: Исторические данные
        portfolio: Портфель
        settings: Настройки бэктеста

    Returns:
        Dict: Результаты бэктеста
    """
    # Создаем бэктестер
    backtester = Backtester(settings)

    # Выполняем бэктест
    return await backtester.backtest(strategy, data, settings)
