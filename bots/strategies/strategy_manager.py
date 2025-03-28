"""
Модуль для управления торговыми стратегиями.
Предоставляет функции для создания, запуска и управления стратегиями.
"""

import importlib
from typing import Any, Dict, List, Type

from project.bots.strategies.base_strategy import BaseStrategy
from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)


class StrategyManager:
    """
    Класс для управления торговыми стратегиями.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "StrategyManager":
        """
        Получает экземпляр менеджера стратегий (Singleton).

        Returns:
            Экземпляр класса StrategyManager
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Инициализирует менеджер стратегий.
        """
        self.config = get_config()
        self.strategies: Dict[str, Type[BaseStrategy]] = {}  # name -> стратегия
        self.running_strategies: Dict[str, BaseStrategy] = {}  # id -> экземпляр
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}  # name -> конфигурация
        self._load_strategies()
        logger.debug("Создан экземпляр StrategyManager")

    def _load_strategies(self) -> None:
        """
        Загружает доступные стратегии.
        """
        try:
            # Загружаем встроенные стратегии
            from project.bots.strategies.cross import CrossStrategy
            from project.bots.strategies.futures import FuturesStrategy
            from project.bots.strategies.main_strategy import MainStrategy
            from project.bots.strategies.mean_revision import MeanReversionStrategy
            from project.bots.strategies.memcoin import MemcoinStrategy
            from project.bots.strategies.scalping import ScalpingStrategy
            from project.bots.strategies.volatility_strategy import VolatilityStrategy

            # Регистрируем стратегии
            self.register_strategy("main", MainStrategy)
            self.register_strategy("volatility", VolatilityStrategy)
            self.register_strategy("scalping", ScalpingStrategy)
            self.register_strategy("memcoin", MemcoinStrategy)
            self.register_strategy("cross", CrossStrategy)
            self.register_strategy("futures", FuturesStrategy)
            self.register_strategy("mean_reversion", MeanReversionStrategy)

            # Загружаем пользовательские стратегии из конфигурации
            custom_strategies = self.config.CUSTOM_STRATEGIES or {}
            for name, strategy_class_path in custom_strategies.items():
                try:
                    module_path, class_name = strategy_class_path.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    strategy_class = getattr(module, class_name)

                    self.register_strategy(name, strategy_class)
                except Exception as e:
                    logger.error(
                        f"Ошибка при загрузке пользовательской стратегии {name}: {str(e)}"
                    )

            logger.info(f"Загружено {len(self.strategies)} стратегий")

        except Exception as e:
            logger.error(f"Ошибка при загрузке стратегий: {str(e)}")

    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Регистрирует стратегию.

        Args:
            name: Имя стратегии
            strategy_class: Класс стратегии
        """
        # Проверяем, что класс наследуется от BaseStrategy
        if not issubclass(strategy_class, BaseStrategy):
            logger.warning(
                f"Класс {strategy_class.__name__} не является подклассом BaseStrategy"
            )
            return

        # Регистрируем стратегию
        self.strategies[name] = strategy_class
        logger.debug(f"Зарегистрирована стратегия: {name}")

    def set_strategy_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Устанавливает конфигурацию для стратегии.

        Args:
            name: Имя стратегии
            config: Конфигурация стратегии
        """
        self.strategy_configs[name] = config
        logger.debug(f"Установлена конфигурация для стратегии {name}")

    def get_available_strategies(self) -> List[str]:
        """
        Получает список доступных стратегий.

        Returns:
            Список имен стратегий
        """
        return list(self.strategies.keys())

    def get_running_strategies(self) -> List[Dict[str, Any]]:
        """
        Получает список запущенных стратегий.

        Returns:
            Список словарей с информацией о стратегиях
        """
        result = []
        for strategy_id, strategy in self.running_strategies.items():
            result.append(
                {
                    "id": strategy_id,
                    "name": strategy.name,
                    "class": strategy.__class__.__name__,
                    "exchange": strategy.exchange_id,
                    "symbols": strategy.symbols,
                    "timeframes": strategy.timeframes,
                    "status": strategy.get_status(),
                }
            )
        return result

    @async_handle_error
    async def start_strategy(self, strategy_name: str, **kwargs) -> str:
        """
        Запускает стратегию.

        Args:
            strategy_name: Имя стратегии
            **kwargs: Параметры для стратегии

        Returns:
            ID запущенной стратегии
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Стратегия {strategy_name} не найдена")

        # Получаем класс стратегии
        strategy_class = self.strategies[strategy_name]

        # Получаем конфигурацию стратегии
        config = dict(self.strategy_configs.get(strategy_name, {}))

        # Объединяем с переданными параметрами
        config.update(kwargs)

        # Создаем экземпляр стратегии
        strategy = strategy_class(
            name=config.get("name", strategy_name),
            exchange_id=config.get("exchange_id", "binance"),
            symbols=config.get("symbols", None),
            timeframes=config.get("timeframes", None),
            config=config,
        )

        # Запускаем стратегию
        success = await strategy.start()
        if not success:
            raise RuntimeError(f"Не удалось запустить стратегию {strategy_name}")

        # Добавляем стратегию в список запущенных
        strategy_id = strategy.strategy_id
        self.running_strategies[strategy_id] = strategy

        logger.info(f"Запущена стратегия {strategy_name} с ID {strategy_id}")
        await send_trading_signal(f"Запущена стратегия {strategy.name}")

        return strategy_id

    @async_handle_error
    async def stop_strategy(self, strategy_id: str) -> bool:
        """
        Останавливает стратегию.

        Args:
            strategy_id: ID стратегии

        Returns:
            True в случае успеха, иначе False
        """
        if strategy_id not in self.running_strategies:
            logger.warning(f"Стратегия с ID {strategy_id} не найдена")
            return False

        # Получаем стратегию
        strategy = self.running_strategies[strategy_id]

        # Останавливаем стратегию
        success = await strategy.stop()
        if not success:
            logger.error(f"Не удалось остановить стратегию с ID {strategy_id}")
            return False

        # Удаляем стратегию из списка запущенных
        del self.running_strategies[strategy_id]

        logger.info(f"Остановлена стратегия {strategy.name} с ID {strategy_id}")
        await send_trading_signal(f"Остановлена стратегия {strategy.name}")

        return True

    @async_handle_error
    async def pause_strategy(self, strategy_id: str) -> bool:
        """
        Приостанавливает работу стратегии.

        Args:
            strategy_id: ID стратегии

        Returns:
            True в случае успеха, иначе False
        """
        if strategy_id not in self.running_strategies:
            logger.warning(f"Стратегия с ID {strategy_id} не найдена")
            return False

        # Получаем стратегию
        strategy = self.running_strategies[strategy_id]

        # Приостанавливаем стратегию
        success = await strategy.pause()

        logger.info(f"Приостановлена стратегия {strategy.name} с ID {strategy_id}")

        return success

    @async_handle_error
    async def resume_strategy(self, strategy_id: str) -> bool:
        """
        Возобновляет работу стратегии.

        Args:
            strategy_id: ID стратегии

        Returns:
            True в случае успеха, иначе False
        """
        if strategy_id not in self.running_strategies:
            logger.warning(f"Стратегия с ID {strategy_id} не найдена")
            return False

        # Получаем стратегию
        strategy = self.running_strategies[strategy_id]

        # Возобновляем стратегию
        success = await strategy.resume()

        logger.info(f"Возобновлена работа стратегии {strategy.name} с ID {strategy_id}")

        return success

    @async_handle_error
    async def update_strategy_config(
        self, strategy_id: str, config: Dict[str, Any]
    ) -> bool:
        """
        Обновляет конфигурацию запущенной стратегии.

        Args:
            strategy_id: ID стратегии
            config: Новая конфигурация

        Returns:
            True в случае успеха, иначе False
        """
        if strategy_id not in self.running_strategies:
            logger.warning(f"Стратегия с ID {strategy_id} не найдена")
            return False

        # Получаем стратегию
        strategy = self.running_strategies[strategy_id]

        # Обновляем конфигурацию
        success = await strategy.update_config(config)

        logger.info(
            f"Обновлена конфигурация стратегии {strategy.name} с ID {strategy_id}"
        )

        return success

    @async_handle_error
    async def get_strategy_state(self, strategy_id: str) -> Dict[str, Any]:
        """
        Получает состояние стратегии.

        Args:
            strategy_id: ID стратегии

        Returns:
            Словарь с состоянием стратегии
        """
        if strategy_id not in self.running_strategies:
            logger.warning(f"Стратегия с ID {strategy_id} не найдена")
            return {}

        # Получаем стратегию
        strategy = self.running_strategies[strategy_id]

        # Получаем состояние
        return strategy.get_state()

    @async_handle_error
    async def process_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Обрабатывает внешний сигнал и передает его всем запущенным стратегиям.

        Args:
            signal: Словарь с данными сигнала

        Returns:
            True, если сигнал обработан хотя бы одной стратегией, иначе False
        """
        if not self.running_strategies:
            logger.warning("Нет запущенных стратегий для обработки сигнала")
            return False

        # Отправляем сигнал всем запущенным стратегиям
        results = []
        for strategy in self.running_strategies.values():
            if strategy.is_running():
                try:
                    result = await strategy.process_signal(signal)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Ошибка при обработке сигнала стратегией {strategy.name}: {str(e)}"
                    )

        # Возвращаем True, если хотя бы одна стратегия обработала сигнал успешно
        return any(results)

    @async_handle_error
    async def shutdown(self) -> None:
        """
        Останавливает все запущенные стратегии.
        """
        # Получаем список запущенных стратегий
        running_strategies = list(self.running_strategies.items())

        # Останавливаем каждую стратегию
        for strategy_id, strategy in running_strategies:
            try:
                await strategy.stop()
                logger.info(f"Остановлена стратегия {strategy.name} с ID {strategy_id}")
            except Exception as e:
                logger.error(
                    f"Ошибка при остановке стратегии {strategy.name}: {str(e)}"
                )

        # Очищаем список запущенных стратегий
        self.running_strategies = {}

        logger.info("Все стратегии остановлены")
