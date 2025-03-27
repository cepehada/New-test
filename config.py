"""
Модуль конфигурации для торгового бота.
Загружает настройки из переменных окружения и .env файла.
"""

import os
import logging
from typing import Dict, List, Optional, Set, Any
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

logger = logging.getLogger(__name__)

# ----------------------------
# Классы настроек отдельных подсистем
# ----------------------------


class DatabaseSettings(BaseModel):
    """Настройки базы данных"""

    URI: str
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_TIMEOUT: int = 30


class ExchangeSettings(BaseModel):
    """Настройки биржи"""

    API_KEY: str
    API_SECRET: str
    TESTNET: bool = False
    RATE_LIMIT_MARGIN: float = 0.9


class TelegramSettings(BaseModel):
    """Настройки Telegram"""

    BOT_TOKEN: str
    CHAT_ID: str
    ALLOWED_USERS: Set[str]

    @validator("ALLOWED_USERS", pre=True)
    def parse_allowed_users(cls, v: Any) -> Set[str]:
        if isinstance(v, str):
            return {user.strip() for user in v.split(",") if user.strip()}
        return set(v)


class LoggingSettings(BaseModel):
    """Настройки логирования"""

    LEVEL: str = "INFO"
    FILE_PATH: Optional[str] = None
    CONSOLE_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    FILE_FORMAT: str = (
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    )


class SystemSettings(BaseModel):
    """Системные настройки"""

    ENABLE_BACKTESTING: bool = False
    ENABLE_PAPER_TRADING: bool = True
    TRADING_CONCURRENCY: int = 5
    DEFAULT_RISK_PERCENTAGE: float = 1.0


class PositionSizingSettings(BaseModel):
    """Настройки адаптивного размера позиции"""

    adaptive_sizing: bool = True
    base_position_size: float = 0.02
    min_position_size: float = 0.01
    max_position_size: float = 0.1
    win_multiplier: float = 1.1
    loss_multiplier: float = 0.9
    volatility_sizing: bool = True
    signal_sizing: bool = True
    martingale: bool = False
    risk_per_trade: float = 0.01


# ----------------------------
# Новые настройки для модулей
# ----------------------------


class GeneticOptimizerSettings(BaseModel):
    POPULATION_SIZE: int = 50
    GENERATIONS: int = 10
    CROSSOVER_RATE: float = 0.7
    MUTATION_RATE: float = 0.2
    ELITISM_RATE: float = 0.1
    TOURNAMENT_SIZE: int = 3
    MAX_WORKERS: int = 4
    BACKTEST_INITIAL_BALANCE: float = 10000.0
    BACKTEST_COMMISSION: float = 0.001
    BACKTEST_SLIPPAGE: float = 0.0001
    FITNESS_METRICS: Dict = {
        "total_return": 1.0,
        "sharpe_ratio": 1.0,
        "max_drawdown": -0.5,
        "win_rate": 0.3,
    }


class DataVisualizerSettings(BaseModel):
    THEME: str = "dark"  # или "light"
    FIGSIZE: tuple = (14, 8)


# ----------------------------
# Основной класс конфигурации
# ----------------------------


class Config(BaseSettings):
    # Существующие настройки
    DATABASE_URI: str
    BINANCE_API_KEY: str
    BINANCE_API_SECRET: str
    BYBIT_API_KEY: str = ""
    BYBIT_API_SECRET: str = ""
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    TELEGRAM_ALLOWED_USERS: str
    LOG_LEVEL: str = "INFO"
    LOG_FILE_PATH: Optional[str] = None
    ENABLE_BACKTESTING: bool = False
    ENABLE_PAPER_TRADING: bool = True
    TRADING_CONCURRENCY: int = 5
    DEFAULT_RISK_PERCENTAGE: float = 1.0
    MESSAGE_BROKER_URI: str = "amqp://guest:guest@localhost:5672/"
    ENCRYPTION_KEY: str
    POSITION_SIZING_ADAPTIVE: bool = True
    POSITION_SIZING_BASE_SIZE: float = 0.02
    POSITION_SIZING_MIN_SIZE: float = 0.01
    POSITION_SIZING_MAX_SIZE: float = 0.1
    POSITION_SIZING_WIN_MULTIPLIER: float = 1.1
    POSITION_SIZING_LOSS_MULTIPLIER: float = 0.9
    POSITION_SIZING_VOLATILITY: bool = True
    POSITION_SIZING_SIGNAL: bool = True
    POSITION_SIZING_MARTINGALE: bool = False
    POSITION_SIZING_RISK_PER_TRADE: float = 0.01

    # Новые настройки
    GENETIC_OPTIMIZER_SETTINGS: GeneticOptimizerSettings = GeneticOptimizerSettings()
    DATA_VISUALIZER_SETTINGS: DataVisualizerSettings = DataVisualizerSettings()

    class Config:
        env_file = ".env"
        case_sensitive = True

    def get_database_settings(self) -> DatabaseSettings:
        return DatabaseSettings(
            URI=self.DATABASE_URI, POOL_SIZE=10, MAX_OVERFLOW=20, POOL_TIMEOUT=30
        )

    def get_exchange_settings(self, exchange_name: str) -> ExchangeSettings:
        if exchange_name.lower() == "binance":
            return ExchangeSettings(
                API_KEY=self.BINANCE_API_KEY,
                API_SECRET=self.BINANCE_API_SECRET,
                TESTNET=self.ENABLE_PAPER_TRADING,
            )
        elif exchange_name.lower() == "bybit":
            return ExchangeSettings(
                API_KEY=self.BYBIT_API_KEY,
                API_SECRET=self.BYBIT_API_SECRET,
                TESTNET=self.ENABLE_PAPER_TRADING,
            )
        else:
            raise ValueError(f"Неизвестная биржа: {exchange_name}")

    def get_telegram_settings(self) -> TelegramSettings:
        return TelegramSettings(
            BOT_TOKEN=self.TELEGRAM_BOT_TOKEN,
            CHAT_ID=self.TELEGRAM_CHAT_ID,
            ALLOWED_USERS=self.TELEGRAM_ALLOWED_USERS,
        )

    def get_logging_settings(self) -> LoggingSettings:
        return LoggingSettings(LEVEL=self.LOG_LEVEL, FILE_PATH=self.LOG_FILE_PATH)

    def get_system_settings(self) -> SystemSettings:
        return SystemSettings(
            ENABLE_BACKTESTING=self.ENABLE_BACKTESTING,
            ENABLE_PAPER_TRADING=self.ENABLE_PAPER_TRADING,
            TRADING_CONCURRENCY=self.TRADING_CONCURRENCY,
            DEFAULT_RISK_PERCENTAGE=self.DEFAULT_RISK_PERCENTAGE,
        )

    def get_position_sizing_settings(self) -> PositionSizingSettings:
        return PositionSizingSettings(
            adaptive_sizing=self.POSITION_SIZING_ADAPTIVE,
            base_position_size=self.POSITION_SIZING_BASE_SIZE,
            min_position_size=self.POSITION_SIZING_MIN_SIZE,
            max_position_size=self.POSITION_SIZING_MAX_SIZE,
            win_multiplier=self.POSITION_SIZING_WIN_MULTIPLIER,
            loss_multiplier=self.POSITION_SIZING_LOSS_MULTIPLIER,
            volatility_sizing=self.POSITION_SIZING_VOLATILITY,
            signal_sizing=self.POSITION_SIZING_SIGNAL,
            martingale=self.POSITION_SIZING_MARTИНГАЛЕ,
            risk_per_trade=self.POSITION_SIZING_RISK_PER_TRADE,
        )


_config_instance = None


def get_config() -> Config:
    """
    Получить экземпляр конфигурации.
    Использует паттерн Singleton для предотвращения многократной загрузки.
    """
    global _config_instance
    if _config_instance is None:
        try:
            _config_instance = Config()
            logger.info("Конфигурация успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
            raise
    return _config_instance
