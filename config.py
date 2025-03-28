"""
Модуль с конфигурацией для торгового бота.
Загружает настройки из файла .env и конфигурационных файлов.
"""

import os
import sys
import logging
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()


@dataclass
class DatabaseConfig:
    """Конфигурация базы данных"""
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    username: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "trading"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "trading_password"))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "trading"))
    connection_pool_size: int = 10
    connection_timeout: int = 30
    enable_ssl: bool = False


@dataclass
class RedisConfig:
    """Конфигурация Redis"""
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))
    db: int = 0
    connection_pool_size: int = 10
    enable_ssl: bool = False


@dataclass
class InfluxDBConfig:
    """Конфигурация InfluxDB"""
    url: str = field(default_factory=lambda: os.getenv("INFLUXDB_URL", "http://localhost:8086"))
    token: str = field(default_factory=lambda: os.getenv("INFLUXDB_TOKEN", ""))
    org: str = field(default_factory=lambda: os.getenv("INFLUXDB_ORG", "trading"))
    bucket: str = field(default_factory=lambda: os.getenv("INFLUXDB_BUCKET", "trading_data"))
    username: str = field(default_factory=lambda: os.getenv("INFLUXDB_USERNAME", "admin"))
    password: str = field(default_factory=lambda: os.getenv("INFLUXDB_PASSWORD", "influxdb_password"))


@dataclass
class TelegramConfig:
    """Конфигурация уведомлений Telegram"""
    enabled: bool = True
    bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))


@dataclass
class EmailConfig:
    """Конфигурация уведомлений по электронной почте"""
    enabled: bool = True
    server: str = field(default_factory=lambda: os.getenv("EMAIL_SERVER", ""))
    port: int = field(default_factory=lambda: int(os.getenv("EMAIL_PORT", "587")))
    username: str = field(default_factory=lambda: os.getenv("EMAIL_USERNAME", ""))
    password: str = field(default_factory=lambda: os.getenv("EMAIL_PASSWORD", ""))
    recipients: List[str] = field(default_factory=lambda: os.getenv("EMAIL_RECIPIENTS", "").split(","))
    use_tls: bool = True


@dataclass
class DiscordConfig:
    """Конфигурация уведомлений Discord"""
    enabled: bool = True
    webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))


@dataclass
class LoggingConfig:
    """Конфигурация журналирования"""
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file_path: str = field(default_factory=lambda: os.getenv("LOG_FILE_PATH", "logs/trading_bot.log"))
    rotation_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class ApiConfig:
    """Конфигурация API"""
    host: str = "0.0.0.0"
    port: int = 8000
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", "your-jwt-secret"))
    jwt_expiration: int = 86400  # 24 hours
    rate_limit: int = 100  # requests per minute
    enable_docs: bool = True


@dataclass
class ExchangeConfig:
    """Конфигурация биржи"""
    name: str = "binance"
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    testnet: bool = True
    request_timeout: int = 30
    rate_limit: Dict[str, int] = field(default_factory=lambda: {"max_requests": 20, "time_window": 1})
    auto_adjust_leverage: bool = True
    default_leverage: int = 3


@dataclass
class GeneticOptimizerSettings:
    """Настройки генетического оптимизатора"""
    POPULATION_SIZE: int = 50
    GENERATIONS: int = 30
    CROSSOVER_RATE: float = 0.8
    MUTATION_RATE: float = 0.2
    ELITISM_RATE: float = 0.1
    TOURNAMENT_SIZE: int = 3
    MAX_WORKERS: int = 8
    BACKTEST_INITIAL_BALANCE: float = 10000.0
    BACKTEST_COMMISSION: float = 0.001
    BACKTEST_SLIPPAGE: float = 0.0005
    FITNESS_METRICS: Dict[str, float] = field(default_factory=lambda: {
        "net_profit": 1.0,
        "win_rate": 1.0,
        "profit_factor": 1.0,
        "max_drawdown": -1.0,
        "sharpe_ratio": 1.0,
        "num_trades": 0.5
    })


@dataclass
class DataVisualizerSettings:
    """Настройки визуализатора данных"""
    THEME: str = "dark"
    FIGSIZE: tuple = (14, 8)


@dataclass
class TradingConfig:
    """Конфигурация торговли"""
    mode: str = field(default_factory=lambda: os.getenv("TRADING_MODE", "paper"))
    initial_balance: float = field(default_factory=lambda: float(os.getenv("INITIAL_BALANCE", "10000")))
    max_drawdown_percent: float = field(default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_PERCENT", "20")))
    max_risk_per_trade_percent: float = field(default_factory=lambda: float(os.getenv("MAX_RISK_PER_TRADE_PERCENT", "1")))
    enable_auto_strategy_rotation: bool = field(default_factory=lambda: os.getenv("ENABLE_AUTO_STRATEGY_ROTATION", "false").lower() == "true")
    enable_dynamic_position_sizing: bool = field(default_factory=lambda: os.getenv("ENABLE_DYNAMIC_POSITION_SIZING", "false").lower() == "true")
    default_timeframe: str = field(default_factory=lambda: os.getenv("DEFAULT_TIMEFRAME", "1h"))
    default_trade_duration: str = field(default_factory=lambda: os.getenv("DEFAULT_TRADE_DURATION", "24h"))
    enable_gpu_acceleration: bool = field(default_factory=lambda: os.getenv("ENABLE_GPU_ACCELERATION", "false").lower() == "true")
    default_symbols: List[str] = field(default_factory=list)
    default_strategies: List[str] = field(default_factory=list)


@dataclass
class Config:
    """Главная конфигурация приложения"""
    DATABASE: DatabaseConfig = field(default_factory=DatabaseConfig)
    REDIS: RedisConfig = field(default_factory=RedisConfig)
    INFLUXDB: InfluxDBConfig = field(default_factory=InfluxDBConfig)
    TELEGRAM: TelegramConfig = field(default_factory=TelegramConfig)
    EMAIL: EmailConfig = field(default_factory=EmailConfig)
    DISCORD: DiscordConfig = field(default_factory=DiscordConfig)
    LOGGING: LoggingConfig = field(default_factory=LoggingConfig)
    API: ApiConfig = field(default_factory=ApiConfig)
    EXCHANGE: ExchangeConfig = field(default_factory=ExchangeConfig)
    TRADING: TradingConfig = field(default_factory=TradingConfig)
    GENETIC_OPTIMIZER_SETTINGS: GeneticOptimizerSettings = field(default_factory=GeneticOptimizerSettings)
    DATA_VISUALIZER_SETTINGS: DataVisualizerSettings = field(default_factory=DataVisualizerSettings)
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Загружает конфигурацию из файла

        Args:
            file_path: Путь к файлу конфигурации

        Returns:
            bool: True, если загрузка успешна, иначе False
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                logging.warning(f"Config file not found: {file_path}")
                return False
                
            if path.suffix == '.json':
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
            elif path.suffix in ('.yaml', '.yml'):
                with open(file_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                logging.error(f"Unsupported config file format: {path.suffix}")
                return False
                
            # Обновляем конфигурацию
            self._update_from_dict(config_data)
            
            logging.info(f"Configuration loaded from {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading config from {file_path}: {str(e)}")
            return False
            
    def _update_from_dict(self, config_data: Dict[str, Any]) -> None:
        """
        Обновляет конфигурацию из словаря

        Args:
            config_data: Словарь с конфигурацией
        """
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                
                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
                            
    def get_bot_config(self, symbol: str, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Возвращает конфигурацию для торгового бота

        Args:
            symbol: Торговая пара
            strategy_id: ID стратегии

        Returns:
            Dict[str, Any]: Конфигурация для бота
        """
        # Базовая конфигурация
        bot_config = {
            "symbol": symbol,
            "exchange_id": self.EXCHANGE.name,
            "timeframe": self.TRADING.default_timeframe,
            "strategy_id": strategy_id,
            "leverage": self.EXCHANGE.default_leverage,
            "position_size": self.TRADING.max_risk_per_trade_percent / 100,
            "is_position_size_percentage": True,
            "max_positions": 1,
            "paper_trading": self.TRADING.mode == "paper",
            "backtest_mode": False,
        }
        
        # Если есть дополнительные настройки для символа или стратегии, добавляем их
        # Здесь можно добавить загрузку специфических настроек из БД или файла
        
        return bot_config


# Создаем глобальный экземпляр конфигурации
_config = None


def get_config() -> Config:
    """
    Возвращает глобальный экземпляр конфигурации

    Returns:
        Config: Экземпляр конфигурации
    """
    global _config
    
    if _config is None:
        _config = Config()
        
        # Пытаемся загрузить конфигурацию из файла
        config_file = os.getenv("CONFIG_FILE")
        if config_file:
            _config.load_from_file(config_file)
        
        # Добавляем значения по умолчанию для trading
        _config.TRADING.default_symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        _config.TRADING.default_strategies = ["moving_average_cross", "rsi_strategy", "macd_strategy"]
            
    return _config


def reload_config() -> Config:
    """
    Перезагружает конфигурацию

    Returns:
        Config: Обновленный экземпляр конфигурации
    """
    global _config
    _config = None
    return get_config()
