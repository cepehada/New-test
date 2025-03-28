"""
Configuration module for the trading system.
Provides centralized access to application settings.
"""

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SettingsConfig:
    """Data class for storing application settings."""
    
    # Database settings
    DATABASE_URI: str = os.environ.get("DATABASE_URI", "postgresql://postgres:postgres@localhost:5432/trading")
    
    # API settings
    API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("API_PORT", 8000))
    
    # WebSocket settings
    WS_HOST: str = os.environ.get("WS_HOST", "0.0.0.0")
    WS_PORT: int = int(os.environ.get("WS_PORT", 8001))
    
    # Security settings
    SECRET_KEY: str = os.environ.get("SECRET_KEY", "your-secret-key-here")
    ENCRYPTION_KEY: str = os.environ.get("ENCRYPTION_KEY", "your-encryption-key-here")
    AUTH_ENABLED: bool = os.environ.get("AUTH_ENABLED", "true").lower() == "true"
    
    # Message broker settings
    MESSAGE_BROKER_URI: str = os.environ.get("MESSAGE_BROKER_URI", "amqp://guest:guest@localhost:5672/")
    
    # Telegram settings
    TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    
    # Exchange API keys
    BINANCE_API_KEY: str = os.environ.get("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.environ.get("BINANCE_API_SECRET", "")
    
    # Logging settings
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Trading settings
    DEFAULT_COMMISSION: float = float(os.environ.get("DEFAULT_COMMISSION", "0.001"))
    DEFAULT_SLIPPAGE: float = float(os.environ.get("DEFAULT_SLIPPAGE", "0.0005"))
    
    def get_telegram_settings(self) -> 'TelegramSettings':
        """Returns Telegram-specific settings."""
        return TelegramSettings(
            CHAT_ID=self.TELEGRAM_CHAT_ID,
            BOT_TOKEN=self.TELEGRAM_BOT_TOKEN,
            ALLOWED_USERS=os.environ.get("TELEGRAM_ALLOWED_USERS", "").split(",")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class TelegramSettings:
    """Telegram-specific settings."""
    CHAT_ID: str = ""
    BOT_TOKEN: str = ""
    ALLOWED_USERS: list = None
    
    def __post_init__(self):
        if self.ALLOWED_USERS is None:
            self.ALLOWED_USERS = []


class Config:
    """
    Configuration class for the trading system.
    Implements the Singleton pattern to ensure consistent configuration across the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._settings = SettingsConfig()
            cls._instance._load_from_file()
        return cls._instance
    
    def _load_from_file(self):
        """Load settings from a configuration file if it exists."""
        config_file = os.environ.get("CONFIG_FILE", "config.json")
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update settings with values from config file
                for key, value in config_data.items():
                    if hasattr(self._settings, key):
                        setattr(self._settings, key, value)
                        
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_file}: {str(e)}")
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the settings object."""
        return getattr(self._settings, name)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary."""
        return self._settings.to_dict()


def get_config() -> Config:
    """
    Get the application configuration.
    
    Returns:
        Config: The configuration instance
    """
    return Config()