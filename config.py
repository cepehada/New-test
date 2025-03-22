from pydantic import BaseSettings, Field
from typing import List

class RedisSettings(BaseSettings):
    host: str = Field(..., env="REDIS__HOST")
    port: int = Field(..., env="REDIS__PORT")
    password: str = Field("", env="REDIS__PASSWORD")

    class Config:
        env_prefix = ""
        env_file = "/app/.env"

class APISettings(BaseSettings):
    BYBIT_API_KEY: str = Field(..., env="API__BYBIT_API_KEY")
    BYBIT_SECRET_KEY: str = Field(..., env="API__BYBIT_SECRET_KEY")
    MEXC_API_KEY: str = Field(..., env="API__MEXC_API_KEY")
    MEXC_SECRET_KEY: str = Field(..., env="API__MEXC_SECRET_KEY")
    PHEMEX_API_KEY: str = Field(..., env="API__PHEMEX_API_KEY")
    PHEMEX_SECRET_KEY: str = Field(..., env="API__PHEMEX_SECRET_KEY")
    HTX_API_KEY: str = Field(..., env="API__HTX_API_KEY")
    HTX_SECRET_KEY: str = Field(..., env="API__HTX_SECRET_KEY")

    class Config:
        env_prefix = ""
        env_file = "/app/.env"

class NewsSettings(BaseSettings):
    SOURCES: str = Field(..., env="NEWS__SOURCES")
    ARTICLE_LIMIT: int = Field(..., env="NEWS__ARTICLE_LIMIT")
    LONG_KEYWORDS: str = Field(..., env="NEWS__LONG_KEYWORDS")
    SHORT_KEYWORDS: str = Field(..., env="NEWS__SHORT_KEYWORDS")
    EXIT_KEYWORDS: str = Field(..., env="NEWS__EXIT_KEYWORDS")
    UPDATE_INTERVAL: int = Field(..., env="NEWS__UPDATE_INTERVAL")

    class Config:
        env_prefix = ""
        env_file = "/app/.env"

class SystemSettings(BaseSettings):
    MODE: str = Field(..., env="SYSTEM__MODE")
    CACHE_TTL: int = Field(..., env="SYSTEM__CACHE_TTL")
    SECRETS_MANAGER_URL: str = Field(..., env="SYSTEM__SECRETS_MANAGER_URL")
    JWT_SECRET: str = Field(..., env="SYSTEM__JWT_SECRET")
    JWT_ALGORITHM: str = Field(..., env="SYSTEM__JWT_ALGORITHM")

    class Config:
        env_prefix = ""
        env_file = "/app/.env"

class Settings(BaseSettings):
    redis: RedisSettings = RedisSettings()
    api: APISettings = APISettings()
    news: NewsSettings = NewsSettings()
    system: SystemSettings = SystemSettings()

    MAX_ACTIVE_TRADES: int = Field(..., env="MAX_ACTIVE_TRADES")
    DAILY_LOSS_LIMIT: float = Field(..., env="DAILY_LOSS_LIMIT")
    TRADE_SIZE_PERCENT: float = Field(..., env="TRADE_SIZE_PERCENT")
    MIN_PROFIT: float = Field(..., env="MIN_PROFIT")
    symbols: List[str] = []

    class Config:
        env_file = "/app/.env"

config = Settings()

def load_config():
    return config
