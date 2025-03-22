"""
Модуль логирования.
Настраивает логгер с ротацией по времени.
"""

import logging
from logging.handlers import TimedRotatingFileHandler

def configure_logging(log_conf: dict) -> None:
    """
    Настраивает корневой логгер согласно параметрам.

    Args:
        log_conf (dict): Словарь с ключами "level", "format", "file".
    """
    logger = logging.getLogger()
    logger.setLevel(log_conf.get("level", "INFO"))
    fmt = logging.Formatter(log_conf.get("format"))
    fh = TimedRotatingFileHandler(
        log_conf.get("file", "app.log"),
        when="midnight",
        backupCount=7
    )
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

