"""
Утилиты для настройки логирования в приложении.
Предоставляет единый интерфейс для логирования во всех модулях.
"""

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

from project.config import get_config

# Настройки логирования по умолчанию
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Константы
MAX_LOG_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ
BACKUP_COUNT = 5


def setup_logging() -> None:
    """
    Настраивает систему логирования для всего приложения.
    Должна вызываться один раз при запуске.
    """
    try:
        config = get_config()
        log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)

        # Создаем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Очищаем существующие обработчики
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Добавляем обработчик для консоли
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Если указан путь к файлу логов, добавляем файловый обработчик
        if config.LOG_FILE_PATH:
            # Создаем директорию для логов, если она не существует
            log_dir = os.path.dirname(config.LOG_FILE_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = RotatingFileHandler(
                config.LOG_FILE_PATH,
                maxBytes=MAX_LOG_FILE_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
                datefmt=DEFAULT_DATE_FORMAT,
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)

        # Настраиваем сторонние библиотеки на уровень WARNING, чтобы снизить шум в логах
        for logger_name in ["asyncio", "matplotlib", "urllib3", "websockets"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        root_logger.info("Logging initialized at level {config.LOG_LEVEL}" %)

    except Exception as e:
        # Запасной вариант при ошибке конфигурации
        print(f"Error setting up logging: {str(e)}")

        # Базовая настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT,
            stream=sys.stdout,
        )

        logging.error(f"Failed to configure logging properly: {str(e)}", exc_info=True)


def get_logger(name: str) -> logging.Logger:
    """
    Получает настроенный логгер для указанного модуля.

    Args:
        name: Имя модуля или компонента

    Returns:
        Настроенный объект логгера
    """
    # Если это __main__, используем имя файла
    if name == "__main__":
        # Извлекаем имя файла из стека вызовов
        import inspect

        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        if module:
            # Используем имя файла без расширения
            name = os.path.splitext(os.path.basename(module.__file__))[0]

    # Если имя начинается с project, используем только относительный путь
    if name.startswith("project."):
        name = name
    elif not name.startswith("project"):
        name = f"project.{name}"

    return logging.getLogger(name)


def log_execution_time(func):
    """
    Декоратор для логирования времени выполнения функции.

    Args:
        func: Декорируемая функция

    Returns:
        Декорированная функция
    """
    logger = get_logger(func.__module__)

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug("Function %s executed in %.4f seconds", func.__name__, elapsed_time)
        return result

    return wrapper


async def log_async_execution_time(func):
    """
    Декоратор для логирования времени выполнения асинхронной функции.

    Args:
        func: Декорируемая асинхронная функция

    Returns:
        Декорированная асинхронная функция
    """
    logger = get_logger(func.__module__)

    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(
            "Async function %s executed in %.4f seconds", func.__name__, elapsed_time
        )
        return result

    return wrapper


def log_error_with_context(logger, message, error, context=None):
    """Логирует ошибку с дополнительным контекстом"""
    try:
        # Заменяем f-string на % форматирование
        logger.error("%s: %s", message, str(error))
        if context:
            logger.error("Контекст: %s", context)
    except Exception:
        logger.exception("Ошибка при логировании с контекстом")


def setup_file_handler(name, log_file, formatter):
    # ...existing code...
    logger.debug("Настроен файловый обработчик для %s", name)
    # ...existing code...


def setup_console_handler(name, formatter):
    # ...existing code...
    logger.debug("Настроен консольный обработчик для %s", name)
    # ...existing code...


# Настройка логгера для текущего модуля
logger = get_logger(__name__)
