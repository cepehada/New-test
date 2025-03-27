"""
Модуль для настройки и управления логированием в приложении.
Предоставляет функции для получения логгеров и настройки форматирования.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Глобальная настройка логирования
_logging_initialized = False
_loggers = {}
logger = logging.getLogger(__name__)

# Цвета для логов в консоли
COLORS = {
    'CRITICAL': '\033[91m',  # Красный
    'ERROR': '\033[91m',     # Красный
    'WARNING': '\033[93m',   # Желтый
    'INFO': '\033[92m',      # Зеленый
    'DEBUG': '\033[94m',     # Синий
    'RESET': '\033[0m',      # Сброс цвета
}


def setup_logging(
    log_level=None, 
    log_file=None, 
    console_format=None, 
    file_format=None
):
    """
    Настраивает логирование приложения.
    
    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу логов
        console_format: Формат логов для консоли
        file_format: Формат логов для файла
    """
    global _logging_initialized
    
    if _logging_initialized:
        return
    
    # Получение настроек из конфигурации
    try:
        from project.config import get_config
        config = get_config()
        
        log_level = log_level or config.LOG_LEVEL
        log_file = log_file or config.LOG_FILE_PATH
        console_format = console_format or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        file_format = file_format or "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        
    except Exception as e:
        # Если конфигурация недоступна, используем значения по умолчанию
        logger.warning("Не удалось загрузить конфигурацию: %s", str(e))
        log_level = log_level or "INFO"
        console_format = console_format or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        file_format = file_format or "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
    
    # Настройка уровня логирования
    level = getattr(logging, log_level.upper())
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Очистка всех обработчиков
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(console_format))
    root_logger.addHandler(console_handler)
    
    # Файловый обработчик (если задан)
    if log_file:
        # Создаем директорию для логов, если она не существует
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)
    
    # Отключаем логи от библиотек
    for log_name in ['urllib3', 'requests', 'ccxt', 'matplotlib', 'asyncio', 'websockets']:
        logging.getLogger(log_name).setLevel(logging.WARNING)
    
    _logging_initialized = True
    logger.info("Логирование настроено. Уровень: %s, Файл логов: %s", log_level, log_file or "не задан")


def get_logger(name=None):
    """
    Возвращает настроенный логгер.
    
    Args:
        name: Имя логгера
        
    Returns:
        Настроенный логгер
    """
    # Определяем имя модуля
    if name is None:
        # Получаем имя вызывающего модуля
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__
    
    # Проверяем, не является ли имя файлом
    if name == '__main__':
        name = name  # Оставляем как есть
    
    # Проверяем, если логгер уже существует
    if name in _loggers:
        return _loggers[name]
    
    # Настраиваем логирование, если еще не настроено
    if not _logging_initialized:
        setup_logging()
    
    # Создаем логгер
    new_logger = logging.getLogger(name)
    _loggers[name] = new_logger
    
    return new_logger


def get_function_logger(current_module=None):
    """
    Получает логгер для текущей функции.
    
    Args:
        current_module: Текущий модуль
        
    Returns:
        Логгер для текущей функции
    """
    import inspect
    
    # Получаем стек вызовов
    stack = inspect.stack()
    
    # Имя вызывающей функции
    calling_func = stack[1].function
    
    # Модуль, из которого вызвана функция
    if current_module is None:
        module = inspect.getmodule(stack[1].frame)
        module_name = module.__name__ if module else "__main__"
    else:
        module_name = current_module.__name__
    
    # Создаем имя логгера
    logger_name = f"{module_name}.{calling_func}"
    
    # Получаем логгер
    function_logger = get_logger(logger_name)
    
    return function_logger


class ColoredFormatter(logging.Formatter):
    """
    Форматтер для цветных логов в консоли.
    """
    
    def format(self, record):
        """
        Форматирует запись лога с цветовым выделением.
        
        Args:
            record: Запись лога
            
        Returns:
            Отформатированная строка лога
        """
        log_message = super().format(record)
        
        # Добавляем цвета только если вывод идет в терминал
        if sys.stdout.isatty():
            levelname = record.levelname
            if levelname in COLORS:
                return f"{COLORS[levelname]}{log_message}{COLORS['RESET']}"
        
        return log_message
