"""
Модуль для обработки исключений и ошибок в приложении.
Предоставляет декораторы для обработки ошибок в асинхронных и синхронных функциях.
"""

import asyncio
import functools
import inspect
import sys
from typing import Any, Callable, Dict, Optional

from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


def handle_error(func):
    """
    Декоратор для обработки ошибок в синхронных функциях.
    
    Args:
        func: Оборачиваемая функция
        
    Returns:
        Обернутая функция с обработкой ошибок
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Обертка функции с обработкой ошибок."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Получаем имя функции для лога
            func_name = func.__qualname__
            logger.error("Ошибка в %s: %s", func_name, str(e))
            
            # Возвращаем None при ошибке
            return None
    return wrapper


def async_handle_error(func):
    """
    Декоратор для обработки ошибок в асинхронных функциях.
    
    Args:
        func: Оборачиваемая корутина
        
    Returns:
        Обернутая корутина с обработкой ошибок
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        """Асинхронная обертка функции с обработкой ошибок."""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Получаем имя функции для лога
            func_name = func.__qualname__
            logger.error("Ошибка в асинхронной функции %s: %s", func_name, str(e))
            
            # Возвращаем None при ошибке
            return None
    return wrapper


def notify_on_error(func=None, *, notification_type="log"):
    """
    Декоратор для уведомления об ошибках.
    
    Args:
        func: Оборачиваемая функция
        notification_type: Тип уведомления (log, telegram, email)
        
    Returns:
        Обернутая функция с уведомлениями об ошибках
    """
    def decorator(func):
        """Внутренний декоратор."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Обертка функции с уведомлениями об ошибках."""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Получаем имя функции для лога
                func_name = func.__qualname__
                error_message = f"Ошибка в {func_name}: {str(e)}"
                
                # Выбираем тип уведомления
                if notification_type == "log":
                    logger.error("%s", error_message)
                elif notification_type == "telegram":
                    # Уведомление через телеграм
                    logger.error("%s", error_message)
                    # TODO: Добавить отправку в телеграм
                elif notification_type == "email":
                    # Уведомление по email
                    logger.error("%s", error_message)
                    # TODO: Добавить отправку на email
                
                # Пробрасываем исключение дальше
                raise
        return wrapper
    
    # Обработка случаев вызова с аргументами и без
    if func is None:
        return decorator
    return decorator(func)


def async_notify_on_error(func=None, *, notification_type="log"):
    """
    Декоратор для уведомления об ошибках в асинхронных функциях.
    
    Args:
        func: Оборачиваемая асинхронная функция
        notification_type: Тип уведомления (log, telegram, email)
        
    Returns:
        Обернутая асинхронная функция с уведомлениями об ошибках
    """
    def decorator(func):
        """Внутренний декоратор."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """Асинхронная обертка функции с уведомлениями об ошибках."""
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Для отслеживания ошибок может понадобиться импортировать время
                import time
                
                # Получаем имя функции для лога
                func_name = func.__qualname__
                error_message = f"Ошибка в асинхронной функции {func_name}: {str(e)}"
                timestamp = time.strftime("%Y-%м-%d %H:%M:%S")
                
                # Выбираем тип уведомления
                if notification_type == "log":
                    logger.error("%s [%s]", error_message, timestamp)
                elif notification_type == "telegram":
                    # Уведомление через телеграм
                    logger.error("%s [%s]", error_message, timestamp)
                    # TODO: Добавить отправку в телеграм
                elif notification_type == "email":
                    # Уведомление по email
                    logger.error("%s [%s]", error_message, timestamp)
                    # TODO: Добавить отправку на email
                
                # Пробрасываем исключение дальше
                raise e
        return wrapper
    
    # Обработка случаев вызова с аргументами и без
    if func is None:
        return decorator
    return decorator(func)


def async_with_retry(
    func=None, 
    *, 
    retries=3, 
    delay=1, 
    backoff=2, 
    exceptions=None
):
    """
    Декоратор для повторных попыток выполнения асинхронной функции при ошибке.
    
    Args:
        func: Оборачиваемая асинхронная функция
        retries: Количество повторных попыток
        delay: Начальная задержка между попытками (в секундах)
        backoff: Коэффициент увеличения задержки
        exceptions: Список исключений для перехвата (по умолчанию все)
        
    Returns:
        Обернутая асинхронная функция с повторными попытками
    """
    def decorator(func):
        """Внутренний декоратор."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            """Асинхронная обертка функции с повторными попытками."""
            local_logger = get_logger(f"{func.__module__}.{func.__qualname__}")
            
            # Настройка перехватываемых исключений
            nonlocal exceptions
            if exceptions is None:
                exceptions = (Exception,)
            
            # Счетчик попыток
            attempt = 0
            current_delay = delay
            last_exception = None
            
            # Попытки выполнения с повторами
            while attempt < retries + 1:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    last_exception = e
                    
                    if attempt > retries:
                        local_logger.error(
                            "Превышено количество попыток (%d) для %s: %s", 
                            retries, func.__qualname__, str(e)
                        )
                        raise last_exception
                    
                    # Логируем информацию о повторе
                    local_logger.warning(
                        "Попытка %d/%d для %s не удалась: %s. Повтор через %.2f с",
                        attempt, retries, func.__qualname__, str(e), current_delay
                    )
                    
                    # Задержка перед следующей попыткой
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    
    # Обработка случаев вызова с аргументами и без
    if func is None:
        return decorator
    return decorator(func)


def log_call(func):
    """
    Декоратор для логирования вызовов функции.
    
    Args:
        func: Оборачиваемая функция
        
    Returns:
        Обернутая функция с логированием вызовов
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Обертка функции с логированием вызовов."""
        # Получаем имя функции
        func_name = func.__qualname__
        
        # Логируем вызов
        logger.debug("Вызов функции %s с аргументами %s, %s", func_name, args, kwargs)
        
        # Выполняем функцию
        result = func(*args, **kwargs)
        
        # Логируем завершение
        logger.debug("Функция %s завершена с результатом: %s", func_name, result)
        
        return result
    return wrapper


def async_log_call(func):
    """
    Декоратор для логирования вызовов асинхронной функции.
    
    Args:
        func: Оборачиваемая асинхронная функция
        
    Returns:
        Обернутая асинхронная функция с логированием вызовов
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        """Асинхронная обертка функции с логированием вызовов."""
        # Получаем имя функции
        func_name = func.__qualname__
        
        # Логируем вызов
        logger.debug("Вызов асинхронной функции %s с аргументами %s, %s", 
                    func_name, args, kwargs)
        
        # Выполняем функцию
        result = await func(*args, **kwargs)
        
        # Логируем завершение
        logger.debug("Асинхронная функция %s завершена с результатом: %s", 
                    func_name, result)
        
        return result
    return wrapper


def report_error(error_message, exc_info=None):
    """
    Сообщает об ошибке и отправляет уведомление.
    
    Args:
        error_message: Сообщение об ошибке
        exc_info: Информация об исключении (sys.exc_info())
    """
    # Логируем ошибку
    logger.error("%s", error_message, exc_info=exc_info)
    
    # TODO: Отправка уведомления через нужный канал
