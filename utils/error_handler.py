"""
Модуль для централизованной обработки ошибок.
Обеспечивает единообразную обработку исключений во всем приложении.
"""

import sys
import traceback
import functools
import logging
import asyncio
from typing import Any, Callable, TypeVar, cast, Optional, Dict, Type, Union

from project.utils.logging_utils import get_logger

# Типовые переменные для типизации функций
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

logger = get_logger(__name__)

# Словарь для регистрации обработчиков ошибок
_error_handlers: Dict[Type[Exception], Callable] = {}


def register_error_handler(
    exception_type: Type[Exception], handler: Callable[[Exception], Any]
) -> None:
    """
    Регистрирует обработчик для определенного типа исключения.

    Args:
        exception_type: Тип исключения, для которого регистрируется обработчик
        handler: Функция-обработчик ошибки
    """
    _error_handlers[exception_type] = handler
    logger.debug(f"Зарегистрирован обработчик для {exception_type.__name__}")


def handle_exception(exc: Exception) -> Any:
    """
    Обрабатывает исключение с использованием зарегистрированных обработчиков.

    Args:
        exc: Экземпляр исключения для обработки

    Returns:
        Результат обработки исключения или None, если обработчик не найден
    """
    # Ищем наиболее специфичный обработчик для данного типа исключения
    for exc_type, handler in _error_handlers.items():
        if isinstance(exc, exc_type):
            logger.debug(
                f"Найден обработчик для {type(exc).__name__}: {handler.__name__}"
            )
            return handler(exc)

    # Если специфичный обработчик не найден, используем общий
    logger.warning(
        f"Обработчик для {type(exc).__name__} не найден, "
        f"используем стандартную обработку"
    )

    # Логируем исключение
    logger.error(f"Необработанное исключение: {str(exc)}", exc_info=exc)

    # Для асинхронных отмен не выполняем дополнительных действий
    if isinstance(exc, asyncio.CancelledError):
        logger.debug("Обработка отмены асинхронной операции")
        return None

    # Для критических исключений завершаем программу
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        logger.critical(f"Критическое исключение: {type(exc).__name__}")
        raise exc

    return None


def handle_error(func: F) -> F:
    """
    Декоратор для обработки ошибок в функциях.

    Args:
        func: Декорируемая функция

    Returns:
        Декорированная функция
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return handle_exception(e)

    return cast(F, wrapper)


def async_handle_error(func: F) -> F:
    """
    Декоратор для обработки ошибок в асинхронных функциях.

    Args:
        func: Декорируемая асинхронная функция

    Returns:
        Декорированная асинхронная функция
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return handle_exception(e)

    return cast(F, wrapper)


def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Декоратор для повторных попыток выполнения функции при возникновении исключения.

    Args:
        max_retries: Максимальное количество повторных попыток
        retry_delay: Задержка между попытками в секундах
        exceptions: Типы исключений, при которых выполняются повторные попытки
        logger: Логгер для записи информации о повторных попытках

    Returns:
        Декоратор для функции
    """

    def decorator(func: F) -> F:
        _logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        _logger.warning(
                            f"Attempt {attempt}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {retry_delay} seconds..."
                        )
                        import time

                        time.sleep(retry_delay)
                    else:
                        _logger.error(f"All {max_retries} attempts failed.")
                        raise last_exception

            # Этот код не должен выполниться, но на всякий случай
            if last_exception:
                raise last_exception
            return None

        return cast(F, wrapper)

    return decorator


def async_with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None,
) -> Callable[[F], F]:
    """
    Декоратор для повторных попыток выполнения асинхронной функции при возникновении исключения.

    Args:
        max_retries: Максимальное количество повторных попыток
        retry_delay: Задержка между попытками в секундах
        exceptions: Типы исключений, при которых выполняются повторные попытки
        logger: Логгер для записи информации о повторных попытках

    Returns:
        Декоратор для асинхронной функции
    """

    def decorator(func: F) -> F:
        _logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        _logger.warning(
                            f"Attempt {attempt}/{max_retries} failed: {str(e)}. "
                            f"Retrying in {retry_delay} seconds..."
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        _logger.error(f"All {max_retries} attempts failed.")
                        raise last_exception

            # Этот код не должен выполниться, но на всякий случай
            if last_exception:
                raise last_exception
            return None

        return cast(F, wrapper)

    return decorator


def setup_error_handlers() -> None:
    """
    Настраивает обработчики ошибок по умолчанию.
    Должна вызываться при инициализации приложения.
    """
    # Регистрируем стандартные обработчики ошибок
    register_error_handler(
        ConnectionError, lambda e: logger.error(f"Ошибка соединения: {str(e)}")
    )
    register_error_handler(
        TimeoutError, lambda e: logger.error(f"Таймаут операции: {str(e)}")
    )
    register_error_handler(
        ValueError, lambda e: logger.error(f"Ошибка значения: {str(e)}")
    )
    register_error_handler(KeyError, lambda e: logger.error(f"Ошибка ключа: {str(e)}"))

    # Настраиваем обработчик неперехваченных исключений
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Для Ctrl+C выводим короткое сообщение и выполняем стандартную обработку
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Неперехваченное исключение:", exc_info=(exc_type, exc_value, exc_traceback)
        )

    # Устанавливаем глобальный обработчик исключений
    sys.excepthook = handle_uncaught_exception

    logger.info("Обработчики ошибок настроены")


# Частые обработчики ошибок для повторного использования
def log_and_ignore_error(e: Exception) -> None:
    """
    Логирует ошибку и игнорирует её.

    Args:
        e: Исключение для логирования
    """
    logger.warning(f"Игнорируемая ошибка: {str(e)}")
    return None


def log_and_raise_error(e: Exception) -> None:
    """
    Логирует ошибку и пробрасывает её дальше.

    Args:
        e: Исключение для логирования и пробрасывания
    """
    logger.error(f"Ошибка требует дальнейшей обработки: {str(e)}")
    raise e


def log_and_return_fallback(fallback_value: T) -> Callable[[Exception], T]:
    """
    Создает обработчик, который логирует ошибку и возвращает запасное значение.

    Args:
        fallback_value: Запасное значение для возврата

    Returns:
        Функция-обработчик ошибки
    """

    def handler(e: Exception) -> T:
        logger.warning(f"Ошибка, возвращаем запасное значение: {str(e)}")
        return fallback_value

    return handler


def handle_exceptions(func):
    """
    Decorator to handle exceptions in a standardized way.
    
    Args:
        func: The function to wrap with exception handling
        
    Returns:
        Wrapped function with exception handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (RequestError, NetworkError) as e:
            # Handle specific API-related errors
            logger.error("API error occurred: %s", str(e))
            raise
        except (ValueError, TypeError) as e:
            # Handle input validation errors
            logger.error("Input validation error: %s", str(e))
            raise
        except Exception as e:
            # Handle unexpected errors
            logger.error("Unexpected error: %s", str(e), exc_info=True)
            raise
    return wrapper
