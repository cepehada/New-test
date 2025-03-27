"""
Модуль для обработки ошибок и исключений.
Предоставляет декораторы и функции для стандартизованной обработки ошибок.
"""
import asyncio
import functools
import logging
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from project.utils.logging_utils import get_logger

F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Глобальный логгер для модуля обработки ошибок
logger = get_logger(__name__)

# Словарь для хранения обработчиков ошибок
# Ключ - класс исключения, значение - функция-обработчик
_error_handlers: Dict[type, Callable[[Exception], Any]] = {}


def register_error_handler(
    exception_type: type, handler: Callable[[Exception], Any]
) -> None:
    """
    Регистрирует обработчик ошибок для определенного типа исключений.

    Args:
        exception_type: Класс исключения
        handler: Функция-обработчик
    """
    _error_handlers[exception_type] = handler
    logger.debug("Зарегистрирован обработчик для %s: %s", exception_type.__name__, handler.__name__)


def get_error_handler(exception_type: type) -> Optional[Callable[[Exception], Any]]:
    """
    Получает обработчик для указанного типа исключения.

    Args:
        exception_type: Класс исключения

    Returns:
        Функция-обработчик или None, если обработчик не найден
    """
    for exc_type, handler in _error_handlers.items():
        if issubclass(exception_type, exc_type):
            logger.debug("Найден обработчик %s для %s", handler.__name__, exception_type.__name__)
            return handler
    logger.debug("Не найден обработчик для %s", exception_type.__name__)
    return None


def handle_error(func: F) -> F:
    """
    Декоратор для обработки исключений с помощью зарегистрированных обработчиков.
    
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
            exc_type = type(e)
            handler = get_error_handler(exc_type)
            
            if handler:
                logger.debug("Применение обработчика %s для исключения %s", handler.__name__, exc_type.__name__)
                return handler(e)
            else:
                logger.error("Необработанное исключение: %s: %s", exc_type.__name__, str(e))
                raise

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
                            "Attempt %s/%s failed: %s. "
                            "Retrying in %s seconds...",
                            attempt, max_retries, str(e), retry_delay
                        )
                        import time

                        time.sleep(retry_delay)
                    else:
                        _logger.error("All %s attempts failed.", max_retries)
                        raise last_exception from e

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
                            "Attempt %s/%s failed: %s. "
                            "Retrying in %s seconds...",
                            attempt, max_retries, str(e), retry_delay
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        _logger.error("All %s attempts failed.", max_retries)
                        raise last_exception from e

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
        ConnectionError, lambda e: logger.error("Ошибка соединения: %s", str(e))
    )
    register_error_handler(
        TimeoutError, lambda e: logger.error("Таймаут операции: %s", str(e))
    )
    register_error_handler(
        ValueError, lambda e: logger.error("Ошибка значения: %s", str(e))
    )
    register_error_handler(KeyError, lambda e: logger.error("Ошибка ключа: %s", str(e)))

    # Настраиваем обработчик неперехваченных исключений
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Для Ctrl+C выводим короткое сообщение и выполняем стандартную обработку
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "Неперехваченное исключение:",
            exc_info=(exc_type, exc_value, exc_traceback)
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
    logger.warning("Игнорируемая ошибка: %s", str(e))
    return None


def log_and_raise_error(e: Exception) -> None:
    """
    Логирует ошибку и пробрасывает её дальше.

    Args:
        e: Исключение для логирования и пробрасывания
    """
    logger.error("Ошибка требует дальнейшей обработки: %s", str(e))
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
        logger.warning("Ошибка, возвращаем запасное значение: %s", str(e))
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


def log_error(message, e, *args, **kwargs):
    """Логирует ошибку с сообщением"""
    # Заменяем f-string на % форматирование
    logger.error("%s: %s", message, str(e))


async def async_retry(func, max_retries=3, delay=1, *args, **kwargs):
    """Выполняет асинхронную функцию с повторными попытками в случае ошибки"""
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            logger.warning("Попытка %d из %d не удалась: %s", 
                          attempt + 1, max_retries, str(e))
            await asyncio.sleep(delay * (2 ** attempt))
    
    # Если все попытки не удались, выбрасываем последнее исключение
    if last_exception:
        raise last_exception
        
    return None  # Это никогда не должно выполняться, но для удовлетворения линтера


# Определяем классы ошибок, которые используются в handle_exceptions
class RequestError(Exception):
    """Ошибка запроса к API"""
    pass


class NetworkError(Exception):
    """Ошибка сети"""
    pass
