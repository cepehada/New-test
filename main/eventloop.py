"""
Модуль для настройки и управления циклом событий asyncio.
Включает в себя функции для оптимизации производительности и мониторинга.
"""

import asyncio
import signal
import sys
import time
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar

from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Типовые переменные для типизации
T = TypeVar("T")


def setup_event_loop(
    loop: asyncio.AbstractEventLoop = None,
) -> asyncio.AbstractEventLoop:
    """
    Настраивает и оптимизирует цикл событий asyncio.

    Args:
        loop: Существующий цикл событий или None для создания нового

    Returns:
        Настроенный цикл событий
    """
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Настройка обработчиков исключений
    loop.set_exception_handler(exception_handler)

    # Увеличение лимита на количество задач
    loop.set_task_factory(lambda loop, coro: asyncio.Task(coro, loop=loop))

    # Оптимизация параметров цикла событий в зависимости от ОС
    if sys.platform == "win32":
        # Windows-специфичные настройки
        if hasattr(loop, "set_event_loop_policy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
        # Linux/Unix/MacOS-специфичные настройки
        import resource

        # Увеличиваем лимит на количество открытых файлов
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))

    logger.debug(f"Цикл событий настроен: {loop}")
    return loop


def exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """
    Обработчик исключений для цикла событий.

    Args:
        loop: Цикл событий, в котором произошло исключение
        context: Контекст исключения
    """
    exception = context.get("exception")
    if exception is None:
        logger.error(f"Ошибка в цикле событий: {context['message']}")
        return

    if isinstance(exception, asyncio.CancelledError):
        # Нормальное отменение задачи, логируем с debug-уровнем
        logger.debug(f"Задача отменена: {context.get('message')}")
        return

    # Получаем информацию о задаче
    task = context.get("task")
    task_name = task.get_name() if task else "Неизвестная задача"

    logger.error(
        f"Необработанное исключение в задаче {task_name}: {exception}",
        exc_info=exception,
    )

    # Проверяем, является ли ошибка критической
    if isinstance(exception, (SystemExit, KeyboardInterrupt)):
        logger.critical("Получен сигнал завершения работы.")
        # Отправляем сигнал для корректного завершения
        signal.raise_signal(signal.SIGTERM)


def with_timeout(
    timeout: float,
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]
]:
    """
    Декоратор для добавления таймаута к корутинам.

    Args:
        timeout: Таймаут в секундах

    Returns:
        Декорированная корутина с таймаутом
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

        return wrapper

    return decorator


async def periodic(
    period: float,
    func: Callable[..., Coroutine[Any, Any, T]],
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Запускает функцию периодически с указанным интервалом.

    Args:
        period: Интервал в секундах
        func: Функция для периодического выполнения
        *args: Аргументы для функции
        **kwargs: Именованные аргументы для функции
    """
    while True:
        try:
            start_time = time.monotonic()
            await func(*args, **kwargs)
            elapsed = time.monotonic() - start_time

            # Логируем, если выполнение заняло больше половины периода
            if elapsed > period / 2:
                logger.warning(
                    f"Функция {func.__name__} выполнялась {elapsed:.2f}с "
                    f"(период: {period}с)"
                )

            # Ждем до следующего запуска
            sleep_time = max(0.0, period - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            # Нормальное отменение задачи
            logger.debug(f"Периодическая задача {func.__name__} отменена")
            break
        except Exception as e:
            logger.error(
                f"Ошибка в периодической задаче {func.__name__}: {str(e)}",
                exc_info=True,
            )
            # Ждем перед следующей попыткой
            await asyncio.sleep(period)


async def gather_with_concurrency(n: int, *tasks: Coroutine[Any, Any, T]) -> list[T]:
    """
    Запускает сопрограммы с ограничением на количество одновременно выполняемых задач.

    Args:
        n: Максимальное количество одновременно выполняемых задач
        *tasks: Сопрограммы для выполнения

    Returns:
        Список результатов выполнения задач
    """
    semaphore = asyncio.Semaphore(n)

    async def task_with_semaphore(task: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*(task_with_semaphore(task) for task in tasks))
