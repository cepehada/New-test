"""
Scheduler.
Планировщик задач для периодического запуска сервисов.
Регистрирует и выполняет задачи с заданным интервалом.
"""

import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger("Scheduler")

class Scheduler:
    """
    Класс Scheduler.
    
    Позволяет регистрировать и запускать задачи,
    выполняемые асинхронно с заданными интервалами.
    """

    def __init__(self) -> None:
        self.tasks = []

    def add_task(self, func: Callable[..., Any], interval: int,
                 *args, **kwargs) -> None:
        """
        Регистрирует задачу для периодического выполнения.
        
        Args:
            func (Callable[..., Any]): Функция задачи.
            interval (int): Интервал выполнения в секундах.
            *args: Позиционные аргументы для функции.
            **kwargs: Именованные аргументы для функции.
        """
        self.tasks.append({
            "func": func,
            "interval": interval,
            "args": args,
            "kwargs": kwargs
        })
        logger.info(
            f"Задача {func.__name__} добавлена с интервалом {interval} сек."
        )

    async def _run_task(self, func: Callable[..., Any],
                        interval: int, *args, **kwargs) -> None:
        """
        Запускает задачу в бесконечном цикле с заданным интервалом.
        
        Args:
            func (Callable[..., Any]): Функция задачи.
            interval (int): Интервал в секундах.
            *args: Позиционные аргументы.
            **kwargs: Именованные аргументы.
        """
        while True:
            try:
                await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Ошибка задачи {func.__name__}: {e}"
                )
            await asyncio.sleep(interval)

    async def start(self) -> None:
        """
        Запускает все зарегистрированные задачи параллельно.
        """
        logger.info("Планировщик задач запущен")
        await asyncio.gather(*[
            self._run_task(task["func"],
                           task["interval"],
                           *task["args"],
                           **task["kwargs"])
            for task in self.tasks
        ])

if __name__ == "__main__":
    async def sample_task():
        print("Задача выполняется:", asyncio.get_running_loop().time())

    sched = Scheduler()
    sched.add_task(sample_task, interval=5)
    asyncio.run(sched.start())
