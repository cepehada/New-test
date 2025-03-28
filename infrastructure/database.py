"""
Модуль для работы с базой данных.
Предоставляет единый интерфейс для выполнения операций с базой данных.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

try:
    import asyncpg
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "asyncpg"])
    import asyncpg

from project.config import get_config
from project.utils.error_handler import async_handle_error, async_with_retry
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Database:
    """
    Класс для взаимодействия с базой данных PostgreSQL.
    Реализует паттерн Singleton для предотвращения создания множества подключений.
    """

    _instance = None

    @classmethod
    def get_instance(cls, connection_string: Optional[str] = None) -> "Database":
        """
        Получает экземпляр базы данных (Singleton).

        Args:
            connection_string: Строка подключения к БД (None для использования из конфигурации)

        Returns:
            Экземпляр класса Database
        """
        if cls._instance is None:
            cls._instance = cls(connection_string)
        return cls._instance

    def __init__(self, connection_string: Optional[str] = None):
        """
        Инициализирует объект базы данных.

        Args:
            connection_string: Строка подключения к БД (None для использования из конфигурации)
        """
        self.config = get_config()
        self.connection_string = connection_string or self.config.DATABASE_URI
        self.pool = None
        logger.debug("Создан экземпляр базы данных")

    @async_with_retry(
        max_retries=3, retry_delay=1.0, exceptions=(asyncpg.PostgresError,)
    )
    async def initialize(self) -> None:
        """
        Инициализирует пул соединений с базой данных.
        """
        if self.pool is None:
            try:
                logger.info("Инициализация пула соединений с базой данных")
                self.pool = await asyncpg.create_pool(
                    dsn=self.connection_string, min_size=5, max_size=20, timeout=30.0
                )
                logger.info("Пул соединений с базой данных успешно создан")
            except Exception as e:
                logger.error(f"Ошибка при инициализации пула соединений: {str(e)}")
                raise

    @asynccontextmanager
    async def connection(self):
        """
        Контекстный менеджер для получения соединения из пула.

        Yields:
            Соединение с базой данных
        """
        if self.pool is None:
            await self.initialize()

        async with self.pool.acquire() as conn:
            try:
                yield conn
            except Exception as e:
                logger.error(f"Ошибка при работе с соединением: {str(e)}")
                raise

    @async_handle_error
    async def execute(self, query: str, *args) -> str:
        """
        Выполняет SQL-запрос без возврата результатов.

        Args:
            query: SQL-запрос
            *args: Параметры запроса

        Returns:
            Статус выполнения запроса
        """
        if self.pool is None:
            await self.initialize()

        async with self.connection() as conn:
            result = await conn.execute(query, *args)
            logger.debug(f"Выполнен запрос: {query[:100]}...")
            return result

    @async_handle_error
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Выполняет SQL-запрос и возвращает все строки результата.

        Args:
            query: SQL-запрос
            *args: Параметры запроса

        Returns:
            Список словарей с результатами запроса
        """
        if self.pool is None:
            await self.initialize()

        async with self.connection() as conn:
            rows = await conn.fetch(query, *args)
            result = [dict(row) for row in rows]
            logger.debug(f"Запрос вернул {len(result)} строк: {query[:100]}...")
            return result

    @async_handle_error
    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """
        Выполняет SQL-запрос и возвращает одну строку результата.

        Args:
            query: SQL-запрос
            *args: Параметры запроса

        Returns:
            Словарь с результатом запроса или None, если результат пуст
        """
        if self.pool is None:
            await self.initialize()

        async with self.connection() as conn:
            row = await conn.fetchrow(query, *args)
            result = dict(row) if row else None
            logger.debug(
                f"Запрос вернул {'одну строку' if result else 'пустой результат'}: {query[:100]}..."
            )
            return result

    @async_handle_error
    async def fetchval(self, query: str, *args) -> Any:
        """
        Выполняет SQL-запрос и возвращает одно значение.

        Args:
            query: SQL-запрос
            *args: Параметры запроса

        Returns:
            Значение результата запроса или None, если результат пуст
        """
        if self.pool is None:
            await self.initialize()

        async with self.connection() as conn:
            result = await conn.fetchval(query, *args)
            logger.debug(f"Запрос вернул одно значение: {query[:100]}...")
            return result

    @async_handle_error
    async def transaction(self, func, *args, **kwargs) -> Any:
        """
        Выполняет функцию в рамках транзакции.

        Args:
            func: Асинхронная функция, принимающая соединение и другие аргументы
            *args: Дополнительные аргументы для функции
            **kwargs: Дополнительные именованные аргументы для функции

        Returns:
            Результат выполнения функции
        """
        if self.pool is None:
            await self.initialize()

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                return await func(conn, *args, **kwargs)

    async def close(self) -> None:
        """
        Закрывает пул соединений с базой данных.
        """
        if self.pool is not None:
            logger.info("Закрытие пула соединений с базой данных")
            await self.pool.close()
            self.pool = None
            logger.info("Пул соединений с базой данных закрыт")

    # Типовые операции с данными

    @async_handle_error
    async def insert(
        self, table: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Вставляет данные в таблицу и возвращает вставленную строку.

        Args:
            table: Имя таблицы
            data: Словарь с данными для вставки

        Returns:
            Словарь с вставленными данными или None при ошибке
        """
        if not data:
            logger.warning(f"Попытка вставки пустых данных в таблицу {table}")
            return None

        columns = list(data.keys())
        values = list(data.values())

        placeholders = [f"${i+1}" for i in range(len(values))]

        query = f"""
        INSERT INTO {table} ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        RETURNING *
        """

        return await self.fetchrow(query, *values)

    @async_handle_error
    async def update(
        self, table: str, data: Dict[str, Any], condition: str, *condition_args
    ) -> Optional[Dict[str, Any]]:
        """
        Обновляет данные в таблице и возвращает обновленную строку.

        Args:
            table: Имя таблицы
            data: Словарь с данными для обновления
            condition: Условие обновления (например, "id = $1")
            *condition_args: Аргументы для условия

        Returns:
            Словарь с обновленными данными или None при ошибке
        """
        if not data:
            logger.warning(f"Попытка обновления пустыми данными в таблице {table}")
            return None

        set_parts = []
        values = []

        for i, (column, value) in enumerate(data.items()):
            set_parts.append(f"{column} = ${i+1}")
            values.append(value)

        # Добавляем аргументы условия в конец списка значений
        placeholder_offset = len(values) + 1
        condition_with_placeholders = condition.replace("$1", f"${placeholder_offset}")
        for i in range(1, len(condition_args)):
            condition_with_placeholders = condition_with_placeholders.replace(
                f"${i+1}", f"${placeholder_offset+i}"
            )

        values.extend(condition_args)

        query = f"""
        UPDATE {table}
        SET {', '.join(set_parts)}
        WHERE {condition_with_placeholders}
        RETURNING *
        """

        return await self.fetchrow(query, *values)

    @async_handle_error
    async def delete(self, table: str, condition: str, *condition_args) -> int:
        """
        Удаляет данные из таблицы и возвращает количество удаленных строк.

        Args:
            table: Имя таблицы
            condition: Условие удаления (например, "id = $1")
            *condition_args: Аргументы для условия

        Returns:
            Количество удаленных строк
        """
        query = f"""
        DELETE FROM {table}
        WHERE {condition}
        """

        result = await self.execute(query, *condition_args)

        # Извлекаем количество удаленных строк из результата
        # Формат результата: "DELETE count"
        try:
            return int(result.split(" ")[1])
        except (IndexError, ValueError):
            logger.error(
                f"Не удалось извлечь количество удаленных строк из результата: {result}"
            )
            return 0

    @async_handle_error
    async def get_by_id(
        self, table: str, id_value: Union[int, str], id_column: str = "id"
    ) -> Optional[Dict[str, Any]]:
        """
        Получает запись из таблицы по её ID.

        Args:
            table: Имя таблицы
            id_value: Значение ID
            id_column: Имя столбца ID (по умолчанию "id")

        Returns:
            Словарь с данными записи или None, если запись не найдена
        """
        query = f"""
        SELECT * FROM {table}
        WHERE {id_column} = $1
        """

        return await self.fetchrow(query, id_value)

    @async_handle_error
    async def exists(self, table: str, condition: str, *condition_args) -> bool:
        """
        Проверяет существование записи в таблице.

        Args:
            table: Имя таблицы
            condition: Условие проверки (например, "id = $1")
            *condition_args: Аргументы для условия

        Returns:
            True, если запись существует, иначе False
        """
        query = f"""
        SELECT EXISTS (
            SELECT 1 FROM {table}
            WHERE {condition}
        ) AS exists
        """

        result = await self.fetchval(query, *condition_args)
        return result
