"""
Модуль для интеграции с Telegram.
Предоставляет функции для отправки сообщений и управления ботом.
"""

import traceback
from typing import Awaitable, Callable, Dict, List, Optional, Union

from aiogram import Bot, Dispatcher, types
from aiogram.types import InputFile, ParseMode
from aiogram.utils.exceptions import TelegramAPIError
from project.config import get_config
from project.utils.error_handler import async_handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Определение типа для обработчиков команд
CommandHandler = Callable[[types.Message, List[str]], Awaitable[Optional[str]]]


class TelegramIntegration:
    """
    Класс для взаимодействия с Telegram API через бота.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "TelegramIntegration":
        """
        Получает экземпляр интеграции с Telegram (Singleton).

        Returns:
            Экземпляр класса TelegramIntegration
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, token: Optional[str] = None):
        """
        Инициализирует интеграцию с Telegram.

        Args:
            token: Токен Telegram-бота (None для использования из конфигурации)
        """
        config = get_config()
        self.token = token or config.TELEGRAM_BOT_TOKEN

        if not self.token:
            logger.warning("Telegram-бот не будет работать - токен не указан")
            self.bot = None
            self.dp = None
            return

        self.bot = Bot(token=self.token)
        self.dp = Dispatcher(self.bot)

        # Загружаем разрешенных пользователей из конфига
        tg_settings = config.get_telegram_settings()
        self.allowed_users = tg_settings.ALLOWED_USERS
        self.default_chat_id = tg_settings.CHAT_ID

        # Словарь для хранения обработчиков команд
        self.command_handlers: Dict[str, CommandHandler] = {}

        # Регистрируем базовые команды
        self._register_base_handlers()

        logger.info(
            f"Интеграция с Telegram инициализирована, разрешенных пользователей: {len(self.allowed_users)}"
        )

    def _register_base_handlers(self) -> None:
        """
        Регистрирует базовые обработчики команд.
        """
        if not self.dp:
            return

        @self.dp.message_handler(commands=["start", "help"])
        async def cmd_start(message: types.Message):
            """Обработчик команд /start и /help"""
            if str(message.from_user.id) not in self.allowed_users:
                await message.reply("У вас нет прав для использования этого бота.")
                return

            help_text = (
                "👋 Добро пожаловать в торгового бота!\n\n"
                "Доступные команды:\n"
                "/help - Показать это сообщение\n"
                "/status - Показать статус системы\n"
                "/orders - Показать активные ордера\n"
                "/balance - Показать баланс\n"
                "/start_strategy <name> - Запустить стратегию\n"
                "/stop_strategy <id> - Остановить стратегию\n"
            )
            await message.reply(help_text)

        @self.dp.message_handler()
        async def handle_message(message: types.Message):
            """Обработчик обычных сообщений"""
            # Проверяем, авторизован ли пользователь
            if str(message.from_user.id) not in self.allowed_users:
                logger.warning(
                    f"Неавторизованный доступ от {message.from_user.id}: {message.text}"
                )
                return

            # Обрабатываем только текстовые сообщения, начинающиеся с /
            if not message.text or not message.text.startswith("/"):
                return

            # Извлекаем команду и аргументы
            parts = message.text.split()
            command = parts[0][1:]  # Убираем слеш
            args = parts[1:]

            # Проверяем наличие обработчика для команды
            if command in self.command_handlers:
                try:
                    handler = self.command_handlers[command]
                    result = await handler(message, args)
                    if result:
                        await self.bot.send_message(
                            chat_id=message.chat.id,
                            text=result,
                            parse_mode=ParseMode.MARKDOWN,
                        )
                except Exception as e:
                    error_msg = f"Ошибка при выполнении команды /{command}:\n```\n{traceback.format_exc()}```"
                    await message.reply(error_msg, parse_mode=ParseMode.MARKDOWN)
                    logger.error(
                        f"Ошибка в обработчике команды {command}: {str(e)}",
                        exc_info=True,
                    )

    def register_command(self, command: str, handler: CommandHandler) -> None:
        """
        Регистрирует обработчик для команды.

        Args:
            command: Название команды (без /)
            handler: Асинхронная функция-обработчик команды
        """
        self.command_handlers[command] = handler
        logger.debug(f"Зарегистрирован обработчик для команды /{command}")

    @async_handle_error
    async def start_polling(self) -> None:
        """
        Запускает бота в режиме polling.
        """
        if not self.dp:
            logger.warning("Telegram-бот не запущен - нет диспетчера")
            return

        logger.info("Запуск Telegram-бота в режиме polling")
        await self.dp.start_polling()

    @async_handle_error
    async def send_message(
        self,
        chat_id: Optional[str] = None,
        text: str = "",
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> bool:
        """
        Отправляет сообщение в указанный чат.

        Args:
            chat_id: ID чата или пользователя (None для использования чата по умолчанию)
            text: Текст сообщения
            parse_mode: Режим форматирования (HTML, Markdown, MarkdownV2)
            disable_notification: Отключить уведомление о сообщении

        Returns:
            True при успешной отправке, иначе False
        """
        if not self.bot:
            logger.warning("Telegram-бот не настроен - сообщение не отправлено")
            return False

        chat_id = chat_id or self.default_chat_id

        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_notification=disable_notification,
            )
            logger.debug("Сообщение отправлено в чат {chat_id}".format(chat_id=chat_id))
            return True
        except TelegramAPIError as e:
            logger.error(
                f"Ошибка Telegram API при отправке сообщения в чат {chat_id}: {str(e)}"
            )
            return False
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в чат {chat_id}: {str(e)}")
            return False

    @async_handle_error
    async def broadcast_message(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        disable_notification: bool = False,
    ) -> int:
        """
        Отправляет сообщение всем разрешенным пользователям.

        Args:
            text: Текст сообщения
            parse_mode: Режим форматирования (HTML, Markdown, MarkdownV2)
            disable_notification: Отключить уведомление о сообщении

        Returns:
            Количество успешно отправленных сообщений
        """
        if not self.bot:
            logger.warning("Telegram-бот не настроен - сообщения не отправлены")
            return 0

        success_count = 0

        for user_id in self.allowed_users:
            try:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=text,
                    parse_mode=parse_mode,
                    disable_notification=disable_notification,
                )
                success_count += 1
            except Exception as e:
                logger.error(
                    f"Ошибка при отправке сообщения пользователю {user_id}: {str(e)}"
                )

        logger.debug(
            f"Групповое сообщение отправлено {success_count}/{len(self.allowed_users)} пользователям"
        )
        return success_count

    @async_handle_error
    async def send_photo(
        self,
        chat_id: Optional[str] = None,
        photo: Union[str, InputFile] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> bool:
        """
        Отправляет фото в указанный чат.

        Args:
            chat_id: ID чата или пользователя (None для использования чата по умолчанию)
            photo: Путь к фото или объект InputFile
            caption: Подпись к фото
            parse_mode: Режим форматирования подписи

        Returns:
            True при успешной отправке, иначе False
        """
        if not self.bot:
            logger.warning("Telegram-бот не настроен - фото не отправлено")
            return False

        chat_id = chat_id or self.default_chat_id

        try:
            # Если photo это строка (путь к файлу), преобразуем в InputFile
            if isinstance(photo, str):
                photo_file = InputFile(photo)
            else:
                photo_file = photo

            await self.bot.send_photo(
                chat_id=chat_id,
                photo=photo_file,
                caption=caption,
                parse_mode=parse_mode,
            )
            logger.debug("Фото отправлено в чат {chat_id}".format(chat_id=chat_id))
            return True
        except Exception as e:
            logger.error("Ошибка при отправке фото в чат {chat_id}: {error}".format(chat_id=chat_id, error=str(e)))
            return False

    @async_handle_error
    async def send_document(
        self,
        chat_id: Optional[str] = None,
        document: Union[str, InputFile] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> bool:
        """
        Отправляет документ в указанный чат.

        Args:
            chat_id: ID чата или пользователя (None для использования чата по умолчанию)
            document: Путь к документу или объект InputFile
            caption: Подпись к документу
            parse_mode: Режим форматирования подписи

        Returns:
            True при успешной отправке, иначе False
        """
        if not self.bot:
            logger.warning("Telegram-бот не настроен - документ не отправлен")
            return False

        chat_id = chat_id or self.default_chat_id

        try:
            # Если document это строка (путь к файлу), преобразуем в InputFile
            if isinstance(document, str):
                document_file = InputFile(document)
            else:
                document_file = document

            await self.bot.send_document(
                chat_id=chat_id,
                document=document_file,
                caption=caption,
                parse_mode=parse_mode,
            )
            logger.debug("Документ отправлен в чат {chat_id}".format(chat_id=chat_id))
            return True
        except Exception as e:
            logger.error("Ошибка при отправке документа в чат {chat_id}: {error}".format(chat_id=chat_id, error=str(e)))
            return False

    async def close(self) -> None:
        """
        Закрывает соединение с Telegram API.
        """
        if self.bot:
            await self.bot.close()
            logger.info("Соединение с Telegram API закрыто")
