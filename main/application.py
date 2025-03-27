"""
Главный модуль приложения, отвечающий за запуск и координацию всех компонентов системы.
"""

import asyncio
import signal
import sys

from project.bots.strategies.strategy_manager import StrategyManager
from project.config import get_config
from project.infrastructure.database import Database
from project.infrastructure.message_broker import MessageBroker
from project.integrations.telegram_integration import TelegramIntegration
from project.main.eventloop import setup_event_loop
from project.utils.error_handler import setup_error_handlers
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Application:
    """
    Главный класс приложения, управляющий жизненным циклом всех компонентов.
    """

    def __init__(self):
        self.config = get_config()
        self.loop = None
        self.is_running = False
        self.components = {}
        self.shutdown_tasks = set()

    async def initialize(self):
        """Инициализация всех компонентов системы"""
        logger.info("Инициализация приложения...")

        # Инициализация базы данных
        self.components["database"] = Database.get_instance()
        await self.components["database"].initialize()

        # Инициализация брокера сообщений
        self.components["message_broker"] = MessageBroker.get_instance()
        await self.components["message_broker"].initialize()

        # Инициализация менеджера стратегий
        self.components["strategy_manager"] = StrategyManager()

        # Инициализация Telegram-интеграции
        if self.config.TELEGRAM_BOT_TOKEN:
            self.components["telegram"] = TelegramIntegration()
            # Запуск в отдельной задаче чтобы не блокировать инициализацию
            asyncio.create_task(self.components["telegram"].start_polling())

        # Регистрация обработчиков для корректного завершения
        for signame in ("SIGINT", "SIGTERM"):
            self.loop.add_signal_handler(
                getattr(signal, signame),
                lambda signame=signame: asyncio.create_task(self.shutdown(signame)),
            )

        logger.info("Приложение успешно инициализировано")
        self.is_running = True

    async def start(self):
        """Запуск приложения и всех его компонентов"""
        try:
            # Получение и настройка цикла событий
            self.loop = asyncio.get_running_loop()
            setup_event_loop(self.loop)

            # Инициализация всех компонентов
            await self.initialize()

            # Если включен режим бэктестирования, запускаем соответствующие стратегии
            if self.config.ENABLE_BACKTESTING:
                await self.start_backtesting()
            # Иначе запускаем торговые стратегии
            else:
                await self.start_trading()

            # Основной цикл приложения
            while self.is_running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error("Ошибка в основном цикле приложения: {str(e)}" %, exc_info=True)
            await self.shutdown("ERROR")

    async def start_trading(self):
        """Запуск торговых стратегий"""
        logger.info("Запуск торговых стратегий...")

        # Получаем менеджер стратегий
        strategy_manager = self.components["strategy_manager"]

        # Настраиваем и запускаем основные стратегии
        # TODO: Загружать стратегии из конфигурации или БД
        strategy_id = await strategy_manager.start_strategy(
            "main_strategy",
            exchange_id="binance",
            symbols=["BTC/USDT", "ETH/USDT"],
            interval="1h",
        )

        logger.info("Стратегия main_strategy запущена с ID: {strategy_id}" %)

        if self.components.get("telegram"):
            await self.components["telegram"].broadcast_message(
                "🚀 Торговый бот запущен в рабочем режиме"
            )

    async def start_backtesting(self):
        """Запуск режима бэктестирования"""
        logger.info("Запуск режима бэктестирования...")

        # Тут код для запуска бэктестирования
        # ...

        if self.components.get("telegram"):
            await self.components["telegram"].broadcast_message(
                "📊 Торговый бот запущен в режиме бэктестирования"
            )

    async def shutdown(self, signal_name=None):
        """Корректное завершение работы приложения"""
        if signal_name:
            logger.info("Получен сигнал {signal_name}, завершаем работу..." %)

        self.is_running = False

        # Остановка всех стратегий
        if "strategy_manager" in self.components:
            try:
                strategy_manager = self.components["strategy_manager"]
                for strategy_id in list(strategy_manager.running_strategies.keys()):
                    await strategy_manager.stop_strategy(strategy_id)
                logger.info("Все стратегии остановлены")
            except Exception as e:
                logger.error("Ошибка при остановке стратегий: {str(e)}" %)

        # Закрытие соединений с брокером сообщений
        if "message_broker" in self.components:
            try:
                await self.components["message_broker"].close()
                logger.info("Соединения с брокером сообщений закрыты")
            except Exception as e:
                logger.error("Ошибка при закрытии брокера сообщений: {str(e)}" %)

        # Закрытие соединений с базой данных
        if "database" in self.components:
            try:
                await self.components["database"].close()
                logger.info("Соединения с базой данных закрыты")
            except Exception as e:
                logger.error("Ошибка при закрытии соединений с БД: {str(e)}" %)

        # Отправка уведомления о завершении работы
        if "telegram" in self.components:
            try:
                await self.components["telegram"].broadcast_message(
                    "🛑 Торговый бот завершил работу"
                )
                await self.components["telegram"].close()
                logger.info("Telegram-бот остановлен")
            except Exception as e:
                logger.error("Ошибка при остановке Telegram-бота: {str(e)}" %)

        # Завершение всех оставшихся задач
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()

        logger.info("Приложение успешно завершило работу")

        # Выход из программы
        if signal_name != "ERROR":
            sys.exit(0)


def main():
    """Точка входа приложения"""
    # Настройка обработчиков ошибок
    setup_error_handlers()

    # Создание и запуск приложения
    app = Application()

    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания, завершаем работу...")
    except Exception as e:
        logger.critical("Критическая ошибка: {str(e)}" %, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
