"""
Модуль telegram_news.
Интегрирует новостной бот с Telegram, предоставляет команды для
просмотра статуса, последних новостей и справки.
"""

from aiogram import Router, types
from aiogram.filters import Command
from datetime import datetime
from project.config import load_config
config = load_config()
from project.bots.news.news_bot import NewsBot
from project.risk_management.portfolio_manager import PortfolioManager

router = Router()
# Инициализируем новостной бот с настройками из конфигурации
news_bot = NewsBot(config)
# Инициализируем менеджер портфеля (для уведомлений, если требуется)
portfolio_manager = PortfolioManager()

@router.message(Command("news_status"))
async def cmd_news_status(message: types.Message) -> None:
    """
    Команда /news_status.
    Возвращает статус новостного бота и время последнего обновления.
    """
    status = "Активен" if news_bot.is_running else "Неактивен"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"Новостной бот: {status}\nПоследнее обновление: {now}"
    await message.answer(text)

@router.message(Command("last_news"))
async def cmd_last_news(message: types.Message) -> None:
    """
    Команда /last_news.
    Возвращает последние N новостей, полученные ботом.
    """
    articles = news_bot.get_last_articles(limit=None)
    if not articles:
        await message.answer("Нет новостей для отображения.")
        return
    text_lines = ["Последние новости:"]
    for art in articles:
        title = art.get("title", "Без заголовка")
        source = art.get("source", "Неизвестный источник")
        text_lines.append(f"- {title} (Источник: {source})")
    await message.answer("\n".join(text_lines))

@router.message(Command("news_help"))
async def cmd_news_help(message: types.Message) -> None:
    """
    Команда /news_help.
    Выводит справочную информацию по командам новостного бота.
    """
    help_text = (
        "Доступные команды:\n"
        "/news_status - Статус новостного бота и последнее обновление\n"
        "/last_news - Последние новости\n"
        "/news_help - Справка по новостным командам"
    )
    await message.answer(help_text)
