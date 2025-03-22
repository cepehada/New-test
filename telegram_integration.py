"""
Модуль интеграции с Telegram.
Реализует команды для получения данных и управления ботом.
Полностью рабочая версия с реальной бизнес-логикой.
"""

import asyncio
from aiogram import Router, types
from aiogram.filters import Command
from project.config import load_config
from project.risk_management.portfolio_manager import PortfolioManager
from project.risk_management.var_calculator import calculate_var
from project.bots.news.news_bot import NewsBot

# Загрузка конфигурации
config = load_config()

# Инициализация роутера
router = Router()

# Инициализация менеджера портфеля
portfolio_manager = PortfolioManager()

# Инициализация новостного бота
news_bot = NewsBot(config)


@router.message(Command("portfolio"))
async def cmd_portfolio(message: types.Message) -> None:
    """
    Команда /portfolio.
    Возвращает текущие активные позиции.
    """
    positions = await portfolio_manager.get_all_positions()
    if not positions:
        await message.answer("📊 Портфель пуст.")
        return

    response = "📊 Текущий портфель:\n"
    for pos in positions:
        response += (
            f"- {pos['symbol']} | Кол-во: {pos['amount']} | "
            f"Цена входа: {pos['entry_price']}\n"
        )

    await message.answer(response)


@router.message(Command("risk"))
async def cmd_risk(message: types.Message) -> None:
    """
    Команда /risk.
    Возвращает отчёт по рискам и расчёт VaR.
    """
    closed_trades = await portfolio_manager.get_closed_trades()
    if not closed_trades:
        await message.answer("🔔 Нет закрытых сделок для расчета VaR.")
        return

    returns = [trade.get("pnl", 0) for trade in closed_trades]
    var_95 = calculate_var(returns, confidence=0.95)
    var_99 = calculate_var(returns, confidence=0.99)

    response = (
        "📈 Риск-отчёт:\n"
        f"- VaR (95%): {var_95:.2f}\n"
        f"- VaR (99%): {var_99:.2f}"
    )

    await message.answer(response)


@router.message(Command("news"))
async def cmd_news(message: types.Message) -> None:
    """
    Команда /news.
    Возвращает последние 5 новостей.
    """
    latest_news = await news_bot.get_last_articles(limit=5)

    if not latest_news:
        await message.answer("📰 Свежих новостей нет.")
        return

    response = "📰 Последние новости:\n\n"
    for article in latest_news:
        response += (
            f"🔹 {article['title']}\n"
            f"🌐 [Читать подробнее]({article['url']})\n\n"
        )

    await message.answer(response, parse_mode="Markdown", disable_web_page_preview=True)


@router.message(Command("trade_status"))
async def cmd_trade_status(message: types.Message) -> None:
    """
    Команда /trade_status.
    Возвращает статус последних сделок (открытых и закрытых).
    """
    open_positions = await portfolio_manager.get_all_positions()
    closed_trades = await portfolio_manager.get_closed_trades(limit=5)

    response = "📌 Статус сделок:\n\n"

    if open_positions:
        response += "🔸 Открытые позиции:\n"
        for pos in open_positions:
            response += (
                f"{pos['symbol']} | Кол-во: {pos['amount']} | "
                f"Цена входа: {pos['entry_price']}\n"
            )
    else:
        response += "🔸 Открытых позиций нет.\n"

    response += "\n🔹 Последние закрытые сделки:\n"
    if closed_trades:
        for trade in closed_trades:
            response += (
                f"{trade['symbol']} | PnL: {trade['pnl']:.2f} | "
                f"Время: {trade['timestamp']}\n"
            )
    else:
        response += "Закрытых сделок нет.\n"

    await message.answer(response)


@router.message(Command("help"))
async def cmd_help(message: types.Message) -> None:
    """
    Команда /help.
    Возвращает список доступных команд.
    """
    response = (
        "📚 Доступные команды:\n"
        "/portfolio — текущие позиции\n"
        "/risk — отчёт по рискам\n"
        "/news — последние новости\n"
        "/trade_status — статус сделок\n"
        "/help — список команд"
    )

    await message.answer(response)
