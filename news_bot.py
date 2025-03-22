"""
News Bot.
Реализует парсинг новостных источников, анализ тональности и
хранение новостных статей для последующего доступа.
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any

import aiohttp
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from project.config import load_config
config = load_config()
from project.utils.notify import NotificationManager

logger = logging.getLogger("NewsBot")

class NewsBot:
    """
    Класс новостного бота.
    
    Парсит новости с заданных источников, анализирует тональность
    и сохраняет последние новости в памяти.
    """

    def __init__(self, conf) -> None:
        """
        Инициализирует новостной бот.
        
        Args:
            conf: Конфигурация, содержащая настройки новостного модуля.
        """
        self.config = conf
        # Разбиваем строку источников по запятым
        self.sources = self.config.news.NEWS_SOURCES.split(",")
        self.article_limit = self.config.news.NEWS_ARTICLE_LIMIT
        self.long_keywords = [s.strip() for s in
                              self.config.news.LONG_KEYWORDS.split(",")]
        self.short_keywords = [s.strip() for s in
                               self.config.news.SHORT_KEYWORDS.split(",")]
        self.exit_keywords = [s.strip() for s in
                              self.config.news.EXIT_KEYWORDS.split(",")]
        self.articles: List[Dict[str, Any]] = []
        self.is_running = False
        self.sia = SentimentIntensityAnalyzer()
        self.notifier = NotificationManager()

    async def fetch_news(self, url: str) -> List[Dict[str, Any]]:
        """
        Получает новости с заданного URL.
        
        Args:
            url (str): URL новостного источника.
        
        Returns:
            List[Dict[str, Any]]: Список новостных статей.
        """
        articles = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            # Пример: ищем заголовки в теге h2 (адаптируйте под сайт)
            for item in soup.find_all("h2"):
                title = item.get_text(strip=True)
                parent = item.find_parent("a")
                link = parent["href"] if parent and parent.has_attr("href") else url
                timestamp = datetime.now().isoformat()
                content = title  # При желании можно расширить парсинг
                article = {
                    "title": title,
                    "link": link,
                    "timestamp": timestamp,
                    "content": content,
                    "source": url
                }
                articles.append(article)
        except Exception as e:
            logger.error(f"Ошибка парсинга с {url}: {e}")
        return articles

    async def update_articles(self) -> None:
        """
        Обновляет список новостных статей, объединяя данные всех источников.
        
        Загружает новости, анализирует тональность и сохраняет последние
        100 статей.
        """
        new_articles = []
        for src in self.sources:
            articles = await self.fetch_news(src.strip())
            for art in articles:
                # Создаем уникальный хэш на основе заголовка и содержания
                art_hash = hashlib.sha256(
                    (art["title"] + art["content"]).encode()
                ).hexdigest()
                art["hash"] = art_hash
                art["analyzed"] = self.sia.polarity_scores(art["content"])
                art["fetched_at"] = datetime.now().isoformat()
                new_articles.append(art)
        # Сортировка по времени получения (новейшие первыми)
        new_articles.sort(
            key=lambda x: x["fetched_at"], reverse=True
        )
        # Сохраняем последние 100 статей
        self.articles = new_articles[:100]

    def get_last_articles(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Возвращает последние N новостей.
        
        Args:
            limit (int, optional): Количество новостей.
        
        Returns:
            List[Dict[str, Any]]: Список последних новостей.
        """
        if limit is None:
            limit = self.article_limit
        return self.articles[:limit]

    async def run(self) -> None:
        """
        Основной цикл работы новостного бота.
        
        Периодически обновляет список новостей с интервалом из настроек.
        """
        self.is_running = True
        update_interval = self.config.news.UPDATE_INTERVAL
        while self.is_running:
            await self.update_articles()
            await asyncio.sleep(update_interval)

    async def stop(self) -> None:
        """
        Останавливает работу новостного бота.
        """
        self.is_running = False
