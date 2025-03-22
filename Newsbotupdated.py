"""
News Bot.
Реализует парсинг новостных источников, анализ тональности и
хранение новостных статей для последующего доступа.
"""

import asyncio
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Type

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from project.config import load_config
from project.utils.notify import NotificationManager
from project.base_bot import BaseBot  # Новый импорт
from project.news.parsers.base_parser import NewsParser  # Новый импорт

config = load_config()
logger = logging.getLogger("NewsBot")


class NewsBot(BaseBot):
    """
    Класс новостного бота.

    Парсит новости с заданных источников, анализирует тональность
    и сохраняет последние новости в памяти.
    """

    def __init__(self, parsers: Optional[List[NewsParser]] = None) -> None:
        """
        Инициализирует новостной бот.

        Args:
            parsers: Список парсеров для получения новостей
        """
        super().__init__(name="NewsBot")
        self.config = config
        self.article_limit = self.config.news.NEWS_ARTICLE_LIMIT
        self.long_keywords = [
            s.strip() for s in self.config.news.LONG_KEYWORDS.split(",")
        ]
        self.short_keywords = [
            s.strip() for s in self.config.news.SHORT_KEYWORDS.split(",")
        ]
        self.exit_keywords = [
            s.strip() for s in self.config.news.EXIT_KEYWORDS.split(",")
        ]
        self.articles: List[Dict[str, Any]] = []
        self.sia = SentimentIntensityAnalyzer()
        self.notifier = NotificationManager()
        
        # Используем переданные парсеры или создаем пустой список
        self.parsers = parsers or []

    async def update_articles(self) -> None:
        """
        Обновляет список новостных статей, объединяя данные всех источников.

        Загружает новости, анализирует тональность и сохраняет последние
        статьи согласно настройкам.
        """
        if not self.parsers:
            self.logger.warning("No parsers configured, can't update articles")
            return
            
        new_articles = []
        
        # Получаем статьи от всех парсеров параллельно
        tasks = [parser.fetch_articles() for parser in self.parsers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем результаты
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Ошибка при получении статей из парсера {i}: {result}")
                continue
                
            for art in result:
                # Создаем уникальный хэш на основе заголовка и содержания
                art_hash = hashlib.sha256(
                    (art["title"] + art["content"]).encode()
                ).hexdigest()
                art["hash"] = art_hash
                art["analyzed"] = self.sia.polarity_scores(art["content"])
                art["fetched_at"] = datetime.now().isoformat()
                new_articles.append(art)
                
                # Проверяем на ключевые слова и отправляем уведомления при необходимости
                self._check_keywords_and_notify(art)
        
        # Сортировка по времени получения (новейшие первыми)
        new_articles.sort(key=lambda x: x["fetched_at"], reverse=True)
        
        # Сохраняем последние статьи согласно лимиту
        self.articles = new_articles[:self.article_limit]
        
        self.logger.info(f"Обновлено {len(new_articles)} статей")

    def _check_keywords_and_notify(self, article: Dict[str, Any]) -> None:
        """
        Проверяет статью на наличие ключевых слов и отправляет уведомления.
        
        Args:
            article: Статья для проверки
        """
        content = article["title"] + " " + article["content"]
        content = content.lower()
        
        # Проверка на ключевые слова
        long_match = any(kw.lower() in content for kw in self.long_keywords)
        short_match = any(kw.lower() in content for kw in self.short_keywords)
        exit_match = any(kw.lower() in content for kw in self.exit_keywords)
        
        if long_match or short_match or exit_match:
            signal_type = []
            if long_match:
                signal_type.append("LONG")
            if short_match:
                signal_type.append("SHORT")
            if exit_match:
                signal_type.append("EXIT")
                
            message = (
                f"🚨 {', '.join(signal_type)} сигнал!\n"
                f"📰 {article['title']}\n"
                f"🔗 {article['link']}\n"
                f"📊 Тональность: {article['analyzed']['compound']:.2f}"
            )
            
            # Отправляем уведомление
            try:
                self.notifier.send_notification(message, level="alert")
            except Exception as e:
                self.logger.error(f"Ошибка при отправке уведомления: {e}")

    def get_last_articles(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Возвращает последние N новостей.

        Args:
            limit: Количество новостей (если None, используется limit из настроек)

        Returns:
            Список последних новостей
        """
        if limit is None:
            limit = self.article_limit
        return self.articles[:limit]
        
    def add_parser(self, parser: NewsParser) -> None:
        """
        Добавляет парсер в список парсеров.
        
        Args:
            parser: Парсер для добавления
        """
        self.parsers.append(parser)
        self.logger.info(f"Добавлен парсер: {parser.source_name}")

    async def run(self) -> None:
        """
        Основной цикл работы новостного бота.

        Периодически обновляет список новостей с интервалом из настроек.
        """
        update_interval = self.config.news.UPDATE_INTERVAL
        self.logger.info(f"Запуск NewsBot с интервалом обновления {update_interval}с")
        
        while self.is_running:
            try:
                await self.update_articles()
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Ошибка в цикле новостного бота: {e}")
                await asyncio.sleep(update_interval)  # Не прерываем цикл при ошибке
