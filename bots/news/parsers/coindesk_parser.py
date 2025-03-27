"""
Парсер для сайта CoinDesk.
Извлекает новости из RSS-ленты и страниц CoinDesk.
"""

import time
from datetime import datetime
from typing import Any, Dict, List

import feedparser
from bs4 import BeautifulSoup
from project.utils.error_handler import async_handle_error
from project.utils.http_utils import aiohttp_session
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CoindeskParser:
    """
    Парсер для извлечения новостей с сайта CoinDesk.
    """

    def __init__(self):
        """
        Инициализирует парсер CoinDesk.
        """
        self.rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        self.base_url = "https://www.coindesk.com"
        self.last_fetch_time = 0
        self.articles_cache = {}  # url -> article_data

        logger.debug("Создан парсер CoinDesk")

    @async_handle_error
    async def fetch_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Получает последние новости из RSS-ленты CoinDesk.

        Args:
            limit: Максимальное количество новостей для получения

        Returns:
            Список словарей с данными статей
        """
        current_time = time.time()

        # Проверяем, не слишком ли часто делаем запросы (не чаще раза в 5 минут)
        if current_time - self.last_fetch_time < 300:
            # Возвращаем кэшированные статьи
            return list(self.articles_cache.values())[:limit]

        try:
            articles = []

            # Получаем RSS-ленту
            async with aiohttp_session() as session:
                async with session.get(self.rss_url) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Не удалось получить RSS-ленту CoinDesk. Статус: {response.status}"
                        )
                        return []

                    rss_content = await response.text()

            # Парсим RSS
            feed = feedparser.parse(rss_content)

            # Обрабатываем каждую запись
            for entry in feed.entries[:limit]:
                # Извлекаем данные из записи
                title = entry.get("title", "")
                link = entry.get("link", "")
                published = entry.get("published", "")
                summary = entry.get("summary", "")

                # Преобразуем дату публикации
                try:
                    published_at = datetime.strptime(
                        published, "%a, %d %b %Y %H:%M:%S %z"
                    )
                except:
                    published_at = datetime.now()

                # Определяем автора
                author = entry.get("author", "CoinDesk")

                # Создаем запись о статье
                article = {
                    "title": title,
                    "url": link,
                    "published_at": published_at,
                    "author": author,
                    "summary": summary,
                    "content": summary,  # Краткое содержание по умолчанию
                    "source": "coindesk",
                }

                # Добавляем в результаты
                articles.append(article)

                # Кэшируем статью
                self.articles_cache[link] = article

                # Получаем полное содержимое статьи асинхронно
                asyncio.create_task(self._fetch_article_content(link))

            # Обновляем время последнего запроса
            self.last_fetch_time = current_time

            logger.debug("Получено {len(articles)} статей из CoinDesk" %)
            return articles

        except Exception as e:
            logger.error("Ошибка при получении новостей из CoinDesk: {str(e)}" %)
            return []

    @async_handle_error
    async def _fetch_article_content(self, url: str) -> None:
        """
        Получает полное содержимое статьи по URL.

        Args:
            url: URL статьи
        """
        if not url or url not in self.articles_cache:
            return

        try:
            # Получаем HTML-страницу
            async with aiohttp_session() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Не удалось получить содержимое статьи {url}. Статус: {response.status}"
                        )
                        return

                    html_content = await response.text()

            # Парсим HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Ищем содержимое статьи
            article_body = soup.select("div.at-body")

            if article_body:
                # Извлекаем текст из всех параграфов
                paragraphs = article_body[0].find_all("p")
                content = " ".join([p.get_text().strip() for p in paragraphs])

                # Обновляем кэшированную статью
                if url in self.articles_cache:
                    self.articles_cache[url]["content"] = content
                    logger.debug("Обновлено содержимое статьи: {url}" %)

        except Exception as e:
            logger.error("Ошибка при получении содержимого статьи {url}: {str(e)}" %)

    @async_handle_error
    async def search_news(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Ищет новости по запросу.

        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов

        Returns:
            Список словарей с данными статей
        """
        try:
            # Формируем URL для поиска
            search_url = f"{self.base_url}/search?q={query}"

            # Получаем HTML-страницу
            async with aiohttp_session() as session:
                async with session.get(search_url) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Не удалось выполнить поиск на CoinDesk. Статус: {response.status}"
                        )
                        return []

                    html_content = await response.text()

            # Парсим HTML
            soup = BeautifulSoup(html_content, "html.parser")

            # Ищем результаты поиска
            search_results = soup.select("div.search-results article")

            articles = []

            # Обрабатываем каждый результат
            for article_elem in search_results[:limit]:
                # Извлекаем данные из результата
                title_elem = article_elem.select_one("h2")
                link_elem = article_elem.select_one("a")
                summary_elem = article_elem.select_one("p")

                if title_elem and link_elem:
                    title = title_elem.get_text().strip()
                    link = link_elem.get("href")

                    if not link.startswith("http"):
                        link = f"{self.base_url}{link}"

                    summary = ""
                    if summary_elem:
                        summary = summary_elem.get_text().strip()

                    # Создаем запись о статье
                    article = {
                        "title": title,
                        "url": link,
                        "published_at": datetime.now(),  # Точная дата неизвестна
                        "author": "CoinDesk",
                        "summary": summary,
                        "content": summary,
                        "source": "coindesk",
                    }

                    # Добавляем в результаты
                    articles.append(article)

                    # Получаем полное содержимое статьи асинхронно
                    asyncio.create_task(self._fetch_article_content(link))

            logger.debug(
                f"Найдено {len(articles)} статей в поиске CoinDesk по запросу '{query}'"
            )
            return articles

        except Exception as e:
            logger.error("Ошибка при поиске новостей на CoinDesk: {str(e)}" %)
            return []
