"""
Парсер Bitcoin Magazine.
Парсит новости с сайта Bitcoin Magazine с поддержкой пагинации 
и настраиваемыми селекторами.
"""

import logging
from bs4 import BeautifulSoup
from project.utils.http_utils import async_get
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger("BitcoinMagParser")

# Константы
BITCOINMAG_BASE_URL = "https://bitcoinmagazine.com"
MAX_PAGES_TO_FETCH = 3  # Максимальное количество страниц для парсинга

# Селекторы сайта - можно легко обновить при изменении структуры сайта
SELECTORS = {
    "article_card": ".post-card",
    "title": ".post-card__title",
    "excerpt": ".post-card__excerpt",
    "link": "a",
    "next_page": ".pagination__next a"
}


def parse_bitcoinmag_html(html: str, source_url: str) -> List[Dict]:
    """
    Парсит HTML-контент сайта Bitcoin Magazine.

    Args:
        html (str): HTML страницы.
        source_url (str): URL источника для включения в результаты.

    Returns:
        List[Dict]: Список статей с заголовками и ссылками.
        Dict с ключом 'next_page': URL следующей страницы, если есть.
    """
    soup = BeautifulSoup(html, "html.parser")
    articles = []
    
    # Поиск ссылки на следующую страницу
    next_page_link = soup.select_one(SELECTORS["next_page"])
    next_page = next_page_link["href"] if next_page_link else None
    
    # Если ссылка относительная, добавляем базовый URL
    if next_page and not next_page.startswith(('http://', 'https://')):
        next_page = BITCOINMAG_BASE_URL + next_page
    
    # Парсинг статей
    for card in soup.select(SELECTORS["article_card"]):
        try:
            title_elem = card.select_one(SELECTORS["title"])
            link_elem = card.find(SELECTORS["link"])
            excerpt_elem = card.select_one(SELECTORS["excerpt"])
            
            if not (title_elem and link_elem):
                continue  # Пропускаем неполные карточки
                
            title = title_elem.text.strip() if title_elem else ""
            url = link_elem["href"] if link_elem else ""
            
            # Если URL относительный, добавляем базовый URL
            if url and not url.startswith(('http://', 'https://')):
                url = BITCOINMAG_BASE_URL + url
                
            content = excerpt_elem.text.strip() if excerpt_elem else ""
            ts = datetime.now().isoformat()
            
            articles.append({
                "source": "bitcoinmag",
                "title": title,
                "url": url,
                "content": content,
                "timestamp": ts,
                "source_url": source_url
            })
        except Exception as e:
            logger.error(f"Ошибка при парсинге BitcoinMag карточки: {e}")
    
    return articles, {"next_page": next_page}


async def fetch_bitcoinmag_articles(max_pages: int = MAX_PAGES_TO_FETCH) -> List[Dict]:
    """
    Асинхронно получает статьи с Bitcoin Magazine с поддержкой пагинации.

    Args:
        max_pages (int): Максимальное количество страниц для парсинга.

    Returns:
        List[Dict]: Список последних статей со всех обработанных страниц.
    """
    all_articles = []
    current_url = BITCOINMAG_BASE_URL
    pages_fetched = 0
    
    while current_url and pages_fetched < max_pages:
        try:
            html = await async_get(current_url)
            if not html:
                break
                
            articles, pagination_data = parse_bitcoinmag_html(html, current_url)
            all_articles.extend(articles)
            
            # Подготовка к следующей итерации
            current_url = pagination_data.get("next_page")
            pages_fetched += 1
            
            logger.info(f"Получено {len(articles)} статей с {current_url}. Всего: {len(all_articles)}")
        except Exception as e:
            logger.error(f"Ошибка при получении статей с {current_url}: {e}")
            break
    
    return all_articles


async def fetch_full_article_content(article_url: str) -> Optional[str]:
    """
    Получает полный текст статьи по URL.
    
    Args:
        article_url (str): URL статьи для получения полного текста.
        
    Returns:
        Optional[str]: Полный текст статьи или None в случае ошибки.
    """
    try:
        html = await async_get(article_url)
        if not html:
            return None
            
        soup = BeautifulSoup(html, "html.parser")
        
        # Селектор может отличаться в зависимости от структуры страницы статьи
        article_content = soup.select_one(".article-content")
        if article_content:
            return article_content.get_text(separator="\n").strip()
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении полного текста статьи {article_url}: {e}")
        return None
