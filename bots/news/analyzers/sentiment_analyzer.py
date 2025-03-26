"""
Анализатор настроения новостей.
Определяет положительное, отрицательное или нейтральное влияние новости на рынок.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from project.utils.logging_utils import get_logger
from project.utils.error_handler import handle_error

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Класс для анализа настроения текста новостей.
    """

    def __init__(self):
        """
        Инициализирует анализатор настроения.
        """
        # Проверяем, загружены ли необходимые данные NLTK
        try:
            # Пытаемся загрузить необходимые компоненты
            nltk.data.find("vader_lexicon")
            nltk.data.find("punkt")
            nltk.data.find("stopwords")
        except LookupError:
            # Если не загружены, скачиваем их
            logger.info("Скачивание необходимых данных NLTK...")
            nltk.download("vader_lexicon")
            nltk.download("punkt")
            nltk.download("stopwords")

        # Инициализируем анализатор VADER
        self.sia = SentimentIntensityAnalyzer()

        # Загружаем стоп-слова
        self.stop_words = set(stopwords.words("english"))

        # Словари положительных и отрицательных слов для крипторынка
        self.crypto_positive_words = [
            "bull",
            "bullish",
            "rally",
            "surge",
            "soar",
            "moon",
            "pump",
            "gain",
            "growth",
            "adoption",
            "institutional",
            "partnership",
            "breakthrough",
            "innovative",
            "support",
            "accumulation",
            "upgrade",
            "milestone",
            "record",
            "high",
            "all-time",
            "peak",
            "overtake",
            "outperform",
            "recovery",
            "momentum",
            "strength",
            "confidence",
            "optimistic",
            "enthusiasm",
            "potential",
            "opportunity",
            "promising",
            "trust",
            "backing",
            "endorsement",
            "approval",
            "legalization",
            "regulation",
            "clarity",
            "development",
            "progress",
            "advancement",
            "expansion",
            "integration",
            "solution",
        ]

        self.crypto_negative_words = [
            "bear",
            "bearish",
            "crash",
            "plunge",
            "dump",
            "sink",
            "drop",
            "fall",
            "decline",
            "loss",
            "sell-off",
            "correction",
            "fear",
            "uncertainty",
            "doubt",
            "ban",
            "prohibit",
            "restrict",
            "crackdown",
            "illegal",
            "criminal",
            "fraud",
            "scam",
            "hack",
            "exploit",
            "vulnerability",
            "attack",
            "concern",
            "worry",
            "risk",
            "threat",
            "danger",
            "warning",
            "investigation",
            "lawsuit",
            "probe",
            "scrutiny",
            "penalty",
            "fine",
            "rejection",
            "disapproval",
            "criticism",
            "skepticism",
            "pessimism",
            "weakness",
            "trouble",
            "issue",
            "problem",
            "delay",
            "postpone",
            "suspend",
            "halt",
        ]

        logger.debug("Создан анализатор настроения")

    @handle_error
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Анализирует настроение текста.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с результатами анализа
        """
        if not text:
            return {
                "compound": 0.0,
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment": "neutral",
                "score": 0.0,
            }

        # Получаем оценки настроения от VADER
        sentiment_scores = self.sia.polarity_scores(text)

        # Применяем дополнительный анализ для криптовалютной специфики
        crypto_sentiment = self._analyze_crypto_sentiment(text)

        # Комбинируем оценки
        compound = sentiment_scores["compound"]
        crypto_score = crypto_sentiment["score"]

        # Взвешенная комбинация
        final_score = (compound * 0.7) + (crypto_score * 0.3)

        # Определяем итоговое настроение
        if final_score >= 0.05:
            sentiment = "positive"
        elif final_score <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "compound": compound,
            "positive": sentiment_scores["pos"],
            "negative": sentiment_scores["neg"],
            "neutral": sentiment_scores["neu"],
            "crypto_score": crypto_score,
            "sentiment": sentiment,
            "score": final_score,
        }

    @handle_error
    def _analyze_crypto_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Выполняет специализированный анализ настроения для криптовалютного контекста.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с результатами анализа
        """
        # Приводим к нижнему регистру
        text = text.lower()

        # Токенизация
        tokens = word_tokenize(text)

        # Удаляем стоп-слова
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Ищем положительные и отрицательные слова
        positive_count = 0
        negative_count = 0

        for word in filtered_tokens:
            if word in self.crypto_positive_words or any(
                pos_word in word for pos_word in self.crypto_positive_words
            ):
                positive_count += 1
            elif word in self.crypto_negative_words or any(
                neg_word in word for neg_word in self.crypto_negative_words
            ):
                negative_count += 1

        # Рассчитываем общее количество найденных слов с настроением
        total_sentiment_words = positive_count + negative_count

        # Рассчитываем оценку настроения
        if total_sentiment_words > 0:
            score = (positive_count - negative_count) / total_sentiment_words
        else:
            score = 0.0

        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_sentiment_words": total_sentiment_words,
            "score": score,
        }

    @handle_error
    def analyze_news_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует настроение новостной статьи.

        Args:
            article: Словарь с данными статьи

        Returns:
            Словарь с результатами анализа
        """
        # Получаем текст для анализа
        title = article.get("title", "")
        content = article.get("content", "")
        summary = article.get("summary", "")

        # Комбинируем текст с приоритетом на заголовок
        text = f"{title} {title} {summary} {content}"

        # Анализируем настроение
        sentiment = self.analyze_sentiment(text)

        # Возвращаем результаты
        result = {
            "article": {
                "title": title,
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "published_at": article.get("published_at", ""),
            },
            "sentiment": sentiment,
            "relevant_coins": self._extract_relevant_coins(text),
        }

        return result

    @handle_error
    def _extract_relevant_coins(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлекает упоминания криптовалют из текста.

        Args:
            text: Текст для анализа

        Returns:
            Список словарей с данными о найденных криптовалютах
        """
        text = text.lower()

        # Словарь криптовалют для поиска
        # Формат: 'поисковый_термин': ('символ', 'полное_название')
        crypto_dict = {
            "bitcoin": ("BTC", "Bitcoin"),
            "btc": ("BTC", "Bitcoin"),
            "ethereum": ("ETH", "Ethereum"),
            "eth": ("ETH", "Ethereum"),
            "litecoin": ("LTC", "Litecoin"),
            "ltc": ("LTC", "Litecoin"),
            "ripple": ("XRP", "Ripple"),
            "xrp": ("XRP", "Ripple"),
            "cardano": ("ADA", "Cardano"),
            "ada": ("ADA", "Cardano"),
            "solana": ("SOL", "Solana"),
            "sol": ("SOL", "Solana"),
            "dogecoin": ("DOGE", "Dogecoin"),
            "doge": ("DOGE", "Dogecoin"),
            "polkadot": ("DOT", "Polkadot"),
            "dot": ("DOT", "Polkadot"),
            "binance coin": ("BNB", "Binance Coin"),
            "bnb": ("BNB", "Binance Coin"),
            "binance": ("BNB", "Binance Coin"),
            "tether": ("USDT", "Tether"),
            "usdt": ("USDT", "Tether"),
            "chainlink": ("LINK", "Chainlink"),
            "link": ("LINK", "Chainlink"),
            "uniswap": ("UNI", "Uniswap"),
            "uni": ("UNI", "Uniswap"),
            "avalanche": ("AVAX", "Avalanche"),
            "avax": ("AVAX", "Avalanche"),
            "polygon": ("MATIC", "Polygon"),
            "matic": ("MATIC", "Polygon"),
        }

        found_coins = {}

        # Ищем упоминания криптовалют
        for term, (symbol, name) in crypto_dict.items():
            # Ищем термин как отдельное слово
            pattern = r"\b" + re.escape(term) + r"\b"
            matches = re.findall(pattern, text)
            count = len(matches)

            if count > 0:
                if symbol in found_coins:
                    found_coins[symbol]["count"] += count
                else:
                    found_coins[symbol] = {
                        "symbol": symbol,
                        "name": name,
                        "count": count,
                    }

        # Преобразуем в список и сортируем по количеству упоминаний
        result = list(found_coins.values())
        result.sort(key=lambda x: x["count"], reverse=True)

        return result

    @handle_error
    def get_trading_signal(
        self, sentiment_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Генерирует торговый сигнал на основе анализа настроения.

        Args:
            sentiment_result: Результат анализа настроения

        Returns:
            Словарь с данными торгового сигнала или None, если сигнал не сформирован
        """
        sentiment = sentiment_result.get("sentiment", {})
        relevant_coins = sentiment_result.get("relevant_coins", [])
        article = sentiment_result.get("article", {})

        # Проверяем наличие данных
        if not sentiment or not relevant_coins or not article:
            return None

        # Получаем оценку настроения
        score = sentiment.get("score", 0)
        sentiment_label = sentiment.get("sentiment", "neutral")

        # Если настроение нейтральное, не формируем сигнал
        if sentiment_label == "neutral" or abs(score) < 0.1:
            return None

        # Определяем действие на основе настроения
        action = "buy" if sentiment_label == "positive" else "sell"

        # Формируем сигнал для наиболее упоминаемой монеты
        if relevant_coins:
            top_coin = relevant_coins[0]

            # Прогнозируем изменение цены
            price_change_pct = abs(score) * 0.05  # 5% при максимальной уверенности

            return {
                "symbol": f"{top_coin['symbol']}/USDT",
                "action": action,
                "confidence": abs(score),
                "prediction": {
                    "direction": "up" if action == "buy" else "down",
                    "price_change_pct": price_change_pct,
                },
                "source": article.get("source"),
                "title": article.get("title"),
                "url": article.get("url"),
                "published_at": article.get("published_at"),
                "signal_type": "news_sentiment",
            }

        return None
