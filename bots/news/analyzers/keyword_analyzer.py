"""
Анализатор ключевых слов в новостях.
Выделяет ключевые слова и фразы, значимые для крипторынка.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from project.utils.logging_utils import get_logger
from project.utils.error_handler import handle_error

logger = get_logger(__name__)


class KeywordAnalyzer:
    """
    Класс для анализа ключевых слов в новостных текстах.
    """

    def __init__(self):
        """
        Инициализирует анализатор ключевых слов.
        """
        # Проверяем, загружены ли необходимые данные NLTK
        try:
            nltk.data.find("punkt")
            nltk.data.find("stopwords")
        except LookupError:
            # Если не загружены, скачиваем их
            logger.info("Скачивание необходимых данных NLTK...")
            nltk.download("punkt")
            nltk.download("stopwords")

        # Загружаем стоп-слова
        self.stop_words = set(stopwords.words("english"))

        # Добавляем дополнительные стоп-слова, характерные для новостей
        self.stop_words.update(
            [
                "says",
                "said",
                "report",
                "reports",
                "according",
                "told",
                "reuters",
                "reported",
                "announced",
                "statement",
                "press",
                "release",
                "today",
                "yesterday",
                "week",
                "month",
                "year",
            ]
        )

        # Ключевые слова и фразы, значимые для крипторынка
        self.market_keywords = {
            "regulation": {
                "weight": 5,
                "terms": [
                    "regulation",
                    "regulatory",
                    "sec",
                    "cftc",
                    "government",
                    "law",
                    "compliance",
                    "legal",
                    "license",
                    "ban",
                    "prohibit",
                    "allow",
                    "approve",
                    "authorization",
                    "requirement",
                ],
            },
            "adoption": {
                "weight": 4,
                "terms": [
                    "adoption",
                    "institutional",
                    "corporate",
                    "mainstream",
                    "retail",
                    "user",
                    "customer",
                    "client",
                    "merchant",
                    "payment",
                    "acceptance",
                    "integration",
                    "implement",
                    "support",
                    "partnership",
                ],
            },
            "technology": {
                "weight": 3,
                "terms": [
                    "blockchain",
                    "protocol",
                    "layer",
                    "scaling",
                    "fork",
                    "upgrade",
                    "update",
                    "version",
                    "release",
                    "launch",
                    "development",
                    "network",
                    "node",
                    "mining",
                    "consensus",
                    "validator",
                    "smart contract",
                ],
            },
            "security": {
                "weight": 5,
                "terms": [
                    "hack",
                    "exploit",
                    "vulnerability",
                    "bug",
                    "attack",
                    "security",
                    "breach",
                    "theft",
                    "stolen",
                    "phishing",
                    "scam",
                    "fraud",
                    "safety",
                    "protect",
                    "fix",
                    "patch",
                    "audit",
                ],
            },
            "market": {
                "weight": 4,
                "terms": [
                    "market",
                    "price",
                    "trading",
                    "volume",
                    "liquidity",
                    "volatility",
                    "bull",
                    "bear",
                    "rally",
                    "crash",
                    "correction",
                    "consolidation",
                    "resistance",
                    "support",
                    "breakout",
                    "all-time high",
                    "all-time low",
                ],
            },
            "macroeconomics": {
                "weight": 4,
                "terms": [
                    "inflation",
                    "deflation",
                    "interest rate",
                    "federal reserve",
                    "central bank",
                    "monetary policy",
                    "fiscal policy",
                    "recession",
                    "depression",
                    "economic",
                    "economy",
                    "gdp",
                    "unemployment",
                    "stimulus",
                    "debt",
                    "crisis",
                ],
            },
        }

        logger.debug("Создан анализатор ключевых слов")

    @handle_error
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Извлекает ключевые слова из текста.

        Args:
            text: Текст для анализа
            top_n: Количество ключевых слов для извлечения

        Returns:
            Список словарей с ключевыми словами и их весами
        """
        if not text:
            return []

        # Приводим к нижнему регистру
        text = text.lower()

        # Токенизация
        tokens = word_tokenize(text)

        # Удаляем стоп-слова и пунктуацию
        filtered_tokens = [
            word
            for word in tokens
            if word not in self.stop_words and word.isalnum() and len(word) > 2
        ]

        # Считаем частоту слов
        word_freq = Counter(filtered_tokens)

        # Также рассматриваем биграммы (фразы из двух слов)
        bigrams_list = list(ngrams(tokens, 2))
        # Фильтруем биграммы, убирая стоп-слова и пунктуацию
        filtered_bigrams = [
            " ".join(bigram)
            for bigram in bigrams_list
            if all(
                word not in self.stop_words and word.isalnum() and len(word) > 2
                for word in bigram
            )
        ]
        bigram_freq = Counter(filtered_bigrams)

        # Объединяем униграммы и биграммы
        all_keywords = []

        # Добавляем униграммы
        for word, freq in word_freq.most_common(top_n):
            all_keywords.append(
                {
                    "keyword": word,
                    "frequency": freq,
                    "weight": self._calculate_word_weight(word),
                    "type": "word",
                }
            )

        # Добавляем биграммы
        for bigram, freq in bigram_freq.most_common(top_n // 2):
            all_keywords.append(
                {
                    "keyword": bigram,
                    "frequency": freq,
                    "weight": self._calculate_word_weight(bigram),
                    "type": "phrase",
                }
            )

        # Сортируем по комбинированному весу (частота * вес)
        all_keywords.sort(key=lambda x: x["frequency"] * x["weight"], reverse=True)

        # Возвращаем top_n keywords
        return all_keywords[:top_n]

    @handle_error
    def _calculate_word_weight(self, word: str) -> float:
        """
        Рассчитывает вес слова или фразы на основе их важности для крипторынка.

        Args:
            word: Слово или фраза для оценки

        Returns:
            Вес слова
        """
        # Базовый вес
        base_weight = 1.0

        # Проверяем, содержится ли слово в категориях ключевых слов
        for category, data in self.market_keywords.items():
            if any(term in word or word in term for term in data["terms"]):
                return data["weight"]

        return base_weight

    @handle_error
    def analyze_news_keywords(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализирует ключевые слова в новостной статье.

        Args:
            article: Словарь с данными статьи

        Returns:
            Словарь с результатами анализа
        """
        # Получаем текст для анализа
        title = article.get("title", "")
        content = article.get("content", "")

        # Комбинируем с приоритетом на заголовок
        text = f"{title} {title} {content}"

        # Извлекаем ключевые слова
        keywords = self.extract_keywords(text, top_n=15)

        # Категоризируем ключевые слова
        categorized_keywords = self._categorize_keywords(keywords)

        # Получаем общую тему на основе категорий
        main_topics = self._identify_main_topics(categorized_keywords)

        # Возвращаем результаты
        return {
            "article": {
                "title": title,
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "published_at": article.get("published_at", ""),
            },
            "keywords": keywords,
            "categorized_keywords": categorized_keywords,
            "main_topics": main_topics,
        }

    @handle_error
    def _categorize_keywords(
        self, keywords: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Распределяет ключевые слова по категориям.

        Args:
            keywords: Список ключевых слов

        Returns:
            Словарь с ключевыми словами по категориям
        """
        categorized = {}

        # Инициализируем категории
        for category in self.market_keywords:
            categorized[category] = []

        # Категория для прочих ключевых слов
        categorized["other"] = []

        # Распределяем ключевые слова по категориям
        for keyword in keywords:
            word = keyword["keyword"]
            assigned = False

            for category, data in self.market_keywords.items():
                if any(term in word or word in term for term in data["terms"]):
                    categorized[category].append(keyword)
                    assigned = True
                    break

            if not assigned:
                categorized["other"].append(keyword)

        return categorized

    @handle_error
    def _identify_main_topics(
        self, categorized_keywords: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Определяет основные темы на основе категоризированных ключевых слов.

        Args:
            categorized_keywords: Словарь с ключевыми словами по категориям

        Returns:
            Список с основными темами
        """
        topics = []

        for category, keywords in categorized_keywords.items():
            if category == "other" or not keywords:
                continue

            # Рассчитываем общий вес категории
            total_weight = sum(
                keyword["frequency"] * keyword["weight"] for keyword in keywords
            )

            # Если вес достаточный, добавляем тему
            if total_weight > 0:
                topics.append(
                    {
                        "category": category,
                        "weight": total_weight,
                        "top_keywords": [
                            keyword["keyword"] for keyword in keywords[:3]
                        ],
                    }
                )

        # Сортируем темы по весу
        topics.sort(key=lambda x: x["weight"], reverse=True)

        return topics

    @handle_error
    def generate_market_insight(
        self, analysis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Генерирует инсайт о влиянии новости на рынок.

        Args:
            analysis_result: Результат анализа ключевых слов

        Returns:
            Словарь с инсайтом
        """
        article = analysis_result.get("article", {})
        main_topics = analysis_result.get("main_topics", [])

        if not main_topics:
            return {
                "article": article,
                "market_impact": "neutral",
                "confidence": 0.0,
                "insight": "Нет достаточной информации для определения влияния на рынок",
            }

        # Оцениваем влияние каждой темы на рынок
        topic_impacts = []

        for topic in main_topics:
            category = topic["category"]
            impact = "neutral"
            confidence = 0.5

            # Определяем влияние в зависимости от категории
            if category == "regulation":
                # Регуляции обычно негативно влияют на рынок
                impact = "negative"
                confidence = 0.7
            elif category == "adoption":
                # Принятие обычно позитивно
                impact = "positive"
                confidence = 0.8
            elif category == "technology":
                # Технологические новости обычно позитивны
                impact = "positive"
                confidence = 0.6
            elif category == "security":
                # Новости о безопасности обычно негативны
                impact = "negative"
                confidence = 0.7
            elif category == "market":
                # Рыночные новости могут быть позитивными или негативными
                # в зависимости от содержания
                if any(kw in " ".join(topic["top_keywords"]) for kw in 
                       ["bull", "rally", "breakout", "surge", "all-time high"]):
                    impact = "positive"
                    confidence = 0.7
                elif any(kw in " ".join(topic["top_keywords"]) for kw in 
                         ["bear", "crash", "correction", "resistance", "all-time low"]):
                    impact = "negative"
                    confidence = 0.7
                else:
                    impact = "neutral"
                    confidence = 0.5
            elif category == "macroeconomics":
                # Макроэкономика сложнее для однозначной оценки
                impact = "neutral"
                confidence = 0.5

            # Добавляем влияние темы
            topic_impacts.append(
                {
                    "category": category,
                    "impact": impact,
                    "confidence": confidence,
                    "weight": topic["weight"],
                }
            )

        # Рассчитываем общее влияние на рынок
        positive_weight = sum(
            topic["weight"] * topic["confidence"]
            for topic in topic_impacts
            if topic["impact"] == "positive"
        )
        negative_weight = sum(
            topic["weight"] * topic["confidence"]
            for topic in topic_impacts
            if topic["impact"] == "negative"
        )

        total_weight = sum(topic["weight"] for topic in topic_impacts)
        total_confidence = (
            sum(topic["weight"] * topic["confidence"] for topic in topic_impacts)
            / total_weight
            if total_weight > 0
            else 0
        )

        # Определяем итоговое влияние
        if positive_weight > negative_weight:
            market_impact = "positive"
        elif negative_weight > positive_weight:
            market_impact = "negative"
        else:
            market_impact = "neutral"

        # Формируем инсайт
        impact_strength = (
            abs(positive_weight - negative_weight) / total_weight
            if total_weight > 0
            else 0
        )
        impact_descriptions = {
            "positive": [
                "может оказать положительное влияние на рынок",
                "вероятно, вызовет рост рынка",
                "может стимулировать бычий тренд",
            ],
            "negative": [
                "может оказать негативное влияние на рынок",
                "вероятно, вызовет падение рынка",
                "может усилить медвежий тренд",
            ],
            "neutral": [
                "вряд ли окажет значительное влияние на рынок",
                "не должна существенно повлиять на тренд",
                "скорее всего, не изменит текущее направление рынка",
            ],
        }

        import random

        impact_description = random.choice(impact_descriptions[market_impact])

        return {
            "article": article,
            "market_impact": market_impact,
            "impact_strength": impact_strength,
            "confidence": total_confidence,
            "topic_impacts": topic_impacts,
            "insight": f"Новость о {', '.join(topic['category'] for topic in main_topics[:2])} {impact_description}.",
        }
