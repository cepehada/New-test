"""
Простой анализатор настроений на основе словаря ключевых слов.
Не использует машинное обучение для экономии ресурсов.
"""

import re
from typing import Dict, List, Tuple, Union

from project.utils.logging_utils import setup_logger

logger = setup_logger("sentiment_analyzer")


class SimpleSentimentAnalyzer:
    """Простой анализатор настроений на основе словаря ключевых слов"""
    
    def __init__(self):
        """Инициализирует анализатор настроений с предустановленными словарями"""
        # Словари позитивных и негативных слов для криптовалют
        self.positive_words = {
            'bullish', 'buy', 'rally', 'surge', 'soar', 'gain', 'rise', 'growth', 'profit',
            'positive', 'partnership', 'adopt', 'launch', 'innovation', 'support', 'upgrade',
            'success', 'breakthrough', 'milestone', 'opportunity', 'promising', 'strong',
            'recovery', 'upward', 'outperform', 'beat', 'exceed', 'momentum', 'win', 'improve'
        }
        
        self.negative_words = {
            'bearish', 'sell', 'crash', 'dump', 'drop', 'plunge', 'fall', 'decline', 'loss',
            'negative', 'ban', 'hack', 'attack', 'scam', 'fraud', 'risk', 'warning', 'concern',
            'fear', 'uncertainty', 'threat', 'weakness', 'weak', 'downward', 'underperform',
            'miss', 'fail', 'trouble', 'problem', 'issue', 'struggle', 'panic', 'worry'
        }
        
        # Словарь с весами для особо важных слов
        self.word_weights = {
            'buy': 1.5, 'sell': 1.5,
            'bullish': 1.8, 'bearish': 1.8,
            'crash': 2.0, 'surge': 2.0,
            'partnership': 1.6, 'ban': 1.7,
            'hack': 2.0, 'adoption': 1.6
        }
        
        logger.info("Simple sentiment analyzer initialized")
        
    def analyze(self, text: str) -> Dict:
        """
        Анализирует текст и возвращает оценку настроения
        
        Args:
            text: Текст для анализа
            
        Returns:
            Dict: Результаты анализа
        """
        if not text or not isinstance(text, str):
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }
        
        # Преобразуем текст в нижний регистр и токенизируем
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Считаем позитивные и негативные слова
        positive_count = 0
        negative_count = 0
        weighted_positive = 0
        weighted_negative = 0
        
        for word in words:
            if word in self.positive_words:
                weight = self.word_weights.get(word, 1.0)
                positive_count += 1
                weighted_positive += weight
            elif word in self.negative_words:
                weight = self.word_weights.get(word, 1.0)
                negative_count += 1
                weighted_negative += weight
        
        # Рассчитываем общий счет
        total_count = len(words)
        if total_count == 0:
            # Пустой текст
            sentiment = 'neutral'
            score = 0.0
            confidence = 0.0
        else:
            # Рассчитываем оценку настроения [-1.0, 1.0]
            sentiment_words = positive_count + negative_count
            if sentiment_words == 0:
                # Нет ключевых слов для определения настроения
                sentiment = 'neutral'
                score = 0.0
                confidence = 0.0
            else:
                weighted_sum = weighted_positive - weighted_negative
                max_possible = max(weighted_positive, weighted_negative) * 2
                score = weighted_sum / max_possible if max_possible > 0 else 0
                
                # Определяем настроение по знаку оценки
                if score > 0.15:
                    sentiment = 'positive'
                elif score < -0.15:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                # Рассчитываем уверенность на основе количества ключевых слов
                confidence = min(sentiment_words / (total_count * 0.5), 1.0)
        
        return {
            'sentiment': sentiment,
            'score': round(score, 3),
            'confidence': round(confidence, 3),
            'details': {
                'positive_words': positive_count,
                'negative_words': negative_count,
                'total_words': total_count
            }
        }
    
    def get_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Упрощенная версия анализа настроения
        
        Args:
            text: Текст для анализа
            
        Returns:
            Tuple[str, float]: Настроение и оценка
        """
        result = self.analyze(text)
        return result['sentiment'], result['score']


# Создаем глобальный экземпляр анализатора
_sentiment_analyzer = None


def get_sentiment_analyzer() -> SimpleSentimentAnalyzer:
    """
    Возвращает глобальный экземпляр анализатора настроений
    
    Returns:
        SimpleSentimentAnalyzer: Анализатор настроений
    """
    global _sentiment_analyzer
    
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SimpleSentimentAnalyzer()
    
    return _sentiment_analyzer


def analyze_sentiment(text: str) -> Dict:
    """
    Анализирует текст с использованием глобального анализатора
    
    Args:
        text: Текст для анализа
        
    Returns:
        Dict: Результаты анализа
    """
    analyzer = get_sentiment_analyzer()
    return analyzer.analyze(text)
