"""
Модуль для определения графических паттернов.
Предоставляет функции для поиска паттернов в ценовых данных.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

from project.utils.logging_utils import get_logger
from project.utils.error_handler import handle_error
from project.technical_analysis.indicators import Indicators

logger = get_logger(__name__)


class Patterns:
    """
    Класс для определения графических паттернов.
    """

    @staticmethod
    @handle_error
    def engulfing(data: pd.DataFrame, bullish: bool = True) -> pd.Series:
        """
        Определяет паттерн поглощения (Engulfing Pattern).

        Args:
            data: DataFrame с данными OHLCV
            bullish: True для бычьего паттерна, False для медвежьего

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 2:
            logger.warning("Недостаточно данных для определения паттерна поглощения")
            return pd.Series(False, index=data.index)

        # Создаем серию результатов
        engulfing = pd.Series(False, index=data.index)

        if bullish:
            # Бычий паттерн поглощения
            # Предыдущая свеча красная, текущая зеленая
            # Тело текущей свечи полностью поглощает тело предыдущей
            engulfing = (
                (
                    data["open"].shift(1) > data["close"].shift(1)
                )  # Предыдущая свеча красная
                & (data["open"] < data["close"])  # Текущая свеча зеленая
                & (
                    data["open"] <= data["close"].shift(1)
                )  # Открытие ниже закрытия предыдущей
                & (
                    data["close"] >= data["open"].shift(1)
                )  # Закрытие выше открытия предыдущей
            )
        else:
            # Медвежий паттерн поглощения
            # Предыдущая свеча зеленая, текущая красная
            # Тело текущей свечи полностью поглощает тело предыдущей
            engulfing = (
                (
                    data["open"].shift(1) < data["close"].shift(1)
                )  # Предыдущая свеча зеленая
                & (data["open"] > data["close"])  # Текущая свеча красная
                & (
                    data["open"] >= data["close"].shift(1)
                )  # Открытие выше закрытия предыдущей
                & (
                    data["close"] <= data["open"].shift(1)
                )  # Закрытие ниже открытия предыдущей
            )

        return engulfing

    @staticmethod
    @handle_error
    def doji(data: pd.DataFrame, threshold: float = 0.05) -> pd.Series:
        """
        Определяет паттерн доджи (Doji Pattern).

        Args:
            data: DataFrame с данными OHLCV
            threshold: Пороговое значение для определения доджи (в процентах)

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 1:
            logger.warning("Недостаточно данных для определения паттерна доджи")
            return pd.Series(False, index=data.index)

        # Рассчитываем размер тела свечи
        body_size = abs(data["close"] - data["open"])

        # Рассчитываем размер всей свечи
        candle_size = data["high"] - data["low"]

        # Обрабатываем случай, когда high == low
        candle_size = candle_size.replace(0, float("nan"))

        # Рассчитываем отношение размера тела к размеру всей свечи
        body_ratio = (body_size / candle_size).fillna(0)

        # Определяем доджи
        doji = body_ratio < threshold

        return doji

    @staticmethod
    @handle_error
    def hammer(data: pd.DataFrame, ratio_threshold: float = 2.0) -> pd.Series:
        """
        Определяет паттерн молот (Hammer Pattern).

        Args:
            data: DataFrame с данными OHLCV
            ratio_threshold: Пороговое значение отношения нижней тени к телу

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 1:
            logger.warning("Недостаточно данных для определения паттерна молот")
            return pd.Series(False, index=data.index)

        # Определяем верхнюю и нижнюю цены тела свечи
        body_high = data[["open", "close"]].max(axis=1)
        body_low = data[["open", "close"]].min(axis=1)

        # Рассчитываем размер тела свечи
        body_size = body_high - body_low

        # Рассчитываем размер верхней тени
        upper_shadow = data["high"] - body_high

        # Рассчитываем размер нижней тени
        lower_shadow = body_low - data["low"]

        # Обрабатываем случай, когда body_size == 0
        body_size = body_size.replace(0, float("nan"))

        # Рассчитываем отношение нижней тени к телу
        lower_shadow_ratio = (lower_shadow / body_size).fillna(0)

        # Рассчитываем отношение верхней тени к телу
        upper_shadow_ratio = (upper_shadow / body_size).fillna(0)

        # Определяем молот
        hammer = (
            lower_shadow_ratio > ratio_threshold
        ) & (  # Нижняя тень больше тела в ratio_threshold раз
            upper_shadow_ratio < 0.5
        )  # Верхняя тень меньше половины тела

        return hammer

    @staticmethod
    @handle_error
    def shooting_star(data: pd.DataFrame, ratio_threshold: float = 2.0) -> pd.Series:
        """
        Определяет паттерн падающая звезда (Shooting Star Pattern).

        Args:
            data: DataFrame с данными OHLCV
            ratio_threshold: Пороговое значение отношения верхней тени к телу

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 1:
            logger.warning(
                "Недостаточно данных для определения паттерна падающая звезда"
            )
            return pd.Series(False, index=data.index)

        # Определяем верхнюю и нижнюю цены тела свечи
        body_high = data[["open", "close"]].max(axis=1)
        body_low = data[["open", "close"]].min(axis=1)

        # Рассчитываем размер тела свечи
        body_size = body_high - body_low

        # Рассчитываем размер верхней тени
        upper_shadow = data["high"] - body_high

        # Рассчитываем размер нижней тени
        lower_shadow = body_low - data["low"]

        # Обрабатываем случай, когда body_size == 0
        body_size = body_size.replace(0, float("nan"))

        # Рассчитываем отношение верхней тени к телу
        upper_shadow_ratio = (upper_shadow / body_size).fillna(0)

        # Рассчитываем отношение нижней тени к телу
        lower_shadow_ratio = (lower_shadow / body_size).fillna(0)

        # Определяем падающую звезду
        shooting_star = (
            upper_shadow_ratio > ratio_threshold
        ) & (  # Верхняя тень больше тела в ratio_threshold раз
            lower_shadow_ratio < 0.5
        )  # Нижняя тень меньше половины тела

        return shooting_star

    @staticmethod
    @handle_error
    def morning_star(data: pd.DataFrame, doji_threshold: float = 0.05) -> pd.Series:
        """
        Определяет паттерн утренняя звезда (Morning Star Pattern).

        Args:
            data: DataFrame с данными OHLCV
            doji_threshold: Пороговое значение для определения доджи

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 3:
            logger.warning(
                "Недостаточно данных для определения паттерна утренняя звезда"
            )
            return pd.Series(False, index=data.index)

        # Создаем серию результатов
        morning_star = pd.Series(False, index=data.index)

        # Определяем доджи
        doji = Patterns.doji(data, doji_threshold)

        # Определяем утреннюю звезду
        for i in range(2, len(data)):
            # Первая свеча - красная
            first_candle_bearish = data["open"].iloc[i - 2] > data["close"].iloc[i - 2]

            # Вторая свеча - доджи или свеча с маленьким телом
            second_candle_doji = doji.iloc[i - 1]

            # Третья свеча - зеленая
            third_candle_bullish = data["open"].iloc[i] < data["close"].iloc[i]

            # Вторая свеча открывается ниже закрытия первой
            gap_down = data["high"].iloc[i - 1] < data["close"].iloc[i - 2]

            # Третья свеча закрывается выше открытия первой
            close_above = data["close"].iloc[i] > data["open"].iloc[i - 2]

            # Утренняя звезда
            if (
                first_candle_bearish
                and second_candle_doji
                and third_candle_bullish
                and gap_down
                and close_above
            ):
                morning_star.iloc[i] = True

        return morning_star

    @staticmethod
    @handle_error
    def evening_star(data: pd.DataFrame, doji_threshold: float = 0.05) -> pd.Series:
        """
        Определяет паттерн вечерняя звезда (Evening Star Pattern).

        Args:
            data: DataFrame с данными OHLCV
            doji_threshold: Пороговое значение для определения доджи

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 3:
            logger.warning(
                "Недостаточно данных для определения паттерна вечерняя звезда"
            )
            return pd.Series(False, index=data.index)

        # Создаем серию результатов
        evening_star = pd.Series(False, index=data.index)

        # Определяем доджи
        doji = Patterns.doji(data, doji_threshold)

        # Определяем вечернюю звезду
        for i in range(2, len(data)):
            # Первая свеча - зеленая
            first_candle_bullish = data["open"].iloc[i - 2] < data["close"].iloc[i - 2]

            # Вторая свеча - доджи или свеча с маленьким телом
            second_candle_doji = doji.iloc[i - 1]

            # Третья свеча - красная
            third_candle_bearish = data["open"].iloc[i] > data["close"].iloc[i]

            # Вторая свеча открывается выше закрытия первой
            gap_up = data["low"].iloc[i - 1] > data["close"].iloc[i - 2]

            # Третья свеча закрывается ниже открытия первой
            close_below = data["close"].iloc[i] < data["open"].iloc[i - 2]

            # Вечерняя звезда
            if (
                first_candle_bullish
                and second_candle_doji
                and third_candle_bearish
                and gap_up
                and close_below
            ):
                evening_star.iloc[i] = True

        return evening_star

    @staticmethod
    @handle_error
    def three_white_soldiers(data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Определяет паттерн три белых солдата (Three White Soldiers Pattern).

        Args:
            data: DataFrame с данными OHLCV
            threshold: Пороговое значение для определения размера тела

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 3:
            logger.warning(
                "Недостаточно данных для определения паттерна три белых солдата"
            )
            return pd.Series(False, index=data.index)

        # Создаем серию результатов
        three_white_soldiers = pd.Series(False, index=data.index)

        # Определяем три белых солдата
        for i in range(2, len(data)):
            # Все три свечи - зеленые
            all_bullish = (
                (data["open"].iloc[i - 2] < data["close"].iloc[i - 2])
                and (data["open"].iloc[i - 1] < data["close"].iloc[i - 1])
                and (data["open"].iloc[i] < data["close"].iloc[i])
            )

            # Каждая свеча открывается внутри тела предыдущей
            opens_inside = (data["open"].iloc[i - 1] > data["open"].iloc[i - 2]) and (
                data["open"].iloc[i] > data["open"].iloc[i - 1]
            )

            # Каждая свеча закрывается выше закрытия предыдущей
            closes_higher = (
                data["close"].iloc[i - 1] > data["close"].iloc[i - 2]
            ) and (data["close"].iloc[i] > data["close"].iloc[i - 1])

            # Размер тела каждой свечи достаточно большой
            body_sizes = [
                data["close"].iloc[i - 2] - data["open"].iloc[i - 2],
                data["close"].iloc[i - 1] - data["open"].iloc[i - 1],
                data["close"].iloc[i] - data["open"].iloc[i],
            ]

            candle_sizes = [
                data["high"].iloc[i - 2] - data["low"].iloc[i - 2],
                data["high"].iloc[i - 1] - data["low"].iloc[i - 1],
                data["high"].iloc[i] - data["low"].iloc[i],
            ]

            large_bodies = all(
                b / c > threshold if c > 0 else False
                for b, c in zip(body_sizes, candle_sizes)
            )

            # Три белых солдата
            if all_bullish and opens_inside and closes_higher and large_bodies:
                three_white_soldiers.iloc[i] = True

        return three_white_soldiers

    @staticmethod
    @handle_error
    def three_black_crows(data: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """
        Определяет паттерн три черных вороны (Three Black Crows Pattern).

        Args:
            data: DataFrame с данными OHLCV
            threshold: Пороговое значение для определения размера тела

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 3:
            logger.warning(
                "Недостаточно данных для определения паттерна три черных вороны"
            )
            return pd.Series(False, index=data.index)

        # Создаем серию результатов
        three_black_crows = pd.Series(False, index=data.index)

        # Определяем три черных вороны
        for i in range(2, len(data)):
            # Все три свечи - красные
            all_bearish = (
                (data["open"].iloc[i - 2] > data["close"].iloc[i - 2])
                and (data["open"].iloc[i - 1] > data["close"].iloc[i - 1])
                and (data["open"].iloc[i] > data["close"].iloc[i])
            )

            # Каждая свеча открывается внутри тела предыдущей
            opens_inside = (data["open"].iloc[i - 1] < data["open"].iloc[i - 2]) and (
                data["open"].iloc[i] < data["open"].iloc[i - 1]
            )

            # Каждая свеча закрывается ниже закрытия предыдущей
            closes_lower = (data["close"].iloc[i - 1] < data["close"].iloc[i - 2]) and (
                data["close"].iloc[i] < data["close"].iloc[i - 1]
            )

            # Размер тела каждой свечи достаточно большой
            body_sizes = [
                data["open"].iloc[i - 2] - data["close"].iloc[i - 2],
                data["open"].iloc[i - 1] - data["close"].iloc[i - 1],
                data["open"].iloc[i] - data["close"].iloc[i],
            ]

            candle_sizes = [
                data["high"].iloc[i - 2] - data["low"].iloc[i - 2],
                data["high"].iloc[i - 1] - data["low"].iloc[i - 1],
                data["high"].iloc[i] - data["low"].iloc[i],
            ]

            large_bodies = all(
                b / c > threshold if c > 0 else False
                for b, c in zip(body_sizes, candle_sizes)
            )

            # Три черных вороны
            if all_bearish and opens_inside and closes_lower and large_bodies:
                three_black_crows.iloc[i] = True

        return three_black_crows

    @staticmethod
    @handle_error
    def piercing_line(data: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Определяет паттерн пирсинг (Piercing Line Pattern).

        Args:
            data: DataFrame с данными OHLCV
            threshold: Пороговое значение для определения проникновения

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 2:
            logger.warning("Недостаточно данных для определения паттерна пирсинг")
            return pd.Series(False, index=data.index)

        # Создаем серию результатов
        piercing_line = pd.Series(False, index=data.index)

        # Рассчитываем середину тела предыдущей свечи
        prev_midpoint = (data["open"].shift(1) + data["close"].shift(1)) / 2

        # Определяем пирсинг
        piercing_line = (
            (data["open"].shift(1) > data["close"].shift(1))  # Предыдущая свеча красная
            & (data["open"] < data["close"])  # Текущая свеча зеленая
            & (
                data["open"] < data["close"].shift(1)
            )  # Открытие ниже закрытия предыдущей
            & (data["close"] > prev_midpoint)  # Закрытие выше середины предыдущей
            & (
                data["close"] < data["open"].shift(1)
            )  # Закрытие ниже открытия предыдущей
        )

        return piercing_line

    @staticmethod
    @handle_error
    def dark_cloud_cover(data: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Определяет паттерн темное облако (Dark Cloud Cover Pattern).

        Args:
            data: DataFrame с данными OHLCV
            threshold: Пороговое значение для определения проникновения

        Returns:
            Series с булевыми значениями (True для свечей с паттерном)
        """
        if len(data) < 2:
            logger.warning("Недостаточно данных для определения паттерна темное облако")
            return pd.Series(False, index=data.index)

        # Создаем серию результатов
        dark_cloud_cover = pd.Series(False, index=data.index)

        # Рассчитываем середину тела предыдущей свечи
        prev_midpoint = (data["open"].shift(1) + data["close"].shift(1)) / 2

        # Определяем темное облако
        dark_cloud_cover = (
            (data["open"].shift(1) < data["close"].shift(1))  # Предыдущая свеча зеленая
            & (data["open"] > data["close"])  # Текущая свеча красная
            & (
                data["open"] > data["close"].shift(1)
            )  # Открытие выше закрытия предыдущей
            & (data["close"] < prev_midpoint)  # Закрытие ниже середины предыдущей
            & (
                data["close"] > data["open"].shift(1)
            )  # Закрытие выше открытия предыдущей
        )

        return dark_cloud_cover
