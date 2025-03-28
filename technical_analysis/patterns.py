"""
Модуль для определения графических паттернов.
Предоставляет функции для поиска паттернов в ценовых данных.
"""

# Standard imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
try:
    import numpy as np
    import pandas as pd
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "pandas"])
    import numpy as np
    import pandas as pd

# Local imports
from project.technical_analysis.indicators import Indicators
from project.utils.error_handler import handle_error
from project.utils.logging_utils import get_logger

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

        @staticmethod
        @handle_error
        def bullish_harami(data: pd.DataFrame) -> pd.Series:
            """
            Определяет бычий паттерн харами (Bullish Harami Pattern).

            Args:
                data: DataFrame с данными OHLCV

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 2:
                logger.warning(
                    "Недостаточно данных для определения паттерна бычий харами"
                )
                return pd.Series(False, index=data.index)

            # Создаем серию результатов
            bullish_harami = pd.Series(False, index=data.index)

            # Определяем бычий харами
            bullish_harami = (
                (
                    data["open"].shift(1) > data["close"].shift(1)
                )  # Предыдущая свеча красная
                & (data["open"] < data["close"])  # Текущая свеча зеленая
                & (
                    data["open"] > data["close"].shift(1)
                )  # Открытие выше закрытия предыдущей
                & (
                    data["close"] < data["open"].shift(1)
                )  # Закрытие ниже открытия предыдущей
            )

            return bullish_harami

        @staticmethod
        @handle_error
        def bearish_harami(data: pd.DataFrame) -> pd.Series:
            """
            Определяет медвежий паттерн харами (Bearish Harami Pattern).

            Args:
                data: DataFrame с данными OHLCV

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 2:
                logger.warning(
                    "Недостаточно данных для определения паттерна медвежий харами"
                )
                return pd.Series(False, index=data.index)

            # Создаем серию результатов
            bearish_harami = pd.Series(False, index=data.index)

            # Определяем медвежий харами
            bearish_harami = (
                (
                    data["open"].shift(1) < data["close"].shift(1)
                )  # Предыдущая свеча зеленая
                & (data["open"] > data["close"])  # Текущая свеча красная
                & (
                    data["open"] < data["close"].shift(1)
                )  # Открытие ниже закрытия предыдущей
                & (
                    data["close"] > data["open"].shift(1)
                )  # Закрытие выше открытия предыдущей
            )

            return bearish_harami

        @staticmethod
        @handle_error
        def tweezer_tops(data: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
            """
            Определяет паттерн шипцы вершины (Tweezer Tops Pattern).

            Args:
                data: DataFrame с данными OHLCV
                tolerance: Допустимое отклонение между максимумами

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 2:
                logger.warning(
                    "Недостаточно данных для определения паттерна шипцы вершины"
                )
                return pd.Series(False, index=data.index)

            # Создаем серию результатов
            tweezer_tops = pd.Series(False, index=data.index)

            # Определяем шипцы вершины
            for i in range(1, len(data)):
                # Первая свеча зеленая, вторая красная
                first_bullish = data["close"].iloc[i - 1] > data["open"].iloc[i - 1]
                second_bearish = data["close"].iloc[i] < data["open"].iloc[i]

                # Максимумы почти совпадают
                highs_close = (
                    abs(data["high"].iloc[i] - data["high"].iloc[i - 1])
                    < tolerance * data["high"].iloc[i - 1]
                )

                if first_bullish and second_bearish and highs_close:
                    tweezer_tops.iloc[i] = True

            return tweezer_tops

        @staticmethod
        @handle_error
        def tweezer_bottoms(data: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
            """
            Определяет паттерн шипцы основания (Tweezer Bottoms Pattern).

            Args:
                data: DataFrame с данными OHLCV
                tolerance: Допустимое отклонение между минимумами

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 2:
                logger.warning(
                    "Недостаточно данных для определения паттерна шипцы основания"
                )
                return pd.Series(False, index=data.index)

            # Создаем серию результатов
            tweezer_bottoms = pd.Series(False, index=data.index)

            # Определяем шипцы основания
            for i in range(1, len(data)):
                # Первая свеча красная, вторая зеленая
                first_bearish = data["close"].iloc[i - 1] < data["open"].iloc[i - 1]
                second_bullish = data["close"].iloc[i] > data["open"].iloc[i]

                # Минимумы почти совпадают
                lows_close = (
                    abs(data["low"].iloc[i] - data["low"].iloc[i - 1])
                    < tolerance * data["low"].iloc[i - 1]
                )

                if first_bearish and second_bullish and lows_close:
                    tweezer_bottoms.iloc[i] = True

            return tweezer_bottoms

        @staticmethod
        @handle_error
        def marubozu(data: pd.DataFrame, body_ratio: float = 0.95) -> pd.Series:
            """
            Определяет паттерн марубозу (Marubozu Pattern).

            Args:
                data: DataFrame с данными OHLCV
                body_ratio: Минимальное отношение тела к полной свече

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 1:
                logger.warning("Недостаточно данных для определения паттерна марубозу")
                return pd.Series(False, index=data.index)

            # Определяем верхнюю и нижнюю цены тела свечи
            body_high = data[["open", "close"]].max(axis=1)
            body_low = data[["open", "close"]].min(axis=1)

            # Рассчитываем размер тела свечи
            body_size = body_high - body_low

            # Рассчитываем размер всей свечи
            candle_size = data["high"] - data["low"]

            # Обрабатываем случай, когда high == low
            candle_size = candle_size.replace(0, float("nan"))

            # Рассчитываем отношение размера тела к размеру всей свечи
            body_to_candle_ratio = (body_size / candle_size).fillna(0)

            # Определяем марубозу
            marubozu = body_to_candle_ratio >= body_ratio

            return marubozu

        @staticmethod
        @handle_error
        def rising_three_methods(data: pd.DataFrame) -> pd.Series:
            """
            Определяет паттерн растущие три метода (Rising Three Methods Pattern).

            Args:
                data: DataFrame с данными OHLCV

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 5:
                logger.warning(
                    "Недостаточно данных для определения паттерна растущие три метода"
                )
                return pd.Series(False, index=data.index)

            # Создаем серию результатов
            rising_three_methods = pd.Series(False, index=data.index)

            # Определяем растущие три метода
            for i in range(4, len(data)):
                # Первая свеча - длинная бычья свеча
                first_bullish = data["close"].iloc[i - 4] > data["open"].iloc[i - 4]
                first_body_size = data["close"].iloc[i - 4] - data["open"].iloc[i - 4]

                # Последняя свеча - длинная бычья свеча, закрывается выше первой
                last_bullish = data["close"].iloc[i] > data["open"].iloc[i]
                last_closes_higher = data["close"].iloc[i] > data["close"].iloc[i - 4]

                # Три свечи между - короткие медвежьи свечи внутри первой
                inside_first = True
                all_bearish = True

                for j in range(1, 4):
                    if (
                        data["high"].iloc[i - j] > data["high"].iloc[i - 4]
                        or data["low"].iloc[i - j] < data["low"].iloc[i - 4]
                    ):
                        inside_first = False
                    if data["open"].iloc[i - j] < data["close"].iloc[i - j]:
                        all_bearish = False

                if (
                    first_bullish
                    and last_bullish
                    and last_closes_higher
                    and inside_first
                    and all_bearish
                ):
                    rising_three_methods.iloc[i] = True

            return rising_three_methods

        @staticmethod
        @handle_error
        def falling_three_methods(data: pd.DataFrame) -> pd.Series:
            """
            Определяет паттерн падающие три метода (Falling Three Methods Pattern).

            Args:
                data: DataFrame с данными OHLCV

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 5:
                logger.warning(
                    "Недостаточно данных для определения паттерна падающие три метода"
                )
                return pd.Series(False, index=data.index)

            # Создаем серию результатов
            falling_three_methods = pd.Series(False, index=data.index)

            # Определяем падающие три метода
            for i in range(4, len(data)):
                # Первая свеча - длинная медвежья свеча
                first_bearish = data["close"].iloc[i - 4] < data["open"].iloc[i - 4]
                first_body_size = data["open"].iloc[i - 4] - data["close"].iloc[i - 4]

                # Последняя свеча - длинная медвежья свеча, закрывается ниже первой
                last_bearish = data["close"].iloc[i] < data["open"].iloc[i]
                last_closes_lower = data["close"].iloc[i] < data["close"].iloc[i - 4]

                # Три свечи между - короткие бычьи свечи внутри первой
                inside_first = True
                all_bullish = True

                for j in range(1, 4):
                    if (
                        data["high"].iloc[i - j] > data["high"].iloc[i - 4]
                        or data["low"].iloc[i - j] < data["low"].iloc[i - 4]
                    ):
                        inside_first = False
                    if data["open"].iloc[i - j] > data["close"].iloc[i - j]:
                        all_bullish = False

                if (
                    first_bearish
                    and last_bearish
                    and last_closes_lower
                    and inside_first
                    and all_bullish
                ):
                    falling_three_methods.iloc[i] = True

            return falling_three_methods

        @staticmethod
        @handle_error
        def spinning_top(data: pd.DataFrame, body_ratio: float = 0.3) -> pd.Series:
            """
            Определяет паттерн волчок (Spinning Top Pattern).

            Args:
                data: DataFrame с данными OHLCV
                body_ratio: Максимальное отношение тела к полной свече

            Returns:
                Series с булевыми значениями (True для свечей с паттерном)
            """
            if len(data) < 1:
                logger.warning("Недостаточно данных для определения паттерна волчок")
                return pd.Series(False, index=data.index)

            # Определяем верхнюю и нижнюю цены тела свечи
            body_high = data[["open", "close"]].max(axis=1)
            body_low = data[["open", "close"]].min(axis=1)

            # Рассчитываем размер тела свечи
            body_size = body_high - body_low

            # Рассчитываем размер всей свечи
            candle_size = data["high"] - data["low"]

            # Обрабатываем случай, когда high == low
            candle_size = candle_size.replace(0, float("nan"))

            # Рассчитываем отношение размера тела к размеру всей свечи
            body_to_candle_ratio = (body_size / candle_size).fillna(0)

            # Проверяем наличие верхней и нижней теней
            has_upper_shadow = data["high"] > body_high
            has_lower_shadow = data["low"] < body_low

            # Определяем волчок
            spinning_top = (
                (body_to_candle_ratio <= body_ratio)
                & has_upper_shadow
                & has_lower_shadow
            )

            return spinning_top
