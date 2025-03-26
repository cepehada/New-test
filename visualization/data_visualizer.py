import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import io
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import os

from project.utils.logging_utils import setup_logger

logger = setup_logger("data_visualizer")


class DataVisualizer:
    """Класс для визуализации данных"""

    def __init__(self, theme: str = "dark", figsize: Tuple[int, int] = (14, 8)):
        """
        Инициализирует визуализатор данных

        Args:
            theme: Тема оформления ('dark', 'light')
            figsize: Размер фигуры по умолчанию
        """
        self.theme = theme
        self.figsize = figsize

        # Настройка стилей для matplotlib
        if theme == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        # Настройка глобальных параметров
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["axes.grid"] = True

        # Параметры для plotly
        self.plotly_config = {
            "dark": {
                "plot_bgcolor": "#1E1E1E",
                "paper_bgcolor": "#1E1E1E",
                "font": {"color": "white"},
                "gridcolor": "rgba(255, 255, 255, 0.1)",
            },
            "light": {
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "font": {"color": "black"},
                "gridcolor": "rgba(0, 0, 0, 0.1)",
            },
        }

        logger.info(f"DataVisualizer initialized with theme: {theme}")

    def plot_ohlc(
        self,
        data: pd.DataFrame,
        title: str = "OHLC Chart",
        indicators: Dict = None,
        volume: bool = True,
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит график OHLC

        Args:
            data: DataFrame с данными OHLCV
            title: Заголовок графика
            indicators: Словарь с индикаторами (имя -> DataFrame с данными)
            volume: Показывать объем
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Проверяем наличие необходимых колонок
            required_columns = ["open", "high", "low", "close"]
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return None

            # Создаем figure и определяем размер подграфиков
            fig, ax = plt.subplots(figsize=self.figsize)

            # Если нужно отображать объем, то создаем дополнительную ось
            if volume and "volume" in data.columns:
                ax_volume = ax.twinx()

            # Строим OHLC график
            ax.plot(data.index, data["close"], color="white", alpha=0.7, linewidth=1)

            # Отображаем Up/Down бары
            up = data[data.close >= data.open]
            down = data[data.close < data.open]

            # Отображаем бары
            bar_width = 0.7

            # Up бары
            ax.bar(
                up.index,
                up.close - up.open,
                bar_width,
                bottom=up.open,
                color="green",
                alpha=0.5,
            )
            ax.bar(
                up.index,
                up.high - up.close,
                bar_width / 5,
                bottom=up.close,
                color="green",
                alpha=0.5,
            )
            ax.bar(
                up.index,
                up.low - up.open,
                bar_width / 5,
                bottom=up.open,
                color="green",
                alpha=0.5,
            )

            # Down бары
            ax.bar(
                down.index,
                down.close - down.open,
                bar_width,
                bottom=down.open,
                color="red",
                alpha=0.5,
            )
            ax.bar(
                down.index,
                down.high - down.open,
                bar_width / 5,
                bottom=down.open,
                color="red",
                alpha=0.5,
            )
            ax.bar(
                down.index,
                down.low - down.close,
                bar_width / 5,
                bottom=down.close,
                color="red",
                alpha=0.5,
            )

            # Строим график объема, если требуется
            if volume and "volume" in data.columns:
                volume_color = (
                    "rgba(0, 0, 255, 0.3)"
                    if self.theme == "light"
                    else "rgba(100, 100, 255, 0.3)"
                )
                ax_volume.bar(data.index, data["volume"], color=volume_color, alpha=0.3)
                ax_volume.set_ylabel("Volume")
                ax_volume.spines["right"].set_position(("outward", 60))

                # Устанавливаем пределы для оси объема
                max_volume = data["volume"].max()
                ax_volume.set_ylim(0, max_volume * 3)

            # Добавляем индикаторы, если они указаны
            if indicators:
                for indicator_name, indicator_data in indicators.items():
                    if isinstance(indicator_data, pd.DataFrame) or isinstance(
                        indicator_data, pd.Series
                    ):
                        if isinstance(indicator_data, pd.Series):
                            ax.plot(
                                indicator_data.index,
                                indicator_data.values,
                                label=indicator_name,
                            )
                        else:
                            for column in indicator_data.columns:
                                ax.plot(
                                    indicator_data.index,
                                    indicator_data[column],
                                    label=f"{indicator_name} ({column})",
                                )
                    elif isinstance(indicator_data, dict):
                        # Если индикатор представлен как словарь, ищем ключи 'data' и 'color'
                        ind_data = indicator_data.get("data")
                        ind_color = indicator_data.get("color", "blue")
                        ind_label = indicator_data.get("label", indicator_name)

                        if isinstance(ind_data, pd.Series):
                            ax.plot(
                                ind_data.index,
                                ind_data.values,
                                color=ind_color,
                                label=ind_label,
                            )
                        elif isinstance(ind_data, pd.DataFrame):
                            for column in ind_data.columns:
                                ax.plot(
                                    ind_data.index,
                                    ind_data[column],
                                    color=ind_color,
                                    label=f"{ind_label} ({column})",
                                )

            # Форматируем оси и заголовок
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")

            # Настройка отображения дат
            if len(data) > 0:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                plt.xticks(rotation=45)

            # Добавляем легенду
            ax.legend(loc="upper left")

            # Настраиваем сетку
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting OHLC chart: {str(e)}")
            return None

    def plot_ohlc_plotly(
        self,
        data: pd.DataFrame,
        title: str = "OHLC Chart",
        indicators: Dict = None,
        volume: bool = True,
        return_fig: bool = False,
    ) -> Optional[Union[Any, str]]:
        """
        Строит интерактивный график OHLC с помощью Plotly

        Args:
            data: DataFrame с данными OHLCV
            title: Заголовок графика
            indicators: Словарь с индикаторами (имя -> DataFrame с данными)
            volume: Показывать объем
            return_fig: Вернуть объект Figure вместо HTML

        Returns:
            Optional[Union[Any, str]]: Объект Figure или строка HTML
        """
        try:
            # Проверяем наличие необходимых колонок
            required_columns = ["open", "high", "low", "close"]
            for col in required_columns:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return None

            # Определяем количество подграфиков
            if volume and "volume" in data.columns:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.8, 0.2],
                    subplot_titles=(title, "Volume"),
                )
            else:
                fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

            # Добавляем свечной график
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name="Price",
                ),
                row=1,
                col=1,
            )

            # Добавляем объем
            if volume and "volume" in data.columns:
                colors = [
                    "green" if row["close"] >= row["open"] else "red"
                    for _, row in data.iterrows()
                ]
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data["volume"],
                        marker={"color": colors, "opacity": 0.5},
                        name="Volume",
                    ),
                    row=2,
                    col=1,
                )

            # Добавляем индикаторы, если они указаны
            if indicators:
                for indicator_name, indicator_data in indicators.items():
                    if isinstance(indicator_data, pd.DataFrame) or isinstance(
                        indicator_data, pd.Series
                    ):
                        if isinstance(indicator_data, pd.Series):
                            fig.add_trace(
                                go.Scatter(
                                    x=indicator_data.index,
                                    y=indicator_data.values,
                                    mode="lines",
                                    name=indicator_name,
                                ),
                                row=1,
                                col=1,
                            )
                        else:
                            for column in indicator_data.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=indicator_data.index,
                                        y=indicator_data[column],
                                        mode="lines",
                                        name=f"{indicator_name} ({column})",
                                    ),
                                    row=1,
                                    col=1,
                                )
                    elif isinstance(indicator_data, dict):
                        # Если индикатор представлен как словарь, ищем ключи 'data' и 'color'
                        ind_data = indicator_data.get("data")
                        ind_color = indicator_data.get("color", "blue")
                        ind_label = indicator_data.get("label", indicator_name)

                        if isinstance(ind_data, pd.Series):
                            fig.add_trace(
                                go.Scatter(
                                    x=ind_data.index,
                                    y=ind_data.values,
                                    mode="lines",
                                    line={"color": ind_color},
                                    name=ind_label,
                                ),
                                row=1,
                                col=1,
                            )
                        elif isinstance(ind_data, pd.DataFrame):
                            for column in ind_data.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=ind_data.index,
                                        y=ind_data[column],
                                        mode="lines",
                                        line={"color": ind_color},
                                        name=f"{ind_label} ({column})",
                                    ),
                                    row=1,
                                    col=1,
                                )

            # Применяем тему
            theme_params = (
                self.plotly_config["dark"]
                if self.theme == "dark"
                else self.plotly_config["light"]
            )

            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Price",
                plot_bgcolor=theme_params["plot_bgcolor"],
                paper_bgcolor=theme_params["paper_bgcolor"],
                font=theme_params["font"],
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                height=self.figsize[1] * 80,
                width=self.figsize[0] * 80,
            )

            fig.update_xaxes(gridcolor=theme_params["gridcolor"], showgrid=True)

            fig.update_yaxes(gridcolor=theme_params["gridcolor"], showgrid=True)

            # Возвращаем результат
            if return_fig:
                return fig

            # Преобразуем в HTML
            html = fig.to_html(full_html=False, include_plotlyjs="cdn")

            return html

        except Exception as e:
            logger.error(f"Error plotting OHLC chart with Plotly: {str(e)}")
            return None

    def plot_equity_curve(
        self,
        equity_curve: List[Dict],
        title: str = "Equity Curve",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит график изменения капитала

        Args:
            equity_curve: Список словарей с данными о капитале
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Преобразуем список словарей в DataFrame
            if isinstance(equity_curve, list) и len(equity_curve) > 0:
                df = pd.DataFrame(equity_curve)

                # Преобразуем временные метки
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
            else:
                logger.error("Invalid equity curve data")
                return None

            # Создаем фигуру
            fig, ax = plt.subplots(figsize=self.figsize)

            # Строим график капитала
            if "equity" in df.columns:
                ax.plot(df.index, df["equity"], label="Equity", linewidth=2)

            # Строим график баланса, если он доступен
            if "balance" in df.columns:
                ax.plot(
                    df.index, df["balance"], label="Balance", linewidth=1, alpha=0.7
                )

            # Строим график просадки, если она доступна
            if "drawdown_pct" in df.columns:
                ax_drawdown = ax.twinx()
                ax_drawdown.fill_between(
                    df.index,
                    0,
                    df["drawdown_pct"] * 100,
                    alpha=0.3,
                    color="red",
                    label="Drawdown %",
                )
                ax_drawdown.set_ylabel("Drawdown %")
                ax_drawdown.set_ylim(
                    0, df["drawdown_pct"].max() * 150
                )  # Устанавливаем пределы для оси просадки

            # Форматируем оси и заголовок
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity")

            # Настройка отображения дат
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)

            # Добавляем легенду
            ax.legend(loc="upper left")

            # Добавляем легенду для просадки, если она отображается
            if "drawdown_pct" in df.columns:
                ax_drawdown.legend(loc="upper right")

            # Настраиваем сетку
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            return None

    def plot_equity_curve_plotly(
        self,
        equity_curve: List[Dict],
        title: str = "Equity Curve",
        return_fig: bool = False,
    ) -> Optional[Union[Any, str]]:
        """
        Строит интерактивный график изменения капитала с помощью Plotly

        Args:
            equity_curve: Список словарей с данными о капитале
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо HTML

        Returns:
            Optional[Union[Any, str]]: Объект Figure или строка HTML
        """
        try:
            # Преобразуем список словарей в DataFrame
            if isinstance(equity_curve, list) и len(equity_curve) > 0:
                df = pd.DataFrame(equity_curve)

                # Преобразуем временные метки
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                logger.error("Invalid equity curve data")
                return None

            # Создаем подграфики для капитала и просадки
            if "drawdown_pct" in df.columns:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(title, "Drawdown %"),
                )
            else:
                fig = make_subplots(rows=1, cols=1, subplot_titles=[title])

            # Добавляем график капитала
            if "equity" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["equity"],
                        mode="lines",
                        name="Equity",
                        line={"width": 2},
                    ),
                    row=1,
                    col=1,
                )

            # Добавляем график баланса, если он доступен
            if "balance" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["balance"],
                        mode="lines",
                        name="Balance",
                        line={"width": 1, "dash": "dot"},
                    ),
                    row=1,
                    col=1,
                )

            # Добавляем график просадки, если она доступна
            if "drawdown_pct" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["drawdown_pct"] * 100,
                        fill="tozeroy",
                        mode="lines",
                        name="Drawdown %",
                        line={"color": "red"},
                        fillcolor="rgba(255, 0, 0, 0.3)",
                    ),
                    row=2,
                    col=1,
                )

            # Применяем тему
            theme_params = (
                self.plotly_config["dark"]
                if self.theme == "dark"
                else self.plotly_config["light"]
            )

            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Equity",
                plot_bgcolor=theme_params["plot_bgcolor"],
                paper_bgcolor=theme_params["paper_bgcolor"],
                font=theme_params["font"],
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                height=self.figsize[1] * 80,
                width=self.figsize[0] * 80,
            )

            fig.update_xaxes(gridcolor=theme_params["gridcolor"], showgrid=True)

            fig.update_yaxes(gridcolor=theme_params["gridcolor"], showgrid=True)

            # Устанавливаем название оси Y для просадки
            if "drawdown_pct" in df.columns:
                fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

            # Возвращаем результат
            if return_fig:
                return fig

            # Преобразуем в HTML
            html = fig.to_html(full_html=False, include_plotlyjs="cdn")

            return html

        except Exception as e:
            logger.error(f"Error plotting equity curve with Plotly: {str(e)}")
            return None

    def plot_trades(
        self,
        data: pd.DataFrame,
        trades: List[Dict],
        title: str = "Trades",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит график с отметками сделок

        Args:
            data: DataFrame с данными OHLCV
            trades: Список словарей с данными о сделках
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Сначала строим OHLC график
            fig = self.plot_ohlc(data, title=title, return_fig=True)

            if fig is None:
                return None

            # Получаем ось для отрисовки сделок
            ax = fig.axes[0]

            # Добавляем сделки на график
            for trade in trades:
                # Получаем дату сделки
                if "timestamp" in trade:
                    timestamp = pd.to_datetime(trade["timestamp"])
                elif "datetime" in trade:
                    timestamp = pd.to_datetime(trade["datetime"])
                else:
                    continue

                # Получаем направление и цену
                direction = trade.get("direction", trade.get("side"))
                price = trade.get("price")

                if timestamp and direction and price:
                    # Определяем цвет и маркер
                    if direction.lower() in ["buy", "long"]:
                        color = "green"
                        marker = "^"  # треугольник вверх
                    else:
                        color = "red"
                        marker = "v"  # треугольник вниз

                    # Добавляем маркер
                    ax.plot(
                        timestamp,
                        price,
                        marker=marker,
                        markersize=10,
                        color=color,
                        alpha=0.7,
                    )

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting trades: {str(e)}")
            return None

    def plot_trades_plotly(
        self,
        data: pd.DataFrame,
        trades: List[Dict],
        title: str = "Trades",
        return_fig: bool = False,
    ) -> Optional[Union[Any, str]]:
        """
        Строит интерактивный график с отметками сделок с помощью Plotly

        Args:
            data: DataFrame с данными OHLCV
            trades: Список словарей с данными о сделках
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо HTML

        Returns:
            Optional[Union[Any, str]]: Объект Figure или строка HTML
        """
        try:
            # Сначала строим OHLC график
            fig = self.plot_ohlc_plotly(data, title=title, return_fig=True)

            if fig is None:
                return None

            # Создаем списки для сделок на покупку и продажу
            buy_timestamps = []
            buy_prices = []
            sell_timestamps = []
            sell_prices = []

            # Заполняем списки данными
            for trade in trades:
                # Получаем дату сделки
                if "timestamp" in trade:
                    timestamp = pd.to_datetime(trade["timestamp"])
                elif "datetime" in trade:
                    timestamp = pd.to_datetime(trade["datetime"])
                else:
                    continue

                direction = trade.get("direction", trade.get("side"))
                price = trade.get("price")

                if timestamp and direction and price:
                    # Распределяем по соответствующим спискам
                    if direction.lower() in ["buy", "long"]:
                        buy_timestamps.append(timestamp)
                        buy_prices.append(price)
                    else:
                        sell_timestamps.append(timestamp)
                        sell_prices.append(price)

            # Добавляем маркеры для сделок на покупку
            if buy_timestamps:
                fig.add_trace(
                    go.Scatter(
                        x=buy_timestamps,
                        y=buy_prices,
                        mode="markers",
                        name="Buy",
                        marker=dict(
                            symbol="triangle-up", size=12, color="green", opacity=0.7
                        ),
                    ),
                    row=1,
                    col=1,
                )

            # Добавляем маркеры для сделок на продажу
            if sell_timestamps:
                fig.add_trace(
                    go.Scatter(
                        x=sell_timestamps,
                        y=sell_prices,
                        mode="markers",
                        name="Sell",
                        marker=dict(
                            symbol="triangle-down", size=12, color="red", opacity=0.7
                        ),
                    ),
                    row=1,
                    col=1,
                )

            # Возвращаем результат
            if return_fig:
                return fig

            # Преобразуем в HTML
            html = fig.to_html(full_html=False, include_plotlyjs="cdn")

            return html

        except Exception as e:
            logger.error(f"Error plotting trades with Plotly: {str(e)}")
            return None

    def plot_drawdown(
        self,
        equity_curve: List[Dict],
        title: str = "Drawdown",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит график просадки

        Args:
            equity_curve: Список словарей с данными о капитале
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Преобразуем список словарей в DataFrame
            if isinstance(equity_curve, list) и len(equity_curve) > 0:
                df = pd.DataFrame(equity_curve)

                # Преобразуем временные метки
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
            else:
                logger.error("Invalid equity curve data")
                return None

            # Проверяем наличие данных о просадке
            if "drawdown_pct" not in df.columns:
                logger.error("Drawdown data not found in equity curve")
                return None

            # Создаем фигуру
            fig, ax = plt.subplots(figsize=self.figsize)

            # Строим график просадки
            ax.fill_between(
                df.index, 0, df["drawdown_pct"] * 100, alpha=0.7, color="red"
            )
            ax.plot(df.index, df["drawdown_pct"] * 100, color="darkred", linewidth=1)

            # Добавляем горизонтальные линии для уровней просадки
            max_dd = df["drawdown_pct"].max() * 100
            levels = [5, 10, 15, 20, 25, 30, 50]
            for level in levels:
                if level <= max_dd * 1.5:
                    ax.axhline(y=level, color="gray", linestyle="--", alpha=0.5)
                    ax.text(df.index[0], level, f"{level}%", fontsize=8)

            # Подписываем максимальную просадку
            max_dd_idx = df["drawdown_pct"].idxmax()
            ax.plot(max_dd_idx, max_dd, "ro")
            ax.annotate(
                f"Max DD: {max_dd:.2f}%",
                xy=(max_dd_idx, max_dd),
                xytext=(
                    max_dd_idx,
                    max_dd - max_dd * 0.2 if max_dd > 10 else max_dd + 5,
                ),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
            )

            # Форматируем оси и заголовок
            ax.set_title(title)
            ax.set_xlabel("Date")
            ax.set_ylabel("Drawdown %")

            # Настройка отображения дат
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)

            # Устанавливаем пределы для оси Y
            ax.set_ylim(0, max_dd * 1.5)

            # Настраиваем сетку
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting drawdown: {str(e)}")
            return None

    def plot_monthly_returns(
        self,
        trades: List[Dict],
        title: str = "Monthly Returns",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит столбчатую диаграмму ежемесячной доходности

        Args:
            trades: Список словарей с данными о сделках
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Преобразуем список сделок в DataFrame
            if not trades:
                logger.error("No trades data provided")
                return None

            # Создаем DataFrame
            df = pd.DataFrame(trades)

            # Преобразуем временные метки
            timestamp_field = "timestamp" if "timestamp" in df.columns else "datetime"
            if timestamp_field in df.columns:
                df[timestamp_field] = pd.to_datetime(df[timestamp_field])
            else:
                logger.error("Timestamp field not found in trades data")
                return None

            # Создаем поле с месяцем и годом
            df["month_year"] = df[timestamp_field].dt.strftime("%Y-%m")

            # Рассчитываем ежемесячную доходность
            if "realized_pnl" in df.columns:
                profit_field = "realized_pnl"
            elif "profit" in df.columns:
                profit_field = "profit"
            else:
                logger.error("Profit field not found in trades data")
                return None

            # Группируем по месяцам и суммируем прибыль
            monthly_returns = df.groupby("month_year")[profit_field].sum()

            # Создаем фигуру
            fig, ax = plt.subplots(figsize=self.figsize)

            # Строим столбчатую диаграмму
            colors = ["green" if x >= 0 else "red" for x in monthly_returns]
            ax.bar(monthly_returns.index, monthly_returns, color=colors, alpha=0.7)

            # Добавляем горизонтальную линию на нуле
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)

            # Добавляем подписи со значениями
            for i, v in enumerate(monthly_returns):
                ax.text(
                    i,
                    v + (0.01 * abs(v)) if v >= 0 else v - (0.03 * abs(v)),
                    f"{v:.2f}",
                    ha="center",
                    fontsize=8,
                )

            # Форматируем оси и заголовок
            ax.set_title(title)
            ax.set_xlabel("Month")
            ax.set_ylabel("Return")

            # Поворачиваем метки оси X
            plt.xticks(rotation=90)

            # Настраиваем сетку
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting monthly returns: {str(e)}")
            return None

    def plot_win_loss_distribution(
        self,
        trades: List[Dict],
        title: str = "Win/Loss Distribution",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит распределение выигрышных и проигрышных сделок

        Args:
            trades: Список словарей с данными о сделках
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Преобразуем список сделок в DataFrame
            if not trades:
                logger.error("No trades data provided")
                return None

            # Создаем DataFrame
            df = pd.DataFrame(trades)

            # Определяем поле с прибылью
            if "realized_pnl" in df.columns:
                profit_field = "realized_pnl"
            elif "profit" in df.columns:
                profit_field = "profit"
            else:
                logger.error("Profit field not found in trades data")
                return None

            # Разделяем на выигрышные и проигрышные сделки
            win_trades = df[df[profit_field] > 0][profit_field]
            loss_trades = df[df[profit_field] < 0][profit_field]

            # Создаем фигуру с двумя подграфиками
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

            # Настройка bins для гистограмм
            win_bins = min(20, len(win_trades)) if len(win_trades) > 0 else 10
            loss_bins = min(20, len(loss_trades)) if len(loss_trades) > 0 else 10

            # Строим гистограмму выигрышных сделок
            if len(win_trades) > 0:
                ax1.hist(win_trades, bins=win_bins, color="green", alpha=0.7)
                ax1.set_title("Winning Trades")
                ax1.set_xlabel("Profit")
                ax1.set_ylabel("Frequency")
            else:
                ax1.text(0.5, 0.5, "No winning trades", ha="center", va="center")
                ax1.set_title("Winning Trades")

            # Строим гистограмму проигрышных сделок
            if len(loss_trades) > 0:
                ax2.hist(loss_trades, bins=loss_bins, color="red", alpha=0.7)
                ax2.set_title("Losing Trades")
                ax2.set_xlabel("Loss")
                ax2.set_ylabel("Frequency")
            else:
                ax2.text(0.5, 0.5, "No losing trades", ha="center", va="center")
                ax2.set_title("Losing Trades")

            # Добавляем статистику
            if len(win_trades) > 0:
                ax1.text(
                    0.05,
                    0.95,
                    f"Count: {len(win_trades)}\nMean: {win_trades.mean():.2f}\nMax: {win_trades.max():.2f}",
                    transform=ax1.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", alpha=0.1),
                )

            if len(loss_trades) > 0:
                ax2.text(
                    0.05,
                    0.95,
                    f"Count: {len(loss_trades)}\nMean: {loss_trades.mean():.2f}\nMin: {loss_trades.min():.2f}",
                    transform=ax2.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", alpha=0.1),
                )

            # Устанавливаем общий заголовок
            fig.suptitle(title)

            # Настраиваем сетку
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout(
                rect=[0, 0, 1, 0.95]
            )  # Оставляем место для общего заголовка

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting win/loss distribution: {str(e)}")
            return None

    def plot_trade_durations(
        self,
        trades: List[Dict],
        title: str = "Trade Durations",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит распределение продолжительности сделок

        Args:
            trades: Список словарей с данными о сделках
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Преобразуем список сделок в DataFrame
            if not trades:
                logger.error("No trades data provided")
                return None

            # Создаем DataFrame
            df = pd.DataFrame(trades)

            # Проверяем наличие необходимых колонок
            if ("open_time" not in df.columns or "close_time" not in df.columns) and (
                "timestamp" not in df.columns or "exit_timestamp" not in df.columns
            ):
                logger.error("Open/close time fields not found in trades data")
                return None

            # Определяем поля с временными метками
            if "open_time" in df.columns and "close_time" in df.columns:
                open_time_field = "open_time"
                close_time_field = "close_time"
            else:
                open_time_field = "timestamp"
                close_time_field = "exit_timestamp"

            # Преобразуем временные метки
            df[open_time_field] = pd.to_datetime(df[open_time_field])
            df[close_time_field] = pd.to_datetime(df[close_time_field])

            # Вычисляем продолжительность сделок в часах
            df["duration_hours"] = (
                df[close_time_field] - df[open_time_field]
            ).dt.total_seconds() / 3600

            # Определяем поле с прибылью
            if "realized_pnl" in df.columns:
                profit_field = "realized_pnl"
            elif "profit" in df.columns:
                profit_field = "profit"
            else:
                profit_field = None

            # Создаем фигуру
            fig, ax = plt.subplots(figsize=self.figsize)

            # Если есть данные о прибыли, раскрашиваем по результату
            if profit_field in df.columns:
                win_durations = df[df[profit_field] > 0]["duration_hours"]
                loss_durations = df[df[profit_field] < 0]["duration_hours"]

                # Настройка bins для гистограмм
                bins = np.linspace(0, df["duration_hours"].max() * 1.1, 20)

                # Строим гистограммы
                ax.hist(
                    win_durations,
                    bins=bins,
                    color="green",
                    alpha=0.5,
                    label="Winning Trades",
                )
                ax.hist(
                    loss_durations,
                    bins=bins,
                    color="red",
                    alpha=0.5,
                    label="Losing Trades",
                )

                # Добавляем легенду
                ax.legend()
            else:
                # Строим общую гистограмму
                ax.hist(df["duration_hours"], bins=20, alpha=0.7)

            # Форматируем оси и заголовок
            ax.set_title(title)
            ax.set_xlabel("Duration (hours)")
            ax.set_ylabel("Frequency")

            # Добавляем статистику
            stats_text = f"Mean: {df['duration_hours'].mean():.2f} hours\nMedian: {df['duration_hours'].median():.2f} hours\nMax: {df['duration_hours'].max():.2f} hours"
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", alpha=0.1),
            )

            # Настраиваем сетку
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting trade durations: {str(e)}")
            return None

    def plot_correlation_matrix(
        self,
        data: Dict[str, pd.DataFrame],
        title: str = "Correlation Matrix",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит матрицу корреляции между различными инструментами или стратегиями

        Args:
            data: Словарь DataFrame с данными (имя -> DataFrame)
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Проверяем входные данные
            if not data:
                logger.error("No data provided")
                return None

            # Создаем DataFrame для корреляции
            correlation_df = pd.DataFrame()

            # Объединяем данные в один DataFrame
            for name, df in data.items():
                if isinstance(df, pd.DataFrame) and "close" in df.columns:
                    # Переименовываем колонку close и добавляем в общий DataFrame
                    correlation_df[name] = df["close"]
                elif isinstance(df, pd.Series):
                    # Если это Series, добавляем как есть
                    correlation_df[name] = df

            # Рассчитываем матрицу корреляции
            corr_matrix = correlation_df.corr()

            # Создаем фигуру
            fig, ax = plt.subplots(figsize=self.figsize)

            # Отображаем матрицу корреляции в виде тепловой карты
            im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

            # Добавляем цветовую шкалу
            plt.colorbar(im, ax=ax)

            # Настраиваем метки осей
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns)
            ax.set_yticklabels(corr_matrix.columns)

            # Поворачиваем метки оси X
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            # Добавляем значения в ячейки
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(
                        j,
                        i,
                        f"{corr_matrix.iloc[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white",
                    )

            # Устанавливаем заголовок
            ax.set_title(title)

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")
            return None

    def plot_optimization_results(
        self,
        results: Dict,
        param1: str,
        param2: str,
        title: str = "Optimization Results",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит результаты оптимизации в виде тепловой карты

        Args:
            results: Результаты оптимизации
            param1: Имя первого параметра
            param2: Имя второго параметра
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Проверяем входные данные
            if not results или "all_results" not in results:
                logger.error("Invalid optimization results")
                return None

            # Получаем все результаты
            all_results = results["all_results"]

            # Проверяем, что параметры существуют
            if not all(
                param1 in result["parameters"] and param2 in result["parameters"]
                for result in all_results
            ):
                logger.error(
                    f"Parameters {param1} and/or {param2} not found in all results"
                )
                return None

            # Создаем DataFrame с результатами
            df = pd.DataFrame(
                [
                    {
                        param1: result["parameters"][param1],
                        param2: result["parameters"][param2],
                        "fitness": result["fitness"],
                    }
                    for result in all_results
                ]
            )

            # Преобразуем в матрицу для тепловой карты
            pivot_table = df.pivot_table(
                values="fitness", index=param1, columns=param2, aggfunc="mean"
            )

            # Создаем фигуру
            fig, ax = plt.subplots(figsize=self.figsize)

            # Отображаем тепловую карту
            im = ax.imshow(pivot_table, cmap="viridis")

            # Добавляем цветовую шкалу
            plt.colorbar(im, ax=ax)

            # Настраиваем метки осей
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns)
            ax.set_yticklabels(pivot_table.index)

            # Поворачиваем метки оси X
            plt.setp(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )

            # Добавляем значения в ячейки
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    value = pivot_table.iloc[i, j]
                    if not np.isnan(value):
                        text = ax.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color=(
                                "black"
                                if value < pivot_table.max().max() * 0.7
                                else "white"
                            ),
                        )

            # Устанавливаем заголовок и метки осей
            ax.set_title(title)
            ax.set_xlabel(param2)
            ax.set_ylabel(param1)

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting optimization results: {str(e)}")
            return None

    def plot_optimization_3d(
        self,
        results: Dict,
        param1: str,
        param2: str,
        title: str = "Optimization Results 3D",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит 3D график результатов оптимизации

        Args:
            results: Результаты оптимизации
            param1: Имя первого параметра
            param2: Имя второго параметра
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Проверяем входные данные
            if not results или "all_results" not in results:
                logger.error("Invalid optimization results")
                return None

            # Получаем все результаты
            all_results = results["all_results"]

            # Проверяем, что параметры существуют
            if not all(
                param1 in result["parameters"] and param2 in result["parameters"]
                for result in all_results
            ):
                logger.error(
                    f"Parameters {param1} and/or {param2} not found in all results"
                )
                return None

            # Создаем DataFrame с результатами
            df = pd.DataFrame(
                [
                    {
                        param1: result["parameters"][param1],
                        param2: result["parameters"][param2],
                        "fitness": result["fitness"],
                    }
                    for result in all_results
                ]
            )

            # Создаем фигуру с 3D-графиком
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection="3d")

            # Строим 3D-поверхность
            x_unique = sorted(df[param1].unique())
            y_unique = sorted(df[param2].unique())

            if len(x_unique) > 1 и len(y_unique) > 1:
                # Создаем сетку значений
                X, Y = np.meshgrid(x_unique, y_unique)

                # Создаем матрицу значений fitness
                Z = np.zeros_like(X, dtype=float)

                # Заполняем матрицу значениями
                for i, x_val in enumerate(x_unique):
                    for j, y_val in enumerate(y_unique):
                        # Получаем среднее значение fitness для данной комбинации параметров
                        fitness_vals = df[
                            (df[param1] == x_val) & (df[param2] == y_val)
                        ]["fitness"]
                        if not fitness_vals.empty:
                            Z[j, i] = fitness_vals.mean()
                        else:
                            Z[j, i] = np.nan

                # Строим поверхность
                surf = ax.plot_surface(
                    X, Y, Z, cmap="viridis", alpha=0.8, linewidth=0, antialiased=True
                )

                # Добавляем цветовую шкалу
                fig.colorbar(surf, shrink=0.5, aspect=5)

                # Находим точку с максимальным значением fitness
                max_idx = np.nanargmax(Z)
                max_i, max_j = np.unravel_index(max_idx, Z.shape)

                # Отмечаем точку с максимальным значением
                ax.scatter(
                    [X[max_i, max_j]],
                    [Y[max_i, max_j]],
                    [Z[max_i, max_j]],
                    color="red",
                    s=100,
                    label="Best Result",
                )

                # Добавляем аннотацию с лучшим результатом
                ax.text(
                    X[max_i, max_j],
                    Y[max_i, max_j],
                    Z[max_i, max_j],
                    f"Best: {Z[max_i, max_j]:.2f}\n{param1}={X[max_i, max_j]}, {param2}={Y[max_i, max_j]}",
                    color="black",
                )
            else:
                # Если недостаточно уникальных значений для сетки, строим точечный график
                ax.scatter(
                    df[param1],
                    df[param2],
                    df["fitness"],
                    c=df["fitness"],
                    cmap="viridis",
                    s=50,
                )

                # Находим лучший результат
                best_result = df.loc[df["fitness"].idxmax()]

                # Отмечаем лучший результат
                ax.scatter(
                    [best_result[param1]],
                    [best_result[param2]],
                    [best_result["fitness"]],
                    color="red",
                    s=100,
                    label="Best Result",
                )

                # Добавляем аннотацию с лучшим результатом
                ax.text(
                    best_result[param1],
                    best_result[param2],
                    best_result["fitness"],
                    f"Best: {best_result['fitness']:.2f}\n{param1}={best_result[param1]}, {param2}={best_result[param2]}",
                    color="black",
                )

            # Устанавливаем метки осей и заголовок
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.set_zlabel("Fitness")
            ax.set_title(title)

            # Добавляем легенду
            ax.legend()

            plt.tight_layout()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting 3D optimization results: {str(e)}")
            return None

    def plot_signals(
        self,
        data: pd.DataFrame,
        signals: List[Dict],
        title: str = "Trading Signals",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит график цены с торговыми сигналами

        Args:
            data: DataFrame с данными OHLCV
            signals: Список словарей с сигналами
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Сначала строим OHLC график
            fig = self.plot_ohlc(data, title=title, return_fig=True)

            if fig is None:
                return None

            # Получаем ось для отрисовки сигналов
            ax = fig.axes[0]

            # Добавляем сигналы на график
            buy_timestamps = []
            buy_prices = []
            sell_timestamps = []
            sell_prices = []
            close_timestamps = []
            close_prices = []

            for signal in signals:
                # Получаем дату сигнала
                if "timestamp" in signal:
                    timestamp = pd.to_datetime(signal["timestamp"])
                elif "datetime" in signal:
                    timestamp = pd.to_datetime(signal["datetime"])
                else:
                    continue

                # Получаем направление и цену
                direction = signal.get("direction")
                price = signal.get("price")

                # Если цена не указана, используем цену закрытия для соответствующей даты
                if price is None и timestamp in data.index:
                    price = data.loc[timestamp, "close"]
                elif price is None:
                    # Находим ближайшую дату
                    nearest_date = min(data.index, key=lambda x: abs(x - timestamp))
                    price = data.loc[nearest_date, "close"]

                if timestamp and direction and price:
                    # Распределяем по спискам в зависимости от направления
                    if direction.lower() == "buy":
                        buy_timestamps.append(timestamp)
                        buy_prices.append(price)
                    elif direction.lower() == "sell":
                        sell_timestamps.append(timestamp)
                        sell_prices.append(price)
                    elif direction.lower() == "close":
                        close_timestamps.append(timestamp)
                        close_prices.append(price)

            # Добавляем маркеры для сигналов покупки
            if buy_timestamps:
                ax.scatter(
                    buy_timestamps,
                    buy_prices,
                    marker="^",
                    color="green",
                    s=100,
                    label="Buy Signal",
                )

            # Добавляем маркеры для сигналов продажи
            if sell_timestamps:
                ax.scatter(
                    sell_timestamps,
                    sell_prices,
                    marker="v",
                    color="red",
                    s=100,
                    label="Sell Signal",
                )

            # Добавляем маркеры для сигналов закрытия
            if close_timestamps:
                ax.scatter(
                    close_timestamps,
                    close_prices,
                    marker="x",
                    color="black",
                    s=100,
                    label="Close Signal",
                )

            # Обновляем легенду
            ax.legend()

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting signals: {str(e)}")
            return None

    def plot_market_dashboard(
        self,
        data: Dict[str, pd.DataFrame],
        title: str = "Market Dashboard",
        return_fig: bool = False,
    ) -> Optional[Union[Figure, str]]:
        """
        Строит панель мониторинга рынка с несколькими графиками

        Args:
            data: Словарь с данными для разных инструментов (символ -> DataFrame)
            title: Заголовок графика
            return_fig: Вернуть объект Figure вместо изображения

        Returns:
            Optional[Union[Figure, str]]: Объект Figure или строка с кодировкой base64
        """
        try:
            # Проверяем входные данные
            if not data:
                logger.error("No data provided")
                return None

            # Определяем количество инструментов
            num_instruments = len(data)

            # Определяем размер сетки
            nrows = int(np.ceil(np.sqrt(num_instruments)))
            ncols = int(np.ceil(num_instruments / nrows))

            # Создаем фигуру
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(self.figsize[0], self.figsize[1] * nrows / ncols)
            )

            # Если только один инструмент, превращаем axes в массив
            if num_instruments == 1:
                axes = np.array([axes])

            # Перебираем инструменты и создаем графики
            for i, (symbol, df) in enumerate(data.items()):
                # Получаем соответствующую ось
                if nrows > 1 и ncols > 1:
                    ax = axes[i // ncols, i % ncols]
                elif nrows > 1 или ncols > 1:
                    ax = axes[i]
                else:
                    ax = axes

                # Проверяем наличие необходимых колонок
                required_columns = ["open", "high", "low", "close"]
                if all(col in df.columns for col in required_columns):
                    # Строим OHLC график
                    ax.plot(
                        df.index,
                        df["close"],
                        color="white" if self.theme == "dark" else "black",
                        alpha=0.7,
                        linewidth=1,
                    )

                    # Отображаем Up/Down бары
                    up = df[df.close >= df.open]
                    down = df[df.close < df.open]

                    # Отображаем бары
                    bar_width = 0.7

                    # Up бары
                    ax.bar(
                        up.index,
                        up.close - up.open,
                        bar_width,
                        bottom=up.open,
                        color="green",
                        alpha=0.5,
                    )

                    # Down бары
                    ax.bar(
                        down.index,
                        down.close - down.open,
                        bar_width,
                        bottom=down.open,
                        color="red",
                        alpha=0.5,
                    )

                    # Добавляем последнюю цену в заголовок
                    last_price = df["close"].iloc[-1]
                    pct_change = ((last_price / df["close"].iloc[0]) - 1) * 100

                    ax.set_title(f"{symbol}: {last_price:.2f} ({pct_change:.2f}%)")
                else:
                    # Если нет OHLCV данных, строим линейный график
                    if "close" in df.columns:
                        ax.plot(df.index, df["close"], label="Close")
                    elif len(df.columns) > 0:
                        # Строим график первой колонки
                        ax.plot(df.index, df[df.columns[0]], label=df.columns[0])

                    ax.set_title(symbol)

                # Форматируем ось X
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
                ax.tick_params(axis="x", rotation=45)

                # Настраиваем сетку
                ax.grid(True, alpha=0.3)

            # Скрываем пустые графики
            for i in range(num_instruments, nrows * ncols):
                if nrows > 1 и ncols > 1:
                    axes[i // ncols, i % ncols].axis("off")
                elif nrows > 1 или ncols > 1:
                    axes[i].axis("off")

            # Устанавливаем общий заголовок
            fig.suptitle(title, fontsize=16)

            plt.tight_layout(
                rect=[0, 0, 1, 0.95]
            )  # Оставляем место для общего заголовка

            # Возвращаем результат
            if return_fig:
                return fig

            # Сохраняем изображение в буфер
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            buf.seek(0)

            # Кодируем в base64
            img_str = base64.b64encode(buf.read()).decode("utf-8")

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            return img_str

        except Exception as e:
            logger.error(f"Error plotting market dashboard: {str(e)}")
            return None

    def save_figure(self, fig: Figure, filename: str, dpi: int = 100) -> bool:
        """
        Сохраняет фигуру в файл

        Args:
            fig: Объект Figure
            filename: Имя файла
            dpi: Разрешение

        Returns:
            bool: True, если сохранение успешно, иначе False
        """
        try:
            # Создаем директорию, если её нет
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

            # Сохраняем фигуру
            fig.savefig(filename, dpi=dpi)

            # Закрываем фигуру, чтобы избежать утечки памяти
            plt.close(fig)

            logger.info(f"Figure saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving figure: {str(e)}")
            return False

    def get_connection_params(self) -> Tuple[str, dict]:
        """
        Возвращает параметры подключения для визуализатора.
        
        Returns:
            Tuple с URL и параметрами подключения.
        """
        # Исправляем ошибку с Tuple
        return "visualization_url", {"param1": "value1"}


def create_visualizer(theme: str = "dark") -> DataVisualizer:
    """
    Создает экземпляр визуализатора данных

    Args:
        theme: Тема оформления ('dark', 'light')

    Returns:
        DataVisualizer: Экземпляр визуализатора данных
    """
    return DataVisualizer(theme=theme)
