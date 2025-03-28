"""
Модуль для управления портфелем.
Предоставляет функции для отслеживания и оптимизации портфеля активов.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from project.config import get_config
from project.risk_management.position_sizer import PositionSizer
from project.risk_management.var_calculator import VarCalculator
from project.utils.ccxt_exchanges import fetch_balance, fetch_ticker
from project.utils.error_handler import async_handle_error, handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Asset:
    """
    Информация об активе в портфеле.
    """

    symbol: str
    exchange_id: str
    amount: float
    price: float
    value: float
    weight: float
    last_update: int  # timestamp


class PortfolioManager:
    """
    Класс для управления портфелем активов.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "PortfolioManager":
        """
        Получает экземпляр класса PortfolioManager (Singleton).

        Returns:
            Экземпляр класса PortfolioManager
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """
        Инициализирует менеджер портфеля.
        """
        self.config = get_config()
        self.assets: Dict[str, Asset] = {}  # ключ: "{exchange_id}:{symbol}"
        self.total_value = 0.0
        self.var_calculator = VarCalculator()
        self.position_sizer = PositionSizer()
        self.returns_history: Dict[str, pd.Series] = {}
        self.last_update_time = 0
        logger.debug("Создан экземпляр PortfolioManager")

    @handle_error
    def add_asset(
        self,
        symbol: str,
        exchange_id: str,
        amount: float,
        price: Optional[float] = None,
    ) -> bool:
        """
        Добавляет актив в портфель.

        Args:
            symbol: Символ актива
            exchange_id: Идентификатор биржи
            amount: Количество актива
            price: Цена актива (None для получения текущей цены)

        Returns:
            True в случае успеха, иначе False
        """
        if amount <= 0:
            logger.warning(f"Некорректное количество актива: {amount}")
            return False

        # Генерируем ключ актива
        asset_key = f"{exchange_id}:{symbol}"

        # Если цена не указана, пытаемся получить текущую
        current_price = price
        if current_price is None:
            try:
                ticker = self._fetch_ticker_sync(exchange_id, symbol)
                current_price = ticker.get("last", 0)
                if current_price <= 0:
                    logger.error(
                        f"Не удалось получить цену для {symbol} на {exchange_id}"
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"Ошибка при получении цены для {symbol} на {exchange_id}: {str(e)}"
                )
                return False

        # Рассчитываем стоимость актива
        value = amount * current_price

        # Добавляем актив в портфель
        self.assets[asset_key] = Asset(
            symbol=symbol,
            exchange_id=exchange_id,
            amount=amount,
            price=current_price,
            value=value,
            weight=0.0,  # Вес будет обновлен при пересчете портфеля
            last_update=int(time.time()),
        )

        # Обновляем общую стоимость и веса
        self._recalculate_portfolio()

        logger.info(
            f"Добавлен актив в портфель: {symbol} на {exchange_id}, "
            f"количество={amount}, цена={current_price}"
        )

        return True

    @handle_error
    def update_asset(
        self,
        symbol: str,
        exchange_id: str,
        amount: Optional[float] = None,
        price: Optional[float] = None,
    ) -> bool:
        """
        Обновляет информацию об активе в портфеле.

        Args:
            symbol: Символ актива
            exchange_id: Идентификатор биржи
            amount: Новое количество актива (None для сохранения текущего)
            price: Новая цена актива (None для получения текущей цены)

        Returns:
            True в случае успеха, иначе False
        """
        # Генерируем ключ актива
        asset_key = f"{exchange_id}:{symbol}"

        # Проверяем наличие актива в портфеле
        if asset_key not in self.assets:
            logger.warning(f"Актив {symbol} на {exchange_id} не найден в портфеле")
            return False

        # Получаем текущий актив
        asset = self.assets[asset_key]

        # Обновляем количество, если указано
        if amount is not None:
            if amount <= 0:
                # Удаляем актив из портфеля
                del self.assets[asset_key]
                logger.info(f"Актив {symbol} на {exchange_id} удален из портфеля")
                self._recalculate_portfolio()
                return True

            asset.amount = amount

        # Обновляем цену, если указана или если запрошено обновление
        update_price = price is not None
        current_price = price

        if update_price:
            if current_price is None:
                try:
                    ticker = self._fetch_ticker_sync(exchange_id, symbol)
                    current_price = ticker.get("last", 0)
                    if current_price <= 0:
                        logger.warning(
                            f"Не удалось получить цену для {symbol} на {exchange_id}"
                        )
                        current_price = asset.price  # Используем старую цену
                except Exception as e:
                    logger.error(
                        f"Ошибка при получении цены для {symbol} на {exchange_id}: {str(e)}"
                    )
                    current_price = asset.price  # Используем старую цену

            asset.price = current_price

        # Обновляем стоимость актива
        asset.value = asset.amount * asset.price

        # Обновляем время последнего обновления
        asset.last_update = int(time.time())

        # Обновляем общую стоимость и веса
        self._recalculate_portfolio()

        logger.debug(
            f"Обновлен актив в портфеле: {symbol} на {exchange_id}, "
            f"количество={asset.amount}, цена={asset.price}"
        )

        return True

    @handle_error
    def remove_asset(self, symbol: str, exchange_id: str) -> bool:
        """
        Удаляет актив из портфеля.

        Args:
            symbol: Символ актива
            exchange_id: Идентификатор биржи

        Returns:
            True в случае успеха, иначе False
        """
        # Генерируем ключ актива
        asset_key = f"{exchange_id}:{symbol}"

        # Проверяем наличие актива в портфеле
        if asset_key not in self.assets:
            logger.warning(f"Актив {symbol} на {exchange_id} не найден в портфеле")
            return False

        # Удаляем актив из портфеля
        del self.assets[asset_key]

        # Обновляем общую стоимость и веса
        self._recalculate_portfolio()

        logger.info(f"Актив {symbol} на {exchange_id} удален из портфеля")

        return True

    @handle_error
    def get_asset(self, symbol: str, exchange_id: str) -> Optional[Asset]:
        """
        Получает информацию об активе из портфеля.

        Args:
            symbol: Символ актива
            exchange_id: Идентификатор биржи

        Returns:
            Информация об активе или None, если актив не найден
        """
        # Генерируем ключ актива
        asset_key = f"{exchange_id}:{symbol}"

        # Возвращаем актив, если найден
        return self.assets.get(asset_key)

    @handle_error
    def get_all_assets(self) -> List[Asset]:
        """
        Получает список всех активов в портфеле.

        Returns:
            Список активов
        """
        return list(self.assets.values())

    @handle_error
    def get_portfolio_value(self) -> float:
        """
        Получает общую стоимость портфеля.

        Returns:
            Стоимость портфеля
        """
        return self.total_value

    @handle_error
    def get_asset_weight(self, symbol: str, exchange_id: str) -> float:
        """
        Получает вес актива в портфеле.

        Args:
            symbol: Символ актива
            exchange_id: Идентификатор биржи

        Returns:
            Вес актива (0.0, если актив не найден)
        """
        # Генерируем ключ актива
        asset_key = f"{exchange_id}:{symbol}"

        # Возвращаем вес актива, если найден
        if asset_key in self.assets:
            return self.assets[asset_key].weight

        return 0.0

    @async_handle_error
    async def update_portfolio_from_exchange(
        self, exchange_id: str, quote_currency: str = "USDT"
    ) -> bool:
        """
        Обновляет портфель на основе данных с биржи.

        Args:
            exchange_id: Идентификатор биржи
            quote_currency: Базовая валюта для оценки стоимости

        Returns:
            True в случае успеха, иначе False
        """
        try:
            # Получаем баланс с биржи
            balance = await fetch_balance(exchange_id)
            if not balance:
                logger.error(f"Не удалось получить баланс на {exchange_id}")
                return False

            # Получаем список активов с ненулевым балансом
            assets_to_add = []
            for currency, details in balance.get("total", {}).items():
                if details > 0:
                    # Пропускаем стейблкоины и фиатные валюты (считаем их как базовую валюту)
                    if currency in ["USDT", "USDC", "BUSD", "DAI", "USD", "EUR"]:
                        # Добавляем как базовую валюту
                        if currency == quote_currency:
                            assets_to_add.append((currency, currency, details, 1.0))
                        else:
                            # Получаем курс обмена для оценки стоимости
                            try:
                                ticker = await fetch_ticker(
                                    exchange_id, f"{currency}/{quote_currency}"
                                )
                                price = ticker.get("last", 0)
                                if price > 0:
                                    assets_to_add.append(
                                        (currency, quote_currency, details, price)
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Не удалось получить курс {currency}/{quote_currency}: {str(e)}"
                                )
                    else:
                        # Формируем символ для получения цены
                        symbol = f"{currency}/{quote_currency}"
                        try:
                            ticker = await fetch_ticker(exchange_id, symbol)
                            price = ticker.get("last", 0)
                            if price > 0:
                                assets_to_add.append(
                                    (currency, quote_currency, details, price)
                                )
                        except Exception as e:
                            logger.warning(
                                f"Не удалось получить цену для {symbol}: {str(e)}"
                            )

            # Удаляем все активы этой биржи из портфеля
            for asset_key in list(self.assets.keys()):
                if asset_key.startswith(f"{exchange_id}:"):
                    del self.assets[asset_key]

            # Добавляем новые активы
            for currency, quote, amount, price in assets_to_add:
                symbol = f"{currency}/{quote}" if currency != quote else currency
                self.add_asset(symbol, exchange_id, amount, price)

            logger.info(
                f"Портфель обновлен на основе данных с {exchange_id}, "
                f"добавлено {len(assets_to_add)} активов"
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка при обновлении портфеля с {exchange_id}: {str(e)}")
            return False

    @handle_error
    def calculate_portfolio_var(
        self, confidence_level: float = 0.95, time_horizon: int = 1
    ) -> float:
        """
        Рассчитывает показатель Value at Risk (VaR) для портфеля.

        Args:
            confidence_level: Уровень доверия
            time_horizon: Горизонт времени в днях

        Returns:
            Значение VaR в абсолютных единицах
        """
        if not self.assets:
            logger.warning("Портфель пуст, невозможно рассчитать VaR")
            return 0.0

        # Настраиваем калькулятор VaR
        self.var_calculator.confidence_level = confidence_level
        self.var_calculator.time_horizon = time_horizon

        # Готовим данные для расчета
        asset_returns = {}
        asset_weights = {}

        for asset_key, asset in self.assets.items():
            if asset_key in self.returns_history:
                asset_returns[asset_key] = self.returns_history[asset_key]
                asset_weights[asset_key] = asset.weight

        # Если нет истории доходностей, используем параметрический VaR для каждого актива отдельно
        if not asset_returns:
            logger.warning("Нет истории доходностей для расчета VaR портфеля")
            total_var = 0.0
            for asset_key, asset in self.assets.items():
                # Предполагаем волатильность 2% в день для всех активов
                daily_volatility = 0.02
                z_score = 1.645  # для 95% доверительного интервала
                asset_var = (
                    asset.value * daily_volatility * z_score * np.sqrt(time_horizon)
                )
                total_var += asset_var

            # Предполагаем, что активы не полностью коррелированы (используем множитель 0.8)
            portfolio_var = total_var * 0.8

            logger.debug(f"Рассчитан приблизительный VaR портфеля: {portfolio_var}")
            return portfolio_var

        # Рассчитываем VaR портфеля с учетом корреляций
        portfolio_var = self.var_calculator.calculate_portfolio_var(
            asset_returns, asset_weights, self.total_value
        )

        logger.debug(f"Рассчитан VaR портфеля: {portfolio_var}")
        return portfolio_var

    @handle_error
    def calculate_risk_contributions(self) -> Dict[str, float]:
        """
        Рассчитывает вклад каждого актива в общий риск портфеля.

        Returns:
            Словарь с вкладом каждого актива в риск
        """
        if not self.assets:
            logger.warning("Портфель пуст, невозможно рассчитать вклады в риск")
            return {}

        # Готовим данные для расчета
        asset_returns = {}
        asset_weights = {}

        for asset_key, asset in self.assets.items():
            if asset_key in self.returns_history:
                asset_returns[asset_key] = self.returns_history[asset_key]
                asset_weights[asset_key] = asset.weight

        # Если нет истории доходностей, возвращаем приблизительные значения
        if not asset_returns:
            logger.warning("Нет истории доходностей для расчета вкладов в риск")
            risk_contributions = {}
            for asset_key, asset in self.assets.items():
                risk_contributions[asset_key] = asset.weight

            return risk_contributions

        # Рассчитываем вклады в VaR
        risk_contributions = self.var_calculator.calculate_var_contribution(
            asset_returns, asset_weights, self.total_value
        )

        logger.debug(f"Рассчитаны вклады в риск портфеля: {risk_contributions}")
        return risk_contributions

    @handle_error
    def optimize_portfolio(
        self, target_risk: Optional[float] = None, target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Оптимизирует портфель для достижения целевого уровня риска или доходности.

        Args:
            target_risk: Целевой уровень риска (None для минимизации риска)
            target_return: Целевой уровень доходности (None для максимизации доходности)

        Returns:
            Словарь с оптимальными весами активов
        """
        if not self.assets:
            logger.warning("Портфель пуст, невозможно оптимизировать")
            return {}

        # Готовим данные для оптимизации
        asset_returns = {}
        asset_weights = {}

        for asset_key, asset in self.assets.items():
            if asset_key in self.returns_history:
                asset_returns[asset_key] = self.returns_history[asset_key]
                asset_weights[asset_key] = asset.weight

        # Если нет истории доходностей, используем текущие веса
        if not asset_returns:
            logger.warning("Нет истории доходностей для оптимизации портфеля")
            return asset_weights

        try:
            # Для простоты используем предустановленные веса для разных сценариев
            # В реальном приложении здесь должен быть алгоритм оптимизации

            if target_risk is not None:
                # Оптимизация для целевого уровня риска
                if target_risk < 0.01:
                    # Консервативный портфель
                    weights = self._generate_conservative_weights()
                elif target_risk < 0.05:
                    # Умеренный портфель
                    weights = self._generate_moderate_weights()
                else:
                    # Агрессивный портфель
                    weights = self._generate_aggressive_weights()
            elif target_return is not None:
                # Оптимизация для целевого уровня доходности
                if target_return < 0.05:
                    # Консервативный портфель
                    weights = self._generate_conservative_weights()
                elif target_return < 0.10:
                    # Умеренный портфель
                    weights = self._generate_moderate_weights()
                else:
                    # Агрессивный портфель
                    weights = self._generate_aggressive_weights()
            else:
                # Минимизация риска
                weights = self._generate_conservative_weights()

            logger.info(
                f"Портфель оптимизирован: target_risk={target_risk}, target_return={target_return}"
            )
            return weights

        except Exception as e:
            logger.error(f"Ошибка при оптимизации портфеля: {str(e)}")
            return asset_weights

    @handle_error
    def update_returns_history(self, asset_returns: Dict[str, pd.Series]) -> None:
        """
        Обновляет историю доходностей активов.

        Args:
            asset_returns: Словарь с сериями исторических доходностей для каждого актива
        """
        self.returns_history.update(asset_returns)
        logger.debug(f"Обновлена история доходностей для {len(asset_returns)} активов")

    def _recalculate_portfolio(self) -> None:
        """
        Пересчитывает общую стоимость портфеля и веса активов.
        """
        # Рассчитываем общую стоимость
        self.total_value = sum(asset.value for asset in self.assets.values())

        # Обновляем веса активов
        if self.total_value > 0:
            for asset in self.assets.values():
                asset.weight = asset.value / self.total_value
        else:
            for asset in self.assets.values():
                asset.weight = 0.0

        # Обновляем время последнего обновления
        self.last_update_time = int(time.time())

        # Fix line 90 - likely a syntax error in portfolio rebalancing method
        self.apply_new_allocation(new_allocation)  # Removed 'await' since this is not an async function

    def _fetch_ticker_sync(self, exchange_id: str, symbol: str) -> Dict[str, Any]:
        """
        Синхронно получает тикер с биржи (для внутреннего использования).

        Args:
            exchange_id: Идентификатор биржи
            symbol: Символ актива

        Returns:
            Данные тикера
        """
        import ccxt

        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Биржа {exchange_id} не поддерживается CCXT")

        exchange = exchange_class()
        return exchange.fetch_ticker(symbol)

    def _generate_conservative_weights(self) -> Dict[str, float]:
        """
        Генерирует консервативные веса для портфеля.

        Returns:
            Словарь с весами активов
        """
        weights = {}

        # Классифицируем активы по типу
        stablecoins = []
        large_caps = []
        mid_caps = []
        small_caps = []

        for asset_key, asset in self.assets.items():
            symbol = asset.symbol.split("/")[0]
            if symbol in ["USDT", "USDC", "BUSD", "DAI"]:
                stablecoins.append(asset_key)
            elif symbol in ["BTC", "ETH"]:
                large_caps.append(asset_key)
            elif symbol in ["BNB", "XRP", "ADA", "SOL", "DOT"]:
                mid_caps.append(asset_key)
            else:
                small_caps.append(asset_key)

        # Распределяем веса в соответствии с консервативной стратегией
        # 50% стейблкоины, 30% крупные монеты, 15% средние, 5% мелкие

        # Стейблкоины
        if stablecoins:
            stablecoin_weight = 0.5 / len(stablecoins)
            for asset_key in stablecoins:
                weights[asset_key] = stablecoin_weight

        # Крупные монеты
        if large_caps:
            large_cap_weight = 0.3 / len(large_caps)
            for asset_key in large_caps:
                weights[asset_key] = large_cap_weight

        # Средние монеты
        if mid_caps:
            mid_cap_weight = 0.15 / len(mid_caps)
            for asset_key in mid_caps:
                weights[asset_key] = mid_cap_weight

        # Мелкие монеты
        if small_caps:
            small_cap_weight = 0.05 / len(small_caps)
            for asset_key in small_caps:
                weights[asset_key] = small_cap_weight

        # Если какая-то категория пуста, перераспределяем веса
        total_weight = sum(weights.values())
        if total_weight < 1.0:
            # Нормализуем веса
            for asset_key in weights:
                weights[asset_key] /= total_weight

        return weights

    def _generate_moderate_weights(self) -> Dict[str, float]:
        """
        Генерирует умеренные веса для портфеля.

        Returns:
            Словарь с весами активов
        """
        weights = {}

        # Классифицируем активы по типу
        stablecoins = []
        large_caps = []
        mid_caps = []
        small_caps = []

        for asset_key, asset in self.assets.items():
            symbol = asset.symbol.split("/")[0]
            if symbol in ["USDT", "USDC", "BUSD", "DAI"]:
                stablecoins.append(asset_key)
            elif symbol in ["BTC", "ETH"]:
                large_caps.append(asset_key)
            elif symbol in ["BNB", "XRP", "ADA", "SOL", "DOT"]:
                mid_caps.append(asset_key)
            else:
                small_caps.append(asset_key)

        # Распределяем веса в соответствии с умеренной стратегией
        # 30% стейблкоины, 40% крупные монеты, 20% средние, 10% мелкие

        # Стейблкоины
        if stablecoins:
            stablecoin_weight = 0.3 / len(stablecoins)
            for asset_key in stablecoins:
                weights[asset_key] = stablecoin_weight

        # Крупные монеты
        if large_caps:
            large_cap_weight = 0.4 / len(large_caps)
            for asset_key in large_caps:
                weights[asset_key] = large_cap_weight

        # Средние монеты
        if mid_caps:
            mid_cap_weight = 0.2 / len(mid_caps)
            for asset_key in mid_caps:
                weights[asset_key] = mid_cap_weight

        # Мелкие монеты
        if small_caps:
            small_cap_weight = 0.1 / len(small_caps)
            for asset_key in small_caps:
                weights[asset_key] = small_cap_weight

        # Если какая-то категория пуста, перераспределяем веса
        total_weight = sum(weights.values())
        if total_weight < 1.0:
            # Нормализуем веса
            for asset_key in weights:
                weights[asset_key] /= total_weight

        return weights

    def _generate_aggressive_weights(self) -> Dict[str, float]:
        """
        Генерирует агрессивные веса для портфеля.

        Returns:
            Словарь с весами активов
        """
        weights = {}

        # Классифицируем активы по типу
        stablecoins = []
        large_caps = []
        mid_caps = []
        small_caps = []

        for asset_key, asset in self.assets.items():
            symbol = asset.symbol.split("/")[0]
            if symbol in ["USDT", "USDC", "BUSD", "DAI"]:
                stablecoins.append(asset_key)
            elif symbol in ["BTC", "ETH"]:
                large_caps.append(asset_key)
            elif symbol in ["BNB", "XRP", "ADA", "SOL", "DOT"]:
                mid_caps.append(asset_key)
            else:
                small_caps.append(asset_key)

        # Распределяем веса в соответствии с агрессивной стратегией
        # 10% стейблкоины, 40% крупные монеты, 30% средние, 20% мелкие

        # Стейблкоины
        if stablecoins:
            stablecoin_weight = 0.1 / len(stablecoins)
            for asset_key in stablecoins:
                weights[asset_key] = stablecoin_weight

        # Крупные монеты
        if large_caps:
            large_cap_weight = 0.4 / len(large_caps)
            for asset_key in large_caps:
                weights[asset_key] = large_cap_weight

        # Средние монеты
        if mid_caps:
            mid_cap_weight = 0.3 / len(mid_caps)
            for asset_key in mid_caps:
                weights[asset_key] = mid_cap_weight

        # Мелкие монеты
        if small_caps:
            small_cap_weight = 0.2 / len(small_caps)
            for asset_key in small_caps:
                weights[asset_key] = small_cap_weight

        # Если какая-то категория пуста, перераспределяем веса
        total_weight = sum(weights.values())
        if total_weight < 1.0:
            # Нормализуем веса
            for asset_key in weights:
                weights[asset_key] /= total_weight

        return weights
