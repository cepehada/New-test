"""
Утилиты для арбитражной торговли.
Предоставляет вспомогательные функции для работы с данными рынка и расчетов.
"""

import time
from typing import Any, Dict, List, Set

import pandas as pd
from project.utils.error_handler import handle_error
from project.utils.logging_utils import get_logger

logger = get_logger(__name__)


@handle_error
def calculate_triangular_path(
    price_matrix: pd.DataFrame, start_currency: str, path_length: int = 3
) -> Dict[str, Any]:
    """
    Рассчитывает наиболее выгодный треугольный путь для арбитража.

    Args:
        price_matrix: Матрица цен для всех пар валют
        start_currency: Начальная валюта
        path_length: Длина пути (по умолчанию 3 для треугольного арбитража)

    Returns:
        Словарь с информацией о найденном пути и коэффициенте прибыли
    """
    if price_matrix.empty:
        logger.warning("Пустая матрица цен")
        return {"path": [], "currencies": [], "rate": 0.0, "profit": 0.0}

    if start_currency not in price_matrix.index:
        logger.warning(f"Начальная валюта {start_currency} не найдена в матрице цен")
        return {"path": [], "currencies": [], "rate": 0.0, "profit": 0.0}

    # Инициализируем лучший путь
    best_path = []
    best_currencies = []
    best_rate = 0.0

    # Функция для рекурсивного поиска пути
    def find_path(
        current: str, path: List[str], visited: Set[str], rate: float, depth: int
    ):
        nonlocal best_path, best_currencies, best_rate

        # Если достигли нужной глубины и вернулись к начальной валюте
        if depth == path_length and current == start_currency:
            if rate > best_rate:
                best_path = path.copy()
                best_currencies = [p.split("/")[0] for p in path] + [start_currency]
                best_rate = rate
            return

        # Если превысили глубину
        if depth >= path_length:
            return

        # Перебираем все возможные следующие валюты
        for next_curr in price_matrix.columns:
            if next_curr in visited:
                continue

            # Получаем цену для перехода
            pair = f"{current}/{next_curr}"
            rev_pair = f"{next_curr}/{current}"

            price = None
            is_reverse = False

            if pair in price_matrix.index and price_matrix.at[pair, next_curr] > 0:
                price = price_matrix.at[pair, next_curr]
            elif (
                rev_pair in price_matrix.index
                and price_matrix.at[rev_pair, current] > 0
            ):
                price = 1 / price_matrix.at[rev_pair, current]
                is_reverse = True

            if price is not None:
                # Рассчитываем новую ставку
                new_rate = rate * price

                # Добавляем пару в путь
                path_pair = rev_pair if is_reverse else pair
                path.append(path_pair)
                visited.add(next_curr)

                # Рекурсивно ищем дальше
                find_path(next_curr, path, visited, new_rate, depth + 1)

                # Возвращаемся назад
                path.pop()
                visited.remove(next_curr)

    # Запускаем рекурсивный поиск
    find_path(start_currency, [], {start_currency}, 1.0, 0)

    # Рассчитываем прибыль
    profit = best_rate - 1.0 if best_rate > 0 else 0.0

    return {
        "path": best_path,
        "currencies": best_currencies,
        "rate": best_rate,
        "profit": profit,
    }


@handle_error
def adjust_for_fees(profit_rate: float, fee_rate: float, num_trades: int) -> float:
    """
    Корректирует коэффициент прибыли с учетом комиссий.

    Args:
        profit_rate: Исходный коэффициент прибыли
        fee_rate: Ставка комиссии за одну сделку
        num_trades: Количество сделок

    Returns:
        Скорректированный коэффициент прибыли
    """
    # Каждая сделка уменьшает результат на (1 - fee_rate)
    adjusted_rate = profit_rate * ((1 - fee_rate) ** num_trades)

    return adjusted_rate


@handle_error
def build_price_matrix(tickers: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Строит матрицу цен из словаря тикеров.

    Args:
        tickers: Словарь тикеров {symbol: {last: price, ...}, ...}

    Returns:
        DataFrame с матрицей цен
    """
    # Собираем все валюты
    currencies = set()

    for symbol in tickers:
        parts = symbol.split("/")
        if len(parts) == 2:
            currencies.add(parts[0])
            currencies.add(parts[1])

    # Создаем пустую матрицу
    currencies = sorted(list(currencies))
    price_matrix = pd.DataFrame(0.0, index=currencies, columns=currencies)

    # Заполняем матрицу ценами
    for symbol, ticker in tickers.items():
        parts = symbol.split("/")
        if len(parts) != 2:
            continue

        base, quote = parts
        price = ticker.get("last") or ticker.get("close")

        if price and price > 0:
            price_matrix.at[base, quote] = price
            price_matrix.at[quote, base] = 1.0 / price

    return price_matrix


@handle_error
def find_all_arbitrage_paths(
    price_matrix: pd.DataFrame, min_profit: float = 0.01, path_length: int = 3
) -> List[Dict[str, Any]]:
    """
    Находит все возможные пути для арбитража.

    Args:
        price_matrix: Матрица цен
        min_profit: Минимальная прибыль для учета пути
        path_length: Длина пути

    Returns:
        Список словарей с информацией о путях
    """
    if price_matrix.empty:
        return []

    # Ищем пути для каждой начальной валюты
    paths = []

    for start_currency in price_matrix.index:
        path_info = calculate_triangular_path(price_matrix, start_currency, path_length)

        if path_info["profit"] >= min_profit:
            path_info["start_currency"] = start_currency
            paths.append(path_info)

    # Сортируем по убыванию прибыли
    paths.sort(key=lambda x: x["profit"], reverse=True)

    return paths


@handle_error
def calculate_expected_profit(
    amount: float, path_info: Dict[str, Any], fee_rate: float
) -> Dict[str, float]:
    """
    Рассчитывает ожидаемую прибыль для заданной суммы.

    Args:
        amount: Начальная сумма
        path_info: Информация о пути арбитража
        fee_rate: Ставка комиссии

    Returns:
        Словарь с расчетами прибыли
    """
    # Рассчитываем чистую ставку с учетом комиссий
    num_trades = len(path_info["path"])
    net_rate = adjust_for_fees(path_info["rate"], fee_rate, num_trades)

    # Рассчитываем результат
    final_amount = amount * net_rate
    profit = final_amount - amount
    profit_pct = (net_rate - 1) * 100

    return {
        "initial_amount": amount,
        "final_amount": final_amount,
        "profit": profit,
        "profit_pct": profit_pct,
        "gross_rate": path_info["rate"],
        "net_rate": net_rate,
    }


@handle_error
def format_arbitrage_path(path_info: Dict[str, Any]) -> str:
    """
    Форматирует информацию о пути арбитража в человекочитаемый вид.

    Args:
        path_info: Информация о пути

    Returns:
        Строка с описанием пути
    """
    if not path_info or not path_info["path"]:
        return "Нет данных о пути"

    path_str = " -> ".join(path_info["currencies"])
    profit_str = f"{path_info['profit'] * 100:.2f}%"

    trades = []
    for pair in path_info["path"]:
        base, quote = pair.split("/")
        trades.append(f"Обмен {base} на {quote}")

    trades_str = "\n".join([f"{i+1}. {trade}" for i, trade in enumerate(trades)])

    return f"Путь: {path_str}\nПрибыль: {profit_str}\n\nСделки:\n{trades_str}"


@handle_error
def calculate_max_trade_size(
    balances: Dict[str, float],
    path_info: Dict[str, Any],
    exchange_info: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    """
    Рассчитывает максимальный размер сделки на основе балансов и ограничений биржи.

    Args:
        balances: Словарь с балансами валют
        path_info: Информация о пути арбитража
        exchange_info: Информация о ограничениях биржи

    Returns:
        Словарь с максимальными размерами для каждой сделки
    """
    if not path_info or not path_info["path"]:
        return {}

    start_currency = path_info["currencies"][0]
    available_balance = balances.get(start_currency, 0)

    if available_balance <= 0:
        return {}

    # Рассчитываем максимальные размеры для каждой сделки
    max_sizes = {}
    current_amount = available_balance

    for i, pair in enumerate(path_info["path"]):
        base, quote = pair.split("/")

        # Получаем ограничения для пары
        pair_limits = exchange_info.get(pair, {})
        min_amount = pair_limits.get("min_amount", 0)
        max_amount = pair_limits.get("max_amount", float("inf"))
        min_cost = pair_limits.get("min_cost", 0)

        # Ограничения размера
        if base == path_info["currencies"][i]:
            # Продаем базовую валюту
            max_trade = min(current_amount, max_amount)
            if max_trade < min_amount:
                return {}  # Недостаточно средств

            max_sizes[pair] = max_trade

            # Рассчитываем полученное количество котируемой валюты
            price = 1.0  # В реальном коде здесь была бы реальная цена
            current_amount = max_trade * price
        else:
            # Покупаем базовую валюту
            price = 1.0  # В реальном коде здесь была бы реальная цена
            max_quote = current_amount
            max_base = max_quote / price

            max_trade = min(max_base, max_amount)
            if max_trade < min_amount or max_trade * price < min_cost:
                return (
                    {}
                )  # Недостаточно средств или не соответствует минимальной стоимости

            max_sizes[pair] = max_trade
            current_amount = max_trade

    return max_sizes


async def calculate_max_trade_sizes(
    buy_exchange: str,
    sell_exchange: str,
    symbol: str,
    buy_price: float,
    sell_price: float,
    balances: Dict,
    min_trade_amount: float = 10.0
) -> Dict[str, float]:
    """
    Вычисляет максимально возможные размеры сделок для арбитража.
    
    Args:
        buy_exchange: Биржа для покупки
        sell_exchange: Биржа для продажи
        symbol: Символ торговой пары
        buy_price: Цена покупки
        sell_price: Цена продажи
        balances: Словарь с балансами на биржах
        min_trade_amount: Минимальная сумма сделки в USD
        
    Returns:
        Словарь с размерами сделок
    """
    try:
        # Разбиваем символ на базовую и котируемую валюты
        base, quote = symbol.split('/')
        
        # Получаем балансы
        buy_balance = balances.get(buy_exchange, {}).get('free', {})
        sell_balance = balances.get(sell_exchange, {}).get('free', {})
        
        # Проверяем доступность валют в балансах
        if quote not in buy_balance or base not in sell_balance:
            logger.debug(
                "Отсутствуют необходимые валюты в балансах: %s/%s", 
                base, quote
            )
            return {}
        
        # Расчет максимального размера сделки
        buy_quote_balance = buy_balance[quote]
        sell_base_balance = sell_balance[base]
        
        max_buy_amount = buy_quote_balance / buy_price
        max_sell_amount = sell_base_balance
        
        # Берем минимальное значение для максимального размера сделки
        max_amount = min(max_buy_amount, max_sell_amount)
        
        # Проверяем минимальную сумму сделки
        if max_amount * buy_price < min_trade_amount:
            return {}
        
        # Расчет комиссий и прибыли
        buy_fee = buy_price * max_amount * 0.001  # Примерная комиссия 0.1%
        sell_fee = sell_price * max_amount * 0.001  # Примерная комиссия 0.1%
        
        buy_total = buy_price * max_amount + buy_fee
        sell_total = sell_price * max_amount - sell_fee
        
        profit = sell_total - buy_total
        profit_percent = (profit / buy_total) * 100
        
        return {
            'buy_amount': max_amount,
            'sell_amount': max_amount,
            'buy_cost': buy_total,
            'sell_proceeds': sell_total,
            'profit': profit,
            'profit_percent': profit_percent
        }
    
    except Exception as e:
        logger.warning("Ошибка при расчете размеров сделок: %s", str(e))
        return {}
