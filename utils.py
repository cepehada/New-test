import logging
import asyncio
import time
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)

class ArbitrageOpportunity:
    """Класс для представления и анализа арбитражной возможности."""
    
    def __init__(
        self, 
        buy_exchange: str, 
        sell_exchange: str, 
        symbol: str, 
        raw_spread: float,
        spread_percent: float,
        buy_price: float,
        sell_price: float,
        buy_volume: float,
        sell_volume: float,
        buy_fee_percent: float,
        sell_fee_percent: float
    ):
        self.buy_exchange = buy_exchange
        self.sell_exchange = sell_exchange
        self.symbol = symbol
        self.raw_spread = raw_spread
        self.spread_percent = spread_percent
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.buy_volume = buy_volume
        self.sell_volume = sell_volume
        self.buy_fee_percent = buy_fee_percent
        self.sell_fee_percent = sell_fee_percent
        self.timestamp = time.time()
        
        # Рассчитываем чистый спред с учетом комиссий
        self.net_spread = self._calculate_net_spread()
        
        # Максимально возможный объем для арбитража
        self.max_volume = min(buy_volume, sell_volume)
        
        # Ожидаемая прибыль с учетом комиссий
        self.expected_profit = self._calculate_expected_profit()
    
    def _calculate_net_spread(self) -> float:
        """Рассчитывает чистый спред с учетом комиссий на обеих биржах."""
        # Эффективная цена покупки с учетом комиссии
        effective_buy = self.buy_price * (1 + self.buy_fee_percent/100)
        # Эффективная цена продажи с учетом комиссии
        effective_sell = self.sell_price * (1 - self.sell_fee_percent/100)
        # Чистый спред в процентах
        return ((effective_sell / effective_buy) - 1) * 100
    
    def _calculate_expected_profit(self) -> float:
        """Рассчитывает ожидаемую прибыль от арбитража с учетом комиссий."""
        buy_cost = self.max_volume * self.buy_price
        buy_fee = buy_cost * (self.buy_fee_percent/100)
        sell_revenue = self.max_volume * self.sell_price
        sell_fee = sell_revenue * (self.sell_fee_percent/100)
        return sell_revenue - sell_fee - buy_cost - buy_fee
    
    def is_profitable(self, min_profit: float = 0.0) -> bool:
        """Проверяет, является ли возможность прибыльной с учетом минимального порога."""
        return self.net_spread > 0 and self.expected_profit > min_profit
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует объект в словарь для сериализации."""
        return {
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "symbol": self.symbol,
            "raw_spread": self.raw_spread,
            "spread_percent": self.spread_percent,
            "net_spread": self.net_spread,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "max_volume": self.max_volume,
            "expected_profit": self.expected_profit,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """Строковое представление арбитражной возможности."""
        return (
            f"{self.symbol}: Buy@{self.buy_exchange}({self.buy_price}), "
            f"Sell@{self.sell_exchange}({self.sell_price}), "
            f"Net: {self.net_spread:.2f}%, "
            f"Volume: {self.max_volume}, "
            f"Profit: {self.expected_profit:.6f}"
        )


async def fetch_orderbook_with_retry(
    exchange, exchange_id: str, symbol: str, max_retries: int = 3, retry_delay: float = 0.5
) -> Optional[Dict]:
    """
    Получает стакан с поддержкой повторных попыток при ошибках.
    
    Args:
        exchange: Экземпляр биржи CCXT
        exchange_id: Идентификатор биржи для логирования
        symbol: Торговая пара
        max_retries: Максимальное количество попыток
        retry_delay: Задержка между попытками в секундах
        
    Returns:
        Dictionary с данными стакана или None в случае ошибки
    """
    for attempt in range(max_retries):
        try:
            return await exchange.fetch_order_book(symbol)
        except Exception as e:
            err_msg = f"Ошибка получения стакана {symbol} на {exchange_id} (попытка {attempt+1}/{max_retries}): {e}"
            if attempt < max_retries - 1:
                logger.warning(err_msg)
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Экспоненциальная задержка
            else:
                logger.error(err_msg)
    return None


async def get_exchange_fees(
    exchange, exchange_id: str, symbol: str, fee_cache: Dict = None
) -> Tuple[float, float]:
    """
    Получает комиссии для указанной биржи и символа.
    
    Args:
        exchange: Экземпляр биржи CCXT
        exchange_id: Идентификатор биржи
        symbol: Торговая пара
        fee_cache: Опциональный кэш комиссий
        
    Returns:
        Tuple[float, float]: Комиссии (maker, taker) в процентах
    """
    # Используем кэш, если доступен
    if fee_cache is not None and exchange_id in fee_cache and symbol in fee_cache[exchange_id]:
        return fee_cache[exchange_id][symbol]
    
    try:
        # Пытаемся получить комиссии из рыночных данных
        markets = await exchange.fetch_markets()
        market = next((m for m in markets if m['symbol'] == symbol), None)
        
        if market and 'fee' in market:
            maker = market['fee'].get('maker', 0.1) * 100  # Конвертируем в проценты
            taker = market['fee'].get('taker', 0.1) * 100
        else:
            # Пробуем альтернативный метод
            try:
                trading_fees = await exchange.fetch_trading_fees()
                if symbol in trading_fees:
                    maker = trading_fees[symbol].get('maker', 0.1) * 100
                    taker = trading_fees[symbol].get('taker', 0.1) * 100
                else:
                    maker = taker = 0.1  # Значения по умолчанию - 0.1%
            except Exception:
                maker = taker = 0.1
        
        # Сохраняем в кэш
        if fee_cache is not None:
            if exchange_id not in fee_cache:
                fee_cache[exchange_id] = {}
            fee_cache[exchange_id][symbol] = (maker, taker)
        
        return maker, taker
    
    except Exception as e:
        logger.error(f"Ошибка получения комиссий для {exchange_id}/{symbol}: {e}")
        return 0.1, 0.1  # Значения по умолчанию в случае ошибки


async def arbitrage_spread_scanner(
    exchanges: Dict[str, object], 
    symbol: str, 
    min_spread: float = 0.0,
    min_volume: float = 0.0,
    depth: int = 3,
    check_fees: bool = True,
    fee_cache: Optional[Dict] = None,
    min_profit: float = 0.0
) -> List[ArbitrageOpportunity]:
    """
    Асинхронно сканирует несколько бирж для поиска арбитражных возможностей.

    Args:
        exchanges: Словарь бирж в формате {'id': ccxt_exchange_instance}
        symbol: Торговая пара (например, "BTC/USDT")
        min_spread: Минимальный сырой спред в процентах для генерации сигнала
        min_volume: Минимальный объем для рассмотрения
        depth: Глубина проверки стакана (уровни цен)
        check_fees: Учитывать ли комиссии при расчете
        fee_cache: Кэш для хранения комиссий по биржам
        min_profit: Минимальная ожидаемая прибыль для фильтрации

    Returns:
        List[ArbitrageOpportunity]: Список объектов арбитражных возможностей
    """
    start_time = time.time()
    opportunities = []
    orderbooks = {}
    
    # Инициализируем кэш комиссий, если не предоставлен
    if check_fees and fee_cache is None:
        fee_cache = {}
    
    # Асинхронно получаем данные о комиссиях, если требуется
    fee_tasks = {}
    if check_fees:
        for exchange_id, exchange in exchanges.items():
            fee_tasks[exchange_id] = asyncio.create_task(
                get_exchange_fees(exchange, exchange_id, symbol, fee_cache)
            )
    
    # Асинхронно получаем стаканы со всех бирж
    orderbook_tasks = {}
    for exchange_id, exchange in exchanges.items():
        orderbook_tasks[exchange_id] = asyncio.create_task(
            fetch_orderbook_with_retry(exchange, exchange_id, symbol)
        )
    
    # Ожидаем завершения получения всех стаканов
    for exchange_id, task in orderbook_tasks.items():
        try:
            ob = await task
            if ob and 'bids' in ob and 'asks' in ob:
                orderbooks[exchange_id] = ob
        except Exception as e:
            logger.error(f"Ошибка при получении стакана {symbol} на {exchange_id}: {e}")
    
    # Получаем результаты задач с комиссиями
    fees = {}
    if check_fees:
        for exchange_id, task in fee_tasks.items():
            try:
                maker_fee, taker_fee = await task
                fees[exchange_id] = (maker_fee, taker_fee)
            except Exception as e:
                logger.error(f"Ошибка при получении комиссий для {exchange_id}: {e}")
                fees[exchange_id] = (0.1, 0.1)  # Значения по умолчанию
    else:
        # Устанавливаем нулевые комиссии, если их не нужно учитывать
        fees = {exchange_id: (0.0, 0.0) for exchange_id in exchanges}
    
    if len(orderbooks) < 2:
        logger.warning(f"Недостаточно данных стаканов для {symbol} (получено {len(orderbooks)} из {len(exchanges)})")
        return []
    
    exchange_ids = list(orderbooks.keys())
    
    # Анализируем арбитражные возможности по всем уровням стакана до указанной глубины
    for i in range(len(exchange_ids)):
        for j in range(i + 1, len(exchange_ids)):
            exch1, exch2 = exchange_ids[i], exchange_ids[j]
            
            try:
                # Проверяем каждый уровень стакана до указанной глубины
                for d in range(min(depth, len(orderbooks[exch1]["bids"]), len(orderbooks[exch2]["asks"]))):
                    # Проверяем возможность: покупка на exch2, продажа на exch1
                    bid1 = orderbooks[exch1]["bids"][d][0]  # Цена покупки на бирже 1
                    bid1_volume = orderbooks[exch1]["bids"][d][1]  # Объем на бирже 1
                    
                    ask2 = orderbooks[exch2]["asks"][d][0]  # Цена продажи на бирже 2
                    ask2_volume = orderbooks[exch2]["asks"][d][1]  # Объем на бирже 2
                    
                    # Рассчитываем спред и процент спреда
                    raw_spread1 = bid1 - ask2
                    spread_percent1 = (bid1 / ask2 - 1) * 100
                    
                    if spread_percent1 > min_spread and bid1_volume >= min_volume and ask2_volume >= min_volume:
                        # Получаем комиссии для обеих бирж
                        _, buy_fee = fees.get(exch2, (0.1, 0.1))  # Используем taker fee для покупки
                        maker_fee, _ = fees.get(exch1, (0.1, 0.1))  # Используем maker fee для продажи
                        
                        # Создаем объект арбитражной возможности
                        opportunity = ArbitrageOpportunity(
                            buy_exchange=exch2,
                            sell_exchange=exch1,
                            symbol=symbol,
                            raw_spread=raw_spread1,
                            spread_percent=spread_percent1,
                            buy_price=ask2,
                            sell_price=bid1,
                            buy_volume=ask2_volume,
                            sell_volume=bid1_volume,
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=maker_fee
                        )
                        
                        # Добавляем возможность, если она прибыльна с учетом комиссий
                        if opportunity.is_profitable(min_profit):
                            opportunities.append(opportunity)
                
                # Проверяем обратное направление: покупка на exch1, продажа на exch2
                for d in range(min(depth, len(orderbooks[exch2]["bids"]), len(orderbooks[exch1]["asks"]))):
                    bid2 = orderbooks[exch2]["bids"][d][0]
                    bid2_volume = orderbooks[exch2]["bids"][d][1]
                    
                    ask1 = orderbooks[exch1]["asks"][d][0]
                    ask1_volume = orderbooks[exch1]["asks"][d][1]
                    
                    raw_spread2 = bid2 - ask1
                    spread_percent2 = (bid2 / ask1 - 1) * 100
                    
                    if spread_percent2 > min_spread and bid2_volume >= min_volume and ask1_volume >= min_volume:
                        # Получаем комиссии для обеих бирж
                        _, buy_fee = fees.get(exch1, (0.1, 0.1))  # Taker fee для покупки
                        maker_fee, _ = fees.get(exch2, (0.1, 0.1))  # Maker fee для продажи
                        
                        opportunity = ArbitrageOpportunity(
                            buy_exchange=exch1,
                            sell_exchange=exch2,
                            symbol=symbol,
                            raw_spread=raw_spread2,
                            spread_percent=spread_percent2,
                            buy_price=ask1,
                            sell_price=bid2,
                            buy_volume=ask1_volume,
                            sell_volume=bid2_volume,
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=maker_fee
                        )
                        
                        if opportunity.is_profitable(min_profit):
                            opportunities.append(opportunity)
            
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Ошибка расчёта спреда между {exch1} и {exch2}: {e}")
    
    # Сортируем возможности по ожидаемой прибыли (от большей к меньшей)
    opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
    
    execution_time = time.time() - start_time
    logger.debug(f"Сканирование {symbol} завершено за {execution_time:.3f}с. Найдено возможностей: {len(opportunities)}")
    
    return opportunities


async def scan_multiple_symbols(
    exchanges: Dict[str, object],
    symbols: List[str],
    min_spread: float = 0.0,
    min_volume: float = 0.0,
    check_fees: bool = True,
    min_profit: float = 0.0
) -> Dict[str, List[ArbitrageOpportunity]]:
    """
    Сканирует несколько символов параллельно.
    
    Args:
        exchanges: Словарь бирж
        symbols: Список символов для сканирования
        min_spread: Минимальный спред
        min_volume: Минимальный объем
        check_fees: Учитывать ли комиссии
        min_profit: Минимальная прибыль
        
    Returns:
        Dict[str, List[ArbitrageOpportunity]]: Словарь возможностей по символам
    """
    start_time = time.time()
    
    # Создаем общий кэш комиссий для всех вызовов
    fee_cache = {}
    
    # Создаем задачи для каждого символа
    tasks = []
    for symbol in symbols:
        tasks.append(
            arbitrage_spread_scanner(
                exchanges=exchanges,
                symbol=symbol,
                min_spread=min_spread,
                min_volume=min_volume,
                check_fees=check_fees,
                fee_cache=fee_cache,
                min_profit=min_profit
            )
        )
    
    # Запускаем все задачи параллельно
    results = await asyncio.gather(*tasks)
    
    # Формируем результат
    opportunities_by_symbol = {}
    for i, symbol in enumerate(symbols):
        if results[i]:  # Если найдены возможности
            opportunities_by_symbol[symbol] = results[i]
    
    execution_time = time.time() - start_time
    total_opps = sum(len(opps) for opps in opportunities_by_symbol.values())
    
    logger.info(
        f"Сканирование {len(symbols)} символов завершено за {execution_time:.3f}с. "
        f"Найдено {total_opps} возможностей в {len(opportunities_by_symbol)} символах."
    )
    
    return opportunities_by_symbol


async def execute_arbitrage(
    buy_exchange: object,
    sell_exchange: object,
    opportunity: ArbitrageOpportunity,
    trade_volume: Optional[float] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Исполняет арбитражную возможность.
    
    Args:
        buy_exchange: CCXT-экземпляр биржи для покупки
        sell_exchange: CCXT-экземпляр биржи для продажи
        opportunity: Объект арбитражной возможности
        trade_volume: Объем для торговли (если None, используется max_volume)
        dry_run: Если True, только симулирует выполнение
        
    Returns:
        Dict[str, Any]: Результат исполнения
    """
    if trade_volume is None:
        # Используем максимально доступный объем, но не больше 0.01 BTC (для безопасности)
        trade_volume = min(opportunity.max_volume, 0.01)
    else:
        # Ограничиваем запрошенный объем доступным объемом
        trade_volume = min(trade_volume, opportunity.max_volume)
    
    logger.info(
        f"{'Симуляция' if dry_run else 'Исполнение'} арбитража: {opportunity.symbol}, "
        f"Покупка: {opportunity.buy_exchange}@{opportunity.buy_price}, "
        f"Продажа: {opportunity.sell_exchange}@{opportunity.sell_price}, "
        f"Объем: {trade_volume}, Ожидаемая прибыль: {opportunity.expected_profit:.6f}"
    )
    
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "opportunity": opportunity.to_dict(),
            "trade_volume": trade_volume,
            "timestamp": time.time()
        }
    
    try:
        # Округляем объем с учетом точности биржи
        try:
            # Для получения точности символа
            markets = await buy_exchange.fetch_markets()
            market = next((m for m in markets if m['symbol'] == opportunity.symbol), None)
            amount_precision = market.get('precision', {}).get('amount', 8) if market else 8
            
            # Округляем немного меньше для избежания ошибок по объему
            rounded_volume = round(trade_volume - 1e-10, amount_precision)
            trade_volume = float(Decimal(str(rounded_volume)).quantize(
                Decimal('0.' + '0' * amount_precision), rounding=ROUND_DOWN))
        except Exception as e:
            logger.warning(f"Ошибка при получении точности: {e}. Округляем до 8 знаков.")
            trade_volume = float(Decimal(str(trade_volume)).quantize(
                Decimal('0.00000001'), rounding=ROUND_DOWN))
        
        # Выполняем сделки параллельно
        start_time = time.time()
        
        # Создаем задачи для обеих сделок
        buy_task = asyncio.create_task(
            buy_exchange.create_market_buy_order(opportunity.symbol, trade_volume)
        )
        sell_task = asyncio.create_task(
            sell_exchange.create_market_sell_order(opportunity.symbol, trade_volume)
        )
        
        # Ждем выполнения обеих сделок
        buy_order, sell_order = await asyncio.gather(buy_task, sell_task)
        
        execution_time = time.time() - start_time
        
        # Расчет фактической прибыли
        buy_cost = float(buy_order.get('cost', trade_volume * opportunity.buy_price))
        sell_revenue = float(sell_order.get('cost', trade_volume * opportunity.sell_price))
        actual_profit = sell_revenue - buy_cost
        
        return {
            "success": True,
            "dry_run": False,
            "opportunity": opportunity.to_dict(),
            "trade_volume": trade_volume,
            "buy_order": buy_order,
            "sell_order": sell_order,
            "actual_profit": actual_profit,
            "execution_time": execution_time,
            "timestamp": time.time()
        }
    
    except Exception as e:
        logger.error(f"Ошибка при исполнении арбитража: {e}")
        return {
            "success": False,
            "dry_run": False,
            "opportunity": opportunity.to_dict(),
            "error": str(e),
            "timestamp": time.time()
        }


# Пример использования:
async def main():
    # Код инициализации бирж здесь
    exchanges = {}
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    # Сканируем возможности для всех символов
    all_opportunities = await scan_multiple_symbols(
        exchanges=exchanges,
        symbols=symbols,
        min_spread=0.2,  # Минимальный спред 0.2%
        min_volume=0.001,  # Минимальный объем 0.001 BTC/ETH/SOL
        check_fees=True,
        min_profit=1.0  # Минимальная прибыль $1.0
    )
    
    # Выводим найденные возможности
    for symbol, opportunities in all_opportunities.items():
        print(f"\nНайдено {len(opportunities)} возможностей для {symbol}:")
        for idx, opp in enumerate(opportunities[:3]):  # Показываем топ-3
            print(f"{idx+1}. {opp}")
    
    # Исполняем лучшую возможность, если есть
    best_symbol = None
    best_opportunity = None
    
    for symbol, opportunities in all_opportunities.items():
        if opportunities and (best_opportunity is None or 
                              opportunities[0].expected_profit > best_opportunity.expected_profit):
            best_symbol = symbol
            best_opportunity = opportunities[0]
    
    if best_opportunity:
        print(f"\nИсполнение лучшей возможности:")
        print(best_opportunity)
        
        # Получаем объекты бирж
        buy_exchange = exchanges.get(best_opportunity.buy_exchange)
        sell_exchange = exchanges.get(best_opportunity.sell_exchange)
        
        # Выполняем в режиме симуляции
        result = await execute_arbitrage(
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            opportunity=best_opportunity,
            trade_volume=None,  # Используем максимально доступный объем
            dry_run=True  # Только симуляция
        )
        
        print(f"Результат: {'Успешно' if result['success'] else 'Ошибка'}")
    else:
        print("\nНе найдено прибыльных возможностей")


if __name__ == "__main__":
    # Для ручного запуска и тестирования
    asyncio.run(main())