"""
Модуль ccxt_exchanges.
Инициализирует биржи через CCXT-async с обработкой
рейтов, ретраев и логированием.
"""

import ccxt.async_support as ccxt
import asyncio
import logging
from project.config import load_config
config = load_config()

logger = logging.getLogger("ExchangeManager")

class ExchangeManager:
    """
    Класс для управления биржами через CCXT.
    
    Инициализирует биржи на основе настроек и обеспечивает безопасные
    вызовы API с ретраями.
    """

    def __init__(self, conf) -> None:
        self.config = conf
        self.exchanges = {}
        self._init_exchanges()

    def _init_exchanges(self) -> None:
        """
        Инициализирует биржи согласно конфигурации.
        Для "htx" используется ccxt.huobi.
        """
        api_conf = self.config.api.dict()
        for exch_id, params in api_conf.items():
            exch_lower = exch_id.lower()
            if exch_lower == "bybit":
                self.exchanges["bybit"] = ccxt.bybit(params)
            elif exch_lower == "mexc":
                self.exchanges["mexc"] = ccxt.mexc(params)
            elif exch_lower == "phemex":
                self.exchanges["phemex"] = ccxt.phemex(params)
            elif exch_lower == "htx":
                self.exchanges["htx"] = ccxt.huobi(params)
            else:
                logger.warning(f"Exchange {exch_id} не поддерживается")

    async def _safe_request(self, method, *args, retries=3, **kwargs):
        """
        Выполняет вызов API с ретраями.
        
        Args:
            method: Функция API биржи.
            args, kwargs: Аргументы вызова.
        
        Returns:
            Результат вызова метода.
        """
        for attempt in range(retries):
            try:
                return await method(*args, **kwargs)
            except Exception as e:
                logger.error(f"Ошибка запроса (попытка {attempt+1}): {e}")
                await asyncio.sleep(1)
        raise Exception("Превышено число попыток запроса")

    def get_exchange(self, exch_id: str):
        """
        Возвращает объект биржи по идентификатору.
        
        Args:
            exch_id (str): Идентификатор биржи.
        
        Returns:
            Объект биржи или None.
        """
        return self.exchanges.get(exch_id)

    async def multi_exchange_balance_sync(self) -> dict:
        """
        Получает балансы со всех бирж.
        
        Returns:
            dict: Балансы для каждой биржи.
        """
        balances = {}
        tasks = []
        for exch_id, exch in self.exchanges.items():
            tasks.append(self._safe_request(exch.fetch_balance))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, exch_id in enumerate(self.exchanges.keys()):
            res = results[idx]
            if isinstance(res, Exception):
                logger.error(f"Ошибка баланса для {exch_id}: {res}")
                balances[exch_id] = {"error": str(res)}
            else:
                balances[exch_id] = res
        return balances

    async def verify_taker_maker_fees(self) -> dict:
        """
        Загружает рынки и возвращает информацию о комиссиях.
        
        Returns:
            dict: Комиссии по биржам.
        """
        fees = {}
        for exch_id, exch in self.exchanges.items():
            try:
                await exch.load_markets()
                fees[exch_id] = exch.fees
            except Exception as e:
                fees[exch_id] = {"error": str(e)}
        return fees

    async def close_all(self) -> None:
        """
        Закрывает сессии всех бирж.
        """
        tasks = [exch.close() for exch in self.exchanges.values()]
        await asyncio.gather(*tasks)
