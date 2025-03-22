"""
Adaptive Manager.
Получает реальные данные с бирж, динамически рассчитывает SL,
TP, комиссии, выбирает стратегию, проводит хеджирование и обновляет
список торговых пар.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from project.config import load_config
config = load_config()
from project.integrations.market_data import MarketData
from project.utils.ccxt_exchanges import ExchangeManager
from project.trade_executor.order_executor import (
    partial_close_trade, dynamic_stop_loss_by_atr
)
from project.trade_executor.risk_aware_executor import (
    execute_risk_aware_order
)
from project.technicals.enhanced_indicators import calculate_atr

logger = logging.getLogger("AdaptiveManager")

class AdaptiveManager:
    """
    Адаптивный менеджер, который:
      - Получает реальные данные с бирж.
      - Определяет оптимальные условия выхода.
      - Фиксирует прибыль частично.
      - Выбирает стратегию динамически.
      - Проводит динамическое хеджирование.
      - Рассчитывает актуальные комиссии.
      - Обновляет список торговых пар.
      - Вызывает функции исполнения ордеров.
    """

    def __init__(self) -> None:
        self.config = config
        self.market = MarketData()
        self.ex_manager = ExchangeManager(config)
        self.active_pairs: List[str] = config.symbols
        self.exit_thresh: float = 0.05
        self.slip_limit: float = 0.02

    async def get_market_metrics(
        self, symbol: str
    ) -> Dict[str, Any]:
        """
        Получает реальные тикеры и ордербуки для пары.
        
        Args:
            symbol (str): Торговая пара.
        
        Returns:
            Dict[str, Any]: Метрики по каждой бирже.
        """
        metrics = {}
        for exch_id, exch in self.ex_manager.exchanges.items():
            try:
                ticker = await asyncio.wait_for(
                    exch.fetch_ticker(symbol), timeout=5
                )
                orderbook = await asyncio.wait_for(
                    exch.fetch_order_book(symbol), timeout=5
                )
                metrics[exch_id] = {"ticker": ticker,
                                    "orderbook": orderbook}
            except Exception as e:
                logger.error(
                    f"Error on {exch_id} {symbol}: {e}"
                )
        return metrics

    async def determine_exit(
        self, symbol: str
    ) -> Dict[str, Any]:
        """
        Определяет, следует ли выйти из сделки для заданной пары.
        
        Возвращает {'exit': bool, 'best_ex': str, 'spread': float}.
        """
        metrics = await self.get_market_metrics(symbol)
        best_ex = None
        best_spread = 0.0
        keys = list(metrics.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                try:
                    bid_i = metrics[keys[i]]["orderbook"]["bids"][0][0]
                    ask_i = metrics[keys[i]]["orderbook"]["asks"][0][0]
                    bid_j = metrics[keys[j]]["orderbook"]["bids"][0][0]
                    ask_j = metrics[keys[j]]["orderbook"]["asks"][0][0]
                    spread1 = bid_i - ask_j
                    if spread1 > best_spread:
                        best_spread = spread1
                        best_ex = keys[i]
                    spread2 = bid_j - ask_i
                    if spread2 > best_spread:
                        best_spread = spread2
                        best_ex = keys[j]
                except Exception as e:
                    logger.error(
                        f"Spread calc error for {keys[i]}, {keys[j]}: {e}"
                    )
        exit_decision = best_spread >= self.exit_thresh
        return {"exit": exit_decision, "best_ex": best_ex,
                "spread": best_spread}

    def calculate_slippage(
        self, orderbook: Dict[str, Any], volume: float
    ) -> float:
        """
        Рассчитывает слиппедж на основе ордербука.
        
        Args:
            orderbook (Dict[str, Any]): Данные стакана.
            volume (float): Объем ордера.
        
        Returns:
            float: Средняя цена с учетом слиппеджа.
        """
        bids = orderbook.get("bids", [])
        total_vol = 0.0
        weighted_price = 0.0
        for price, vol in bids:
            if total_vol + vol >= volume:
                needed = volume - total_vol
                weighted_price += price * needed
                total_vol += needed
                break
            else:
                weighted_price += price * vol
                total_vol += vol
        return weighted_price / total_vol if total_vol > 0 else 0.0

    async def update_trading_pairs(self) -> None:
        """
        Обновляет список торговых пар, исключая неактуальные.
        """
        new_pairs = []
        for pair in self.active_pairs:
            try:
                ticker = await self.market.get_ticker("bybit", pair)
                if ticker and ticker.get("last", 0) > 0:
                    new_pairs.append(pair)
                else:
                    logger.info(f"Pair removed: {pair}")
            except Exception as e:
                logger.error(f"Error updating pair {pair}: {e}")
        self.active_pairs = new_pairs

    async def partial_take_profit(
        self, symbol: str, percentage: float
    ) -> Dict[str, Any]:
        """
        Частично фиксирует прибыль по заданному проценту.
        
        Args:
            symbol (str): Торговая пара.
            percentage (float): Процент для фиксации TP.
        
        Returns:
            Dict[str, Any]: Результат частичного закрытия.
        """
        try:
            result = await partial_close_trade(
                "bybit", symbol, "sell", 0.1, percentage
            )
            logger.info(f"Partial TP executed for {symbol}: {result}")
            return result
        except Exception as e:
            logger.error(f"Partial TP error for {symbol}: {e}")
            return {}

    async def select_dynamic_strategy(
        self, market_data: Dict[str, Any]
    ) -> str:
        """
        Выбирает стратегию на основе рыночных данных.
        
        Args:
            market_data (Dict[str, Any]): Данные рынка.
        
        Returns:
            str: Название стратегии.
        """
        vol = market_data.get("volatility", 0)
        strategy = "volatility_strategy" if vol > 1.0 else "scalping"
        logger.info(f"Strategy selected: {strategy}")
        return strategy

    async def _select_hedge_symbol(self) -> str:
        """
        Выбирает символ для хеджирования на основе активных позиций.
        
        Returns:
            str: Выбранный символ.
        """
        positions = await self.get_all_positions()
        if positions:
            sorted_pos = sorted(
                positions, key=lambda p: p.get("amount", 0), reverse=True
            )
            return sorted_pos[0].get("symbol", config.symbols[0])
        return config.symbols[0] if config.symbols else "BTC/USDT"

    async def activate_hedging(
        self, hedge_symbol: Optional[str] = None
    ) -> None:
        """
        Активирует динамическое хеджирование для заданного символа.
        
        Если hedge_symbol не задан, выбирается символ из портфеля.
        Коэффициент хеджирования рассчитывается динамически:
          hedge_ratio = max(0.1, 0.5*(max_dd - threshold)/threshold)
        
        Args:
            hedge_symbol (Optional[str]): Символ для хеджирования.
        """
        if hedge_symbol is None:
            hedge_symbol = await self._select_hedge_symbol()
        history = await self._get_portfolio_history(days=30)
        max_dd = self._calculate_max_drawdown(history)
        threshold = 5.0
        if max_dd <= threshold:
            logger.info(
                "Просадка не превышает порог, хеджирование не требуется"
            )
            return
        raw_ratio = 0.5 * (max_dd - threshold) / threshold
        hedge_ratio = max(0.1, raw_ratio)
        logger.info(
            f"Динамический коэффициент хеджирования для {hedge_symbol}: "
            f"{hedge_ratio:.2f}"
        )
        hedge_params = {
            "symbol": hedge_symbol,
            "side": "sell",
            "amount": hedge_ratio,
            "exchange": "bybit"
        }
        try:
            hedge_order = await execute_risk_aware_order(hedge_params)
            logger.info(
                f"Хеджирующий ордер для {hedge_symbol} выполнен: {hedge_order}"
            )
        except Exception as e:
            logger.error(
                f"Ошибка динамического хеджирования для {hedge_symbol}: {e}"
            )

    async def telegram_alerts_on_drawdown(
        self, threshold: float = 5.0
    ) -> None:
        """
        Мониторит просадку портфеля, отправляет уведомления и
        активирует хеджирование при превышении порога.
        
        Args:
            threshold (float): Порог просадки (%).
        """
        history = await self._get_portfolio_history(days=30)
        max_dd = self._calculate_max_drawdown(history)
        if max_dd >= threshold:
            alert = (
                f"🚨 Просадка портфеля: {max_dd:.2f}% (порог: {threshold}%)"
            )
            await self.notifier.send_alert(alert)
            await self.activate_hedging()

    async def _get_strategy_performance(self) -> Dict[str, Dict]:
        """
        Собирает статистику по стратегиям из Redis.
        
        Returns:
            Dict[str, Dict]: Данные по стратегиям, например:
            {"strategy_name": {"sharpe": 1.2, "returns": [...]}}.
        """
        strategies = {}
        keys = await self.redis.keys("strategy:*")
        for key in keys:
            name = key.split(":")[1]
            data = json.loads(await self.redis.get(key) or "{}")
            returns = data.get("returns", [])
            strategies[name] = {
                "sharpe": self._calculate_sharpe(returns),
                "returns": returns
            }
        return strategies

    def _calculate_sharpe(
        self, returns: List[float], risk_free: float = 0.02/365
    ) -> float:
        """
        Рассчитывает годовой коэффициент Шарпа.
        
        Args:
            returns (List[float]): Доходности.
            risk_free (float): Безрисковая ставка (дневная).
        
        Returns:
            float: Годовой коэффициент Шарпа.
        """
        if len(returns) < 2:
            return 0.0
        excess = [r - risk_free for r in returns]
        mean_ex = np.mean(excess)
        std_ex = np.std(excess)
        if std_ex == 0:
            return 0.0
        daily_sharpe = mean_ex / std_ex
        return daily_sharpe * np.sqrt(252)

    async def _adjust_strategy_weight(
        self, name: str, new_weight: float
    ) -> None:
        """
        Корректирует вес стратегии в Redis.
        
        Args:
            name (str): Имя стратегии.
            new_weight (float): Новый вес.
        """
        key = f"strategy_weight:{name}"
        await self.redis.set(key, new_weight)

    async def _get_portfolio_history(self, days: int = 30) -> List[float]:
        """
        Получает историю значений портфеля за заданное число дней.
        
        Args:
            days (int): Количество дней.
        
        Returns:
            List[float]: Список значений портфеля.
        """
        history_key = "portfolio_history"
        history_json = await self.redis.get(history_key)
        if history_json:
            history = json.loads(history_json)
            cutoff = (datetime.now() - timedelta(days=days)).timestamp()
            return [h["value"] for h in history if h["timestamp"] >= cutoff]
        return []

    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """
        Рассчитывает максимальную просадку портфеля.
        
        Args:
            values (List[float]): История значений.
        
        Returns:
            float: Максимальная просадка (%).
        """
        if not values:
            return 0.0
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd * 100

    async def telegram_alerts_on_drawdown_prod(
        self, threshold: float = 5.0
    ) -> None:
        """
        Метод для продакшена: уведомляет о просадке и вызывает хеджирование.
        
        Args:
            threshold (float): Порог просадки (%).
        """
        await self.telegram_alerts_on_drawdown(threshold)
