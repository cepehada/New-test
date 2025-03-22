"""
Portfolio Manager
Manages portfolio positions and trades.
Handles opening, closing, rebalancing, risk metrics, and hedging.
"""

import json
import logging
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

import redis.asyncio as redis
from project.config import load_config
config = load_config()
from project.utils.notify import NotificationManager
from project.trade_executor.risk_aware_executor import execute_risk_aware_order

logger = logging.getLogger("PortfolioManager")


class PortfolioManager:
    """
    Manages the trading portfolio, positions and risk metrics.
    """

    def __init__(self) -> None:
        """
        Initializes Redis connection, notifier and key names.
        """
        self.redis = redis.Redis(
            host=config.system.redis.host,
            port=config.system.redis.port,
            password=config.system.redis.password,
            db=0,
            decode_responses=True
        )
        self.notifier = NotificationManager()
        self.positions_key = "active_positions"
        self.trades_key = "closed_trades"
        self.daily_loss_key = "daily_loss"

    async def get_all_positions(self, offset: int = 0,
                                limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves all active positions from Redis with pagination.

        Args:
            offset (int): Starting index.
            limit (int, optional): Number of positions to return.

        Returns:
            List[Dict[str, Any]]: List of positions.
        """
        pos_list = await self.redis.lrange(self.positions_key, offset, -1)
        positions = [json.loads(p) for p in pos_list]
        if limit is not None:
            positions = positions[:limit]
        return positions

    async def can_open_trade(self, symbol: str) -> bool:
        """
        Checks if a new trade can be opened for a given symbol.

        Args:
            symbol (str): Trading pair.

        Returns:
            bool: True if trade can be opened.
        """
        positions = await self.get_all_positions()
        if len(positions) >= config.MAX_ACTIVE_TRADES:
            return False
        daily_loss_val = await self.redis.get(self.daily_loss_key)
        daily_loss = float(daily_loss_val or "0")
        if daily_loss <= -config.DAILY_LOSS_LIMIT:
            return False
        sym_pos = [p for p in positions if p.get("symbol") == symbol]
        if len(sym_pos) >= 3:
            return False
        return True

    async def open_position(self, position: Dict[str, Any]) -> bool:
        """
        Opens a new position and stores it in Redis.

        Args:
            position (Dict[str, Any]): Position data.

        Returns:
            bool: True if position is opened successfully.
        """
        pos_json = json.dumps(position)
        try:
            await self.redis.lpush(self.positions_key, pos_json)
            return True
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False

    async def close_position(self, position_id: str,
                             exit_price: float) -> Dict[str, Any]:
        """
        Closes a position, calculates PnL, updates loss, removes the
        position and registers the trade.

        Args:
            position_id (str): Position identifier.
            exit_price (float): Exit price.

        Returns:
            Dict[str, Any]: Closed position data with PnL.
        """
        positions = await self.get_all_positions()
        target = next((p for p in positions if p.get("id") ==
                       position_id), None)
        if not target:
            raise ValueError(f"Position {position_id} not found")
        pnl = (exit_price - target["entry_price"]) * target["amount"]
        if target.get("side", "buy").lower() == "sell":
            pnl = -pnl
        curr_loss_val = await self.redis.get(self.daily_loss_key)
        curr_loss = float(curr_loss_val or "0")
        await self.redis.set(self.daily_loss_key, curr_loss + pnl)
        closed_pos = {**target,
                      "exit_price": exit_price,
                      "pnl": pnl,
                      "closed_at": datetime.now().isoformat()}
        all_pos = await self.get_all_positions()
        for pos in all_pos:
            if pos.get("id") == position_id:
                pos_str = json.dumps(pos)
                await self.redis.lrem(self.positions_key, 0, pos_str)
                break
        await self.register_trade(closed_pos)
        return closed_pos

    async def register_trade(self, trade: Dict[str, Any]) -> None:
        """
        Registers a closed trade in Redis.

        Args:
            trade (Dict[str, Any]): Trade data.
        """
        trade_json = json.dumps(trade)
        pipe = self.redis.pipeline()
        pipe.lpush(self.trades_key, trade_json)
        pipe.ltrim(self.trades_key, 0, 999)
        await pipe.execute()

    async def smart_rebalance_portfolio(self) -> None:
        """
        Rebalances portfolio based on strategy performance.
        """
        strategies = await self._get_strategy_performance()
        total_sharpe = sum(s["sharpe"] for s in strategies.values())
        if total_sharpe == 0:
            logger.warning("No data for rebalancing")
            return
        for name, data in strategies.items():
            new_weight = data["sharpe"] / total_sharpe
            await self._adjust_strategy_weight(name, new_weight)
            logger.info(f"Strategy {name} weight set to {new_weight:.2%}")

    async def telegram_alerts_on_drawdown(self,
                                          threshold: float = 5.0) -> None:
        """
        Monitors portfolio drawdown and sends alerts.
        Activates hedging if drawdown exceeds threshold.

        Args:
            threshold (float): Drawdown threshold (%).
        """
        history = await self._get_portfolio_history(days=30)
        max_dd = self._calculate_max_drawdown(history)
        if max_dd >= threshold:
            alert = (f"🚨 Portfolio drawdown: {max_dd:.2f}% "
                     f"(threshold: {threshold}%)")
            await self.notifier.send_alert(alert)
            await self.activate_hedging()

    async def _get_strategy_performance(self) -> Dict[str, Dict]:
        """
        Collects strategy performance data from Redis.

        Returns:
            Dict[str, Dict]: Data per strategy.
        """
        strategies = {}
        keys = await self.redis.keys("strategy:*")
        for key in keys:
            name = key.split(":")[1]
            data_val = await self.redis.get(key) or "{}"
            data = json.loads(data_val)
            returns = data.get("returns", [])
            strategies[name] = {
                "sharpe": self._calculate_sharpe(returns),
                "returns": returns
            }
        return strategies

    def _calculate_sharpe(self, returns: List[float],
                          risk_free: float = 0.02/252) -> float:
        """
        Calculates the annual Sharpe ratio.

        Args:
            returns (List[float]): Trade returns.
            risk_free (float): Daily risk-free rate.

        Returns:
            float: Annual Sharpe ratio.
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

    async def _adjust_strategy_weight(self, name: str,
                                      new_weight: float) -> None:
        """
        Adjusts strategy weight in Redis.

        Args:
            name (str): Strategy name.
            new_weight (float): New weight.
        """
        key = f"strategy_weight:{name}"
        await self.redis.set(key, new_weight)

    async def _get_portfolio_history(self,
                                     days: int = 30) -> List[float]:
        """
        Retrieves portfolio history for the past days.

        Args:
            days (int): Number of days.

        Returns:
            List[float]: Portfolio values.
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
        Calculates maximum portfolio drawdown.

        Args:
            values (List[float]): Portfolio values.

        Returns:
            float: Maximum drawdown (%).
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

    async def _select_hedge_symbol(self) -> str:
        """
        Selects a symbol for hedging based on active positions.

        Returns:
            str: Hedge symbol.
        """
        positions = await self.get_all_positions()
        if positions:
            sorted_positions = sorted(
                positions,
                key=lambda p: p.get("amount", 0),
                reverse=True
            )
            return sorted_positions[0].get("symbol", config.symbols[0])
        return config.symbols[0] if config.symbols else "BTC/USDT"

    async def activate_hedging(self, hedge_symbol: str = None) -> None:
        """
        Activates dynamic hedging for the given symbol.
        Computes hedge ratio with limits between 0.1 and 1.0.

        Args:
            hedge_symbol (str, optional): Hedge symbol.
        """
        if hedge_symbol is None:
            hedge_symbol = await self._select_hedge_symbol()
        history = await self._get_portfolio_history(days=30)
        max_dd = self._calculate_max_drawdown(history)
        threshold = 5.0
        if max_dd <= threshold:
            logger.info("Drawdown below threshold, hedging not needed")
            return
        raw_ratio = 0.5 * (max_dd - threshold) / threshold
        hedge_ratio = min(max(0.1, raw_ratio), 1.0)
        logger.info(f"Dynamic hedge ratio for {hedge_symbol}: "
                    f"{hedge_ratio:.2f}")
        hedge_params = {
            "symbol": hedge_symbol,
            "side": "sell",
            "amount": hedge_ratio
        }
        try:
            hedge_order = await execute_risk_aware_order(hedge_params)
            logger.info(f"Hedge order for {hedge_symbol} executed: "
                        f"{hedge_order}")
        except Exception as e:
            logger.error(f"Hedging error for {hedge_symbol}: {e}")

    async def telegram_alerts_on_drawdown_prod(
            self, threshold: float = 5.0) -> None:
        """
        Production method: alerts on drawdown and activates hedging.

        Args:
            threshold (float): Drawdown threshold (%).
        """
        await self.telegram_alerts_on_drawdown(threshold)
