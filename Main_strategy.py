"""
Main Strategy.
A threshold-based trading strategy with commission consideration.
"""

import logging
from typing import Any, Dict, Optional

from project.bots.strategies.base_strategy import BaseStrategy
from project.config import load_config

logger = logging.getLogger("MainStrategy")


class MainStrategy(BaseStrategy):
    """
    Implements a threshold-based strategy with commission cost consideration.
    
    This strategy generates signals based on comparing current price with 
    a threshold value, taking into account the trading commission.
    """

    def __init__(self, name: str = "MainStrategy", config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the strategy with given parameters.

        Args:
            name: Strategy name
            config: Strategy configuration parameters
        """
        super().__init__(name=name, config=config or {})
        
        # Load application config
        self.app_config = load_config()
        
        # Default parameters
        self.default_params = {
            "threshold": 10000.0,  # Default price threshold
            "buy_zone_size": 0.02,  # 2% zone above threshold for neutral signal
            "sell_zone_size": 0.02,  # 2% zone below threshold for neutral signal
            "symbol": "BTC/USDT",
            "commission_rate": getattr(self.app_config, "COMMISSION_RATE", 0.001),  # Default 0.1%
        }
        
        # Override defaults with provided config
        if config:
            self.default_params.update(config)
            
        # Required market data keys
        self.required_data_keys = {"price"}
        
        self.logger.info(f"{self.name} initialized with parameters: {self.default_params}")

    async def run(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the strategy based on market data and commission costs.

        Args:
            market_data: Dictionary containing market data including:
                - price: Current price
                - symbol (optional): Trading pair
                - commission (optional): Override default commission rate

        Returns:
            Dictionary with strategy result including action, confidence, and parameters
        """
        self.performance_stats["runs"] += 1
        
        # Validate input data
        if not self.validate_input(market_data):
            self.performance_stats["errors"] += 1
            return {
                "action": "none",
                "symbol": self.default_params["symbol"],
                "confidence": 0.0,
                "reason": "invalid_input_data",
                "parameters": {"missing_fields": list(self.required_data_keys - set(market_data.keys()))}
            }
            
        try:
            # Extract data with defaults
            price = market_data.get("price", 0)
            if price <= 0:
                self.performance_stats["errors"] += 1
                return {
                    "action": "none",
                    "symbol": market_data.get("symbol", self.default_params["symbol"]),
                    "confidence": 0.0,
                    "reason": "invalid_price",
                    "parameters": {"price": price}
                }
                
            # Get other parameters
            symbol = market_data.get("symbol", self.default_params["symbol"])
            threshold = market_data.get("threshold", self.default_params["threshold"])
            commission = market_data.get("commission", self.default_params["commission_rate"])
            buy_zone = self.default_params["buy_zone_size"]
            sell_zone = self.default_params["sell_zone_size"]
            
            # Adjust threshold to account for commission cost
            buy_threshold = threshold * (1 + commission)
            sell_threshold = threshold * (1 - commission)
            
            # Calculate neutral zones
            buy_neutral_max = buy_threshold * (1 + buy_zone)
            sell_neutral_min = sell_threshold * (1 - sell_zone)
            
            # Calculate price deviation from threshold as percentage
            price_deviation = (price - threshold) / threshold
            
            # Determine signal based on price position
            action = "none"
            reason = ""
            confidence = 0.0
            
            if price > buy_neutral_max:
                action = "buy"
                reason = "price_above_buy_threshold"
                # Higher confidence as price moves further from threshold
                confidence = min(1.0, (price - buy_neutral_max) / (threshold * 0.1))
            elif price < sell_neutral_min:
                action = "sell"
                reason = "price_below_sell_threshold"
                # Higher confidence as price moves further from threshold
                confidence = min(1.0, (sell_neutral_min - price) / (threshold * 0.1))
            else:
                action = "hold"
                reason = "price_in_neutral_zone"
                
                # Confidence depends on position within neutral zone
                if price > threshold:
                    # In buy-leaning neutral zone
                    zone_size = buy_neutral_max - threshold
                    position = (price - threshold) / zone_size if zone_size > 0 else 0
                    confidence = 0.5 * (1 - position)  # Lower confidence closer to buy threshold
                else:
                    # In sell-leaning neutral zone
                    zone_size = threshold - sell_neutral_min
                    position = (threshold - price) / zone_size if zone_size > 0 else 0
                    confidence = 0.5 * (1 - position)  # Lower confidence closer to sell threshold
            
            # Update performance stats
            if action in ("buy", "sell"):
                self.performance_stats["signals_generated"] += 1
            self.performance_stats["successful_runs"] += 1
            
            # Create detailed result
            result = {
                "action": action,
                "symbol": symbol,
                "confidence": confidence,
                "reason": reason,
                "parameters": {
                    "price": price,
                    "threshold": threshold,
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "deviation": price_deviation,
                    "commission": commission
                }
            }
            
            self.logger.info(
                f"{self.name} signal: {action.upper()} with confidence {confidence:.2f}, "
                f"price: {price}, threshold: {threshold}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {e}", exc_info=True)
            self.performance_stats["errors"] += 1
            
            return {
                "action": "none",
                "symbol": market_data.get("symbol", self.default_params["symbol"]),
                "confidence": 0.0,
                "reason": "error",
                "parameters": {"error": str(e)}
            }
            
    async def optimize_threshold(self, historical_data: list) -> Dict[str, Any]:
        """
        Analyzes historical data to find optimal threshold value.
        
        Args:
            historical_data: List of historical price data points
            
        Returns:
            Dictionary with optimization results and recommended threshold
        """
        if not historical_data or len(historical_data) < 10:
            return {"error": "Insufficient historical data for optimization"}
            
        # Extract prices
        prices = []
        for data_point in historical_data:
            try:
                price = data_point.get("price", None)
                if price and price > 0:
                    prices.append(price)
            except (KeyError, TypeError):
                continue
                
        if not prices or len(prices) < 10:
            return {"error": "Insufficient valid price points for optimization"}
            
        # Calculate statistics
        avg_price = sum(prices) / len(prices)
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price
        
        # Calculate potential thresholds
        potential_thresholds = [
            avg_price,
            avg_price * 0.95,
            avg_price * 1.05,
            avg_price - (price_range * 0.1),
            avg_price + (price_range * 0.1)
        ]
        
        # Simple evaluation of each threshold
        threshold_results = []
        for threshold in potential_thresholds:
            buy_signals = 0
            sell_signals = 0
            hold_signals = 0
            
            for price in prices:
                # Use a simplified version of the strategy logic
                if price > threshold * (1 + self.default_params["commission_rate"] + self.default_params["buy_zone_size"]):
                    buy_signals += 1
                elif price < threshold * (1 - self.default_params["commission_rate"] - self.default_params["sell_zone_size"]):
                    sell_signals += 1
                else:
                    hold_signals += 1
                    
            # Calculate signal balance - we want a good mix
            signal_balance = min(buy_signals, sell_signals) / max(buy_signals, sell_signals) if max(buy_signals, sell_signals) > 0 else 0
            hold_ratio = hold_signals / len(prices)
            
            # Score is higher when signal balance is good and hold ratio is reasonable
            score = signal_balance * (1 - abs(hold_ratio - 0.6))
            
            threshold_results.append({
                "threshold": threshold,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals,
                "signal_balance": signal_balance,
                "hold_ratio": hold_ratio,
                "score": score
            })
            
        # Sort by score, highest first
        threshold_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Get best threshold
        best_threshold = threshold_results[0] if threshold_results else {"threshold": avg_price, "score": 0}
        
        return {
            "recommended_threshold": best_threshold["threshold"],
            "score": best_threshold["score"],
            "price_statistics": {
                "average": avg_price,
                "minimum": min_price,
                "maximum": max_price,
                "range": price_range
            },
            "evaluation_results": threshold_results,
            "sample_size": len(prices)
        }
