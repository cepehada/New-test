"""
Cross Strategy.
Implements cross-exchange arbitrage trading logic with fees calculation.
"""

import logging
from typing import Any, Dict, Optional, Set

from project.bots.strategies.base_strategy import BaseStrategy

logger = logging.getLogger("CrossStrategy")


class CrossStrategy(BaseStrategy):
    """
    Implements a cross-exchange arbitrage strategy.
    
    Compares prices between two exchanges and generates 
    signals when the price spread exceeds a threshold,
    accounting for trading fees.
    """

    def __init__(self, name: str = "CrossStrategy", config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the strategy with given parameters.

        Args:
            name: Strategy name
            config: Strategy configuration parameters
        """
        super().__init__(name=name, config=config or {})
        
        # Default parameters
        self.default_params = {
            "spread_threshold": 0.01,  # 1% minimum spread
            "default_symbol": "BTC/USDT",
            "default_amount": 0.1,
            "exchange_a_fee": 0.001,  # 0.1% fee
            "exchange_b_fee": 0.001,  # 0.1% fee
            "slippage_factor": 0.001,  # 0.1% slippage estimation
            "min_confidence": 0.7,
            "enable_real_orders": False
        }
        
        # Update defaults with provided config
        if config:
            self.default_params.update(config)
            
        # Required market data keys
        self.required_data_keys = {"exchange_a", "exchange_b"}
        
        self.logger.info(f"{self.name} initialized with parameters: {self.default_params}")
        
    async def run(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the cross-exchange arbitrage strategy using market data.

        Args:
            market_data: Dictionary containing market data with the following structure:
                {
                    "exchange_a": {
                        "name": "Exchange A Name",
                        "price": 10000.0,
                        "fees": 0.001 (optional, overrides default)
                    },
                    "exchange_b": {
                        "name": "Exchange B Name",
                        "price": 10100.0,
                        "fees": 0.001 (optional, overrides default)
                    },
                    "symbol": "BTC/USDT" (optional),
                    "amount": 0.1 (optional)
                }

        Returns:
            Dictionary with strategy result including action, confidence, 
            and execution parameters
        """
        self.performance_stats["runs"] += 1
        
        # Validate input data
        if not self.validate_input(market_data):
            self.performance_stats["errors"] += 1
            return {
                "action": "none",
                "symbol": self.default_params["default_symbol"],
                "confidence": 0.0,
                "reason": "invalid_input_data",
                "parameters": {"required_keys": list(self.required_data_keys)}
            }
            
        try:
            # Extract data with defaults
            exchange_a_data = market_data["exchange_a"]
            exchange_b_data = market_data["exchange_b"]
            symbol = market_data.get("symbol", self.default_params["default_symbol"])
            amount = market_data.get("amount", self.default_params["default_amount"])
            
            # Extract prices
            price_a = exchange_a_data.get("price", 0)
            price_b = exchange_b_data.get("price", 0)
            
            # Check for valid prices
            if price_a <= 0 or price_b <= 0:
                self.performance_stats["errors"] += 1
                return {
                    "action": "none",
                    "symbol": symbol,
                    "confidence": 0.0,
                    "reason": "invalid_prices",
                    "parameters": {
                        "price_a": price_a,
                        "price_b": price_b
                    }
                }
                
            # Get exchange names
            exchange_a_name = exchange_a_data.get("name", "exchange_a")
            exchange_b_name = exchange_b_data.get("name", "exchange_b")
            
            # Get fees (use provided or defaults)
            fee_a = exchange_a_data.get("fees", self.default_params["exchange_a_fee"])
            fee_b = exchange_b_data.get("fees", self.default_params["exchange_b_fee"])
            
            # Calculate price difference and direction
            price_diff = price_b - price_a
            diff_percentage = abs(price_diff) / min(price_a, price_b)
            
            # Determine spread direction
            if price_diff > 0:
                # Price on exchange B is higher
                buy_exchange = exchange_a_name
                sell_exchange = exchange_b_name
                buy_price = price_a
                sell_price = price_b
                buy_fee = fee_a
                sell_fee = fee_b
            else:
                # Price on exchange A is higher
                buy_exchange = exchange_b_name
                sell_exchange = exchange_a_name
                buy_price = price_b
                sell_price = price_a
                buy_fee = fee_b
                sell_fee = fee_a
                
            # Calculate total cost including fees and slippage
            slippage = self.default_params["slippage_factor"]
            buy_cost = buy_price * (1 + buy_fee + slippage) * amount
            sell_revenue = sell_price * (1 - sell_fee - slippage) * amount
            
            # Calculate net profit
            profit = sell_revenue - buy_cost
            profit_percentage = profit / buy_cost
            
            # Calculate confidence based on profit vs threshold
            threshold = self.default_params["spread_threshold"]
            confidence = min(1.0, profit_percentage / (threshold * 2)) if profit_percentage > 0 else 0.0
            
            # Determine if arbitrage is worth executing
            if profit_percentage > threshold and confidence >= self.default_params["min_confidence"]:
                action = "arbitrage"
                reason = "profitable_spread"
                self.performance_stats["signals_generated"] += 1
            else:
                action = "hold"
                reason = "insufficient_spread"
                
            # Record successful run
            self.performance_stats["successful_runs"] += 1
            
            # Create result
            result = {
                "action": action,
                "symbol": symbol,
                "confidence": confidence,
                "reason": reason,
                "parameters": {
                    "buy_exchange": buy_exchange,
                    "sell_exchange": sell_exchange,
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "amount": amount,
                    "spread_percentage": diff_percentage,
                    "profit_percentage": profit_percentage,
                    "profit_amount": profit,
                    "threshold": threshold
                }
            }
            
            # Log result
            self.logger.info(
                f"Cross strategy result: {action} with confidence {confidence:.2f}, "
                f"profit: {profit_percentage:.2%}, spread: {diff_percentage:.2%}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in cross strategy: {e}", exc_info=True)
            self.performance_stats["errors"] += 1
            
            return {
                "action": "none",
                "symbol": market_data.get("symbol", self.default_params["default_symbol"]),
                "confidence": 0.0,
                "reason": "error",
                "parameters": {"error": str(e)}
            }
    
    async def analyze_historical_spreads(self, historical_data: list) -> Dict[str, Any]:
        """
        Analyzes historical spread data to optimize strategy parameters.
        
        Args:
            historical_data: List of historical market data points
            
        Returns:
            Dictionary with analysis results and recommended parameters
        """
        if not historical_data:
            return {"error": "No historical data provided"}
            
        spreads = []
        profits = []
        
        for data_point in historical_data:
            try:
                price_a = data_point["exchange_a"]["price"]
                price_b = data_point["exchange_b"]["price"]
                
                # Calculate spread percentage
                spread = abs(price_a - price_b) / min(price_a, price_b)
                spreads.append(spread)
                
                # Estimate profit with fees
                fee_a = data_point["exchange_a"].get(
                    "fees", self.default_params["exchange_a_fee"]
                )
                fee_b = data_point["exchange_b"].get(
                    "fees", self.default_params["exchange_b_fee"]
                )
                
                # Simple profit calculation
                profit = spread - (fee_a + fee_b)
                profits.append(profit)
                
            except (KeyError, ZeroDivisionError) as e:
                self.logger.warning(f"Skipping data point due to error: {e}")
                continue
        
        if not spreads:
            return {"error": "No valid data points for analysis"}
            
        # Calculate statistics
        avg_spread = sum(spreads) / len(spreads)
        max_spread = max(spreads)
        min_spread = min(spreads)
        
        avg_profit = sum(profits) / len(profits)
        
        # Recommend threshold based on historical data
        # Use a value slightly lower than average profit when positive
        recommended_threshold = max(0.001, avg_profit * 0.8) if avg_profit > 0 else 0.01
        
        return {
            "avg_spread": avg_spread,
            "max_spread": max_spread,
            "min_spread": min_spread,
            "avg_profit": avg_profit,
            "recommendations": {
                "spread_threshold": recommended_threshold,
                "min_confidence": 0.75
            },
            "sample_size": len(spreads)
        }
