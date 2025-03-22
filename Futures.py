"""
Futures Strategy.
Реализует стратегию для фьючерсной торговли, используя реальные данные.
Анализирует текущую цену и funding rate, учитывает кредитное плечо,
генерируя сигнал на открытие/закрытие позиции.
"""

import logging
from typing import Dict, Any, Optional, List, Set

from project.strategies.base_strategy import BaseStrategy
from project.config import load_config
from project.utils.ccxt_exchanges import ExchangeManager
from project.trade_executor.risk_aware_executor import execute_risk_aware_order

logger = logging.getLogger("FuturesStrategy")


class FuturesStrategy(BaseStrategy):
    """
    Стратегия для фьючерсной торговли.
    
    Анализирует текущую цену и funding rate, учитывает кредитное плечо,
    генерируя сигнал на открытие/закрытие позиции.
    """

    def __init__(self, name: str = "FuturesStrategy", config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализирует стратегию фьючерсной торговли.
        
        Args:
            name: Имя стратегии
            config: Конфигурация стратегии
        """
        super().__init__(name=name, config=config)
        self.app_config = load_config()
        self.exchange_manager = ExchangeManager(self.app_config)
        
        # Настройки стратегии по умолчанию
        self.default_settings = {
            "exchange": "bybit",
            "symbol": "BTC/USDT",
            "default_side": "buy",
            "default_amount": 1.0,
            "default_leverage": 10,
            "funding_rate_threshold": 0.01,
            "max_positions": 3
        }
        
        # Обновляем настройки из конфигурации
        if self.config:
            self.default_settings.update(self.config)
            
        # Задаем необходимые ключи для входных данных
        self.required_data_keys = {"exchange", "symbol", "side"}
        
        # Храним информацию об открытых позициях
        self.open_positions: List[Dict[str, Any]] = []

    async def run(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет стратегию фьючерсной торговли на основе входящих данных.
        
        Args:
            market_data: Данные рынка с параметрами стратегии
        
        Returns:
            Результат исполнения стратегии
        """
        self.performance_stats["runs"] += 1
        
        # Проверяем входные данные
        if not self.validate_input(market_data):
            # Используем настройки по умолчанию для отсутствующих полей
            for key in self.required_data_keys:
                if key not in market_data:
                    market_data[key] = self.default_settings.get(f"default_{key}", None)
        
        # Извлекаем параметры
        exchange_id = market_data.get("exchange", self.default_settings["exchange"])
        symbol = market_data.get("symbol", self.default_settings["symbol"])
        side = market_data.get("side", self.default_settings["default_side"])
        amount = market_data.get("amount", self.default_settings["default_amount"])
        leverage = market_data.get("leverage", self.default_settings["default_leverage"])
        funding_rate_threshold = market_data.get(
            "funding_rate_threshold", self.default_settings["funding_rate_threshold"]
        )
        
        # Получаем экземпляр биржи
        exchange = self.exchange_manager.get_exchange(exchange_id)
        if not exchange:
            self.logger.error(f"Биржа {exchange_id} не найдена")
            self.performance_stats["errors"] += 1
            return self._create_result(success=False, action="none", reason="exchange_not_found")
        
        try:
            # Получаем текущую цену фьючерсного контракта
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker.get("last", 0)
            if current_price == 0:
                self.logger.error("Текущая цена равна 0")
                self.performance_stats["errors"] += 1
                return self._create_result(success=False, action="none", reason="invalid_price")
            
            # Попытка получить funding rate через API биржи
            funding_rate = 0.0
            try:
                funding_info = await exchange.fetch_funding_rate(symbol)
                funding_rate = float(funding_info.get("fundingRate", 0))
            except Exception as e:
                self.logger.warning(f"Не удалось получить funding rate: {e}")
            
            self.logger.info(f"Текущая цена: {current_price}, Funding rate: {funding_rate}")
            
            # Проверяем, если позиция по данному символу уже открыта
            existing_position = self._find_open_position(exchange_id, symbol)
            
            # Если абсолютное значение funding rate превышает порог,
            # генерируется торговой сигнал для исполнения ордера.
            if abs(funding_rate) >= funding_rate_threshold:
                # Проверяем, не превышает ли количество открытых позиций лимит
                if not existing_position and len(self.open_positions) >= self.default_settings["max_positions"]:
                    self.logger.warning(f"Достигнут максимум открытых позиций: {self.default_settings['max_positions']}")
                    return self._create_result(
                        success=False,
                        action="none",
                        reason="max_positions_reached",
                        parameters={
                            "funding_rate": funding_rate,
                            "current_price": current_price,
                            "open_positions": len(self.open_positions)
                        }
                    )
                
                # Если позиция уже существует и сторона та же, пропускаем
                if existing_position and existing_position["side"] == side:
                    self.logger.info(f"Позиция {symbol} со стороной {side} уже открыта")
                    return self._create_result(
                        success=True,
                        action="hold",
                        reason="position_already_exists",
                        parameters={
                            "funding_rate": funding_rate,
                            "current_price": current_price,
                            "existing_position": existing_position
                        }
                    )
                
                # Если позиция существует, но с противоположной стороной, закрываем ее
                if existing_position and existing_position["side"] != side:
                    self.logger.info(f"Закрываем позицию {symbol} со стороной {existing_position['side']}")
                    close_params = {
                        "exchange": exchange_id,
                        "symbol": symbol,
                        "side": "sell" if existing_position["side"] == "buy" else "buy",
                        "amount": existing_position["amount"],
                        "leverage": existing_position["leverage"],
                        "action": "close"
                    }
                    
                    close_result = await execute_risk_aware_order(close_params)
                    if close_result and close_result.get("success"):
                        self._remove_position(exchange_id, symbol)
                    
                # Формируем параметры ордера
                order_params = {
                    "exchange": exchange_id,
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "leverage": leverage,
                    "action": "open"
                }
                
                # Выполняем ордер
                order_result = await execute_risk_aware_order(order_params)
                
                if order_result and order_result.get("success"):
                    # Сохраняем информацию о новой позиции
                    self._add_position(
                        exchange_id=exchange_id,
                        symbol=symbol,
                        side=side,
                        amount=amount,
                        leverage=leverage,
                        price=current_price,
                        order_id=order_result.get("order_id", "unknown")
                    )
                    self.performance_stats["signals_generated"] += 1
                    self.performance_stats["successful_runs"] += 1
                
                return self._create_result(
                    success=bool(order_result and order_result.get("success")),
                    action=side,
                    reason="funding_rate_trigger",
                    parameters={
                        "funding_rate": funding_rate,
                        "current_price": current_price,
                        "order_result": order_result
                    }
                )
            else:
                self.logger.info("Funding rate ниже порога, сигнал не генерируется")
                self.performance_stats["successful_runs"] += 1
                return self._create_result(
                    success=True,
                    action="hold",
                    reason="funding_rate_below_threshold",
                    parameters={
                        "funding_rate": funding_rate,
                        "threshold": funding_rate_threshold,
                        "current_price": current_price
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Ошибка в futures_strategy: {e}")
            self.performance_stats["errors"] += 1
            return self._create_result(success=False, action="none", reason=str(e))
    
    def _create_result(self, success: bool, action: str, reason: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Создает стандартизированный результат стратегии.
        
        Args:
            success: Успешно ли выполнена стратегия
            action: Действие (buy, sell, hold, none)
            reason: Причина действия
            parameters: Дополнительные параметры
            
        Returns:
            Стандартизированный результат
        """
        return {
            "success": success,
            "action": action,
            "symbol": self.default_settings["symbol"],
            "strategy": self.name,
            "confidence": 1.0 if success else 0.0,
            "reason": reason,
            "parameters": parameters or {}
        }
    
    def _find_open_position(self, exchange_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Находит открытую позицию по бирже и символу.
        
        Args:
            exchange_id: ID биржи
            symbol: Торговая пара
            
        Returns:
            Информация о позиции или None
        """
        for position in self.open_positions:
            if position["exchange"] == exchange_id and position["symbol"] == symbol:
                return position
        return None
    
    def _add_position(self, exchange_id: str, symbol: str, side: str, amount: float, 
                     leverage: float, price: float, order_id: str) -> None:
        """
        Добавляет информацию о новой открытой позиции.
        
        Args:
            exchange_id: ID биржи
            symbol: Торговая пара
            side: Сторона (buy/sell)
            amount: Объем
            leverage: Кредитное плечо
            price: Цена открытия
            order_id: ID ордера
        """
        position = {
            "exchange": exchange_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "leverage": leverage,
            "open_price": price,
            "order_id": order_id,
            "opened_at": self.exchange_manager.get_current_timestamp()
        }
        self.open_positions.append(position)
        self.logger.info(f"Добавлена новая позиция: {position}")
    
    def _remove_position(self, exchange_id: str, symbol: str) -> None:
        """
        Удаляет информацию о закрытой позиции.
        
        Args:
            exchange_id: ID биржи
            symbol: Торговая пара
        """
        self.open_positions = [
            p for p in self.open_positions 
            if not (p["exchange"] == exchange_id and p["symbol"] == symbol)
        ]
        self.logger.info(f"Удалена позиция для {exchange_id}:{symbol}")
    
    async def close_all_positions(self) -> List[Dict[str, Any]]:
        """
        Закрывает все открытые позиции.
        
        Returns:
            Список результатов закрытия позиций
        """
        results = []
        for position in self.open_positions[:]:  # Копируем список, чтобы безопасно изменять его
            close_params = {
                "exchange": position["exchange"],
                "symbol": position["symbol"],
                "side": "sell" if position["side"] == "buy" else "buy",
                "amount": position["amount"],
                "leverage": position["leverage"],
                "action": "close"
            }
            
            try:
                result = await execute_risk_aware_order(close_params)
                if result and result.get("success"):
                    self._remove_position(position["exchange"], position["symbol"])
                results.append(result)
            except Exception as e:
                self.logger.error(f"Ошибка при закрытии позиции {position['symbol']}: {e}")
                results.append({"success": False, "error": str(e)})
                
        return results
