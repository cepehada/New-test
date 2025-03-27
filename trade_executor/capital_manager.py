from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta
import math
import logging

from project.utils.logging_utils import setup_logger
from project.config import get_config

logger = setup_logger("capital_manager")

class CapitalManager:
    """Модуль для управления капиталом"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or get_config()
        self.max_risk_per_trade = self.config.get("max_risk_per_trade", 0.02)  # 2% риска на сделку
        self.min_balance_threshold = self.config.get("min_balance_threshold", 100)  # Минимальный баланс
        self.max_positions = self.config.get("max_positions", 10)  # Максимум одновременных позиций
        self.position_history = []  # История позиций для анализа эффективности
        self.current_positions = {}  # Текущие открытые позиции
        self.win_rate = 0.5  # Начальное значение винрейта
        self.kelly_fraction = self.config.get("kelly_fraction", 0.5)  # Доля от критерия Келли
        self._load_history()

    def calculate_allocation(self, balance: float, volatility: float, slippage: float) -> float:
        """
        Рассчитывает сумму для торговли на основе баланса, волатильности и слиппейджа.

        Args:
            balance: Текущий баланс.
            volatility: Волатильность рынка.
            slippage: Слиппейдж.

        Returns:
            float: Рекомендуемая сумма для торговли.
        """
        # Учитываем количество открытых позиций
        current_positions_count = len(self.current_positions)
        if current_positions_count >= self.max_positions:
            logger.warning(f"Достигнут лимит позиций ({self.max_positions}), новые позиции не открываются")
            return 0.0
        
        # Учитываем риск по методу Келли
        kelly_risk = self._calculate_kelly_criterion(balance)
        
        # Максимальный риск (меньшее из фиксированного % или критерия Келли)
        max_risk = min(self.max_risk_per_trade, kelly_risk)
        
        # Учитываем риск и волатильность
        risk_adjusted_balance = balance * max_risk
        volatility_factor = max(0.1, 1 - volatility)  # Чем выше волатильность, тем меньше сумма
        slippage_factor = max(0.1, 1 - slippage)  # Чем выше слиппейдж, тем меньше сумма
        
        # Учитываем количество открытых позиций
        position_factor = 1 - (current_positions_count / self.max_positions)
        
        allocation = risk_adjusted_balance * volatility_factor * slippage_factor * position_factor

        # Убедимся, что сумма не меньше минимального порога
        allocation = max(allocation, self.min_balance_threshold)
        
        logger.info(f"Рассчитанное распределение: {allocation:.2f} (баланс: {balance:.2f}, риск: {max_risk:.4f}, "
                    f"волатильность: {volatility:.2f}, слиппейдж: {slippage:.2f}, позиции: {current_positions_count}/{self.max_positions})")
        
        return allocation

    def register_position(self, position_id: str, symbol: str, entry_price: float, 
                          amount: float, side: str, timestamp: Optional[float] = None) -> None:
        """Регистрирует новую позицию"""
        if position_id in self.current_positions:
            logger.warning(f"Позиция {position_id} уже существует, обновляем данные")
        
        self.current_positions[position_id] = {
            "id": position_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "amount": amount,
            "side": side,
            "open_time": timestamp or datetime.now().timestamp(),
            "pnl": 0.0,
            "pnl_percent": 0.0
        }
        
        logger.info(f"Зарегистрирована новая позиция {position_id}: {symbol} {side} {amount} по {entry_price}")

    def close_position(self, position_id: str, exit_price: float, pnl: float) -> None:
        """Закрывает позицию и записывает результат в историю"""
        if position_id not in self.current_positions:
            logger.warning(f"Попытка закрыть несуществующую позицию {position_id}")
            return
        
        position = self.current_positions[position_id]
        position["exit_price"] = exit_price
        position["close_time"] = datetime.now().timestamp()
        position["pnl"] = pnl
        
        # Расчет процентного изменения
        entry_value = position["entry_price"] * position["amount"]
        if entry_value > 0:
            position["pnl_percent"] = (pnl / entry_value) * 100
        
        # Добавляем в историю
        self.position_history.append(position)
        
        # Удаляем из текущих позиций
        del self.current_positions[position_id]
        
        # Обновляем винрейт
        self._update_win_rate()
        
        # Сохраняем историю
        self._save_history()
        
        logger.info(f"Закрыта позиция {position_id} с P&L: {pnl:.2f} ({position['pnl_percent']:.2f}%)")

    def _calculate_kelly_criterion(self, balance: float) -> float:
        """Рассчитывает размер ставки по критерию Келли"""
        # Для расчета Келли нам нужны исторические данные о выигрышах и проигрышах
        if len(self.position_history) < 20:
            # Если недостаточно данных, используем значение по умолчанию
            return self.max_risk_per_trade
        
        # Средний выигрыш в % (только для прибыльных сделок)
        profitable_trades = [t for t in self.position_history[-50:] if t.get("pnl_percent", 0) > 0]
        if not profitable_trades:
            return self.max_risk_per_trade
            
        avg_win = sum(t.get("pnl_percent", 0) / 100 for t in profitable_trades) / len(profitable_trades)
        
        # Средний проигрыш в % (только для убыточных сделок)
        losing_trades = [t for t in self.position_history[-50:] if t.get("pnl_percent", 0) < 0]
        if not losing_trades:
            return self.max_risk_per_trade
            
        avg_loss = abs(sum(t.get("pnl_percent", 0) / 100 for t in losing_trades) / len(losing_trades))
        
        # Формула Келли: f = (bp - q) / b
        # где b = средний выигрыш / средний проигрыш, p = вероятность выигрыша, q = вероятность проигрыша
        b = avg_win / avg_loss if avg_loss > 0 else 1
        p = self.win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b if b > 0 else 0
        
        # Используем часть от критерия Келли для уменьшения риска
        fraction_kelly = kelly * self.kelly_fraction
        
        # Ограничиваем максимальный риск
        kelly_risk = min(max(fraction_kelly, 0.005), 0.1)  # От 0.5% до 10%
        
        logger.info(f"Критерий Келли: {kelly:.4f}, используем {kelly_risk:.4f} "
                    f"(винрейт: {p:.2f}, выигрыш/проигрыш: {b:.2f})")
        
        return kelly_risk

    def _update_win_rate(self) -> None:
        """Обновляет винрейт на основе последних 50 сделок"""
        if not self.position_history:
            return
            
        # Берем последние 50 сделок
        recent_trades = self.position_history[-50:]
        if not recent_trades:
            return
            
        # Подсчитываем прибыльные сделки
        profitable_trades = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        
        # Обновляем винрейт
        self.win_rate = profitable_trades / len(recent_trades)

    def _save_history(self) -> None:
        """Сохраняет историю позиций в файл"""
        try:
            # Ограничиваем историю до 500 записей для экономии памяти
            history_to_save = self.position_history[-500:]
            
            with open("data/position_history.json", "w") as f:
                json.dump(history_to_save, f, indent=4)
        except Exception as e:
            logger.error(f"Ошибка при сохранении истории позиций: {str(e)}")

    def _load_history(self) -> None:
        """Загружает историю позиций из файла"""
        try:
            with open("data/position_history.json", "r") as f:
                self.position_history = json.load(f)
            
            # Обновляем винрейт
            self._update_win_rate()
            
            logger.info(f"Загружена история позиций: {len(self.position_history)} записей")
        except FileNotFoundError:
            logger.info("Файл истории позиций не найден, используем пустую историю")
        except Exception as e:
            logger.error(f"Ошибка при загрузке истории позиций: {str(e)}")

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """Возвращает статистику портфеля"""
        stats = {
            "win_rate": self.win_rate,
            "open_positions": len(self.current_positions),
            "total_trades": len(self.position_history),
            "current_exposure": 0.0,
            "best_symbol": None,
            "worst_symbol": None
        }
        
        # Рассчитываем текущую экспозицию
        for pos in self.current_positions.values():
            stats["current_exposure"] += pos["entry_price"] * pos["amount"]
        
        # Анализируем эффективность по символам
        symbol_stats = {}
        for trade in self.position_history:
            symbol = trade.get("symbol")
            if not symbol:
                continue
                
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"count": 0, "pnl": 0.0, "win_count": 0}
                
            symbol_stats[symbol]["count"] += 1
            symbol_stats[symbol]["pnl"] += trade.get("pnl", 0)
            
            if trade.get("pnl", 0) > 0:
                symbol_stats[symbol]["win_count"] += 1
                
        # Находим лучший и худший символы
        if symbol_stats:
            best_symbol = max(symbol_stats.items(), key=lambda x: x[1]["pnl"])
            worst_symbol = min(symbol_stats.items(), key=lambda x: x[1]["pnl"])
            
            stats["best_symbol"] = {
                "symbol": best_symbol[0],
                "pnl": best_symbol[1]["pnl"],
                "win_rate": best_symbol[1]["win_count"] / best_symbol[1]["count"] if best_symbol[1]["count"] > 0 else 0
            }
            
            stats["worst_symbol"] = {
                "symbol": worst_symbol[0],
                "pnl": worst_symbol[1]["pnl"],
                "win_rate": worst_symbol[1]["win_count"] / worst_symbol[1]["count"] if worst_symbol[1]["count"] > 0 else 0
            }
            
        return stats
