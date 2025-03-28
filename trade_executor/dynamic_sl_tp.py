import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math

from project.utils.logging_utils import setup_logger
from project.data.market_data import MarketDataProvider
from project.config import get_config
from project.trade_executor.advanced_order_manager import (
    OrderType,
    OrderSide,
    AdvancedOrder,
    OrderStatus,  # Added import for OrderStatus
)

logger = setup_logger("dynamic_sl_tp")


class DynamicSLTPManager:
    """Менеджер для управления динамическими стоп-лоссами и тейк-профитами"""

    def __init__(self, order_manager, config: Dict[str, Any] = None):
        self.order_manager = order_manager
        self.config = config or get_config()
        self.market_data = MarketDataProvider()

        # Настройки по умолчанию
        self.default_sl_pct = self.config.get("default_stop_loss_pct", 0.02)  # 2%
        self.default_tp_pct = self.config.get("default_take_profit_pct", 0.05)  # 5%
        self.trailing_activation_pct = self.config.get(
            "trailing_activation_pct", 0.01
        )  # 1%
        self.trailing_step_pct = self.config.get("trailing_step_pct", 0.005)  # 0.5%
        self.atr_multiplier = self.config.get(
            "atr_multiplier", 1.5
        )  # Множитель ATR для стопов

        # Отслеживание трейлинг-стопов
        self._stop_requested = False
        self._tracking_task = None
        self.trailing_stops = {}

    async def start(self):
        """Запускает отслеживание трейлинг-стопов"""
        if self._tracking_task is not None:
            logger.warning("Отслеживание трейлинг-стопов уже запущено")
            return

        self._stop_requested = False
        self._tracking_task = asyncio.create_task(self._tracking_loop())
        logger.info("Запущено отслеживание трейлинг-стопов")

    async def stop(self):
        """Останавливает отслеживание трейлинг-стопов"""
        if self._tracking_task is None:
            logger.warning("Отслеживание трейлинг-стопов не запущено")
            return

        self._stop_requested = True
        self._tracking_task.cancel()
        try:
            await self._tracking_task
        except asyncio.CancelledError:
            pass
        self._tracking_task = None
        logger.info("Остановлено отслеживание трейлинг-стопов")

    async def _tracking_loop(self):
        """Фоновая задача для отслеживания и обновления трейлинг-стопов"""
        while not self._stop_requested:
            try:
                # Копируем словарь, чтобы избежать изменения во время итерации
                stops_to_check = dict(self.trailing_stops)

                for order_id, stop_data in stops_to_check.items():
                    try:
                        await self._check_and_update_trailing_stop(order_id, stop_data)
                    except Exception as e:
                        logger.error(
                            f"Ошибка при обновлении трейлинг-стопа для {order_id}: {str(e)}"
                        )

                # Пауза между проверками
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info("Задача отслеживания трейлинг-стопов отменена")
                break
            except Exception as e:
                logger.error(f"Ошибка в цикле отслеживания трейлинг-стопов: {str(e)}")
                await asyncio.sleep(5)

    async def _check_and_update_trailing_stop(
        self, order_id: str, stop_data: Dict[str, Any]
    ):
        """Проверяет и обновляет трейлинг-стоп"""
        symbol = stop_data.get("symbol")
        exchange_id = stop_data.get("exchange_id")
        side = stop_data.get("side")
        current_stop = stop_data.get("current_stop")

        # Получаем текущую цену
        ticker = await self.market_data.get_ticker(symbol, exchange_id)
        if not ticker:
            return

        current_price = ticker.get("last", 0)
        if not current_price:
            return

        # Проверяем условия для обновления стопа
        if side == OrderSide.BUY.value:  # Для длинной позиции
            # Проверяем, поднялась ли цена выше предыдущего максимума
            if current_price > stop_data.get("highest_price", 0):
                # Обновляем максимальную цену
                stop_data["highest_price"] = current_price

                # Рассчитываем новый уровень стопа
                new_stop = current_price * (
                    1 - stop_data.get("trail_pct", self.trailing_step_pct)
                )

                # Если новый стоп выше текущего, обновляем
                if new_stop > current_stop:
                    stop_data["current_stop"] = new_stop
                    logger.info(
                        f"Обновлен трейлинг-стоп для {order_id}: {current_stop:.6f} -> {new_stop:.6f} (цена: {current_price:.6f})"
                    )

                    # Обновляем стоп-ордер, если он уже создан
                    if stop_data.get("stop_order_id"):
                        await self._update_stop_order(stop_data)

        else:  # Для короткой позиции
            # Проверяем, опустилась ли цена ниже предыдущего минимума
            if current_price < stop_data.get("lowest_price", float("inf")):
                # Обновляем минимальную цену
                stop_data["lowest_price"] = current_price

                # Рассчитываем новый уровень стопа
                new_stop = current_price * (
                    1 + stop_data.get("trail_pct", self.trailing_step_pct)
                )

                # Если новый стоп ниже текущего, обновляем
                if new_stop < current_stop:
                    stop_data["current_stop"] = new_stop
                    logger.info(
                        f"Обновлен трейлинг-стоп для {order_id}: {current_stop:.6f} -> {new_stop:.6f} (цена: {current_price:.6f})"
                    )

                    # Обновляем стоп-ордер, если он уже создан
                    if stop_data.get("stop_order_id"):
                        await self._update_stop_order(stop_data)

        # Проверяем, достигнут ли стоп
        is_triggered = False
        if side == OrderSide.BUY.value:  # Для длинной позиции
            is_triggered = current_price <= current_stop
        else:  # Для короткой позиции
            is_triggered = current_price >= current_stop

        if is_triggered:
            logger.info(
                f"Трейлинг-стоп для {order_id} сработал: цена {current_price:.6f}, стоп {current_stop:.6f}"
            )

            # Удаляем из отслеживания
            if order_id in self.trailing_stops:
                del self.trailing_stops[order_id]

            # Создаем рыночный ордер для закрытия позиции
            await self._execute_stop(stop_data)

    async def _update_stop_order(self, stop_data: Dict[str, Any]):
        """Обновляет стоп-ордер на бирже"""
        try:
            # Отменяем текущий стоп-ордер
            await self.order_manager.cancel_order(stop_data.get("stop_order_id"))

            # Создаем новый стоп-ордер
            new_stop_order = await self._create_stop_order(
                stop_data.get("symbol"),
                stop_data.get("exchange_id"),
                (
                    OrderSide.SELL
                    if stop_data.get("side") == OrderSide.BUY.value
                    else OrderSide.BUY
                ),
                stop_data.get("amount"),
                stop_data.get("current_stop"),
                stop_data.get("parent_order_id"),
            )

            # Обновляем ID стоп-ордера
            stop_data["stop_order_id"] = new_stop_order.order_id

        except Exception as e:
            logger.error(f"Ошибка при обновлении стоп-ордера: {str(e)}")

    async def _execute_stop(self, stop_data: Dict[str, Any]):
        """Исполняет стоп по рыночной цене"""
        try:
            # Создаем рыночный ордер для закрытия позиции
            close_order = await self.order_manager.create_order(
                symbol=stop_data.get("symbol"),
                order_type=OrderType.MARKET,
                side=(
                    OrderSide.SELL
                    if stop_data.get("side") == OrderSide.BUY.value
                    else OrderSide.BUY
                ),
                amount=stop_data.get("amount"),
                price=None,
                exchange_id=stop_data.get("exchange_id"),
                params={"reduceOnly": True},
            )

            logger.info(f"Создан ордер для исполнения стопа: {close_order.order_id}")

        except Exception as e:
            logger.error(f"Ошибка при исполнении стопа: {str(e)}")

    async def create_dynamic_sl_tp(
        self, order: AdvancedOrder, market_data: Dict[str, Any]
    ) -> Dict[str, AdvancedOrder]:
        """
        Создает динамические стоп-лосс и тейк-профит ордера

        Args:
            order: Основной ордер
            market_data: Данные о рынке

        Returns:
            Dict[str, AdvancedOrder]: Словарь с созданными ордерами (sl, tp)
        """
        if order.status != OrderStatus.FILLED:
            logger.warning(
                f"Нельзя создать SL/TP для невыполненного ордера {order.order_id}"
            )
            return {}

        result = {}

        # Получаем параметры из рыночных данных
        volatility = market_data.get("volatility", 0.02)
        atr = market_data.get("atr", None)

        # Рассчитываем стоп-лосс и тейк-профит
        sl_level, tp_level = await self._calculate_sl_tp(order, volatility, atr)

        # Создаем стоп-лосс ордер
        sl_params = {"reduceOnly": True, "parentId": order.order_id}

        sl_order = await self._create_stop_order(
            symbol=order.symbol,
            exchange_id=order.exchange_id,
            side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
            amount=order.executed_amount,
            stop_price=sl_level,
            parent_order_id=order.order_id,
        )

        if sl_order:
            result["sl"] = sl_order
            order.related_orders.append(sl_order.order_id)

        # Создаем тейк-профит ордер
        tp_params = {
            "reduceOnly": True,
            "parentId": order.order_id,
            "stopPrice": tp_level,
        }

        tp_order = await self.order_manager.create_order(
            symbol=order.symbol,
            order_type=OrderType.TAKE_PROFIT,
            side=OrderSide.SELL if order.side == OrderSide.BUY else OrderSide.BUY,
            amount=order.executed_amount,
            price=tp_level,
            exchange_id=order.exchange_id,
            params=tp_params,
        )

        if tp_order:
            result["tp"] = tp_order
            order.related_orders.append(tp_order.order_id)

        # Сохраняем трейлинг-стоп
        if self.config.get("use_trailing_stop", True):
            # Добавляем в отслеживание
            self.trailing_stops[order.order_id] = {
                "symbol": order.symbol,
                "exchange_id": order.exchange_id,
                "side": order.side.value,
                "amount": order.executed_amount,
                "entry_price": order.average_price,
                "current_stop": sl_level,
                "trail_pct": self.trailing_step_pct,
                "highest_price": (
                    order.average_price if order.side == OrderSide.BUY else float("inf")
                ),
                "lowest_price": (
                    order.average_price if order.side == OrderSide.SELL else 0
                ),
                "parent_order_id": order.order_id,
                "stop_order_id": sl_order.order_id if sl_order else None,
            }

        # Обновляем основной ордер в БД
        if self.order_manager.db:
            await self.order_manager.db.save_order(order.to_dict())

        return result

    async def _create_stop_order(
        self,
        symbol: str,
        exchange_id: str,
        side: OrderSide,
        amount: float,
        stop_price: float,
        parent_order_id: str,
    ) -> Optional[AdvancedOrder]:
        """Создает стоп-ордер"""
        try:
            stop_order = await self.order_manager.create_order(
                symbol=symbol,
                order_type=OrderType.STOP_LOSS,
                side=side,
                amount=amount,
                price=None,
                exchange_id=exchange_id,
                params={
                    "reduceOnly": True,
                    "parentId": parent_order_id,
                    "stopPrice": stop_price,
                },
            )

            logger.info(
                f"Создан стоп-ордер {stop_order.order_id} по цене {stop_price:.6f}"
            )
            return stop_order

        except Exception as e:
            logger.error(f"Ошибка при создании стоп-ордера: {str(e)}")
            return None

    async def _calculate_sl_tp(
        self, order: AdvancedOrder, volatility: float, atr: Optional[float]
    ) -> Tuple[float, float]:
        """
        Рассчитывает уровни стоп-лосса и тейк-профита на основе рыночных данных

        Args:
            order: Ордер
            volatility: Волатильность в %
            atr: ATR (Average True Range)

        Returns:
            Tuple[float, float]: (стоп-лосс, тейк-профит)
        """
        entry_price = order.average_price

        # Если доступен ATR, используем его
        if atr:
            sl_distance = atr * self.atr_multiplier
            tp_distance = atr * self.atr_multiplier * 2  # TP обычно в 2 раза больше SL
        else:
            # Иначе используем волатильность и процентные настройки
            # Масштабируем стоп-лосс и тейк-профит с учетом волатильности
            volatility_factor = max(
                1.0, volatility / 0.02
            )  # Нормализуем относительно 2%

            sl_pct = self.default_sl_pct * volatility_factor
            tp_pct = self.default_tp_pct * volatility_factor

            sl_distance = entry_price * sl_pct
            tp_distance = entry_price * tp_pct

        # Рассчитываем уровни в зависимости от стороны
        if order.side == OrderSide.BUY:
            sl_level = entry_price - sl_distance
            tp_level = entry_price + tp_distance
        else:  # SELL
            sl_level = entry_price + sl_distance
            tp_level = entry_price - tp_distance

        logger.info(
            f"Рассчитаны уровни для {order.order_id}: SL={sl_level:.6f}, TP={tp_level:.6f} "
            f"(вход: {entry_price:.6f}, волатильность: {volatility:.4f}, ATR: {atr or 'N/A'})"
        )

        return sl_level, tp_level
