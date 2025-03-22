import asyncio
import logging
import numpy as np
from typing import Dict, Any

from project.config import load_config

config = load_config()
from project.integrations.market_data import MarketData

logger = logging.getLogger("AdvancedFeatures")


async def auto_dynamic_leverage(
    exchange_id: str, symbol: str, current_leverage: float, target_volatility: float, 
    max_leverage: float = 50.0  # Добавлен параметр максимального плеча
) -> Dict[str, Any]:
    """
    Автоматически корректирует плечо на основе текущей волатильности.

    Получает исторические данные для расчета стандартного
    отклонения, сравнивает с целевым значением.

    Args:
        exchange_id (str): Идентификатор биржи.
        symbol (str): Торговая пара.
        current_leverage (float): Текущее плечо.
        target_volatility (float): Целевое значение волатильности.
        max_leverage (float): Максимально допустимое плечо.

    Returns:
        Dict[str, Any]: Новое плечо и рассчитанную волатильность.
    """
    try:
        market = MarketData()
        ticker = await market.get_ticker(exchange_id, symbol)
        # Предположим, что тикер содержит исторические цены
        prices = ticker.get("historical", [])
        
        if not prices or len(prices) < 2:
            logger.error("Нет данных для расчета волатильности")
            return {"new_leverage": current_leverage, "error": "No data"}
            
        # Расчет дневной волатильности (более точный подход)
        log_returns = np.diff(np.log(prices))
        vol = np.std(log_returns) * np.sqrt(24)  # Примерное масштабирование до суточной волатильности
        
        # Пропорциональное изменение плеча на основе разницы волатильностей
        volatility_ratio = target_volatility / vol if vol > 0 else 1.0
        leverage_adjustment = volatility_ratio - 1.0
        
        # Более плавное изменение плеча с ограничением
        new_leverage = current_leverage * (1 + leverage_adjustment * 0.1)  # Коэффициент 0.1 для плавности
        new_leverage = max(1.0, min(max_leverage, round(new_leverage, 1)))  # Ограничение сверху и снизу
        
        logger.info(f"Volatility: {vol:.4f}, Target: {target_volatility:.4f}, " 
                   f"New leverage: {new_leverage} (was {current_leverage})")
        
        return {"new_leverage": new_leverage, "volatility": vol, "target_volatility": target_volatility}
    
    except Exception as e:
        logger.error(f"Ошибка при расчете динамического плеча: {str(e)}")
        return {"new_leverage": current_leverage, "error": str(e)}


if __name__ == "__main__":

    async def main():
        result = await auto_dynamic_leverage(
            exchange_id="bybit",
            symbol="BTC/USDT",
            current_leverage=10,
            target_volatility=0.05,  # Примерная целевая волатильность (5%)
            max_leverage=25  # Максимальное плечо
        )
        print(result)

    asyncio.run(main())