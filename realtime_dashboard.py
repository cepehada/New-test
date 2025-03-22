"""
Realtime Dashboard.
FastAPI-приложение для получения реальных торговых метрик.
Предоставляет эндпоинты для получения текущих торговых метрик,
исторических данных и агрегированной статистики.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import List, Dict, Optional, Union, Any
from datetime import datetime, timedelta
import asyncio
from functools import lru_cache

# Импорты из проекта
from project.analytics.analytics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_roi,
)
from project.analytics.metrics import (
    calculate_sortino_ratio, 
    calculate_calmar_ratio, 
    calculate_metrics_pack  # Предполагаем, что эта функция добавлена в metrics.py
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RealtimeDashboard")

# Модели данных
class MetricsResponse(BaseModel):
    """Модель ответа с торговыми метриками."""
    sharpe_ratio: float = Field(..., description="Коэффициент Шарпа (годовой)")
    max_drawdown: float = Field(..., description="Максимальная просадка в процентах")
    roi: float = Field(..., description="Доходность инвестиций в процентах")
    sortino_ratio: float = Field(..., description="Коэффициент Сортино")
    calmar_ratio: float = Field(..., description="Коэффициент Калмара")
    last_updated: datetime = Field(..., description="Время последнего обновления")
    
    class Config:
        schema_extra = {
            "example": {
                "sharpe_ratio": 1.234,
                "max_drawdown": 5.67,
                "roi": 20.0,
                "sortino_ratio": 1.890,
                "calmar_ratio": 3.53,
                "last_updated": "2025-03-22T19:38:01Z"
            }
        }

class ExtendedMetricsResponse(MetricsResponse):
    """Расширенная модель с дополнительными метриками."""
    win_rate: float = Field(..., description="Процент выигрышных сделок")
    profit_loss_ratio: float = Field(..., description="Соотношение средней прибыли к среднему убытку")
    omega_ratio: float = Field(..., description="Коэффициент Омега")
    annual_volatility: float = Field(..., description="Годовая волатильность в процентах")

class HistoricalDataRequest(BaseModel):
    """Модель запроса исторических данных."""
    start_date: datetime = Field(..., description="Начальная дата периода")
    end_date: datetime = Field(..., description="Конечная дата периода")
    interval: str = Field("daily", description="Интервал агрегации данных (daily, weekly, monthly)")

class HistoricalDataPoint(BaseModel):
    """Модель точки исторических данных."""
    timestamp: datetime
    sharpe_ratio: float
    max_drawdown: float
    roi: float

# Инициализация FastAPI приложения
app = FastAPI(
    title="Realtime Trading Dashboard",
    description="API для получения текущих и исторических торговых метрик",
    version="1.1",
)

# Добавление CORS middleware для поддержки cross-origin запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Кэшированная функция для получения данных
# TTL (time-to-live) кэша: 5 минут
@lru_cache(maxsize=1)
def get_cached_data():
    """Кэширует данные на 5 минут для снижения нагрузки."""
    expiry = datetime.now() + timedelta(minutes=5)
    return {
        "data": fetch_live_data(),
        "expiry": expiry
    }

def fetch_live_data():
    """
    Получает живые данные из источника данных.
    В продакшене эта функция должна получать реальные данные.
    """
    # Пример: данные для расчета. В продакшене
    # данные извлекаются из базы или сервиса.
    sample_returns = [0.01, -0.005, 0.015, -0.02, 0.005, 0.01]
    portfolio_values = [100, 105, 102, 110, 108, 107, 112, 109]
    initial_equity = 10000.0
    final_equity = 12000.0

    return {
        "returns": sample_returns,
        "portfolio_values": portfolio_values,
        "initial_equity": initial_equity,
        "final_equity": final_equity
    }

async def get_data_source():
    """
    Dependency для получения данных.
    Проверяет кэш и обновляет при необходимости.
    """
    cached = get_cached_data()
    if datetime.now() > cached["expiry"]:
        # Очистить кэш, чтобы следующий вызов получил свежие данные
        get_cached_data.cache_clear()
        cached = get_cached_data()
    return cached["data"]

@app.get("/dashboard", response_model=MetricsResponse, tags=["Metrics"])
async def get_dashboard(data: Dict = Depends(get_data_source)) -> Dict:
    """
    Возвращает текущие базовые торговые метрики.
    
    Returns:
        dict: Собранные метрики.
    
    Raises:
        HTTPException: При ошибке получения данных.
    """
    try:
        sample_returns = data["returns"]
        portfolio_values = data["portfolio_values"]
        initial_equity = data["initial_equity"]
        final_equity = data["final_equity"]

        sharpe = calculate_sharpe_ratio(sample_returns)
        max_dd = calculate_max_drawdown(portfolio_values)[0]  # Первый элемент - процентная просадка
        roi = calculate_roi(initial_equity, final_equity)["roi"]  # Предполагаем, что ROI возвращает словарь
        sortino = calculate_sortino_ratio(sample_returns)
        # Убедитесь, что эта версия calmar_ratio принимает скалярные значения
        calmar = calculate_calmar_ratio(roi, max_dd) 

        return {
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "roi": roi,
            "sortino_ratio": sortino if sortino is not None else float('nan'),
            "calmar_ratio": calmar if calmar is not None else float('nan'),
            "last_updated": datetime.now()
        }
    except Exception as e:
        logger.error(f"Ошибка получения дашборда: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных: {str(e)}")

@app.get("/dashboard/extended", response_model=ExtendedMetricsResponse, tags=["Metrics"])
async def get_extended_dashboard(data: Dict = Depends(get_data_source)) -> Dict:
    """
    Возвращает расширенный набор торговых метрик,
    включая win rate, profit/loss ratio и другие.
    
    Returns:
        dict: Расширенный набор метрик.
    
    Raises:
        HTTPException: При ошибке получения данных.
    """
    try:
        # Получить базовые метрики
        base_metrics = await get_dashboard(data)
        
        # Добавить расширенные метрики
        sample_returns = data["returns"]
        
        # Предполагаем, что есть функция для расчета win_rate и profit_loss_ratio
        # В реальном проекте вам нужно реализовать эту функцию или импортировать из модуля
        trade_stats = {"win_rate": 65.0, "profit_loss_ratio": 2.1}  # Пример значений
        
        # Рассчитать Omega Ratio и Annual Volatility
        # Предполагаем, что эти функции есть в вашем модуле metrics
        omega_ratio = 1.75  # Пример значения
        annual_volatility = 12.5  # Пример значения
        
        # Объединить базовые и расширенные метрики
        extended_metrics = {
            **base_metrics,
            "win_rate": trade_stats["win_rate"],
            "profit_loss_ratio": trade_stats["profit_loss_ratio"],
            "omega_ratio": omega_ratio,
            "annual_volatility": annual_volatility
        }
        
        return extended_metrics
    except Exception as e:
        logger.error(f"Ошибка получения расширенного дашборда: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка получения расширенных данных: {str(e)}")

@app.get("/dashboard/history", tags=["Historical Data"])
async def get_historical_data(
    start_date: datetime = Query(..., description="Начальная дата (ISO формат)"),
    end_date: datetime = Query(..., description="Конечная дата (ISO формат)"),
    interval: str = Query("daily", description="Интервал данных (daily, weekly, monthly)")
) -> List[Dict]:
    """
    Возвращает исторические данные метрик за указанный период.
    
    Args:
        start_date: Начальная дата периода
        end_date: Конечная дата периода
        interval: Интервал агрегации (daily, weekly, monthly)
    
    Returns:
        List[Dict]: Список точек данных с метриками
    """
    try:
        # В реальном приложении здесь нужно получить исторические данные
        # из базы данных или другого хранилища
        
        # Генерируем тестовые данные для демонстрации
        result = []
        current = start_date
        while current <= end_date:
            # Генерируем случайные метрики для демонстрации
            import random
            point = {
                "timestamp": current.isoformat(),
                "sharpe_ratio": random.uniform(0.5, 2.5),
                "max_drawdown": random.uniform(1.0, 10.0),
                "roi": random.uniform(5.0, 25.0),
                "sortino_ratio": random.uniform(0.7, 3.0)
            }
            result.append(point)
            
            # Увеличиваем текущую дату в зависимости от интервала
            if interval == "daily":
                current += timedelta(days=1)
            elif interval == "weekly":
                current += timedelta(weeks=1)
            elif interval == "monthly":
                # Примерно месяц
                current += timedelta(days=30)
        
        return result
    except Exception as e:
        logger.error(f"Ошибка получения исторических данных: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка получения исторических данных: {str(e)}")

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, str]:
    """
    Проверка работоспособности API.
    
    Returns:
        Dict[str, str]: Статус API
    """
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)