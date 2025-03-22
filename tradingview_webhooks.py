"""
TradingView Webhooks.
Handles POST requests from TradingView and initiates trade signals.
"""

from fastapi import APIRouter, Request, HTTPException
import logging
from project.config import load_config
config = load_config()
from project.trade_executor.strategy_executor import execute_strategy

logger = logging.getLogger("TradingViewWebhooks")
router = APIRouter()


@router.post("/webhook")
async def webhook_handler(request: Request) -> dict:
    """
    Handles POST requests from TradingView.

    Args:
        request (Request): Incoming request with JSON payload.

    Returns:
        dict: Result of processing.
    """
    try:
        data = await request.json()
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    logger.info(f"Webhook received: {data}")

    try:
        result = await execute_strategy(data)
    except Exception as e:
        logger.error(f"Strategy execution error: {e}")
        raise HTTPException(status_code=500,
                            detail="Strategy execution failed")

    return {"status": "success", "result": result}
