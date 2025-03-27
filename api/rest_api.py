"""
Модуль для RESTful API.
Предоставляет REST эндпоинты для взаимодействия с системой.
"""

import os
import time
from functools import wraps

import jwt
from aiohttp import web
from project.bots.arbitrage.core import ArbitrageCore
from project.bots.bot_manager import BotManager
from project.bots.strategies.strategy_manager import StrategyManager
from project.config import get_config
from project.data.market_data import MarketData
from project.trade_executor.order_executor import OrderExecutor
from project.utils.logging_utils import get_logger
from project.utils.notify import send_trading_signal

logger = get_logger(__name__)

# Глобальные объекты
market_data = MarketData.get_instance()
order_executor = OrderExecutor.get_instance()
bot_manager = BotManager.get_instance()
strategy_manager = StrategyManager.get_instance()
arbitrage_core = ArbitrageCore.get_instance()
config = get_config()

# Настройки JWT
JWT_SECRET = config.JWT_SECRET or os.environ.get("JWT_SECRET", "super-secret-key-change-this")
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION = 86400  # 24 часа

# Защита эндпоинтов с помощью JWT


def jwt_required(func):
    @wraps(func)
    async def wrapper(request):
        try:
            # Получаем токен из заголовка Authorization
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return web.json_response({"error": "Invalid or missing token"}, status=401)

            token = auth_header.split(' ')[1]

            try:
                # Проверяем токен
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                request.user = payload.get("username")
            except jwt.PyJWTError:
                return web.json_response({"error": "Invalid token"}, status=401)

            # Проверяем срок действия токена
            if "exp" in payload and time.time() > payload["exp"]:
                return web.json_response({"error": "Token expired"}, status=401)

            # Вызываем исходную функцию
            return await func(request)

        except Exception as e:
            logger.error("Error in JWT middleware: {str(e)}" %)
            return web.json_response({"error": "Authentication error"}, status=500)

    return wrapper

# Эндпоинты для аутентификации


async def login(request):
    """
    Эндпоинт для аутентификации пользователя.
    """
    try:
        # Получаем учетные данные
        data = await request.json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return web.json_response({"error": "Missing username or password"}, status=400)

        # Проверяем учетные данные
        # В реальном приложении здесь должна быть проверка в базе данных
        valid_users = {
            "admin": "adminpassword",
            "user": "userpassword"
        }

        if username not in valid_users or valid_users[username] != password:
            return web.json_response({"error": "Invalid credentials"}, status=401)

        # Создаем JWT токен
        payload = {
            "username": username,
            "exp": time.time() + JWT_EXPIRATION
        }

        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        # Отправляем токен
        return web.json_response({
            "token": token,
            "username": username,
            "expires": time.time() + JWT_EXPIRATION
        })

    except Exception as e:
        logger.error("Error in login: {str(e)}" %)
        return web.json_response({"error": "Login error"}, status=500)

# Эндпоинты для рыночных данных


@jwt_required
async def get_tickers(request):
    """
    Получает тикеры для указанных символов.
    """
    try:
        # Получаем параметры запроса
        exchange_id = request.query.get("exchange", "binance")
        symbols_str = request.query.get("symbols", "")

        # Разделяем символы
        if symbols_str:
            symbols = symbols_str.split(",")
        else:
            # Если символы не указаны, используем популярные
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

        # Получаем тикеры
        results = {}
        for symbol in symbols:
            ticker = await market_data.get_ticker(exchange_id, symbol)
            if ticker is not None:
                results[symbol] = ticker

        return web.json_response(results)

    except Exception as e:
        logger.error("Error in get_tickers: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def get_orderbook(request):
    """
    Получает ордербук для указанного символа.
    """
    try:
        # Получаем параметры запроса
        exchange_id = request.query.get("exchange", "binance")
        symbol = request.query.get("symbol")
        limit = int(request.query.get("limit", 10))

        if not symbol:
            return web.json_response({"error": "Symbol parameter is required"}, status=400)

        # Получаем ордербук
        orderbook = await market_data.get_orderbook(exchange_id, symbol, limit=limit)

        if orderbook is None:
            return web.json_response({"error": f"Failed to get orderbook for {symbol}"}, status=404)

        return web.json_response(orderbook)

    except Exception as e:
        logger.error("Error in get_orderbook: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def get_ohlcv(request):
    """
    Получает OHLCV-данные для указанного символа и таймфрейма.
    """
    try:
        # Получаем параметры запроса
        exchange_id = request.query.get("exchange", "binance")
        symbol = request.query.get("symbol")
        timeframe = request.query.get("timeframe", "1h")
        limit = int(request.query.get("limit", 100))

        if not symbol:
            return web.json_response({"error": "Symbol parameter is required"}, status=400)

        # Получаем OHLCV-данные
        ohlcv = await market_data.get_ohlcv(exchange_id, symbol, timeframe, limit=limit)

        if ohlcv is None or ohlcv.empty:
            return web.json_response({"error": f"Failed to get OHLCV data for {symbol}"}, status=404)

        # Преобразуем DataFrame в список
        data = []
        for index, row in ohlcv.iterrows():
            data.append({
                "timestamp": index.timestamp() * 1000,  # в миллисекундах для JS
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"]
            })

        return web.json_response(data)

    except Exception as e:
        logger.error("Error in get_ohlcv: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)

# Эндпоинты для торговли


@jwt_required
async def execute_order(request):
    """
    Выполняет торговый ордер.
    """
    try:
        # Получаем данные ордера
        data = await request.json()

        # Проверяем обязательные параметры
        required_fields = ["symbol", "side", "amount"]
        for field in required_fields:
            if field not in data:
                return web.json_response({"error": f"Missing required field: {field}"}, status=400)

        # Получаем параметры ордера
        symbol = data["symbol"]
        side = data["side"]
        amount = float(data["amount"])
        order_type = data.get("type", "market")
        price = data.get("price")
        exchange_id = data.get("exchange", "binance")

        # Выполняем ордер
        result = await order_executor.execute_order(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type=order_type,
            price=price,
            exchange_id=exchange_id
        )

        if not result.success:
            return web.json_response({"error": result.error}, status=400)

        # Отправляем уведомление
        await send_trading_signal(
            f"Ордер выполнен: {side} {amount} {symbol} по цене {price or 'рыночной'}"
        )

        return web.json_response({
            "success": True,
            "order_id": result.order_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "order_type": order_type,
            "exchange": exchange_id,
            "status": result.status,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("Error in execute_order: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def get_open_orders(request):
    """
    Получает список открытых ордеров.
    """
    try:
        # Получаем параметры запроса
        exchange_id = request.query.get("exchange", "binance")
        symbol = request.query.get("symbol")

        # Получаем открытые ордера
        orders = await order_executor.get_open_orders(exchange_id, symbol)

        return web.json_response(orders)

    except Exception as e:
        logger.error("Error in get_open_orders: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def cancel_order(request):
    """
    Отменяет ордер.
    """
    try:
        # Получаем данные запроса
        data = await request.json()

        # Проверяем обязательные параметры
        required_fields = ["order_id", "symbol"]
        for field in required_fields:
            if field not in data:
                return web.json_response({"error": f"Missing required field: {field}"}, status=400)

        # Получаем параметры
        order_id = data["order_id"]
        symbol = data["symbol"]
        exchange_id = data.get("exchange", "binance")

        # Отменяем ордер
        result = await order_executor.cancel_order(
            order_id=order_id,
            symbol=symbol,
            exchange_id=exchange_id
        )

        if not result.success:
            return web.json_response({"error": result.error}, status=400)

        return web.json_response({
            "success": True,
            "order_id": order_id,
            "symbol": symbol,
            "exchange": exchange_id,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("Error in cancel_order: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)

# Эндпоинты для ботов


@jwt_required
async def list_bots(request):
    """
    Получает список ботов.
    """
    try:
        # Получаем список ботов
        bots = bot_manager.get_bots()

        # Преобразуем в JSON-совместимый формат
        bot_list = []
        for bot_id, bot in bots.items():
            bot_info = {
                "id": bot_id,
                "name": bot.name,
                "status": bot.get_status(),
                "type": bot.__class__.__name__,
                "exchange": bot.exchange_id,
                "symbols": bot.symbols,
                "started_at": bot.start_time,
                "uptime": time.time() - bot.start_time if bot.start_time > 0 else 0
            }
            bot_list.append(bot_info)

        return web.json_response(bot_list)

    except Exception as e:
        logger.error("Error in list_bots: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def get_bot_state(request):
    """
    Получает состояние бота.
    """
    try:
        # Получаем ID бота из параметров
        bot_id = request.match_info.get("bot_id")

        if not bot_id:
            return web.json_response({"error": "Bot ID is required"}, status=400)

        # Получаем состояние бота
        state = await bot_manager.get_bot_state(bot_id)

        if not state:
            return web.json_response({"error": f"Bot with ID {bot_id} not found"}, status=404)

        return web.json_response(state)

    except Exception as e:
        logger.error("Error in get_bot_state: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def start_bot(request):
    """
    Запускает бота.
    """
    try:
        # Получаем данные запроса
        data = await request.json()

        # Получаем параметры
        bot_type = data.get("type")
        bot_name = data.get("name")
        exchange_id = data.get("exchange", "binance")
        symbols = data.get("symbols", ["BTC/USDT"])
        config = data.get("config", {})

        if not bot_type or not bot_name:
            return web.json_response({"error": "Bot type and name are required"}, status=400)

        # Запускаем бота
        bot_id = await bot_manager.start_bot(
            bot_type=bot_type,
            name=bot_name,
            exchange_id=exchange_id,
            symbols=symbols,
            config=config
        )

        if not bot_id:
            return web.json_response({"error": "Failed to start bot"}, status=400)

        return web.json_response({
            "success": True,
            "bot_id": bot_id,
            "name": bot_name,
            "type": bot_type,
            "exchange": exchange_id,
            "symbols": symbols,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("Error in start_bot: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def stop_bot(request):
    """
    Останавливает бота.
    """
    try:
        # Получаем ID бота из параметров
        bot_id = request.match_info.get("bot_id")

        if not bot_id:
            return web.json_response({"error": "Bot ID is required"}, status=400)

        # Останавливаем бота
        success = await bot_manager.stop_bot(bot_id)

        if not success:
            return web.json_response({"error": f"Failed to stop bot with ID {bot_id}"}, status=400)

        return web.json_response({
            "success": True,
            "bot_id": bot_id,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("Error in stop_bot: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)

# Эндпоинты для стратегий


@jwt_required
async def list_strategies(request):
    """
    Получает список доступных стратегий.
    """
    try:
        # Получаем список стратегий
        available_strategies = strategy_manager.get_available_strategies()
        running_strategies = strategy_manager.get_running_strategies()

        return web.json_response({
            "available_strategies": available_strategies,
            "running_strategies": running_strategies
        })

    except Exception as e:
        logger.error("Error in list_strategies: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def start_strategy(request):
    """
    Запускает стратегию.
    """
    try:
        # Получаем данные запроса
        data = await request.json()

        # Проверяем обязательные параметры
        if "strategy_name" not in data:
            return web.json_response({"error": "Strategy name is required"}, status=400)

        # Получаем параметры
        strategy_name = data["strategy_name"]
        exchange_id = data.get("exchange", "binance")
        symbols = data.get("symbols")
        timeframes = data.get("timeframes")
        config = data.get("config", {})

        # Запускаем стратегию
        strategy_id = await strategy_manager.start_strategy(
            strategy_name=strategy_name,
            exchange_id=exchange_id,
            symbols=symbols,
            timeframes=timeframes,
            config=config
        )

        if not strategy_id:
            return web.json_response({"error": f"Failed to start strategy {strategy_name}"}, status=400)

        return web.json_response({
            "success": True,
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "exchange": exchange_id,
            "symbols": symbols,
            "timeframes": timeframes,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("Error in start_strategy: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def stop_strategy(request):
    """
    Останавливает стратегию.
    """
    try:
        # Получаем ID стратегии из параметров
        strategy_id = request.match_info.get("strategy_id")

        if not strategy_id:
            return web.json_response({"error": "Strategy ID is required"}, status=400)

        # Останавливаем стратегию
        success = await strategy_manager.stop_strategy(strategy_id)

        if not success:
            return web.json_response({"error": f"Failed to stop strategy with ID {strategy_id}"}, status=400)

        return web.json_response({
            "success": True,
            "strategy_id": strategy_id,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("Error in stop_strategy: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def get_strategy_state(request):
    """
    Получает состояние стратегии.
    """
    try:
        # Получаем ID стратегии из параметров
        strategy_id = request.match_info.get("strategy_id")

        if not strategy_id:
            return web.json_response({"error": "Strategy ID is required"}, status=400)

        # Получаем состояние стратегии
        state = await strategy_manager.get_strategy_state(strategy_id)

        if not state:
            return web.json_response({"error": f"Strategy with ID {strategy_id} not found"}, status=404)

        return web.json_response(state)

    except Exception as e:
        logger.error("Error in get_strategy_state: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)

# Эндпоинты для арбитража


@jwt_required
async def scan_arbitrage(request):
    """
    Сканирует арбитражные возможности.
    """
    try:
        # Получаем параметры запроса
        exchange_ids_str = request.query.get("exchanges", "binance,kucoin,huobi,okex")
        symbols_str = request.query.get("symbols", "BTC/USDT,ETH/USDT,XRP/USDT")

        # Разделяем параметры
        exchanges = exchange_ids_str.split(",")
        symbols = symbols_str.split(",")

        # Сканируем возможности
        opportunities = await arbitrage_core.scan_opportunities(symbols, exchanges)

        # Преобразуем в JSON-совместимый формат
        opps_list = []
        for opp in opportunities:
            opps_list.append({
                "symbol": opp.symbol,
                "buy_exchange": opp.buy_exchange,
                "sell_exchange": opp.sell_exchange,
                "buy_price": opp.buy_price,
                "sell_price": opp.sell_price,
                "price_diff": opp.price_diff,
                "price_diff_pct": opp.price_diff_pct,
                "profit_margin_pct": opp.profit_margin_pct,
                "buy_volume": opp.buy_volume,
                "sell_volume": opp.sell_volume,
                "timestamp": opp.timestamp
            })

        return web.json_response(opps_list)

    except Exception as e:
        logger.error("Error in scan_arbitrage: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)


@jwt_required
async def execute_arbitrage(request):
    """
    Выполняет арбитражную сделку.
    """
    try:
        # Получаем данные запроса
        data = await request.json()

        # Проверяем обязательные параметры
        required_fields = ["symbol", "buy_exchange", "sell_exchange"]
        for field in required_fields:
            if field not in data:
                return web.json_response({"error": f"Missing required field: {field}"}, status=400)

        # Получаем параметры
        symbol = data["symbol"]
        buy_exchange = data["buy_exchange"]
        sell_exchange = data["sell_exchange"]

        # Сканируем возможности для указанного символа
        opportunities = await arbitrage_core.scan_opportunities([symbol], [buy_exchange, sell_exchange])

        if not opportunities:
            return web.json_response({"error": f"No arbitrage opportunities found for {symbol}"}, status=404)

        # Находим нужную возможность
        target_opp = None
        for opp in opportunities:
            if opp.buy_exchange == buy_exchange and opp.sell_exchange == sell_exchange:
                target_opp = opp
                break

        if not target_opp:
            return web.json_response({
                "error": f"No matching arbitrage opportunity found for {symbol} between {buy_exchange} and {sell_exchange}"
            }, status=404)

        # Проверяем актуальность
        is_valid = await arbitrage_core.verify_opportunity(target_opp)
        if not is_valid:
            return web.json_response({"error": "Opportunity is no longer valid"}, status=400)

        # Выполняем арбитраж
        success = await arbitrage_core.execute_arbitrage(target_opp)

        if not success:
            return web.json_response({"error": "Failed to execute arbitrage"}, status=400)

        return web.json_response({
            "success": True,
            "symbol": symbol,
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "buy_price": target_opp.buy_price,
            "sell_price": target_opp.sell_price,
            "profit_margin_pct": target_opp.profit_margin_pct,
            "timestamp": time.time()
        })

    except Exception as e:
        logger.error("Error in execute_arbitrage: {str(e)}" %)
        return web.json_response({"error": str(e)}, status=500)

# Создание приложения aiohttp


def create_app():
    """Создает и конфигурирует приложение FastAPI"""
    from fastapi import FastAPI
    app = FastAPI(title="Trading API", description="API для торгового бота", version="1.0.0")
    
    # Регистрация маршрутов
    app.include_router(router)
    app.include_router(strategy_router)
    app.include_router(market_router)
    app.include_router(user_router)
    
    # Регистрация WebSocket эндпоинтов
    app.include_router(ws_router)
    app.include_router(data_ws_router)
    app.include_router(auth_router)
    app.include_router(order_router)
    
    # Регистрация административных маршрутов
    app.include_router(admin_router)
    app.include_router(stats_router)
    
    return app

# Создаем приложение
app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск API сервера...")
    uvicorn.run("rest_api:app", host="0.0.0.0", port=8000)
