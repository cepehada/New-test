"""
Модуль для асинхронных HTTP запросов.
Использует aiohttp для GET и POST запросов.
"""

import aiohttp
import asyncio
import logging

async def async_get(url: str) -> str:
    """
    Выполняет асинхронный GET запрос к указанному URL.

    Args:
        url (str): URL для запроса.

    Returns:
        str: Текст ответа или пустая строка при ошибке.
    """
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    data = await resp.json()
                    return str(data)
                else:
                    return await resp.text()
    except Exception as e:
        logging.error(f"async_get error for {url}: {e}")
        return ""

async def async_post(url: str, data: dict) -> str:
    """
    Выполняет асинхронный POST запрос с JSON-данными.

    Args:
        url (str): URL для запроса.
        data (dict): Данные для POST запроса.

    Returns:
        str: Текст ответа или пустая строка при ошибке.
    """
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=data) as resp:
                return await resp.text()
    except Exception as e:
        logging.error(f"async_post error for {url}: {e}")
        return ""
