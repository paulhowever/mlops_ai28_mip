import asyncio
import logging
import re

import asyncpg

from dota_winprob.config import Settings

log = logging.getLogger(__name__)

_VERSION_BANNER = re.compile(r"PostgreSQL (?P<version>\d+(?:\.\d+)*)")


async def create_pool(settings: Settings) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(
        settings.database_url,
        # Нулевой минимум: пул не открывает соединений заранее, поэтому
        # приложение поднимается даже с лежащей базой.
        min_size=0,
        max_size=5,
        command_timeout=settings.health_probe_timeout,
    )
    log.info("Пул соединений с Postgres создан")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    await pool.close()
    log.info("Пул соединений с Postgres закрыт")


def parse_version(banner: str) -> str:
    match = _VERSION_BANNER.search(banner)
    return match.group("version") if match else banner


async def fetch_version(pool: asyncpg.Pool) -> str:
    connection = await pool.acquire()
    try:
        banner = await connection.fetchval("SELECT version()")
    except (asyncio.CancelledError, TimeoutError):
        # Без обрыва release ждёт, пока зависший сервер подтвердит отмену запроса, и таймаут пробы не срабатывает.
        connection.terminate()
        raise
    finally:
        await pool.release(connection)
    return parse_version(str(banner))
