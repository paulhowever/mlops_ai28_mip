import logging
import re

import asyncpg

from dota_winprob.config import Settings

log = logging.getLogger(__name__)

_VERSION_BANNER = re.compile(r"PostgreSQL (?P<version>\d+(?:\.\d+)*)")


async def create_pool(settings: Settings) -> asyncpg.Pool | None:
    try:
        pool = await asyncpg.create_pool(
            settings.database_url,
            # Нулевой минимум: пул не открывает соединений заранее, поэтому
            # приложение поднимается даже с лежащей базой.
            min_size=0,
            max_size=5,
            command_timeout=settings.health_probe_timeout,
        )
    except (OSError, asyncpg.PostgresError) as exc:
        log.warning(
            "Не удалось создать пул соединений с Postgres",
            extra={"error": f"{type(exc).__name__}: {exc}"},
        )
        return None

    log.info("Пул соединений с Postgres создан")
    return pool


async def close_pool(pool: asyncpg.Pool | None) -> None:
    if pool is None:
        return
    await pool.close()
    log.info("Пул соединений с Postgres закрыт")


def parse_version(banner: str) -> str:
    match = _VERSION_BANNER.search(banner)
    return match.group("version") if match else banner


async def fetch_version(pool: asyncpg.Pool | None) -> str:
    if pool is None:
        message = "пул соединений не создан при старте приложения"
        raise ConnectionError(message)

    banner = await pool.fetchval("SELECT version()")
    return parse_version(str(banner))
