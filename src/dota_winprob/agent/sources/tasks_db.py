from typing import Any

import asyncpg

from dota_winprob.agent.config import SERVICE_NAME, AgentSettings
from dota_winprob.agent.errors import SourceUnavailable

NAME = "postgres"
ERRORS = (OSError, TimeoutError, asyncpg.PostgresError, asyncpg.InterfaceError)

FETCH_TASK = """
SELECT task_id, source, status, win_probability, model_version, error,
       attempts, worker, published_at, completed_at, created_at
FROM tasks
WHERE task_id = $1
"""
FETCH_FAILED = """
SELECT task_id, source, status, win_probability, model_version, error,
       attempts, worker, published_at, completed_at, created_at
FROM tasks
WHERE status = 'failed'
ORDER BY completed_at DESC NULLS LAST
LIMIT $1
"""
COUNT_BY_STATUS = "SELECT status, count(*) AS total FROM tasks GROUP BY status"


async def create_pool(settings: AgentSettings) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        settings.readonly_database_url,
        min_size=0,
        max_size=3,
        command_timeout=settings.agent_source_timeout,
        timeout=settings.agent_source_timeout,
        server_settings={
            "default_transaction_read_only": "on",
            "application_name": SERVICE_NAME,
        },
    )


class TasksReader:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def _fetch(self, query: str, *args: object) -> list[dict[str, Any]]:
        try:
            async with (
                self._pool.acquire() as connection,
                connection.transaction(readonly=True),
            ):
                return [dict(row) for row in await connection.fetch(query, *args)]
        except ERRORS as error:
            raise SourceUnavailable(NAME, type(error).__name__) from error

    async def task(self, task_id: str) -> dict[str, Any] | None:
        rows = await self._fetch(FETCH_TASK, task_id)
        return rows[0] if rows else None

    async def failed(self, limit: int) -> list[dict[str, Any]]:
        return await self._fetch(FETCH_FAILED, limit)

    async def totals(self) -> dict[str, int]:
        rows = await self._fetch(COUNT_BY_STATUS)
        return {row["status"]: row["total"] for row in rows}

    async def check(self) -> None:
        await self._fetch("SELECT 1")
