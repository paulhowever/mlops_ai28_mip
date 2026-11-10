import logging

import asyncpg

from dota_winprob.schemas import TaskMessage, TaskSource, TaskView

log = logging.getLogger(__name__)

SCHEMA_LOCK = 4_211_917

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'done', 'failed')),
    win_probability DOUBLE PRECISION,
    model_version TEXT,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    worker TEXT,
    published_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

INSERT_PENDING = """
INSERT INTO tasks (task_id, source, status, published_at)
VALUES ($1, $2, 'pending', $3)
ON CONFLICT (task_id) DO NOTHING
"""

SAVE_RESULT = """
INSERT INTO tasks (
    task_id, source, status, win_probability, model_version,
    attempts, worker, published_at, completed_at
)
VALUES ($1, $2, 'done', $3, $4, $5, $6, $7, now())
ON CONFLICT (task_id) DO UPDATE SET
    status = 'done',
    win_probability = EXCLUDED.win_probability,
    model_version = EXCLUDED.model_version,
    attempts = EXCLUDED.attempts,
    worker = EXCLUDED.worker,
    error = NULL,
    completed_at = EXCLUDED.completed_at
WHERE tasks.status <> 'done'
RETURNING task_id
"""

MARK_FAILED = """
INSERT INTO tasks (task_id, source, status, error, attempts, worker, completed_at)
VALUES ($1, $2, 'failed', $3, $4, $5, now())
ON CONFLICT (task_id) DO UPDATE SET
    status = 'failed',
    error = EXCLUDED.error,
    attempts = EXCLUDED.attempts,
    worker = EXCLUDED.worker,
    completed_at = EXCLUDED.completed_at
WHERE tasks.status = 'pending'
"""

FETCH = """
SELECT task_id, source, status, win_probability, model_version,
       error, attempts, published_at, completed_at
FROM tasks
WHERE task_id = $1
"""


async def ensure_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection, connection.transaction():
        await connection.execute("SELECT pg_advisory_xact_lock($1)", SCHEMA_LOCK)
        await connection.execute(SCHEMA)
    log.info("Таблица tasks готова")


class TaskRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def insert_pending(self, message: TaskMessage) -> None:
        await self._pool.execute(
            INSERT_PENDING,
            message.task_id,
            message.source.value,
            message.published_at,
        )

    async def save_result(
        self,
        message: TaskMessage,
        win_probability: float,
        model_version: str,
        attempt: int,
        worker: str,
    ) -> bool:
        saved = await self._pool.fetchval(
            SAVE_RESULT,
            message.task_id,
            message.source.value,
            win_probability,
            model_version,
            attempt,
            worker,
            message.published_at,
        )
        return saved is not None

    async def mark_failed(
        self,
        task_id: str,
        source: TaskSource,
        error: str,
        attempt: int,
        worker: str,
    ) -> None:
        await self._pool.execute(MARK_FAILED, task_id, source.value, error, attempt, worker)

    async def fetch(self, task_id: str) -> TaskView | None:
        row = await self._pool.fetchrow(FETCH, task_id)
        return None if row is None else TaskView.model_validate(dict(row))
