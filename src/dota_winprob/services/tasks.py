import asyncio
import logging
import math
import uuid
from datetime import UTC, datetime
from typing import Protocol

import asyncpg
from faststream.rabbit import RabbitBroker

from dota_winprob.clients import broker as broker_client
from dota_winprob.config import Settings
from dota_winprob.schemas import MatchState, TaskMessage, TaskSource

log = logging.getLogger(__name__)


class Overloaded(Exception):
    def __init__(self, retry_after: int, reason: str) -> None:
        super().__init__(reason)
        self.retry_after = retry_after
        self.reason = reason


class PendingWriter(Protocol):
    async def insert_pending(self, message: TaskMessage) -> None: ...


def estimate_retry_after(messages: int, consumers: int, settings: Settings) -> int:
    if consumers <= 0:
        return settings.retry_after_max
    throughput = consumers * settings.worker_concurrency / max(settings.inference_delay, 0.001)
    return max(1, min(settings.retry_after_max, math.ceil(messages / throughput)))


async def retry_after(broker: RabbitBroker, settings: Settings) -> int:
    try:
        async with asyncio.timeout(settings.publish_timeout):
            messages, consumers = await broker_client.queue_load(broker)
    except broker_client.PUBLISH_ERRORS:
        return settings.retry_after_max
    return estimate_retry_after(messages, consumers, settings)


async def submit(
    broker: RabbitBroker,
    broker_ready: bool,
    repository: PendingWriter,
    settings: Settings,
    features: MatchState,
    correlation_id: str | None = None,
) -> TaskMessage:
    if not broker_ready:
        raise Overloaded(math.ceil(settings.broker_reconnect_interval), "нет связи с брокером")

    message = TaskMessage(
        task_id=uuid.uuid4().hex,
        source=TaskSource.api,
        features=features,
        published_at=datetime.now(UTC),
    )
    accepted = await broker_client.publish_task(
        broker,
        message,
        confirm_timeout=settings.publish_timeout,
        correlation_id=correlation_id,
    )
    if not accepted:
        raise Overloaded(await retry_after(broker, settings), "очередь задач заполнена")

    try:
        await repository.insert_pending(message)
    except (asyncpg.PostgresError, OSError, TimeoutError) as error:
        log.warning(
            "Задача в очереди, но строка pending не записана: результат запишет воркер",
            extra={"task_id": message.task_id, "error": repr(error)},
        )
    return message
