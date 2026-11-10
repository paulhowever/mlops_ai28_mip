import logging
from collections.abc import Mapping
from typing import Any

from aio_pika.exceptions import AMQPError, ChannelInvalidStateError
from faststream.exceptions import FastStreamException
from faststream.rabbit import ExchangeType, QueueType, RabbitBroker, RabbitExchange, RabbitQueue
from faststream.rabbit.opentelemetry import RabbitTelemetryMiddleware
from pamqp.commands import Basic

from dota_winprob.config import Settings
from dota_winprob.schemas import TaskMessage

log = logging.getLogger(__name__)

ROUTING_KEY = "score"
QUEUE_NAME = "tasks.score"
RETRY_QUEUE_NAME = "tasks.retry"
DEAD_LETTER_QUEUE_NAME = "tasks.dlq"
ATTEMPT_HEADER = "x-attempt"
ACCESS_LOGGER = "faststream.access.rabbit"

TASKS_EXCHANGE = RabbitExchange("tasks", type=ExchangeType.DIRECT, durable=True)
DEAD_LETTER_EXCHANGE = RabbitExchange("tasks.dlx", type=ExchangeType.FANOUT, durable=True)

PUBLISH_ERRORS = (TimeoutError, AMQPError, ChannelInvalidStateError, FastStreamException)


def tasks_queue(settings: Settings) -> RabbitQueue:
    arguments: dict[str, Any] = {
        "x-delivery-limit": settings.delivery_limit,
        "x-dead-letter-exchange": DEAD_LETTER_EXCHANGE.name,
    }
    if settings.queue_max_length:
        arguments |= {
            "x-max-length": settings.queue_max_length,
            "x-overflow": "reject-publish",
            "x-dead-letter-strategy": "at-least-once",
        }
    return RabbitQueue(
        QUEUE_NAME,
        queue_type=QueueType.QUORUM,
        durable=True,
        routing_key=ROUTING_KEY,
        arguments=arguments,
    )


def retry_queue(settings: Settings) -> RabbitQueue:
    arguments: dict[str, Any] = {
        "x-message-ttl": settings.retry_delay_ms,
        "x-dead-letter-exchange": TASKS_EXCHANGE.name,
        "x-dead-letter-routing-key": ROUTING_KEY,
        "x-dead-letter-strategy": "at-least-once",
        "x-overflow": "reject-publish",
    }
    if settings.queue_max_length:
        arguments["x-max-length"] = settings.queue_max_length
    return RabbitQueue(
        RETRY_QUEUE_NAME,
        queue_type=QueueType.QUORUM,
        durable=True,
        arguments=arguments,
    )


def dead_letter_queue() -> RabbitQueue:
    return RabbitQueue(DEAD_LETTER_QUEUE_NAME, queue_type=QueueType.QUORUM, durable=True)


def create_broker(settings: Settings) -> RabbitBroker:
    return RabbitBroker(
        settings.rabbitmq_url,
        graceful_timeout=settings.graceful_timeout,
        logger=logging.getLogger(ACCESS_LOGGER),
        log_level=logging.DEBUG,
        middlewares=(RabbitTelemetryMiddleware(),),
    )


async def declare_topology(
    broker: RabbitBroker,
    settings: Settings,
    *,
    with_tasks_queue: bool = True,
) -> None:
    tasks = await broker.declare_exchange(TASKS_EXCHANGE)
    dead_letters = await broker.declare_exchange(DEAD_LETTER_EXCHANGE)

    if with_tasks_queue:
        queue = await broker.declare_queue(tasks_queue(settings))
        await queue.bind(tasks, routing_key=ROUTING_KEY)

    await broker.declare_queue(retry_queue(settings))

    dlq = await broker.declare_queue(dead_letter_queue())
    await dlq.bind(dead_letters)
    log.info(
        "Топология RabbitMQ объявлена",
        extra={
            "queue": QUEUE_NAME,
            "queue_max_length": settings.queue_max_length,
            "delivery_limit": settings.delivery_limit,
        },
    )


async def stop_consuming(broker: RabbitBroker) -> None:
    for subscriber in broker.subscribers:
        queue, tag = subscriber._queue_obj, subscriber._consumer_tag
        if queue is None or tag is None or queue.channel.is_closed:
            continue
        await queue.cancel(tag)
        subscriber._consumer_tag = None
    log.info("Подписка отменена, новые сообщения брокер не отдаёт")


async def connect_and_declare(broker: RabbitBroker, settings: Settings) -> None:
    await broker.connect()
    await declare_topology(broker, settings)


def _accepted(confirmation: object) -> bool:
    return isinstance(confirmation, Basic.Ack)


async def publish_task(
    broker: RabbitBroker,
    message: TaskMessage,
    confirm_timeout: float,
    correlation_id: str | None = None,
) -> bool:
    try:
        confirmation = await broker.publish(
            message.model_dump(mode="json"),
            exchange=TASKS_EXCHANGE,
            routing_key=ROUTING_KEY,
            message_id=message.task_id,
            correlation_id=correlation_id,
            persist=True,
            mandatory=True,
            timeout=confirm_timeout,
        )
    except PUBLISH_ERRORS as error:
        log.debug(
            "Брокер не принял задачу", extra={"task_id": message.task_id, "error": repr(error)}
        )
        return False
    return _accepted(confirmation)


async def publish_retry(
    broker: RabbitBroker,
    body: bytes,
    headers: Mapping[str, Any],
    task_id: str,
    correlation_id: str | None,
    confirm_timeout: float,
) -> bool:
    try:
        confirmation = await broker.publish(
            body,
            queue=RETRY_QUEUE_NAME,
            headers=dict(headers),
            message_id=task_id,
            correlation_id=correlation_id,
            content_type="application/json",
            persist=True,
            mandatory=True,
            timeout=confirm_timeout,
        )
    except PUBLISH_ERRORS:
        return False
    return _accepted(confirmation)


async def queue_load(broker: RabbitBroker) -> tuple[int, int]:
    queue = await broker.declare_queue(RabbitQueue(QUEUE_NAME, declare=False))
    declared = await queue.declare()
    return declared.message_count or 0, declared.consumer_count or 0
