import asyncio
import hashlib
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

import aio_pika
from aio_pika.abc import AbstractChannel, AbstractExchange, AbstractIncomingMessage, AbstractQueue
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.trace import Link, SpanKind, Status, StatusCode
from pamqp.commands import Basic
from pydantic import ValidationError

from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.clients.broker import (
    ATTEMPT_HEADER,
    DEAD_LETTER_QUEUE_NAME,
    PUBLISH_ERRORS,
    QUEUE_NAME,
    ROUTING_KEY,
    TASKS_EXCHANGE,
)
from dota_winprob.schemas import TaskMessage

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

NAME = "rabbitmq"
RETURN_SPAN = f"{QUEUE_NAME} publish"
DELIVERY_COUNT = "x-delivery-count"
DEATH_REASON = "x-last-death-reason"
RISKY_DELIVERIES = 15
RPC_PATIENCE = 3
UNKNOWN_REASON = "unknown"
DEATH_REASONS = frozenset({"rejected", "expired", "maxlen", "delivery_limit"})
BROKER_ERRORS = (*PUBLISH_ERRORS, OSError, RuntimeError)

Key = tuple[str, str]
T = TypeVar("T")


@dataclass(frozen=True)
class DeadTask:
    key: Key
    source: str
    reason: str
    attempts: int | None

    @property
    def task_id(self) -> str:
        return self.key[0]


@dataclass(frozen=True)
class Inspection:
    depth: int
    scanned: int
    garbage: int
    tasks: list[DeadTask]


@dataclass
class Returned:
    task_ids: list[str] = field(default_factory=list)
    missing: int = 0
    uncertain: list[str] = field(default_factory=list)
    stopped: str | None = None
    left: int | None = None


class MessageAtRisk(Exception):
    pass


def _int(value: object) -> int | None:
    try:
        return int(str(value))
    except ValueError:
        return None


def _deliveries(message: AbstractIncomingMessage) -> int:
    return _int(message.headers.get(DELIVERY_COUNT, 0)) or 0


def identify(message: AbstractIncomingMessage) -> DeadTask | None:
    try:
        task = TaskMessage.model_validate_json(message.body)
    except ValidationError:
        return None
    reason = str(message.headers.get(DEATH_REASON))
    return DeadTask(
        key=(task.task_id, hashlib.sha256(message.body).hexdigest()),
        source=task.source.value,
        reason=reason if reason in DEATH_REASONS else UNKNOWN_REASON,
        attempts=_int(message.headers.get(ATTEMPT_HEADER, 1)),
    )


def _origin(headers: Mapping[str, object]) -> list[Link]:
    carrier = {
        key: value
        for key, value in headers.items()
        if isinstance(key, str) and isinstance(value, str)
    }
    original = trace.get_current_span(propagate.extract(carrier, Context())).get_span_context()
    current = trace.get_current_span().get_span_context()
    return [Link(context) for context in (original, current) if context.is_valid]


class DeadLetters:
    def __init__(
        self,
        url: str,
        *,
        scan_limit: int,
        timeout: float,
        publish_timeout: float,
        execution_timeout: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = url
        self._scan_limit = scan_limit
        self._timeout = timeout
        self._publish_timeout = publish_timeout
        self._execution_timeout = execution_timeout
        self._rpc_timeout = timeout * RPC_PATIENCE
        self._clock = clock
        self._running: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()

    async def inspect(self) -> Inspection:
        return await self._detached(self._inspect())

    async def requeue(self, planned: Counter[Key]) -> Returned:
        return await self._detached(self._return_planned(Counter(planned)))

    async def close(self, timeout_s: float) -> None:
        if self._running:
            await asyncio.wait(self._running, timeout=timeout_s)
        for running in list(self._running):
            running.cancel()
        await asyncio.gather(*self._running, return_exceptions=True)

    async def _detached(self, operation: Coroutine[None, None, T]) -> T:
        running = asyncio.create_task(operation)
        self._running.add(running)
        running.add_done_callback(self._forget)
        return await asyncio.shield(running)

    def _forget(self, running: asyncio.Task[Any]) -> None:
        self._running.discard(running)
        if not running.cancelled() and running.exception() is not None:
            log.debug("Операция с DLQ не удалась", extra={"reason": str(running.exception())})

    async def _inspect(self) -> Inspection:
        async with self._lock, self._session() as (channel, queue):
            depth = queue.declaration_result.message_count or 0
            async with self._returning(channel):
                messages = await self._acquire(queue, depth)
                if any(_deliveries(message) >= RISKY_DELIVERIES for message in messages):
                    raise MessageAtRisk
                found = [identify(message) for message in messages]
        tasks = [task for task in found if task is not None]
        return Inspection(depth, len(found), len(found) - len(tasks), tasks)

    async def _return_planned(self, wanted: Counter[Key]) -> Returned:
        outcome = Returned()
        async with self._lock, self._session() as (channel, queue):
            depth = queue.declaration_result.message_count or 0
            try:
                exchange = await channel.get_exchange(TASKS_EXCHANGE.name, ensure=True)
                async with self._returning(channel):
                    messages = await self._acquire(queue, depth)
                    await self._requeue(channel, exchange, messages, wanted, outcome)
            except BROKER_ERRORS as error:
                log.warning("Возврат из DLQ прерван", extra={"reason": type(error).__name__})
                outcome.stopped = "связь с RabbitMQ потеряна"
        outcome.left = max(0, depth - len(outcome.task_ids))
        if outcome.stopped is None:
            outcome.missing = wanted.total()
        return outcome

    async def _acquire(self, queue: AbstractQueue, depth: int) -> list[AbstractIncomingMessage]:
        messages: list[AbstractIncomingMessage] = []
        stop_at = self._clock() + self._timeout
        while len(messages) < min(depth, self._scan_limit) and self._clock() < stop_at:
            message = await queue.get(fail=False, timeout=self._rpc_timeout)
            if message is None:
                break
            messages.append(message)
        return messages

    async def _requeue(
        self,
        channel: AbstractChannel,
        exchange: AbstractExchange,
        messages: list[AbstractIncomingMessage],
        wanted: Counter[Key],
        outcome: Returned,
    ) -> None:
        stop_at = self._clock() + self._execution_timeout
        for message in messages:
            task = identify(message)
            if task is None or not wanted[task.key]:
                continue
            if self._clock() >= stop_at:
                outcome.stopped = "вышло время на возврат"
                return
            outcome.stopped = await self._publish(exchange, message, task, outcome)
            if outcome.stopped is not None:
                return
            wanted[task.key] -= 1
            outcome.task_ids.append(task.task_id)
            await self._settle(channel, message, task, outcome)

    async def _settle(
        self,
        channel: AbstractChannel,
        message: AbstractIncomingMessage,
        task: DeadTask,
        outcome: Returned,
    ) -> None:
        try:
            await message.ack()
            await channel.declare_queue(
                DEAD_LETTER_QUEUE_NAME, passive=True, timeout=self._rpc_timeout
            )
        except BROKER_ERRORS:
            outcome.uncertain.append(task.task_id)
            raise

    async def _publish(
        self,
        exchange: AbstractExchange,
        message: AbstractIncomingMessage,
        task: DeadTask,
        outcome: Returned,
    ) -> str | None:
        with tracer.start_as_current_span(
            RETURN_SPAN,
            context=Context(),
            kind=SpanKind.PRODUCER,
            links=_origin(message.headers),
            attributes={
                "messaging.system": "rabbitmq",
                "messaging.destination.name": QUEUE_NAME,
                "messaging.operation.name": "publish",
                "task.id": task.task_id,
                "task.source": task.source,
                "task.returned_from_dlq": True,
            },
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            headers: dict[str, str] = {}
            propagate.inject(headers)
            copy = aio_pika.Message(
                message.body,
                headers=dict(headers),
                message_id=task.task_id,
                correlation_id=message.correlation_id,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            try:
                confirmation = await exchange.publish(
                    copy, routing_key=ROUTING_KEY, mandatory=True, timeout=self._publish_timeout
                )
            except aio_pika.exceptions.PublishError:
                refusal = "рабочей очереди нет"
            except aio_pika.exceptions.DeliveryError:
                refusal = "брокер отклонил публикацию: рабочая очередь заполнена"
            except TimeoutError:
                outcome.uncertain.append(task.task_id)
                refusal = "брокер не подтвердил публикацию вовремя"
            except (*BROKER_ERRORS, asyncio.CancelledError):
                outcome.uncertain.append(task.task_id)
                span.set_status(Status(StatusCode.ERROR, "connection"))
                raise
            else:
                accepted = isinstance(confirmation, Basic.Ack)
                refusal = None if accepted else "рабочая очередь не принимает сообщения"
            if refusal is not None:
                span.set_status(Status(StatusCode.ERROR, refusal))
            return refusal

    @asynccontextmanager
    async def _returning(self, channel: AbstractChannel) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await asyncio.shield(asyncio.ensure_future(self._return_all(channel)))

    async def _return_all(self, channel: AbstractChannel) -> None:
        try:
            underlay = await channel.get_underlay_channel()
            await underlay.basic_nack(delivery_tag=0, multiple=True, requeue=True)
        except BROKER_ERRORS as error:
            log.warning(
                "Осмотренные сообщения вернёт брокер при закрытии соединения",
                extra={"reason": type(error).__name__},
            )

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[tuple[AbstractChannel, AbstractQueue]]:
        try:
            connection = await aio_pika.connect(
                self._url,
                timeout=self._timeout,
                client_properties={"connection_name": "dota-agent-dlq"},
            )
        except BROKER_ERRORS as error:
            raise SourceUnavailable(NAME, type(error).__name__) from error
        try:
            async with asyncio.timeout(self._timeout):
                channel = await connection.channel(publisher_confirms=True, on_return_raises=True)
                queue = await channel.declare_queue(DEAD_LETTER_QUEUE_NAME, passive=True)
            yield channel, queue
        except BROKER_ERRORS as error:
            raise SourceUnavailable(NAME, type(error).__name__) from error
        finally:
            await asyncio.shield(asyncio.ensure_future(_close(connection)))


async def _close(connection: aio_pika.abc.AbstractConnection) -> None:
    try:
        async with asyncio.timeout(2):
            await connection.close()
    except BROKER_ERRORS as error:
        log.debug("Соединение уже закрыто", extra={"reason": type(error).__name__})
