import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NoReturn, Protocol

from faststream.exceptions import RejectMessage
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import ValidationError

from dota_winprob import instruments
from dota_winprob.clients.broker import ATTEMPT_HEADER
from dota_winprob.logging_config import bind, log_context
from dota_winprob.schemas import TaskMessage, TaskSource
from dota_winprob.services.scoring import ATTEMPT_ATTRIBUTE, Scorer

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

RetryPublisher = Callable[[bytes, Mapping[str, Any], str, str | None], Awaitable[bool]]

TASK_ID_ATTRIBUTE = "task.id"
DB_ATTRIBUTES = {"db.system.name": "postgresql", "db.collection.name": "tasks"}


class Repository(Protocol):
    async def save_result(
        self,
        message: TaskMessage,
        win_probability: float,
        model_version: str,
        attempt: int,
        worker: str,
    ) -> bool: ...

    async def mark_failed(
        self,
        task_id: str,
        source: TaskSource,
        error: str,
        attempt: int,
        worker: str,
    ) -> None: ...


def _attempt(headers: Mapping[str, Any]) -> int:
    try:
        return max(1, int(headers.get(ATTEMPT_HEADER, 1)))
    except (TypeError, ValueError):
        return 1


def _identity(body: bytes) -> tuple[str, TaskSource] | None:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    task_id, source = payload.get("task_id"), payload.get("source")
    if not isinstance(task_id, str) or source not in set(TaskSource):
        return None
    return task_id, TaskSource(source)


def _describe(error: BaseException) -> str:
    message = str(error)
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message.splitlines()[0]}"


def _mark_error(span: Span, error: BaseException, attempt: int) -> None:
    span.record_exception(error, attributes={ATTEMPT_ATTRIBUTE: attempt})
    span.set_status(Status(StatusCode.ERROR, _describe(error)))


def _dead_letter(span: Span, reason: str, attempt: int) -> None:
    span.add_event("dlq", {"reason": reason, ATTEMPT_ATTRIBUTE: attempt})
    span.set_attribute("task.dead_lettered", True)
    span.set_status(Status(StatusCode.ERROR, reason))
    instruments.worker_tasks.add(1, {"result": "dlq"})


@dataclass
class Pipeline:
    repository: Repository
    scorer: Scorer
    publish_retry: RetryPublisher
    max_attempts: int
    worker: str
    in_flight: int = 0
    processed: int = 0
    duplicates: int = 0
    retried: int = 0
    failed: int = 0

    async def handle(
        self,
        body: bytes,
        headers: Mapping[str, Any],
        correlation_id: str | None = None,
        *,
        redelivered: bool = False,
    ) -> None:
        attempt = _attempt(headers)
        span = trace.get_current_span()
        span.set_attribute(ATTEMPT_ATTRIBUTE, attempt)
        if attempt > 1:
            instruments.redeliveries.add(1, {"reason": "retry"})
        if redelivered:
            instruments.redeliveries.add(1, {"reason": "redelivered"})
            span.set_attribute("messaging.rabbitmq.redelivered", True)

        with log_context(attempt=attempt):
            try:
                message = TaskMessage.model_validate_json(body)
            except ValidationError as error:
                await self._reject_invalid(body, error, attempt)

            bind(task_id=message.task_id)
            span.set_attribute(TASK_ID_ATTRIBUTE, message.task_id)
            waited = (datetime.now(UTC) - message.published_at).total_seconds()
            instruments.queue_wait.record(max(0.0, waited))

            self.in_flight += 1
            instruments.in_flight.add(1)
            try:
                await self._process(message, body, attempt, correlation_id)
            finally:
                self.in_flight -= 1
                instruments.in_flight.add(-1)

    async def _process(
        self,
        message: TaskMessage,
        body: bytes,
        attempt: int,
        correlation_id: str | None,
    ) -> None:
        try:
            probability = await self.scorer.score(message.features, attempt)
            with tracer.start_as_current_span(
                "db save_result",
                kind=trace.SpanKind.CLIENT,
                attributes={**DB_ATTRIBUTES, "db.operation.name": "save_result"},
            ):
                saved = await self.repository.save_result(
                    message,
                    probability,
                    self.scorer.model.version,
                    attempt,
                    self.worker,
                )
        except Exception as error:
            final = attempt >= self.max_attempts
            _mark_error(trace.get_current_span(), error, attempt)
            log.log(
                logging.ERROR if final else logging.WARNING,
                "Задача не обработана",
                exc_info=error,
                extra={"max_attempts": self.max_attempts},
            )
            await self._retry_or_fail(message, body, attempt, correlation_id, error)
            return

        if saved:
            self.processed += 1
            instruments.worker_tasks.add(1, {"result": "ok"})
            log.debug("Задача выполнена", extra={"win_probability": probability})
        else:
            self.duplicates += 1
            instruments.worker_tasks.add(1, {"result": "duplicate"})
            log.info("Повторная доставка: результат уже записан")

    async def _retry_or_fail(
        self,
        message: TaskMessage,
        body: bytes,
        attempt: int,
        correlation_id: str | None,
        error: BaseException,
    ) -> None:
        span = trace.get_current_span()
        if attempt >= self.max_attempts:
            await self._fail(message, _describe(error), attempt)
            _dead_letter(span, "попытки исчерпаны", attempt)
            log.error("Попытки исчерпаны, задача уходит в DLQ")
            raise RejectMessage

        retried = await self.publish_retry(
            body,
            {ATTEMPT_HEADER: attempt + 1},
            message.task_id,
            correlation_id,
        )
        if not retried:
            await self._fail(message, f"очередь повторов недоступна: {_describe(error)}", attempt)
            _dead_letter(span, "очередь повторов не приняла задачу", attempt)
            log.error("Очередь повторов не приняла задачу, она уходит в DLQ")
            raise RejectMessage

        self.retried += 1
        instruments.worker_tasks.add(1, {"result": "error"})
        span.add_event("retry", {"next_attempt": attempt + 1})
        log.warning("Задача отправлена на повтор", extra={"next_attempt": attempt + 1})

    async def _mark_failed(
        self, task_id: str, source: TaskSource, error: str, attempt: int
    ) -> None:
        with tracer.start_as_current_span(
            "db mark_failed",
            kind=trace.SpanKind.CLIENT,
            attributes={**DB_ATTRIBUTES, "db.operation.name": "mark_failed"},
        ):
            await self.repository.mark_failed(task_id, source, error, attempt, self.worker)

    async def _fail(self, message: TaskMessage, error: str, attempt: int) -> None:
        self.failed += 1
        await self._mark_failed(message.task_id, message.source, error, attempt)

    async def _reject_invalid(
        self,
        body: bytes,
        error: ValidationError,
        attempt: int,
    ) -> NoReturn:
        self.failed += 1
        identity = _identity(body)
        span = trace.get_current_span()
        if identity is not None:
            bind(task_id=identity[0])
            span.set_attribute(TASK_ID_ATTRIBUTE, identity[0])
        _mark_error(span, error, attempt)
        _dead_letter(span, "невалидное сообщение", attempt)
        log.error(
            "Невалидное сообщение уходит в DLQ без повторов",
            extra={"error": _describe(error)},
        )
        if identity is not None:
            task_id, source = identity
            await self._mark_failed(
                task_id,
                source,
                f"невалидное сообщение: {_describe(error)}",
                attempt,
            )
        raise RejectMessage
