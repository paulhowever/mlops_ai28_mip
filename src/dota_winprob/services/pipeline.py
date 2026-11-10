import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

from faststream.exceptions import RejectMessage
from pydantic import ValidationError

from dota_winprob.clients.broker import ATTEMPT_HEADER
from dota_winprob.schemas import TaskMessage, TaskSource
from dota_winprob.services.scoring import Scorer

log = logging.getLogger(__name__)

RetryPublisher = Callable[[bytes, Mapping[str, Any], str, str | None], Awaitable[bool]]


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
    ) -> None:
        attempt = _attempt(headers)
        try:
            message = TaskMessage.model_validate_json(body)
        except ValidationError as error:
            await self._reject_invalid(body, error, attempt)

        self.in_flight += 1
        try:
            await self._process(message, body, attempt, correlation_id)
        finally:
            self.in_flight -= 1

    async def _process(
        self,
        message: TaskMessage,
        body: bytes,
        attempt: int,
        correlation_id: str | None,
    ) -> None:
        context = {"task_id": message.task_id, "attempt": attempt}
        try:
            probability = await self.scorer.score(message.features)
            saved = await self.repository.save_result(
                message,
                probability,
                self.scorer.model.version,
                attempt,
                self.worker,
            )
        except Exception as error:
            log.exception("Задача не обработана", extra=context)
            await self._retry_or_fail(message, body, attempt, correlation_id, error)
            return

        if saved:
            self.processed += 1
            log.debug("Задача выполнена", extra={**context, "win_probability": probability})
        else:
            self.duplicates += 1
            log.info("Повторная доставка: результат уже записан", extra=context)

    async def _retry_or_fail(
        self,
        message: TaskMessage,
        body: bytes,
        attempt: int,
        correlation_id: str | None,
        error: BaseException,
    ) -> None:
        context = {"task_id": message.task_id, "attempt": attempt}
        if attempt >= self.max_attempts:
            await self._fail(message, _describe(error), attempt)
            log.error("Попытки исчерпаны, задача уходит в DLQ", extra=context)
            raise RejectMessage

        retried = await self.publish_retry(
            body,
            {ATTEMPT_HEADER: attempt + 1},
            message.task_id,
            correlation_id,
        )
        if not retried:
            await self._fail(message, f"очередь повторов недоступна: {_describe(error)}", attempt)
            log.error("Очередь повторов не приняла задачу, она уходит в DLQ", extra=context)
            raise RejectMessage

        self.retried += 1
        log.warning("Задача отправлена на повтор", extra=context)

    async def _fail(self, message: TaskMessage, error: str, attempt: int) -> None:
        self.failed += 1
        await self.repository.mark_failed(
            message.task_id,
            message.source,
            error,
            attempt,
            self.worker,
        )

    async def _reject_invalid(
        self,
        body: bytes,
        error: ValidationError,
        attempt: int,
    ) -> NoReturn:
        self.failed += 1
        identity = _identity(body)
        log.error(
            "Невалидное сообщение уходит в DLQ без повторов",
            extra={"task_id": identity[0] if identity else None, "error": _describe(error)},
        )
        if identity is not None:
            task_id, source = identity
            await self.repository.mark_failed(
                task_id,
                source,
                f"невалидное сообщение: {_describe(error)}",
                attempt,
                self.worker,
            )
        raise RejectMessage
