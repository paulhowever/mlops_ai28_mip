import asyncio
import logging
from collections import Counter

from opentelemetry import trace
from opentelemetry.trace import Link, Status, StatusCode

from dota_winprob.agent import metrics
from dota_winprob.agent.actions.models import (
    REQUEUE_DLQ,
    ActionResult,
    ActionStatus,
    PlannedTask,
    RequeuePlan,
)
from dota_winprob.agent.actions.store import Action, ActionStore
from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.agent.sources.dead_letters import DeadLetters, Inspection
from dota_winprob.agent.tracing import current_span_context, current_trace_id
from dota_winprob.clients.broker import DEAD_LETTER_QUEUE_NAME, QUEUE_NAME

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

EXECUTE_SPAN = f"execute_action {REQUEUE_DLQ}"
INTERRUPTED = "возврат прерван остановкой агента"
REPEAT_NOTE = "Задача вернётся в DLQ, если причина отказа не устранена."
SCAN_NOTE = "Просмотрено только начало DLQ: остальные сообщения план не затрагивает."
LIMIT_NOTE = "Валидных задач больше, чем разрешает limit: остальные останутся в DLQ."


def count(action: Action, result: str) -> None:
    metrics.actions.add(1, {"action": action.kind, "result": result})


def expired(action: Action) -> None:
    count(action, "expired")


def build_plan(inspection: Inspection, limit: int) -> tuple[RequeuePlan, Counter[tuple[str, str]]]:
    chosen = inspection.tasks[:limit]
    notes = [REPEAT_NOTE]
    if inspection.depth > inspection.scanned:
        notes.append(SCAN_NOTE)
    if len(inspection.tasks) > limit:
        notes.append(LIMIT_NOTE)
    plan = RequeuePlan(
        queue=DEAD_LETTER_QUEUE_NAME,
        target=QUEUE_NAME,
        depth=inspection.depth,
        scanned=inspection.scanned,
        valid=len(inspection.tasks),
        garbage=inspection.garbage,
        tasks=[
            PlannedTask(
                task_id=task.task_id,
                source=task.source,
                reason=task.reason,
                attempts=task.attempts,
            )
            for task in chosen
        ],
        notes=notes,
    )
    return plan, Counter(task.key for task in chosen)


class ActionService:
    def __init__(self, store: ActionStore, dead_letters: DeadLetters) -> None:
        self.store = store
        self._dead_letters = dead_letters
        self._running: set[asyncio.Task[Action]] = set()

    async def plan_requeue(self, limit: int) -> tuple[RequeuePlan, Action | None]:
        plan, keys = build_plan(await self._dead_letters.inspect(), limit)
        if not plan.tasks:
            return plan, None
        action = self.store.propose(REQUEUE_DLQ, plan, keys, current_span_context())
        count(action, "proposed")
        log.info("Создано предложение вернуть задачи из DLQ", extra={"tasks": len(plan.tasks)})
        return plan, action

    def reject(self, action_id: str) -> Action:
        action = self.store.decide(action_id, ActionStatus.rejected)
        count(action, "rejected")
        log.info("Предложение отклонено человеком")
        return action

    async def confirm(self, action_id: str) -> Action:
        action = self.store.decide(action_id, ActionStatus.executing)
        count(action, "confirmed")
        running = asyncio.create_task(self._execute(action))
        self._running.add(running)
        running.add_done_callback(self._running.discard)
        return await asyncio.shield(running)

    async def close(self, timeout_s: float) -> None:
        if self._running:
            await asyncio.wait(self._running, timeout=timeout_s)
        for running in list(self._running):
            running.cancel()
        await asyncio.gather(*self._running, return_exceptions=True)

    async def _execute(self, action: Action) -> Action:
        with tracer.start_as_current_span(
            EXECUTE_SPAN,
            links=[Link(action.origin)] if action.origin is not None else [],
            attributes={"agent.action.kind": action.kind},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            action.execution_trace_id = current_trace_id()
            result = ActionResult(stopped=INTERRUPTED)
            try:
                result = await self._requeue(action)
            finally:
                action.result = result
                action.status = _status(result)
                returned = len(result.returned)
                count(action, action.status.value)
                metrics.returned_tasks.add(returned)
                span.set_attribute("agent.action.result", action.status.value)
                span.set_attribute("agent.action.returned", returned)
                if action.status is not ActionStatus.done:
                    span.set_status(Status(StatusCode.ERROR, action.status.value))
            log.info(
                "Предложение выполнено",
                extra={
                    "result": action.status.value,
                    "returned": returned,
                    "reason": result.stopped,
                },
            )
            return action

    async def _requeue(self, action: Action) -> ActionResult:
        try:
            outcome = await self._dead_letters.requeue(action.keys)
        except SourceUnavailable as error:
            return ActionResult(stopped=f"RabbitMQ недоступен: {error.reason}")
        return ActionResult(
            returned=outcome.task_ids,
            missing=outcome.missing,
            uncertain=outcome.uncertain,
            stopped=outcome.stopped,
            dlq_left=outcome.left,
        )


def _status(result: ActionResult) -> ActionStatus:
    if result.stopped is None:
        return ActionStatus.done
    return ActionStatus.partial if result.returned or result.uncertain else ActionStatus.failed
