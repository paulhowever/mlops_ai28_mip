from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

REQUEUE_DLQ = "requeue_dlq"
MAX_REQUEUE = 50


class ActionStatus(StrEnum):
    proposed = "proposed"
    executing = "executing"
    done = "done"
    partial = "partial"
    failed = "failed"
    rejected = "rejected"
    expired = "expired"


class PlannedTask(BaseModel):
    task_id: str
    source: str
    reason: str = Field(description="Почему RabbitMQ перевёл задачу в DLQ.")
    attempts: int | None = Field(description="Номер последней попытки воркера.")


class RequeuePlan(BaseModel):
    queue: str
    target: str
    depth: int = Field(description="Сообщений в DLQ на момент осмотра.")
    scanned: int = Field(description="Сколько из них просмотрено.")
    valid: int = Field(description="Сколько просмотренных разбирается как задача.")
    garbage: int = Field(description="Сколько просмотренных не задача: они останутся в DLQ.")
    tasks: list[PlannedTask] = Field(description="Что вернётся в рабочую очередь.")
    notes: list[str] = Field(default_factory=list)


class ActionResult(BaseModel):
    returned: list[str] = Field(default_factory=list, description="Вернулись в рабочую очередь.")
    missing: int = Field(default=0, description="Сколько показанных в плане уже нет в DLQ.")
    uncertain: list[str] = Field(
        default_factory=list,
        description="Без подтверждения брокера: могли попасть в очередь и остаться в DLQ.",
    )
    stopped: str | None = Field(default=None, description="Почему возврат остановлен.")
    dlq_left: int | None = Field(default=None, description="Осталось сообщений в DLQ.")


class ActionView(BaseModel):
    id: str
    kind: str
    status: ActionStatus
    plan: RequeuePlan
    created_at: datetime
    expires_at: datetime
    decided_at: datetime | None = None
    result: ActionResult | None = None
    trace_id: str | None = Field(default=None, description="Трейс вопроса, создавшего предложение.")
    execution_trace_id: str | None = None
