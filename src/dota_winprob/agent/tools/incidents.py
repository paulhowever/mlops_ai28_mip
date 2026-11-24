from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field

from dota_winprob.agent import queries, spans
from dota_winprob.agent.envelope import ToolResult
from dota_winprob.agent.errors import SourceUnavailable, ToolInputError
from dota_winprob.agent.sources.loki import Loki, grouped
from dota_winprob.agent.sources.tasks_db import TasksReader
from dota_winprob.agent.sources.tempo import Tempo
from dota_winprob.agent.tools.monitoring import Minutes

SCAN_LINES = 300
NO_FAILED = (
    "статус failed не записывается, если сбой был в самом Postgres: проверь dlq_ready в pipeline_status "
    "и search_logs service=dota-worker level=WARNING"
)

REPUBLISHED = (
    "задачу публиковали несколько раз: показан трейс, после которого записан результат, остальные — в "
    "other_trace_ids"
)

LogService = Literal[
    "dota-api",
    "dota-worker",
    "dota-source",
    "dota-agent",
    "rabbitmq",
    "postgres",
    "mlflow",
    "minio",
    "grafana",
    "prometheus",
    "loki",
    "tempo",
    "alloy",
    "otel-collector",
]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
TaskId = Annotated[
    str, Field(max_length=200, description="Идентификатор задачи, например batch:566.")
]
Limit = Annotated[int, Field(ge=1, le=50, description="Сколько записей вернуть.")]


@dataclass(frozen=True)
class IncidentTools:
    loki: Loki
    tempo: Tempo
    tasks: TasksReader

    async def search_logs(
        self,
        service: Annotated[
            LogService | None, Field(description="Сервис. Без него — все сервисы.")
        ] = None,
        level: Annotated[
            LogLevel | None,
            Field(description="Минимальный уровень: ERROR вернёт и CRITICAL."),
        ] = None,
        task_id: TaskId | None = None,
        text: Annotated[
            str | None,
            Field(
                max_length=1000,
                description="Подстрока в строке лога без учёта регистра, обычный текст, не regex.",
            ),
        ] = None,
        minutes: Minutes = 15,
        limit: Limit = 10,
    ) -> ToolResult:
        query = queries.logql(service, level, task_id, text)
        found = await self.loki.search(query, minutes, SCAN_LINES)
        data: dict[str, Any] = {
            "minutes": minutes,
            "scanned": len(found),
            "scan_limit_reached": len(found) >= SCAN_LINES,
            "groups": grouped(found),
            "order": "сначала новые",
            "entries": found[:limit],
        }
        if not found:
            return ToolResult("loki", data, f"за {minutes} мин записей нет", "empty")
        more = " или больше" if data["scan_limit_reached"] else ""
        return ToolResult("loki", data, f"записей: {len(found)}{more} за {minutes} мин")

    async def get_task(self, task_id: TaskId) -> ToolResult:
        task_id = queries.task_id(task_id)
        row = await self.tasks.task(task_id)
        if row is None:
            return ToolResult(
                "postgres",
                {"task_id": task_id, "found": False},
                f"задачи {task_id} нет в таблице tasks",
                "empty",
            )
        return ToolResult("postgres", row, f"{task_id}: {row['status']}, попыток {row['attempts']}")

    async def failed_tasks(self, limit: Limit = 10) -> ToolResult:
        rows = await self.tasks.failed(limit)
        totals = await self.tasks.totals()
        data = {"totals_by_status": totals, "order": "сначала новые", "tasks": rows}
        if not rows:
            data["note"] = NO_FAILED
            return ToolResult("postgres", data, "задач failed нет", "empty")
        return ToolResult("postgres", data, f"failed всего {totals.get('failed', len(rows))}")

    async def get_trace(
        self,
        trace_id: Annotated[
            str | None, Field(max_length=200, description="trace_id, 32 шестнадцатеричных символа.")
        ] = None,
        task_id: TaskId | None = None,
    ) -> ToolResult:
        others: list[str] = []
        if trace_id is not None:
            chosen = queries.trace_id(trace_id)
        elif task_id is not None:
            chosen, *others = await self._traces_of(queries.task_id(task_id)) or [None]
        else:
            raise ToolInputError("нужен trace_id или task_id")
        payload = await self.tempo.fetch(chosen) if chosen else None
        if payload is None:
            return ToolResult("tempo", {"found": False}, "трейс не найден", "empty")
        data = spans.tree(chosen, payload)
        if others:
            data = {"note": REPUBLISHED, **data, "other_trace_ids": others}
        if not data["spans"]:
            return ToolResult("tempo", data, "трейс пуст", "empty")
        summary = f"спанов {data['span_count']}, ошибок {data['error_count']}, всего {data['total_ms']} мс"
        return ToolResult("tempo", data, summary)

    async def _traces_of(self, task_id: str) -> list[str]:
        found = await self.tempo.find_by_task(task_id)
        if len(found) > 1:
            completed = await self._completed_at(task_id)
            if completed is not None:
                completed_ns = completed.timestamp() * 1e9
                found.sort(key=lambda trace: trace.started_ns > completed_ns)
        return [trace.trace_id for trace in found]

    async def _completed_at(self, task_id: str) -> datetime | None:
        try:
            row = await self.tasks.task(task_id)
        except SourceUnavailable:
            return None
        return row["completed_at"] if row else None
