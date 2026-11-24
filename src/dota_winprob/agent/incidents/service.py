import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import partial

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from dota_winprob.agent import metrics
from dota_winprob.agent.config import AgentSettings
from dota_winprob.agent.envelope import ToolResult, clip, fit, render
from dota_winprob.agent.errors import AskError, LLMNotConfigured
from dota_winprob.agent.incidents.models import (
    Alert,
    AlertsAccepted,
    AlertsPayload,
    AnalysisStatus,
    Fact,
    Incident,
    IncidentAnalysis,
    IncidentStatus,
)
from dota_winprob.agent.incidents.store import IncidentStore
from dota_winprob.agent.llm.gate import background
from dota_winprob.agent.prompt import INCIDENT_QUESTION
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.service import AgentService
from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.agent.tools.guard import safely
from dota_winprob.agent.tools.incidents import IncidentTools
from dota_winprob.agent.tools.monitoring import MonitoringTools
from dota_winprob.agent.tracing import current_trace_id

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

ANALYZE_SPAN = "analyze_incident"
FACT_CHARS = 1500
SUMMARY_CHARS = 300
FACT_MINUTES = 15
FACT_LIMIT = 5
DEAD_LETTERS = "tasks.dlq"
OVERFLOW = "очередь разборов полна"
RESOLVED_EARLY = "алерт погас до разбора"
CRASHED = "разбор прерван ошибкой"
STOPPED = "агент остановлен до конца разбора"


class IncidentService:
    def __init__(
        self,
        settings: AgentSettings,
        sources: Sources,
        redactor: Redactor,
        answers: AgentService,
        store: IncidentStore,
    ) -> None:
        self._settings = settings
        self._redactor = redactor
        self._answers = answers
        self._monitoring = MonitoringTools(sources.prometheus)
        self._tools = IncidentTools(sources.loki, sources.tempo, sources.tasks)
        self._queue: asyncio.Queue[Incident] = asyncio.Queue(settings.agent_incident_queue)
        self._worker: asyncio.Task[None] | None = None
        self.store = store

    def start(self) -> None:
        self._worker = asyncio.create_task(self._work(), name="incident-analysis")

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    async def drain(self) -> None:
        await self._queue.join()

    def receive(self, payload: AlertsPayload) -> AlertsAccepted:
        accepted = AlertsAccepted()
        for alert in payload.alerts:
            result = self._resolved(alert) if alert.status == "resolved" else self._firing(alert)
            setattr(accepted, result, getattr(accepted, result) + 1)
            metrics.alerts.add(1, {"result": result})
        log.info("Алерты приняты", extra=accepted.model_dump())
        return accepted

    def _resolved(self, alert: Alert) -> str:
        incident = self.store.latest(alert.fingerprint)
        if incident is None or incident.status is IncidentStatus.resolved:
            return "unknown"
        self.store.resolve(incident)
        return "resolved"

    def _firing(self, alert: Alert) -> str:
        incident = self.store.latest(alert.fingerprint)
        if incident is not None and incident.status is IncidentStatus.firing:
            self.store.repeat(incident)
            return "repeated"
        if incident is not None and self.store.cooling(incident):
            self.store.repeat(incident)
            return "cooldown"
        return self._enqueue(self.store.open(alert))

    def _enqueue(self, incident: Incident) -> str:
        try:
            self._queue.put_nowait(incident)
        except asyncio.QueueFull:
            self._finish(
                incident, IncidentAnalysis(status=AnalysisStatus.skipped, reason=OVERFLOW), 0.0
            )
            return "overflow"
        return "opened"

    async def _work(self) -> None:
        while True:
            incident = await self._queue.get()
            started = time.perf_counter()
            analysis = IncidentAnalysis(status=AnalysisStatus.failed, reason=STOPPED)
            try:
                incident.analysis.status = AnalysisStatus.running
                analysis = await self._analyze(incident)
            except Exception:
                log.exception("Разбор инцидента прерван")
                analysis = IncidentAnalysis(status=AnalysisStatus.failed, reason=CRASHED)
            finally:
                self._finish(incident, analysis, time.perf_counter() - started)
                self._queue.task_done()

    def _finish(self, incident: Incident, analysis: IncidentAnalysis, duration: float) -> None:
        analysis.finished_at = datetime.now(UTC)
        incident.analysis = analysis
        if analysis.status is not AnalysisStatus.skipped:
            self.store.analyzed(incident)
        result = {"result": analysis.status.value}
        metrics.incidents.add(1, result)
        metrics.incident_duration.record(duration, result)
        log.info(
            "Инцидент разобран",
            extra={
                "result": analysis.status.value,
                "reason": analysis.reason,
                "duration_s": round(duration, 3),
            },
        )

    async def _analyze(self, incident: Incident) -> IncidentAnalysis:
        with tracer.start_as_current_span(
            ANALYZE_SPAN,
            attributes={"agent.incident.severity": incident.severity or "unknown"},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            results = await self._collect(incident)
            analysis = IncidentAnalysis(
                facts=[self._fact(tool, result) for tool, result in results],
                trace_id=current_trace_id(),
            )
            if incident.status is IncidentStatus.resolved:
                analysis.status, analysis.reason = AnalysisStatus.facts_only, RESOLVED_EARLY
            else:
                await self._explain(incident, results, analysis)
            span.set_attribute("agent.incident.result", analysis.status.value)
            if analysis.status is AnalysisStatus.failed:
                span.set_status(Status(StatusCode.ERROR, analysis.reason))
            return analysis

    async def _explain(
        self,
        incident: Incident,
        results: list[tuple[str, ToolResult]],
        analysis: IncidentAnalysis,
    ) -> None:
        question = INCIDENT_QUESTION.format(
            alert=self._render("alert", _as_result(incident)),
            facts="\n".join(self._render(tool, result) for tool, result in results),
        )
        try:
            with background():
                answer = await self._answers.analyze(question)
        except LLMNotConfigured as error:
            analysis.status, analysis.reason = AnalysisStatus.facts_only, error.detail
        except AskError as error:
            analysis.status, analysis.reason = AnalysisStatus.failed, error.detail
        else:
            analysis.status = AnalysisStatus.done
            analysis.text = answer.text
            analysis.steps = answer.steps
            analysis.warnings = answer.warnings

    async def _collect(self, incident: Incident) -> list[tuple[str, ToolResult]]:
        metric = "dlq_ready" if incident.labels.get("queue") == DEAD_LETTERS else "queue_ready"
        calls: dict[str, Callable[[], Awaitable[ToolResult]]] = {
            "pipeline_status": self._monitoring.pipeline_status,
            "metric_history": partial(self._monitoring.metric_history, metric, FACT_MINUTES),
            "failed_tasks": partial(self._tools.failed_tasks, FACT_LIMIT),
            "search_logs": partial(
                self._tools.search_logs, level="ERROR", minutes=FACT_MINUTES, limit=FACT_LIMIT
            ),
        }
        results = await asyncio.gather(
            *(safely(tool, call, self._settings.agent_tool_timeout) for tool, call in calls.items())
        )
        for tool, result in zip(calls, results, strict=True):
            metrics.tool_calls.add(1, {"tool": tool, "result": result.status})
        return list(zip(calls, results, strict=True))

    def _fact(self, tool: str, result: ToolResult) -> Fact:
        return Fact(
            tool=tool,
            source=result.source,
            status=result.status,
            summary=clip(self._redactor.text(result.summary), SUMMARY_CHARS),
            data=fit(self._redactor.data(result.data), FACT_CHARS),
        )

    def _render(self, tool: str, result: ToolResult) -> str:
        return render(tool, result, self._redactor, FACT_CHARS)


def _as_result(incident: Incident) -> ToolResult:
    return ToolResult(
        "grafana",
        {
            "title": incident.title,
            "severity": incident.severity,
            "summary": incident.summary,
            "runbook": incident.runbook,
            "opened_at": incident.opened_at,
            "labels": incident.labels,
        },
    )
