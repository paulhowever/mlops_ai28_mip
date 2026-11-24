from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from dota_winprob.agent.schemas import Step

MAX_ALERTS = 100
MAX_LABELS = 64

Fingerprint = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,64}$")]
Labels = Annotated[dict[str, str], Field(max_length=MAX_LABELS)]


class Alert(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Literal["firing", "resolved"]
    fingerprint: Fingerprint
    labels: Labels = Field(default_factory=dict)
    annotations: Labels = Field(default_factory=dict)


class AlertsPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alerts: list[Alert] = Field(max_length=MAX_ALERTS)


class AlertsAccepted(BaseModel):
    opened: int = 0
    repeated: int = 0
    cooldown: int = 0
    resolved: int = 0
    unknown: int = 0
    overflow: int = Field(default=0, description="Инцидент сохранён, но очередь разборов полна.")


class IncidentStatus(StrEnum):
    firing = "firing"
    resolved = "resolved"


class AnalysisStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    facts_only = "facts_only"
    failed = "failed"
    skipped = "skipped"


class Fact(BaseModel):
    tool: str
    source: str
    status: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class IncidentAnalysis(BaseModel):
    status: AnalysisStatus = AnalysisStatus.queued
    text: str | None = None
    reason: str | None = Field(default=None, description="Почему нет текста модели.")
    facts: list[Fact] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    trace_id: str | None = None
    finished_at: datetime | None = None


class Incident(BaseModel):
    id: str
    fingerprint: str
    title: str
    severity: str | None
    status: IncidentStatus = IncidentStatus.firing
    summary: str | None
    runbook: str | None
    labels: dict[str, str]
    opened_at: datetime
    last_seen_at: datetime
    resolved_at: datetime | None = None
    repeats: int = 0
    analysis: IncidentAnalysis = Field(default_factory=IncidentAnalysis)
