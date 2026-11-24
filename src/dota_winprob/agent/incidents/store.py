import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime

from dota_winprob.agent.envelope import clip
from dota_winprob.agent.incidents.models import (
    Alert,
    AnalysisStatus,
    Incident,
    IncidentStatus,
)
from dota_winprob.agent.redaction import Redactor

ID_BYTES = 12
MAX_LABEL = 200
MAX_TEXT = 1000
UNTITLED = "Алерт без названия"
UNFINISHED = frozenset({AnalysisStatus.queued, AnalysisStatus.running})


class IncidentStore:
    def __init__(
        self,
        *,
        capacity: int,
        cooldown: float,
        redactor: Redactor,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._cooldown = cooldown
        self._redactor = redactor
        self._clock = clock
        self._incidents: OrderedDict[str, Incident] = OrderedDict()
        self._analyzed_at: dict[str, float] = {}

    def all(self) -> list[Incident]:
        return list(reversed(self._incidents.values()))

    def latest(self, fingerprint: str) -> Incident | None:
        return next((item for item in self.all() if item.fingerprint == fingerprint), None)

    def open(self, alert: Alert) -> Incident:
        self._make_room()
        now = datetime.now(UTC)
        labels = {
            self._clean(key, MAX_LABEL): self._clean(value, MAX_LABEL)
            for key, value in alert.labels.items()
        }
        incident = Incident(
            id=secrets.token_urlsafe(ID_BYTES),
            fingerprint=alert.fingerprint,
            title=labels.get("alertname") or UNTITLED,
            severity=labels.get("severity"),
            summary=self._text(alert, "summary"),
            runbook=self._text(alert, "runbook"),
            labels=labels,
            opened_at=now,
            last_seen_at=now,
        )
        self._incidents[incident.id] = incident
        return incident

    def repeat(self, incident: Incident) -> None:
        incident.repeats += 1
        incident.last_seen_at = datetime.now(UTC)
        incident.status = IncidentStatus.firing
        incident.resolved_at = None

    def resolve(self, incident: Incident) -> None:
        incident.status = IncidentStatus.resolved
        incident.resolved_at = incident.last_seen_at = datetime.now(UTC)

    def analyzed(self, incident: Incident) -> None:
        self._analyzed_at[incident.id] = self._clock()

    def cooling(self, incident: Incident) -> bool:
        if incident.analysis.status in UNFINISHED:
            return True
        finished = self._analyzed_at.get(incident.id)
        return finished is not None and self._clock() - finished < self._cooldown

    def _make_room(self) -> None:
        if len(self._incidents) < self._capacity:
            return
        resolved = (
            incident
            for incident in self._incidents.values()
            if incident.status is IncidentStatus.resolved
        )
        victim = next(resolved, None) or next(iter(self._incidents.values()))
        del self._incidents[victim.id]
        self._analyzed_at.pop(victim.id, None)

    def _text(self, alert: Alert, name: str) -> str | None:
        value = alert.annotations.get(name)
        return None if value is None else self._clean(value, MAX_TEXT)

    def _clean(self, value: str, limit: int) -> str:
        return clip(self._redactor.text(value), limit)
