from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

MAX_SPANS = 60
ATTRIBUTES = (
    "task.id",
    "task.source",
    "attempt",
    "attempts",
    "task.queue_wait_s",
    "task.dead_lettered",
    "model.version",
    "http.route",
    "http.response.status_code",
    "retry_after_s",
    "db.operation.name",
    "messaging.rabbitmq.redelivered",
)
QUEUE_WAIT = "task.queue_wait_s"
PROCESS_SUFFIX = " process"
INFERENCE = "inference"
ERROR_CODES = frozenset({2, "2", "STATUS_CODE_ERROR"})


@dataclass(frozen=True)
class Span:
    span_id: str
    parent_id: str
    name: str
    service: str
    start_ns: int
    end_ns: int
    error: str | None
    attributes: dict[str, Any]
    events: dict[str, int]

    @property
    def duration_ms(self) -> float:
        return round((self.end_ns - self.start_ns) / 1e6, 1)


def _value(raw: dict[str, Any]) -> str | int | float | bool | None:
    for kind, value in raw.items():
        if kind == "intValue":
            return int(value)
        if kind in {"stringValue", "doubleValue", "boolValue"}:
            return value
    return None


def _attributes(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {item["key"]: _value(item.get("value", {})) for item in items or []}


def _error(status: dict[str, Any]) -> str | None:
    if status.get("code") not in ERROR_CODES:
        return None
    return status.get("message") or "ERROR"


def _span(raw: dict[str, Any], service: str) -> Span:
    attributes = _attributes(raw.get("attributes"))
    return Span(
        span_id=raw.get("spanId", ""),
        parent_id=raw.get("parentSpanId", ""),
        name=raw.get("name", ""),
        service=service,
        start_ns=int(raw["startTimeUnixNano"]),
        end_ns=int(raw["endTimeUnixNano"]),
        error=_error(raw.get("status") or {}),
        attributes={name: attributes[name] for name in ATTRIBUTES if name in attributes},
        events=dict(Counter(event.get("name", "") for event in raw.get("events") or [])),
    )


def parse(payload: dict[str, Any]) -> list[Span]:
    body = payload.get("trace", payload)
    spans: list[Span] = []
    for batch in body.get("batches") or body.get("resourceSpans") or []:
        resource = _attributes((batch.get("resource") or {}).get("attributes"))
        service = str(resource.get("service.name", "unknown"))
        for scope in batch.get("scopeSpans") or []:
            spans.extend(_span(raw, service) for raw in scope.get("spans") or [])
    return spans


def _ordered(spans: list[Span]) -> list[tuple[int, Span]]:
    known = {span.span_id for span in spans}
    children: dict[str, list[Span]] = defaultdict(list)
    for span in sorted(spans, key=lambda item: item.start_ns):
        children[span.parent_id if span.parent_id in known else ""].append(span)
    ordered: list[tuple[int, Span]] = []
    seen: set[str] = set()
    stack = [(0, span) for span in reversed(children[""])]
    while stack:
        depth, span = stack.pop()
        if span.span_id in seen:
            continue
        seen.add(span.span_id)
        ordered.append((depth, span))
        stack.extend((depth + 1, child) for child in reversed(children[span.span_id]))
    return ordered


def _node(depth: int, span: Span, origin_ns: int) -> dict[str, Any]:
    node: dict[str, Any] = {
        "depth": depth,
        "name": span.name,
        "service": span.service,
        "offset_ms": round((span.start_ns - origin_ns) / 1e6, 1),
        "duration_ms": span.duration_ms,
    }
    if span.error is not None:
        node["error"] = span.error
    if span.attributes:
        node["attributes"] = span.attributes
    if span.events:
        node["events"] = span.events
    return node


def tree(trace_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    spans = parse(payload)
    if not spans:
        return {"trace_id": trace_id, "spans": []}
    origin = min(span.start_ns for span in spans)
    ordered = _ordered(spans)
    return {
        "trace_id": trace_id,
        "started_at": datetime.fromtimestamp(origin / 1e9, UTC).isoformat(timespec="seconds"),
        "total_ms": round((max(span.end_ns for span in spans) - origin) / 1e6, 1),
        "span_count": len(spans),
        "error_count": sum(span.error is not None for span in spans),
        "services": sorted({span.service for span in spans}),
        "queue_wait_s": [
            span.attributes[QUEUE_WAIT]
            for _, span in ordered
            if span.name.endswith(PROCESS_SUFFIX) and QUEUE_WAIT in span.attributes
        ],
        "inference_ms": [span.duration_ms for _, span in ordered if span.name == INFERENCE],
        "spans": [_node(depth, span, origin) for depth, span in ordered[:MAX_SPANS]],
    }
