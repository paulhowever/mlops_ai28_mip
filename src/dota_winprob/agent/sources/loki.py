import json
import time
from collections.abc import Callable
from typing import Any

from dota_winprob.agent.sources.http import HTTPSource

KEPT_FIELDS = (
    "timestamp",
    "level",
    "service",
    "message",
    "task_id",
    "trace_id",
    "attempt",
    "max_attempts",
    "worker",
    "error",
    "reason",
    "exception",
    "queue_wait_s",
    "win_probability",
    "model_version",
    "status_code",
    "retry_after",
    "suppressed",
)
EXCEPTION_TAIL = 400


def entry(stream: dict[str, str], stamp_ns: str, line: str) -> dict[str, Any]:
    try:
        record = json.loads(line)
    except ValueError:
        record = None
    if not isinstance(record, dict):
        return {
            "time_ns": stamp_ns,
            "service": stream.get("service"),
            "level": stream.get("level"),
            "line": line,
        }
    kept = {name: record[name] for name in KEPT_FIELDS if record.get(name) is not None}
    if isinstance(kept.get("exception"), str):
        kept["exception"] = kept["exception"][-EXCEPTION_TAIL:]
    return kept


class Loki(HTTPSource):
    ready_path = "/ready"
    clock: Callable[[], float] = staticmethod(time.time)

    async def search(self, query: str, minutes: int, limit: int) -> list[dict[str, Any]]:
        end = self.clock()
        body = await self.json(
            "POST",
            "/loki/api/v1/query_range",
            data={
                "query": query,
                "start": str(int((end - minutes * 60) * 1e9)),
                "end": str(int(end * 1e9)),
                "limit": str(limit),
                "direction": "backward",
            },
        )
        lines = [
            (value[0], stream["stream"], value[1])
            for stream in body["data"]["result"]
            for value in stream["values"]
        ]
        lines.sort(key=lambda line: int(line[0]), reverse=True)
        return [entry(stream, stamp, line) for stamp, stream, line in lines[:limit]]
