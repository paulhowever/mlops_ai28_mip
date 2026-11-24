import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dota_winprob.agent import queries
from dota_winprob.agent.sources.http import HTTPSource

SEARCH_LIMIT = 5
LOOKBACK_S = 48 * 3600
JSON_ONLY = {"Accept": "application/json"}


@dataclass(frozen=True)
class FoundTrace:
    trace_id: str
    started_ns: int


class Tempo(HTTPSource):
    ready_path = "/ready"
    clock: Callable[[], float] = staticmethod(time.time)

    async def find_by_task(self, task_id: str) -> list[FoundTrace]:
        end = int(self.clock())
        body = await self.json(
            "GET",
            "/api/search",
            params={
                "q": queries.traceql_by_task(task_id),
                "limit": SEARCH_LIMIT,
                "start": end - LOOKBACK_S,
                "end": end,
            },
        )
        found = [
            FoundTrace(queries.trace_id(trace["traceID"]), int(trace.get("startTimeUnixNano", 0)))
            for trace in body.get("traces") or []
        ]
        return sorted(found, key=lambda trace: trace.started_ns, reverse=True)

    async def fetch(self, trace_id: str) -> dict[str, Any] | None:
        path = f"/api/traces/{queries.trace_id(trace_id)}"
        response = await self.send("GET", path, headers=JSON_ONLY)
        if response.status_code == 404:
            return None
        return self.payload(response)
