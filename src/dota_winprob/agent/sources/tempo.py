import time
from collections.abc import Callable
from typing import Any

from dota_winprob.agent import queries
from dota_winprob.agent.sources.http import HTTPSource

SEARCH_LIMIT = 5
LOOKBACK_S = 48 * 3600
JSON_ONLY = {"Accept": "application/json"}


class Tempo(HTTPSource):
    ready_path = "/ready"
    clock: Callable[[], float] = staticmethod(time.time)

    async def find_by_task(self, task_id: str) -> list[str]:
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
        found = body.get("traces") or []
        found.sort(key=lambda trace: int(trace.get("startTimeUnixNano", 0)), reverse=True)
        return [queries.trace_id(trace["traceID"]) for trace in found]

    async def fetch(self, trace_id: str) -> dict[str, Any] | None:
        path = f"/api/traces/{queries.trace_id(trace_id)}"
        response = await self.send("GET", path, headers=JSON_ONLY)
        if response.status_code == 404:
            return None
        return self.payload(response)
