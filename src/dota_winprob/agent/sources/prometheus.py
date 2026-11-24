import math
import time
from collections.abc import Callable
from typing import Any

from dota_winprob.agent.sources.http import HTTPSource

Point = tuple[float, float]


def number(raw: object) -> float | None:
    value = float(str(raw))
    return value if math.isfinite(value) else None


class Prometheus(HTTPSource):
    ready_path = "/-/ready"
    clock: Callable[[], float] = staticmethod(time.time)

    async def instant(self, query: str) -> float | None:
        body = await self.json("POST", "/api/v1/query", data={"query": query})
        series = body["data"]["result"]
        return number(series[0]["value"][1]) if series else None

    async def range(self, query: str, minutes: int, step_s: int) -> list[Point]:
        end = self.clock()
        form: dict[str, Any] = {
            "query": query,
            "start": f"{end - minutes * 60:.3f}",
            "end": f"{end:.3f}",
            "step": str(step_s),
        }
        body = await self.json("POST", "/api/v1/query_range", data=form)
        series = body["data"]["result"]
        if not series:
            return []
        points = ((float(stamp), number(raw)) for stamp, raw in series[0]["values"])
        return [(stamp, value) for stamp, value in points if value is not None]
