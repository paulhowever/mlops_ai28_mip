import asyncio
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field

from dota_winprob.agent.envelope import ToolResult
from dota_winprob.agent.errors import ToolInputError
from dota_winprob.agent.promql import METRICS, MetricName
from dota_winprob.agent.sources.prometheus import Point, Prometheus

STATUS_WINDOW_S = 60
MIN_STEP_S = 5
FINE_POINTS = 200
MAX_POINTS = 20
MAX_MINUTES = 24 * 60
MIN_WINDOW_S = 60

STATUS_METRICS: tuple[str, ...] = (
    "queue_ready",
    "queue_unacked",
    "queue_retry",
    "queue_limit",
    "consumers",
    "dlq_ready",
    "source_rate",
    "worker_rate",
    "worker_errors",
    "api_503_share",
    "inference_p95",
    "queue_wait_p95",
)

Minutes = Annotated[
    int,
    Field(ge=1, le=MAX_MINUTES, description="Глубина окна в минутах, считая от текущего момента."),
]


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def downsample(points: list[Point]) -> list[list[float]]:
    if not points:
        return []
    count = min(MAX_POINTS, len(points))
    last = len(points) - 1
    indexes = {round(index * last / max(1, count - 1)) for index in range(count)}
    indexes.add(last)
    return [[round(points[index][0]), round(points[index][1], 4)] for index in sorted(indexes)]


@dataclass(frozen=True)
class MonitoringTools:
    prometheus: Prometheus

    async def pipeline_status(self) -> ToolResult:
        values = await asyncio.gather(
            *(
                self.prometheus.instant(METRICS[name].query(STATUS_WINDOW_S))
                for name in STATUS_METRICS
            )
        )
        current = dict(zip(STATUS_METRICS, values, strict=True))
        ready, limit = current["queue_ready"], current["queue_limit"]
        metrics = [
            {
                "name": name,
                "value": rounded(value),
                "unit": METRICS[name].unit,
                "meaning": METRICS[name].meaning,
            }
            for name, value in current.items()
        ]
        data: dict[str, Any] = {
            "rate_window_s": STATUS_WINDOW_S,
            "queue_fill_share": rounded(ready / limit) if ready is not None and limit else None,
            "note": "null означает, что за окно данных нет, это не ноль",
            "metrics": metrics,
        }
        if all(value is None for value in values):
            return ToolResult("prometheus", data, "метрик конвейера нет", "empty")
        summary = (
            f"очередь {rounded(ready)} из {rounded(limit)}, "
            f"потребителей {rounded(current['consumers'])}, DLQ {rounded(current['dlq_ready'])}"
        )
        return ToolResult("prometheus", data, summary)

    async def metric_history(
        self,
        name: Annotated[MetricName, Field(description="Какую метрику показать.")],
        minutes: Minutes = 15,
    ) -> ToolResult:
        metric = METRICS.get(name)
        if metric is None:
            raise ToolInputError(f"name: одно из {', '.join(METRICS)}")
        step = max(MIN_STEP_S, minutes * 60 // FINE_POINTS)
        points = await self.prometheus.range(metric.query(max(MIN_WINDOW_S, step)), minutes, step)
        data: dict[str, Any] = {
            "metric": name,
            "meaning": metric.meaning,
            "unit": metric.unit,
            "minutes": minutes,
        }
        if not points:
            return ToolResult("prometheus", data, f"{name}: точек за окно нет", "empty")
        values = [value for _, value in points]
        data |= {
            "min": rounded(min(values)),
            "max": rounded(max(values)),
            "avg": rounded(sum(values) / len(values)),
            "last": rounded(values[-1]),
            "measured_points": len(points),
            "points_unix_time_value": downsample(points),
        }
        summary = (
            f"{name} за {minutes} мин: min {data['min']}, max {data['max']}, last {data['last']}"
        )
        return ToolResult("prometheus", data, summary)
