import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from dota_winprob.schemas import ComponentHealth, ComponentStatus, HealthReport, ReportStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Probe:
    name: str
    critical: bool
    run: Callable[[], Awaitable[str | None]]
    note: str | None = None


async def _run_timed(
    probe: Probe,
    probe_timeout: float,
    elapsed_ms: dict[str, float],
) -> str | None:
    started = time.perf_counter()
    # Замер в finally срабатывает и для упавшей пробы: время до отказа
    # тоже нужно в отчёте.
    try:
        async with asyncio.timeout(probe_timeout):
            return await probe.run()
    finally:
        elapsed_ms[probe.name] = round((time.perf_counter() - started) * 1000, 1)


def _describe_failure(error: BaseException, probe_timeout: float) -> str:
    if isinstance(error, TimeoutError):
        return f"превышен таймаут {probe_timeout} с"
    message = str(error)
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message.splitlines()[0]}"


async def build_report(probes: list[Probe], probe_timeout: float) -> HealthReport:
    elapsed_ms: dict[str, float] = {}
    started = time.perf_counter()

    outcomes = await asyncio.gather(
        *(_run_timed(probe, probe_timeout, elapsed_ms) for probe in probes),
        # Исключения приезжают значениями и разбираются ниже — поэтому
        # в коде нет ни одного except Exception.
        return_exceptions=True,
    )

    total_latency_ms = round((time.perf_counter() - started) * 1000, 1)

    components: dict[str, ComponentHealth] = {}
    for probe, outcome in zip(probes, outcomes, strict=True):
        failed = isinstance(outcome, BaseException)
        if failed:
            detail = _describe_failure(outcome, probe_timeout)
            log.warning(
                "Зависимость недоступна",
                extra={"component": probe.name, "critical": probe.critical, "error": detail},
            )
        components[probe.name] = ComponentHealth(
            status=ComponentStatus.unavailable if failed else ComponentStatus.healthy,
            version=None if failed else outcome,
            latency_ms=elapsed_ms[probe.name],
            critical=probe.critical,
            detail=detail if failed else probe.note,
        )

    all_healthy = all(
        component.status is ComponentStatus.healthy for component in components.values()
    )

    return HealthReport(
        status=ReportStatus.ok if all_healthy else ReportStatus.degraded,
        total_latency_ms=total_latency_ms,
        components=components,
    )


def has_critical_failure(report: HealthReport) -> bool:
    return any(
        component.critical and component.status is ComponentStatus.unavailable
        for component in report.components.values()
    )
