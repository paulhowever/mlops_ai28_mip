import asyncio

import pytest

from dota_winprob.schemas import ComponentStatus, ReportStatus
from dota_winprob.services.health import Probe, build_report, has_critical_failure


def sleeping_probe(name: str, seconds: float, critical: bool = True) -> Probe:
    async def run() -> str:
        await asyncio.sleep(seconds)
        return "1.0"

    return Probe(name=name, critical=critical, run=run)


def failing_probe(name: str, error: Exception, critical: bool, delay: float = 0.0) -> Probe:
    async def run() -> str:
        await asyncio.sleep(delay)
        raise error

    return Probe(name=name, critical=critical, run=run)


async def test_probes_run_in_parallel():
    probes = [sleeping_probe("a", 0.15), sleeping_probe("b", 0.15)]

    report = await build_report(probes, probe_timeout=1.0)

    assert report.status is ReportStatus.ok
    assert all(c.latency_ms >= 150 for c in report.components.values())
    assert report.total_latency_ms < 250


async def test_failed_probe_latency_is_measured():
    probes = [failing_probe("db", RuntimeError("boom"), critical=True, delay=0.05)]

    report = await build_report(probes, probe_timeout=1.0)

    component = report.components["db"]
    assert component.status is ComponentStatus.unavailable
    assert component.detail == "RuntimeError: boom"
    assert component.latency_ms >= 50


async def test_hanging_probe_is_reported_as_timeout():
    probes = [sleeping_probe("slow", 10.0)]

    report = await build_report(probes, probe_timeout=0.05)

    assert report.components["slow"].detail == "превышен таймаут 0.05 с"


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (RuntimeError("first line\nsecond line"), "RuntimeError: first line"),
        (RuntimeError(), "RuntimeError: RuntimeError"),
    ],
)
async def test_failure_detail_is_one_line(error, detail):
    report = await build_report([failing_probe("x", error, critical=False)], probe_timeout=1.0)

    assert report.components["x"].detail == detail


async def test_healthy_probe_carries_its_note():
    async def run() -> None:
        return None

    report = await build_report(
        [Probe(name="api", critical=False, run=run, note="версии нет")],
        probe_timeout=1.0,
    )

    assert report.components["api"].version is None
    assert report.components["api"].detail == "версии нет"


async def test_critical_failure_is_detected():
    probes = [
        failing_probe("db", OSError("down"), critical=True),
        sleeping_probe("api", 0, critical=False),
    ]

    report = await build_report(probes, probe_timeout=1.0)

    assert report.status is ReportStatus.degraded
    assert has_critical_failure(report)


async def test_non_critical_failure_is_not_critical():
    probes = [
        sleeping_probe("db", 0, critical=True),
        failing_probe("api", OSError("down"), critical=False),
    ]

    report = await build_report(probes, probe_timeout=1.0)

    assert report.status is ReportStatus.degraded
    assert not has_critical_failure(report)
