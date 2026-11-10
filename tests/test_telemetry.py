import asyncio
import json
import signal
import time

import pytest

from dota_winprob import source, telemetry
from dota_winprob.config import Settings
from dota_winprob.logging_config import setup_logging


@pytest.fixture
def fresh_telemetry(monkeypatch):
    installed = {}
    monkeypatch.setattr(telemetry, "_ACTIVE", {})
    monkeypatch.setattr(
        telemetry.trace, "set_tracer_provider", installed.setdefault("tracer", []).append
    )
    monkeypatch.setattr(
        telemetry.metrics, "set_meter_provider", installed.setdefault("meter", []).append
    )
    monkeypatch.setattr(
        telemetry.SystemMetricsInstrumentor, "instrument", lambda self, **kwargs: None
    )
    return installed


def test_disabled_sdk_installs_nothing(fresh_telemetry, monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    settings = Settings(_env_file=None)

    active = telemetry.setup_telemetry(settings)

    assert settings.otel_sdk_disabled
    assert not active.enabled
    assert fresh_telemetry == {"tracer": [], "meter": []}
    assert active.shutdown(0.1)


def test_unreachable_collector_neither_breaks_nor_blocks(fresh_telemetry, monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "dota-worker")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "parentbased_always_on")
    settings = Settings(_env_file=None, inference_delay_ms=500)

    started = time.perf_counter()
    active = telemetry.setup_telemetry(settings)
    [tracer_provider] = fresh_telemetry["tracer"]
    [meter_provider] = fresh_telemetry["meter"]
    with tracer_provider.get_tracer("test").start_as_current_span("work"):
        meter_provider.get_meter("test").create_counter("pipeline.test").add(1)
    setup_took = time.perf_counter() - started

    started = time.perf_counter()
    active.shutdown(1.0)
    shutdown_took = time.perf_counter() - started

    assert active.enabled
    assert telemetry.setup_telemetry(settings) is active
    assert setup_took < 1.0
    assert shutdown_took < 1.5
    assert tracer_provider.resource.attributes["service.name"] == "dota-worker"
    assert type(tracer_provider.sampler).__name__ == "ParentBased"
    assert meter_provider._sdk_config.views[2]._aggregation._boundaries == tuple(
        telemetry.inference_buckets(0.5)
    )


def test_resource_defaults_come_from_settings(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.instance.id=worker-2")
    settings = Settings(_env_file=None, environment="stand")

    attributes = telemetry.build_resource(settings).attributes

    assert attributes["service.name"] == settings.app_name
    assert attributes["deployment.environment.name"] == "stand"
    assert attributes["service.instance.id"] == "worker-2"
    assert attributes["service.version"]


def test_rejections_are_summarised_not_logged_one_by_one(settings, capsys):
    setup_logging(settings)
    moments = iter([0.0, 1.0, 2.0, 3.0, 11.0, 12.0, 13.0])
    rejections = source.RejectionLog(10.0, clock=lambda: next(moments))

    for _ in range(5):
        rejections.record(0.1)
    rejections.record(0.2)
    rejections.flush()

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["level"] for record in records] == ["WARNING"] * 3
    assert [record["rejected"] for record in records] == [1, 4, 1]
    assert records[1]["waited_s"] == pytest.approx(0.4)


async def test_stop_signal_is_logged_once(settings, capsys):
    setup_logging(settings)
    stop = asyncio.Event()

    source.request_stop(stop, signal.SIGTERM)
    source.request_stop(stop, signal.SIGTERM)

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert stop.is_set()
    assert len(records) == 1
    assert records[0]["level"] == "INFO"
    assert records[0]["signal"] == "SIGTERM"
