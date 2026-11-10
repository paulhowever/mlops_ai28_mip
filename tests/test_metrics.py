import random

import pytest
from faststream.rabbit import TestRabbitBroker
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader, Sum

from dota_winprob import source
from dota_winprob.clients import broker as broker_client
from dota_winprob.schemas import MatchState
from dota_winprob.services.pipeline import Pipeline
from dota_winprob.services.scoring import Scorer
from dota_winprob.telemetry import (
    LATENCY_BUCKETS,
    PROCESS_METRICS,
    inference_buckets,
)
from dota_winprob.worker import WorkerState, build_broker
from tests.fakes import (
    MATCH_STATE,
    BrokenPredictor,
    FakePredictor,
    FakePublisher,
    FakeTasks,
    HealthyPool,
    loaded_model,
    opendota_ok,
    task_message,
)


def collect(reader):
    found = {}
    data = reader.get_metrics_data()
    for resource_metrics in data.resource_metrics if data else ():
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                found[metric.name] = metric
    return found


def value(metric, **attributes):
    total = 0
    for point in metric.data.data_points:
        if all(point.attributes.get(key) == expected for key, expected in attributes.items()):
            total += point.count if isinstance(metric.data, Histogram) else point.value
    return total


def snapshot(reader, name, **attributes):
    metric = collect(reader).get(name)
    return 0 if metric is None else value(metric, **attributes)


def pipeline_for(predictor, max_attempts=3):
    async def accept_retry(body, headers, task_id, correlation_id):
        return True

    return Pipeline(
        repository=FakeTasks(),
        scorer=Scorer(loaded_model(predictor), 2, 0.0),
        publish_retry=accept_retry,
        max_attempts=max_attempts,
        worker="worker-test",
    )


async def deliver(settings, predictor, attempt=1):
    worker = build_broker(settings, WorkerState(pipeline=pipeline_for(predictor)))
    async with TestRabbitBroker(worker) as test_broker:
        await test_broker.publish(
            task_message().model_dump(mode="json"),
            exchange=broker_client.TASKS_EXCHANGE,
            routing_key=broker_client.ROUTING_KEY,
            message_id="batch:1",
            headers={broker_client.ATTEMPT_HEADER: attempt},
        )


async def test_worker_counts_tasks_by_result(settings, metric_reader):
    before = {
        result: snapshot(metric_reader, "pipeline.worker.tasks", result=result)
        for result in ("ok", "error", "dlq")
    }

    await deliver(settings, FakePredictor())
    await deliver(settings, BrokenPredictor(), attempt=1)
    await deliver(settings, BrokenPredictor(), attempt=3)

    metric = collect(metric_reader)["pipeline.worker.tasks"]
    assert isinstance(metric.data, Sum)
    assert metric.data.is_monotonic
    for result in ("ok", "error", "dlq"):
        assert value(metric, result=result) - before[result] == 1
    assert {key for point in metric.data.data_points for key in point.attributes} == {"result"}


async def test_worker_histograms_use_configured_buckets(settings, metric_reader):
    await deliver(settings, FakePredictor())

    found = collect(metric_reader)
    inference = found["pipeline.worker.inference.duration"]
    assert isinstance(inference.data, Histogram)
    assert inference.unit == "s"
    [point] = inference.data.data_points
    assert tuple(point.explicit_bounds) == inference_buckets(settings.inference_delay)
    assert point.count >= 1

    wait = found["pipeline.worker.queue_wait"]
    assert isinstance(wait.data, Histogram)
    assert wait.data.data_points[0].attributes == {}


async def test_in_flight_is_up_down_counter_back_to_zero(settings, metric_reader):
    await deliver(settings, FakePredictor())

    metric = collect(metric_reader)["pipeline.worker.in_flight"]
    assert isinstance(metric.data, Sum)
    assert not metric.data.is_monotonic
    assert value(metric) == 0


async def test_retry_delivery_is_counted(settings, metric_reader):
    before = snapshot(metric_reader, "pipeline.worker.redeliveries", reason="retry")

    await deliver(settings, FakePredictor(), attempt=2)

    assert snapshot(metric_reader, "pipeline.worker.redeliveries", reason="retry") == before + 1


async def test_source_counts_published_rejected_and_wait(metric_reader):
    answers = [False, False, True, True]

    async def publisher(message):
        return answers.pop(0)

    async def no_sleep(delay):
        return None

    before = {
        name: snapshot(metric_reader, name)
        for name in (
            "pipeline.source.published",
            "pipeline.source.rejected",
            "pipeline.source.wait",
        )
    }
    stats = source.SourceStats()
    tasks = [(f"batch:{index}", MatchState.model_validate(MATCH_STATE)) for index in range(2)]

    await source.run(
        tasks, publisher, source.Backoff(0.02, 0.5, rng=random.Random(3)), stats, no_sleep
    )

    found = collect(metric_reader)
    assert value(found["pipeline.source.published"]) - before["pipeline.source.published"] == 2
    assert value(found["pipeline.source.rejected"]) - before["pipeline.source.rejected"] == 2
    waited = value(found["pipeline.source.wait"]) - before["pipeline.source.wait"]
    assert waited == pytest.approx(stats.waited)
    assert found["pipeline.source.wait"].unit == "s"


async def test_api_duration_histogram_has_route_status_and_buckets(make_client, metric_reader):
    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, publisher=FakePublisher(accept=False)
    )
    before = snapshot(
        metric_reader,
        "http.server.request.duration",
        **{"http.route": "/api/v1/tasks", "http.response.status_code": 503},
    )

    await client.post("/api/v1/tasks", json=MATCH_STATE)
    await client.get("/healthz")

    metric = collect(metric_reader)["http.server.request.duration"]
    assert metric.unit == "s"
    rejected = [
        point
        for point in metric.data.data_points
        if point.attributes.get("http.route") == "/api/v1/tasks"
        and point.attributes.get("http.response.status_code") == 503
    ]
    assert sum(point.count for point in rejected) == before + 1
    assert tuple(rejected[0].explicit_bounds) == LATENCY_BUCKETS
    assert rejected[0].attributes["http.request.method"] == "POST"
    routes = {point.attributes.get("http.route") for point in metric.data.data_points}
    assert "/healthz" not in routes


def test_inference_buckets_follow_configured_delay():
    fast = inference_buckets(0.2)
    slow = inference_buckets(0.5)

    assert fast != slow
    assert 0.2 in fast
    assert 0.5 in slow
    assert all(0.2 <= bound <= 1.2 for bound in fast)
    assert len([bound for bound in fast if bound < 0.25]) >= 5
    assert list(fast) == sorted(set(fast))
    assert inference_buckets(0.0)[0] == 0.001


def test_process_memory_usage_is_rss_in_bytes():
    from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    instrumentor = SystemMetricsInstrumentor(config=PROCESS_METRICS)
    instrumentor.instrument(meter_provider=provider)
    try:
        metric = collect(reader)["process.memory.usage"]
    finally:
        instrumentor.uninstrument()
        provider.shutdown()

    assert metric.unit == "By"
    assert value(metric) > 10_000_000
