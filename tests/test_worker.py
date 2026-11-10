import asyncio
import json
import threading
import time

import pytest
from faststream.exceptions import RejectMessage
from faststream.rabbit import TestRabbitBroker

from dota_winprob.clients import broker as broker_client
from dota_winprob.schemas import TaskSource
from dota_winprob.services.loop_lag import LoopLagMonitor
from dota_winprob.services.pipeline import Pipeline
from dota_winprob.services.scoring import Scorer
from dota_winprob.worker import WorkerState, build_broker, make_health
from tests.fakes import (
    BrokenPredictor,
    FakeBroker,
    FakePredictor,
    FakeTasks,
    loaded_model,
    task_message,
)


class RetryRecorder:
    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.calls: list[tuple[bytes, dict]] = []
        self.ids: list[tuple[str, str | None]] = []

    async def __call__(self, body: bytes, headers: dict, task_id: str, correlation_id) -> bool:
        self.calls.append((body, dict(headers)))
        self.ids.append((task_id, correlation_id))
        return self.accept


def make_pipeline(predictor=None, delay=0.0, concurrency=4, accept_retry=True):
    tasks = FakeTasks()
    retry = RetryRecorder(accept_retry)
    scorer = Scorer(loaded_model(predictor or FakePredictor()), concurrency, delay)
    pipeline = Pipeline(
        repository=tasks,
        scorer=scorer,
        publish_retry=retry,
        max_attempts=3,
        worker="worker-test",
    )
    return pipeline, tasks, retry


def body_of(task_id: str = "batch:1") -> bytes:
    return task_message(task_id).model_dump_json().encode()


async def test_result_is_saved_once():
    pipeline, tasks, retry = make_pipeline()

    await pipeline.handle(body_of(), {})

    assert tasks.results["batch:1"] == {
        "win_probability": 0.73,
        "model_version": "4",
        "attempt": 1,
        "worker": "worker-test",
    }
    assert pipeline.processed == 1
    assert retry.calls == []


async def test_redelivery_does_not_duplicate_result():
    pipeline, tasks, _ = make_pipeline()

    await pipeline.handle(body_of(), {})
    await pipeline.handle(body_of(), {})

    assert list(tasks.results) == ["batch:1"]
    assert pipeline.processed == 1
    assert pipeline.duplicates == 1


async def test_invalid_message_is_rejected_without_retry():
    pipeline, tasks, retry = make_pipeline()
    body = json.dumps({"task_id": "batch:7", "source": "batch", "features": {}}).encode()

    with pytest.raises(RejectMessage):
        await pipeline.handle(body, {})

    assert retry.calls == []
    assert tasks.failed["batch:7"]["source"] is TaskSource.batch
    assert tasks.failed["batch:7"]["error"].startswith("невалидное сообщение")
    assert pipeline.failed == 1


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b'{"task_id": 5}'])
async def test_garbage_is_rejected_without_db_write(body):
    pipeline, tasks, _ = make_pipeline()

    with pytest.raises(RejectMessage):
        await pipeline.handle(body, {})

    assert tasks.failed == {}


async def test_failure_goes_to_retry_queue_with_next_attempt():
    pipeline, tasks, retry = make_pipeline(predictor=BrokenPredictor())

    await pipeline.handle(body_of(), {"x-attempt": 2}, "req-7")

    assert retry.calls == [(body_of(), {"x-attempt": 3})]
    assert retry.ids == [("batch:1", "req-7")]
    assert tasks.results == {}
    assert pipeline.retried == 1


async def test_database_failure_is_retried_too():
    pipeline, tasks, retry = make_pipeline()
    tasks.broken = True

    await pipeline.handle(body_of(), {})

    assert retry.calls == [(body_of(), {"x-attempt": 2})]


async def test_last_attempt_marks_failed_and_rejects():
    pipeline, tasks, retry = make_pipeline(predictor=BrokenPredictor())

    with pytest.raises(RejectMessage):
        await pipeline.handle(body_of(), {"x-attempt": 3})

    assert retry.calls == []
    assert tasks.failed["batch:1"]["attempt"] == 3
    assert "feature schema mismatch" in tasks.failed["batch:1"]["error"]


async def test_refused_retry_rejects_to_dlq_and_marks_failed():
    pipeline, tasks, _ = make_pipeline(predictor=BrokenPredictor(), accept_retry=False)

    with pytest.raises(RejectMessage):
        await pipeline.handle(body_of(), {})

    assert tasks.failed["batch:1"]["error"].startswith("очередь повторов недоступна")
    assert pipeline.failed == 1


@pytest.mark.parametrize("header", ["abc", None, 0])
async def test_broken_attempt_header_counts_as_first(header):
    pipeline, tasks, _ = make_pipeline()

    await pipeline.handle(body_of(), {"x-attempt": header})

    assert tasks.results["batch:1"]["attempt"] == 1


async def test_inference_runs_in_pool_in_parallel_without_blocking_loop():
    predictor = FakePredictor()
    pipeline, tasks, _ = make_pipeline(predictor=predictor, delay=0.2, concurrency=4)
    monitor = LoopLagMonitor(interval=0.01)
    monitor.start()

    started = time.perf_counter()
    await asyncio.gather(*(pipeline.handle(body_of(f"batch:{i}"), {}) for i in range(4)))
    elapsed = time.perf_counter() - started
    await monitor.stop()

    assert len(tasks.results) == 4
    assert elapsed < 0.6
    assert all(thread is not threading.main_thread() for thread in predictor.threads)
    assert {thread.name.split("_")[0] for thread in predictor.threads} == {"inference"}
    assert monitor.max_ms < 100
    pipeline.scorer.close()


async def test_subscriber_feeds_pipeline(settings):
    pipeline, tasks, _ = make_pipeline()
    broker = build_broker(settings, WorkerState(pipeline=pipeline))

    async with TestRabbitBroker(broker) as test_broker:
        await test_broker.publish(
            task_message("batch:42").model_dump(mode="json"),
            exchange=broker_client.TASKS_EXCHANGE,
            routing_key=broker_client.ROUTING_KEY,
            message_id="batch:42",
        )

    assert "batch:42" in tasks.results


async def test_subscriber_refuses_work_before_startup(settings):
    broker = build_broker(settings, WorkerState())

    async with TestRabbitBroker(broker) as test_broker:
        with pytest.raises(RuntimeError, match="не инициализирован"):
            await test_broker.publish(
                task_message().model_dump(mode="json"),
                exchange=broker_client.TASKS_EXCHANGE,
                routing_key=broker_client.ROUTING_KEY,
            )


def test_prefetch_follows_pool_size(settings):
    assert settings.prefetch_count == settings.worker_concurrency == 4
    assert settings.model_copy(update={"worker_prefetch": 0}).prefetch_count == 0
    assert settings.model_copy(update={"worker_prefetch": 8}).prefetch_count == 8


async def call_health(handler) -> tuple[int, dict]:
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "GET", "path": "/health", "headers": [], "query_string": b""}
    await handler(scope, receive, send)
    status = sent[0]["status"]
    body = b"".join(message.get("body", b"") for message in sent[1:])
    return status, json.loads(body)


async def test_health_reports_worker_state(settings):
    pipeline, _, _ = make_pipeline()
    state = WorkerState(pipeline=pipeline)
    handler = make_health(FakeBroker(), state, settings)

    status, body = await call_health(handler)

    assert status == 200
    assert body["status"] == "ok"
    assert body["model_version"] == "4"
    assert body["prefetch_count"] == 4
    assert body["in_flight"] == 0


async def test_health_is_503_before_startup(settings):
    handler = make_health(FakeBroker(alive=False), WorkerState(), settings)

    status, body = await call_health(handler)

    assert status == 503
    assert body["status"] == "unavailable"
    assert body["model_version"] is None


class FakeQueue:
    def __init__(self, closed: bool = False) -> None:
        self.channel = type("Channel", (), {"is_closed": closed})()
        self.cancelled: list[str] = []

    async def cancel(self, tag: str) -> None:
        self.cancelled.append(tag)


class FakeSubscriber:
    def __init__(self, queue, tag) -> None:
        self._queue_obj = queue
        self._consumer_tag = tag


async def test_stop_consuming_cancels_only_live_consumers():
    live, closed = FakeQueue(), FakeQueue(closed=True)
    subscribers = [
        FakeSubscriber(live, "ctag-1"),
        FakeSubscriber(closed, "ctag-2"),
        FakeSubscriber(None, None),
    ]
    broker = type("Broker", (), {"subscribers": subscribers})()

    await broker_client.stop_consuming(broker)

    assert live.cancelled == ["ctag-1"]
    assert subscribers[0]._consumer_tag is None
    assert closed.cancelled == []
    assert subscribers[1]._consumer_tag == "ctag-2"
