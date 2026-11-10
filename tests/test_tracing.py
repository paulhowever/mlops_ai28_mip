import json
import random

import pytest
from faststream.rabbit import TestRabbitBroker
from opentelemetry.trace import SpanKind, StatusCode

from dota_winprob import source
from dota_winprob.clients import broker as broker_client
from dota_winprob.schemas import MatchState
from dota_winprob.services.pipeline import Pipeline
from dota_winprob.services.scoring import Scorer
from dota_winprob.worker import WorkerState, build_broker
from tests.fakes import (
    MATCH_STATE,
    BrokenPredictor,
    FakePredictor,
    FakeTasks,
    HealthyPool,
    loaded_model,
    opendota_ok,
    task_message,
)

PROCESS_SPAN = "tasks.score process"
PUBLISH_SPAN = "tasks.score publish"


def by_name(finished, name):
    return [span for span in finished if span.name == name]


def one(finished, name):
    [span] = by_name(finished, name)
    return span


def event_names(span):
    return [event.name for event in span.events]


def pipeline_for(predictor, retry=None, max_attempts=3):
    async def accept_retry(body, headers, task_id, correlation_id):
        return True

    return Pipeline(
        repository=FakeTasks(),
        scorer=Scorer(loaded_model(predictor), 2, 0.0),
        publish_retry=retry or accept_retry,
        max_attempts=max_attempts,
        worker="worker-test",
    )


async def publish(test_broker, task_id="batch:1", attempt=None, body=None):
    await test_broker.publish(
        body if body is not None else task_message(task_id).model_dump(mode="json"),
        exchange=broker_client.TASKS_EXCHANGE,
        routing_key=broker_client.ROUTING_KEY,
        message_id=task_id,
        headers={} if attempt is None else {broker_client.ATTEMPT_HEADER: attempt},
    )


async def test_api_task_is_one_trace_through_broker_worker_and_db(make_client, settings, spans):
    pipeline = pipeline_for(FakePredictor())
    worker = build_broker(settings, WorkerState(pipeline=pipeline))
    real_publish = broker_client.publish_task

    async def publish_through_worker(broker, message, confirm_timeout, correlation_id=None):
        await real_publish(broker, message, confirm_timeout, correlation_id)
        return True

    async with TestRabbitBroker(worker) as test_broker:
        client = await make_client(
            pool=HealthyPool(),
            opendota_handler=opendota_ok,
            broker=test_broker,
            publisher=publish_through_worker,
        )
        response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.status_code == 202
    task_id = response.json()["task_id"]
    assert task_id in pipeline.repository.results

    finished = spans.get_finished_spans()
    server = one(finished, "POST /api/v1/tasks")
    submit = one(finished, "task submit")
    publish_span = one(finished, PUBLISH_SPAN)
    process = one(finished, PROCESS_SPAN)
    inference = one(finished, "inference")
    save = one(finished, "db save_result")

    trace_ids = {span.context.trace_id for span in (server, submit, publish_span, process)}
    trace_ids |= {inference.context.trace_id, save.context.trace_id}
    assert len(trace_ids) == 1
    assert server.kind is SpanKind.SERVER
    assert submit.parent.span_id == server.context.span_id
    assert publish_span.kind is SpanKind.PRODUCER
    assert process.kind is SpanKind.CONSUMER
    assert inference.parent.span_id == process.context.span_id
    assert save.parent.span_id == process.context.span_id
    assert submit.attributes["task.id"] == task_id
    assert process.attributes["task.id"] == task_id
    assert inference.attributes["model.alias"] == "champion"
    assert inference.attributes["model.version"] == "4"
    assert inference.attributes["attempt"] == 1


async def test_inference_in_thread_pool_is_child_of_process(settings, spans):
    predictor = FakePredictor()
    worker = build_broker(settings, WorkerState(pipeline=pipeline_for(predictor)))

    async with TestRabbitBroker(worker) as test_broker:
        await publish(test_broker, "batch:5")

    process = one(spans.get_finished_spans(), PROCESS_SPAN)
    inference = one(spans.get_finished_spans(), "inference")
    assert predictor.threads[0].name.startswith("inference")
    assert inference.parent is not None
    assert inference.parent.span_id == process.context.span_id
    assert inference.context.trace_id == process.context.trace_id


async def test_failed_attempt_marks_process_span_as_error(settings, spans):
    worker = build_broker(settings, WorkerState(pipeline=pipeline_for(BrokenPredictor())))

    async with TestRabbitBroker(worker) as test_broker:
        await publish(test_broker, attempt=2)

    finished = spans.get_finished_spans()
    process = one(finished, PROCESS_SPAN)
    inference = one(finished, "inference")
    assert process.status.status_code is StatusCode.ERROR
    assert process.attributes["attempt"] == 2
    [exception] = [event for event in process.events if event.name == "exception"]
    assert exception.attributes["exception.type"] == "ValueError"
    assert exception.attributes["attempt"] == 2
    assert "retry" in event_names(process)
    assert inference.status.status_code is StatusCode.ERROR


async def test_poison_message_attempts_share_one_trace_and_end_in_dlq(settings, spans):
    state = WorkerState()
    worker = build_broker(settings, state)

    async with TestRabbitBroker(worker) as test_broker:

        async def retry_via_tasks_queue(body, headers, task_id, correlation_id):
            await test_broker.publish(
                body,
                exchange=broker_client.TASKS_EXCHANGE,
                routing_key=broker_client.ROUTING_KEY,
                message_id=task_id,
                headers=dict(headers),
            )
            return True

        state.pipeline = pipeline_for(BrokenPredictor(), retry=retry_via_tasks_queue)
        await publish(test_broker, "batch:9")

    finished = spans.get_finished_spans()
    attempts = sorted(by_name(finished, PROCESS_SPAN), key=lambda span: span.attributes["attempt"])
    assert [span.attributes["attempt"] for span in attempts] == [1, 2, 3]
    assert len({span.context.trace_id for span in attempts}) == 1
    assert all(span.status.status_code is StatusCode.ERROR for span in attempts)
    assert "dlq" in event_names(attempts[-1])
    assert attempts[-1].attributes["task.dead_lettered"] is True
    assert "dlq" not in event_names(attempts[0])
    assert one(finished, "db mark_failed").context.trace_id == attempts[0].context.trace_id
    assert state.pipeline.repository.failed["batch:9"]["attempt"] == 3


async def test_invalid_message_goes_to_dlq_with_error_span(settings, spans):
    worker = build_broker(settings, WorkerState(pipeline=pipeline_for(FakePredictor())))
    body = json.dumps({"task_id": "batch:7", "source": "batch", "features": {}}).encode()

    async with TestRabbitBroker(worker) as test_broker:
        await publish(test_broker, "batch:7", body=body)

    process = one(spans.get_finished_spans(), PROCESS_SPAN)
    assert process.status.status_code is StatusCode.ERROR
    assert process.attributes["task.id"] == "batch:7"
    assert "dlq" in event_names(process)
    assert "exception" in event_names(process)


async def test_overloaded_api_gives_error_trace(make_client, spans):
    from tests.fakes import FakePublisher

    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, publisher=FakePublisher(accept=False)
    )

    response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.status_code == 503
    finished = spans.get_finished_spans()
    server = one(finished, "POST /api/v1/tasks")
    submit = one(finished, "task submit")
    assert submit.context.trace_id == server.context.trace_id
    assert submit.status.status_code is StatusCode.ERROR
    assert "очередь задач заполнена" in submit.status.description
    assert submit.attributes["retry_after_s"] >= 1
    [exception] = [event for event in submit.events if event.name == "exception"]
    assert exception.attributes["exception.type"].endswith("Overloaded")
    assert server.status.status_code is StatusCode.ERROR
    assert server.attributes["http.response.status_code"] == 503


class Refusals:
    def __init__(self, answers):
        self.answers = answers

    async def __call__(self, message):
        return self.answers.pop(0) if self.answers else True


async def test_source_makes_one_trace_per_task_with_nack_and_backoff_events(spans):
    async def no_sleep(delay):
        return None

    tasks = [(f"batch:{index}", MatchState.model_validate(MATCH_STATE)) for index in range(3)]
    stats = source.SourceStats()

    await source.run(
        tasks,
        Refusals([False, False, True]),
        source.Backoff(0.02, 0.5, rng=random.Random(1)),
        stats,
        no_sleep,
    )

    roots = by_name(spans.get_finished_spans(), source.TASK_SPAN)
    assert [span.attributes["task.id"] for span in roots] == ["batch:0", "batch:1", "batch:2"]
    assert all(span.parent is None for span in roots)
    assert len({span.context.trace_id for span in roots}) == 3
    first = roots[0]
    assert event_names(first) == ["broker.nack", "backoff", "broker.nack", "backoff"]
    assert first.events[1].attributes["delay_s"] > 0
    assert first.attributes["attempts"] == 3
    assert event_names(roots[1]) == []
    assert stats.waited == pytest.approx(
        sum(event.attributes["delay_s"] for event in first.events if event.name == "backoff")
    )
