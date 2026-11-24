import asyncio
import contextlib
from urllib.parse import urlsplit

import aio_pika
import pytest
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from dota_winprob.agent.actions.models import ActionStatus
from dota_winprob.agent.actions.service import ActionService
from dota_winprob.agent.actions.store import ActionConflict, ActionStore
from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.agent.sources import dead_letters as source
from dota_winprob.agent.sources.dead_letters import DeadLetters, MessageAtRisk
from dota_winprob.clients import broker as broker_client
from tests.agent.test_store import Clock
from tests.fakes import BrokenPredictor, FakePredictor
from tests.test_integration_pipeline import (
    broker,
    clean,
    config,
    eventually,
    fill_queue,
    message,
    postgres_url,
    queue_depth,
    rabbitmq_url,
    start_worker,
)

__all__ = ["broker", "clean", "config", "postgres_url", "rabbitmq_url"]

pytestmark = pytest.mark.integration

DLQ = broker_client.DEAD_LETTER_QUEUE_NAME
WORK = broker_client.QUEUE_NAME
ORIGINAL_TRACE = "0af7651916cd43dd8448eb211c80319c"
TRACEPARENT = f"00-{ORIGINAL_TRACE}-b7ad6b7169203331-01"


class Proxy:
    def __init__(self, url):
        target = urlsplit(url)
        self._target = (target.hostname, target.port)
        self._writers = set()
        self._server = None
        self.url = None

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"amqp://guest:guest@127.0.0.1:{port}/"
        return self

    async def __aexit__(self, *_):
        self.drop()
        self._server.close()

    def drop(self):
        for writer in self._writers:
            writer.transport.abort()
        self._writers.clear()

    async def _serve(self, reader, writer):
        upstream, downstream = await asyncio.open_connection(*self._target)
        self._writers |= {writer, downstream}
        await asyncio.gather(self._pump(reader, downstream), self._pump(upstream, writer))

    async def _pump(self, reader, writer):
        with contextlib.suppress(OSError):
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()


def make_dead_letters(url, **overrides):
    options = {
        "scan_limit": 200,
        "timeout": 5.0,
        "publish_timeout": 2.0,
        "execution_timeout": 20.0,
    }
    return DeadLetters(url, **(options | overrides))


def make_service(url, clock=None, **overrides):
    options = {} if clock is None else {"clock": clock}
    store = ActionStore(ttl=300, capacity=20, **options)
    return ActionService(store, make_dead_letters(url, **overrides))


@contextlib.asynccontextmanager
async def channel_to(url):
    connection = await aio_pika.connect(url)
    try:
        yield await connection.channel()
    finally:
        await connection.close()


async def bury(url, *items):
    async with channel_to(url) as channel:
        for item in items:
            body = item if isinstance(item, bytes) else message(item).model_dump_json().encode()
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body,
                    headers={
                        "traceparent": TRACEPARENT,
                        broker_client.ATTEMPT_HEADER: 3,
                        "x-last-death-reason": "rejected",
                    },
                    message_id="m",
                    correlation_id="request-1",
                    delivery_mode=2,
                ),
                routing_key=DLQ,
            )


async def take(queue):
    async def taken():
        return await queue.get(fail=False) is not None

    await eventually(taken)


async def drain(url, name):
    async with channel_to(url) as channel:
        queue = await channel.get_queue(name)
        found = []
        while (item := await queue.get(fail=False, no_ack=True)) is not None:
            found.append(item)
        return found


def ids(messages):
    found = []
    for item in messages:
        with contextlib.suppress(ValueError):
            found.append(source.TaskMessage.model_validate_json(item.body).task_id)
    return found


async def propose(service, limit=50):
    _, action = await service.plan_requeue(limit)
    return action


async def test_plan_counts_and_leaves_dlq_untouched(broker, config):
    await bury(config.rabbitmq_url, "batch:1", b"not json", "batch:2", b"{}", "batch:3")
    service = make_service(config.rabbitmq_url)

    plan, action = await service.plan_requeue(2)

    assert (plan.depth, plan.scanned, plan.valid, plan.garbage) == (5, 5, 3, 2)
    assert [task.task_id for task in plan.tasks] == ["batch:1", "batch:2"]
    assert (plan.tasks[0].reason, plan.tasks[0].attempts) == ("rejected", 3)
    assert action.status is ActionStatus.proposed
    assert await queue_depth(config.rabbitmq_url, WORK) == 0
    left = await drain(config.rabbitmq_url, DLQ)
    assert [item.body for item in left][1::2] == [b"not json", b"{}"]
    assert ids(left) == ["batch:1", "batch:2", "batch:3"]


async def test_inspections_never_use_up_delivery_limit(broker, config):
    await bury(config.rabbitmq_url, "batch:1", b"junk", "batch:2")
    dead = make_dead_letters(config.rabbitmq_url)

    for _ in range(30):
        inspection = await dead.inspect()
        assert (inspection.depth, inspection.garbage) == (3, 1)
        assert [task.task_id for task in inspection.tasks] == ["batch:1", "batch:2"]

    left = await drain(config.rabbitmq_url, DLQ)
    assert [item.headers.get("x-delivery-count") for item in left] == [None] * 3
    assert {item.headers["x-acquired-count"] for item in left} == {30}


async def test_message_near_delivery_limit_stops_inspection(broker, config):
    await bury(config.rabbitmq_url, "batch:1")
    for _ in range(source.RISKY_DELIVERIES):
        async with channel_to(config.rabbitmq_url) as channel:
            queue = await channel.get_queue(DLQ)
            await take(queue)

    async def back():
        return await queue_depth(config.rabbitmq_url, DLQ) == 1

    await eventually(back)
    dead = make_dead_letters(config.rabbitmq_url)
    for _ in range(10):
        with pytest.raises(MessageAtRisk):
            await dead.inspect()

    [left] = await drain(config.rabbitmq_url, DLQ)
    assert left.headers["x-delivery-count"] == source.RISKY_DELIVERIES


async def test_confirmed_plan_returns_valid_once_and_keeps_garbage(broker, config, spans):
    await bury(config.rabbitmq_url, b"not json", "batch:1", b"{}", "batch:2")
    service = make_service(config.rabbitmq_url)
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("вопрос"):
        action = await propose(service)

    await service.confirm(action.id)

    assert action.status is ActionStatus.done
    assert action.result.returned == ["batch:1", "batch:2"]
    assert (action.result.missing, action.result.uncertain, action.result.dlq_left) == (0, [], 2)
    left = await drain(config.rabbitmq_url, DLQ)
    assert [item.body for item in left] == [b"not json", b"{}"]
    returned = await drain(config.rabbitmq_url, WORK)
    assert ids(returned) == ["batch:1", "batch:2"]
    for item in returned:
        assert set(item.headers) == {"traceparent"}
        assert item.correlation_id == "request-1"
        assert item.delivery_mode == aio_pika.DeliveryMode.PERSISTENT
    assert returned[0].message_id == "batch:1"

    publishes = [span for span in spans.get_finished_spans() if span.name == source.RETURN_SPAN]
    [execution] = [
        span for span in spans.get_finished_spans() if span.name == "execute_action requeue_dlq"
    ]
    carried = {
        trace.get_current_span(
            TraceContextTextMapPropagator().extract({"traceparent": item.headers["traceparent"]})
        )
        .get_span_context()
        .trace_id
        for item in returned
    }
    assert carried == {span.context.trace_id for span in publishes}
    assert int(ORIGINAL_TRACE, 16) not in carried
    for span in publishes:
        assert {link.context.trace_id for link in span.links} == {
            int(ORIGINAL_TRACE, 16),
            execution.context.trace_id,
        }
    assert format(execution.links[0].context.trace_id, "032x") == action.view().trace_id


async def test_returned_tasks_are_computed_by_the_worker(broker, config):
    broken, _, _ = await start_worker(config, BrokenPredictor())
    for index in range(2):
        assert await broker_client.publish_task(broker, message(f"batch:{index}"), 2.0)
    await bury(config.rabbitmq_url, b"not a task")

    async def dead():
        return await queue_depth(config.rabbitmq_url, DLQ) == 3

    await eventually(dead)
    await broken.stop()
    service = make_service(config.rabbitmq_url)
    action = await propose(service)

    await service.confirm(action.id)
    worker, _, tasks = await start_worker(config, FakePredictor())

    async def computed():
        return len(tasks.results) == 2

    await eventually(computed)
    await worker.stop()
    assert await queue_depth(config.rabbitmq_url, DLQ) == 1
    assert await queue_depth(config.rabbitmq_url, WORK) == 0


async def test_dead_letters_of_delivery_limit_are_returnable(broker, config):
    assert await broker_client.publish_task(broker, message("batch:7"), 2.0)
    for _ in range(config.delivery_limit + 1):
        async with channel_to(config.rabbitmq_url) as channel:
            queue = await channel.get_queue(WORK)
            await take(queue)

    async def dead():
        return await queue_depth(config.rabbitmq_url, DLQ) == 1

    await eventually(dead)
    plan, _ = await make_service(config.rabbitmq_url).plan_requeue(5)

    assert [(task.task_id, task.reason) for task in plan.tasks] == [("batch:7", "delivery_limit")]


async def test_full_work_queue_stops_and_keeps_the_rest(broker, config):
    filled = await fill_queue(broker, config)
    async with channel_to(config.rabbitmq_url) as channel:
        queue = await channel.get_queue(WORK)
        for _ in range(filled - config.queue_max_length + 2):
            await queue.get(no_ack=True)
    buried = [f"batch:{index}" for index in range(100, 110)]
    await bury(config.rabbitmq_url, *buried)
    service = make_service(config.rabbitmq_url)
    action = await propose(service)

    await service.confirm(action.id)

    assert action.status is ActionStatus.partial
    assert "отклонил" in action.result.stopped
    assert action.result.uncertain == []
    returned = action.result.returned
    assert 0 < len(returned) < len(buried)
    assert ids(await drain(config.rabbitmq_url, DLQ)) == buried[len(returned) :]
    work = ids(await drain(config.rabbitmq_url, WORK))
    assert [task for task in work if task in buried] == returned


async def test_changed_dlq_returns_only_what_was_shown(broker, config):
    await bury(config.rabbitmq_url, "batch:1", "batch:2")
    service = make_service(config.rabbitmq_url)
    action = await propose(service)
    async with channel_to(config.rabbitmq_url) as channel:
        queue = await channel.get_queue(DLQ)
        await (await queue.get()).ack()
    await bury(config.rabbitmq_url, "batch:1", "batch:9", b"junk")

    await service.confirm(action.id)

    assert action.status is ActionStatus.done
    assert action.result.returned == ["batch:2"]
    assert action.result.missing == 1
    assert ids(await drain(config.rabbitmq_url, WORK)) == ["batch:2"]
    left = await drain(config.rabbitmq_url, DLQ)
    assert ids(left) == ["batch:1", "batch:9"]
    assert left[-1].body == b"junk"


async def test_repeated_and_parallel_confirmations_run_once(broker, config):
    await bury(config.rabbitmq_url, *[f"batch:{index}" for index in range(4)])
    service = make_service(config.rabbitmq_url)
    action = await propose(service)

    results = await asyncio.gather(
        *(service.confirm(action.id) for _ in range(5)), return_exceptions=True
    )

    assert sum(isinstance(result, ActionConflict) for result in results) == 4
    with pytest.raises(ActionConflict):
        await service.confirm(action.id)
    with pytest.raises(ActionConflict):
        service.reject(action.id)
    assert len(ids(await drain(config.rabbitmq_url, WORK))) == 4
    assert await queue_depth(config.rabbitmq_url, DLQ) == 0


async def test_two_overlapping_plans_do_not_duplicate(broker, config):
    await bury(config.rabbitmq_url, "batch:1", "batch:2")
    service = make_service(config.rabbitmq_url)
    first, second = await propose(service), await propose(service)

    await asyncio.gather(service.confirm(first.id), service.confirm(second.id))

    assert sorted(ids(await drain(config.rabbitmq_url, WORK))) == ["batch:1", "batch:2"]
    assert {first.result.missing, second.result.missing} == {0, 2}


async def test_expired_plan_touches_nothing(broker, config):
    await bury(config.rabbitmq_url, "batch:1")
    clock = Clock()
    service = make_service(config.rabbitmq_url, clock)
    action = await propose(service)
    clock.now += 301

    with pytest.raises(ActionConflict) as conflict:
        await service.confirm(action.id)

    assert conflict.value.status is ActionStatus.expired
    assert await queue_depth(config.rabbitmq_url, DLQ) == 1
    assert await queue_depth(config.rabbitmq_url, WORK) == 0


@pytest.mark.parametrize("when", ["before_publish", "after_confirm"])
async def test_lost_connection_never_loses_a_task(broker, config, monkeypatch, when):
    buried = ["batch:1", "batch:2", "batch:3"]
    await bury(config.rabbitmq_url, *buried)
    async with Proxy(config.rabbitmq_url) as proxy:
        service = make_service(proxy.url)
        action = await propose(service)
        original = DeadLetters._publish
        published = []

        async def flaky(self, exchange, incoming, task, outcome):
            if len(published) == 1 and when == "before_publish":
                proxy.drop()
            result = await original(self, exchange, incoming, task, outcome)
            published.append(task.task_id)
            if len(published) == 2 and when == "after_confirm":
                proxy.drop()
                await asyncio.sleep(0.1)
            return result

        monkeypatch.setattr(DeadLetters, "_publish", flaky)
        async with asyncio.timeout(15):
            await service.confirm(action.id)

    assert action.status is ActionStatus.partial
    assert action.result.stopped is not None
    assert action.result.uncertain == ["batch:2"]

    async def returned():
        return await queue_depth(config.rabbitmq_url, DLQ) == 2

    await eventually(returned)
    assert ids(await drain(config.rabbitmq_url, DLQ)) == ["batch:2", "batch:3"]
    expected = ["batch:1"] if when == "before_publish" else ["batch:1", "batch:2"]
    assert ids(await drain(config.rabbitmq_url, WORK)) == expected


async def test_unconfirmed_publish_keeps_the_task(broker, config, monkeypatch):
    await bury(config.rabbitmq_url, "batch:1", "batch:2")
    service = make_service(config.rabbitmq_url, publish_timeout=0.2)
    action = await propose(service)

    async def silent(self, *args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(aio_pika.Exchange, "publish", silent)
    await service.confirm(action.id)

    assert action.status is ActionStatus.partial
    assert (action.result.returned, action.result.uncertain) == ([], ["batch:1"])
    assert ids(await drain(config.rabbitmq_url, DLQ)) == ["batch:1", "batch:2"]


async def test_missing_work_queue_stops_the_return(broker, config):
    await bury(config.rabbitmq_url, "batch:1")
    service = make_service(config.rabbitmq_url)
    action = await propose(service)
    async with channel_to(config.rabbitmq_url) as channel:
        await channel.queue_delete(WORK)

    await service.confirm(action.id)

    assert action.status is ActionStatus.failed
    assert "очереди нет" in action.result.stopped
    assert ids(await drain(config.rabbitmq_url, DLQ)) == ["batch:1"]


async def test_cancelled_inspection_returns_everything(broker, config):
    await bury(config.rabbitmq_url, *[f"batch:{index}" for index in range(30)])
    dead = make_dead_letters(config.rabbitmq_url)
    for delay in (0.02, 0.03, 0.04, 0.05):
        running = asyncio.create_task(dead.inspect())
        await asyncio.sleep(delay)
        running.cancel()
        await asyncio.gather(running, return_exceptions=True)

    await dead.close(5.0)

    async def back():
        return await queue_depth(config.rabbitmq_url, DLQ) == 30

    await eventually(back)
    left = await drain(config.rabbitmq_url, DLQ)
    assert ids(left) == [f"batch:{index}" for index in range(30)]
    assert {item.headers.get("x-delivery-count") for item in left} == {None}


async def test_execution_timeout_keeps_the_rest(broker, config, monkeypatch):
    await bury(config.rabbitmq_url, "batch:1", "batch:2")
    service = make_service(config.rabbitmq_url, execution_timeout=0.2)
    action = await propose(service)
    original = DeadLetters._publish

    async def slow(self, exchange, incoming, task, outcome):
        result = await original(self, exchange, incoming, task, outcome)
        await asyncio.sleep(0.3)
        return result

    monkeypatch.setattr(DeadLetters, "_publish", slow)
    await service.confirm(action.id)

    assert action.status is ActionStatus.partial
    assert "время" in action.result.stopped
    assert ids(await drain(config.rabbitmq_url, DLQ)) == ["batch:2"]


async def test_dead_broker_is_unavailable_not_lost(broker, config):
    await bury(config.rabbitmq_url, "batch:1")
    dead = make_dead_letters("amqp://guest:guest@127.0.0.1:1/")

    with pytest.raises(SourceUnavailable):
        await dead.inspect()
    service = make_service(config.rabbitmq_url)
    action = await propose(service)
    service._dead_letters = dead
    await service.confirm(action.id)

    assert action.status is ActionStatus.failed
    assert await queue_depth(config.rabbitmq_url, DLQ) == 1


async def test_missing_dlq_is_unavailable(broker, config):
    async with channel_to(config.rabbitmq_url) as channel:
        await channel.queue_delete(DLQ)

    with pytest.raises(SourceUnavailable):
        await make_dead_letters(config.rabbitmq_url).inspect()


async def test_scan_limit_is_announced(broker, config):
    await bury(config.rabbitmq_url, *[f"batch:{index}" for index in range(6)])
    service = make_service(config.rabbitmq_url, scan_limit=3)

    plan, action = await service.plan_requeue(50)
    await service.confirm(action.id)

    assert (plan.depth, plan.scanned, len(plan.tasks)) == (6, 3, 3)
    assert any("Просмотрено только начало" in note for note in plan.notes)
    assert ids(await drain(config.rabbitmq_url, DLQ)) == ["batch:3", "batch:4", "batch:5"]
