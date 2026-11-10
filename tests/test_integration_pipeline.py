import asyncio
import time
from functools import partial

import aio_pika
import asyncpg
import docker
import httpx
import pytest
from asgi_lifespan import LifespanManager
from testcontainers.community.postgres import PostgresContainer
from testcontainers.community.rabbitmq import RabbitMqContainer

from dota_winprob import source
from dota_winprob.app import create_app
from dota_winprob.clients import broker as broker_client
from dota_winprob.clients import tasks_repo
from dota_winprob.schemas import MatchState, TaskMessage, TaskSource, TaskStatus
from dota_winprob.services.pipeline import Pipeline
from dota_winprob.services.scoring import Scorer
from dota_winprob.worker import WorkerState, build_broker
from tests.fakes import MATCH_STATE, BrokenPredictor, FakePredictor, FakeTasks, loaded_model

pytestmark = pytest.mark.integration

RABBITMQ_IMAGE = "rabbitmq:4.3-management"
POSTGRES_IMAGE = "postgres:16-alpine"


def docker_available() -> bool:
    try:
        docker.from_env().ping()
    except (docker.errors.DockerException, OSError):
        return False
    return True


@pytest.fixture(scope="session")
def rabbitmq_url():
    if not docker_available():
        pytest.skip("Docker недоступен")
    with RabbitMqContainer(RABBITMQ_IMAGE) as container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(container.port)
        yield f"amqp://guest:guest@{host}:{port}/"


@pytest.fixture(scope="session")
def postgres_url():
    if not docker_available():
        pytest.skip("Docker недоступен")
    with PostgresContainer(POSTGRES_IMAGE, driver=None) as container:
        yield container.get_connection_url()


@pytest.fixture
async def clean(rabbitmq_url, postgres_url):
    connection = await aio_pika.connect(rabbitmq_url)
    channel = await connection.channel()
    for name in (
        broker_client.QUEUE_NAME,
        broker_client.RETRY_QUEUE_NAME,
        broker_client.DEAD_LETTER_QUEUE_NAME,
    ):
        await channel.queue_delete(name)
    await connection.close()

    db = await asyncpg.connect(postgres_url)
    await db.execute("DROP TABLE IF EXISTS tasks")
    await db.close()


@pytest.fixture
def config(settings, rabbitmq_url, postgres_url, clean):
    return settings.model_copy(
        update={
            "rabbitmq_url": rabbitmq_url,
            "database_url": postgres_url,
            "queue_max_length": 5,
            "retry_delay_ms": 100,
            "inference_delay_ms": 0,
            "publish_timeout": 2.0,
            "graceful_timeout": 5.0,
        }
    )


@pytest.fixture
async def broker(config):
    instance = broker_client.create_broker(config)
    await broker_client.connect_and_declare(instance, config)
    yield instance
    await instance.stop()


async def queue_depth(url: str, name: str) -> int:
    connection = await aio_pika.connect(url)
    channel = await connection.channel()
    queue = await channel.declare_queue(name, passive=True)
    await connection.close()
    return queue.declaration_result.message_count or 0


def message(task_id: str, task_source: TaskSource = TaskSource.batch) -> TaskMessage:
    return source._message(task_id, MatchState.model_validate(MATCH_STATE)).model_copy(
        update={"source": task_source}
    )


async def fill_queue(broker, config) -> int:
    accepted = 0
    for index in range(config.queue_max_length * 3):
        if not await broker_client.publish_task(broker, message(f"batch:{index}"), 2.0):
            return accepted
        accepted += 1
    return accepted


async def eventually(predicate, attempts: int = 200) -> None:
    for _ in range(attempts):
        if await predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("условие не выполнилось за отведённое время")


async def test_full_queue_nacks_publish(broker, config, rabbitmq_url):
    results = [
        await broker_client.publish_task(broker, message(f"batch:{index}"), 2.0)
        for index in range(12)
    ]

    accepted = results.count(True)
    assert config.queue_max_length <= accepted <= config.queue_max_length + 1
    assert results == [True] * accepted + [False] * (12 - accepted)
    assert await queue_depth(rabbitmq_url, broker_client.QUEUE_NAME) == accepted


async def test_api_sheds_load_without_pending_rows(broker, config, postgres_url):
    await fill_queue(broker, config)
    application = create_app(config)

    async with LifespanManager(application):
        await asyncio.wait_for(application.state.broker_ready.wait(), 10)
        await tasks_repo.ensure_schema(application.state.db_pool)
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://testserver"
        )

        started = time.perf_counter()
        response = await client.post("/api/v1/tasks", json=MATCH_STATE)
        elapsed = time.perf_counter() - started

        assert response.status_code == 503
        assert elapsed < config.publish_timeout
        assert response.headers["Retry-After"] == str(config.retry_after_max)
        db = await asyncpg.connect(postgres_url)
        assert await db.fetchval("SELECT count(*) FROM tasks") == 0

        channel = await (await aio_pika.connect(config.rabbitmq_url)).channel()
        await (await channel.get_queue(broker_client.QUEUE_NAME)).purge()

        response = await client.post("/api/v1/tasks", json=MATCH_STATE)
        assert response.status_code == 202
        statuses = await db.fetch("SELECT status FROM tasks")
        assert [row["status"] for row in statuses] == [TaskStatus.pending.value]
        await db.close()
        await client.aclose()


async def test_source_waits_for_space_and_delivers_everything(broker, config):
    total = 20
    stats = source.SourceStats()
    tasks = ((f"batch:{index}", MatchState.model_validate(MATCH_STATE)) for index in range(total))
    running = asyncio.create_task(
        source.run(
            tasks,
            partial(broker_client.publish_task, broker, confirm_timeout=2.0),
            source.Backoff(0.01, 0.1),
            stats,
        )
    )

    await asyncio.sleep(1.0)
    assert not running.done()
    assert stats.published <= config.queue_max_length + 1
    assert 0 < stats.rejected < 60

    consumed: list[str] = []
    connection = await aio_pika.connect(config.rabbitmq_url)
    channel = await connection.channel()
    queue = await channel.get_queue(broker_client.QUEUE_NAME)

    async def consume(incoming: aio_pika.abc.AbstractIncomingMessage) -> None:
        consumed.append(TaskMessage.model_validate_json(incoming.body).task_id)
        await incoming.ack()

    await queue.consume(consume)
    await asyncio.wait_for(running, 15)

    async def all_consumed() -> bool:
        return len(consumed) == total

    await eventually(all_consumed)
    await connection.close()
    assert consumed == [f"batch:{index}" for index in range(total)]
    assert stats.published == total


async def start_worker(config, predictor, delay=0.0):
    tasks = FakeTasks()
    state = WorkerState()
    worker = build_broker(config, state)
    await worker.connect()
    await broker_client.declare_topology(worker, config, with_tasks_queue=False)
    state.pipeline = Pipeline(
        repository=tasks,
        scorer=Scorer(loaded_model(predictor), config.worker_concurrency, delay),
        publish_retry=partial(broker_client.publish_retry, worker, confirm_timeout=2.0),
        max_attempts=config.max_attempts,
        worker="worker-it",
    )
    await worker.start()
    return worker, state, tasks


async def test_poison_and_failing_tasks_end_in_dlq(config, rabbitmq_url):
    worker, _, tasks = await start_worker(config, BrokenPredictor())
    connection = await aio_pika.connect(rabbitmq_url)
    channel = await connection.channel()
    exchange = await channel.get_exchange(broker_client.TASKS_EXCHANGE.name)
    await exchange.publish(aio_pika.Message(b"not json"), routing_key=broker_client.ROUTING_KEY)
    await exchange.publish(
        aio_pika.Message(message("batch:1").model_dump_json().encode()),
        routing_key=broker_client.ROUTING_KEY,
    )

    async def both_dead() -> bool:
        return await queue_depth(rabbitmq_url, broker_client.DEAD_LETTER_QUEUE_NAME) == 2

    await eventually(both_dead)
    await worker.stop()

    dlq = await channel.get_queue(broker_client.DEAD_LETTER_QUEUE_NAME)
    dead = [await dlq.get(no_ack=True), await dlq.get(no_ack=True)]
    by_body = {item.body: item for item in dead}
    assert by_body[b"not json"].headers["x-last-death-reason"] == "rejected"
    failing = next(item for item in dead if item.body != b"not json")
    assert failing.headers["x-last-death-reason"] == "rejected"
    assert failing.headers[broker_client.ATTEMPT_HEADER] == config.max_attempts
    assert tasks.failed["batch:1"]["attempt"] == config.max_attempts
    await connection.close()


async def test_worker_takes_only_prefetch_count(config, broker, rabbitmq_url):
    for index in range(config.queue_max_length):
        assert await broker_client.publish_task(broker, message(f"batch:{index}"), 2.0)
    worker, state, _ = await start_worker(config, FakePredictor(), delay=1.0)

    async def busy() -> bool:
        return state.require_pipeline().in_flight > 0

    await eventually(busy)
    await asyncio.sleep(0.3)
    in_flight = state.require_pipeline().in_flight
    ready = await queue_depth(rabbitmq_url, broker_client.QUEUE_NAME)
    await worker.stop()

    assert in_flight == config.prefetch_count
    assert ready == config.queue_max_length - config.prefetch_count


async def test_graceful_stop_finishes_taken_tasks(config, broker, rabbitmq_url):
    worker, state, tasks = await start_worker(config, FakePredictor(), delay=0.6)
    half = config.prefetch_count // 2

    async def taken(count: int):
        async def check() -> bool:
            return state.require_pipeline().in_flight == count

        return check

    for index in range(half):
        assert await broker_client.publish_task(broker, message(f"batch:{index}"), 2.0)
    await eventually(await taken(half))
    await asyncio.sleep(0.3)
    for index in range(half, config.queue_max_length):
        assert await broker_client.publish_task(broker, message(f"batch:{index}"), 2.0)
    await eventually(await taken(config.prefetch_count))

    await broker_client.stop_consuming(worker)
    await worker.stop()

    assert len(tasks.results) == config.prefetch_count
    assert state.require_pipeline().in_flight == 0
    remaining = config.queue_max_length - config.prefetch_count
    assert await queue_depth(rabbitmq_url, broker_client.QUEUE_NAME) == remaining

    connection = await aio_pika.connect(rabbitmq_url)
    queue = await (await connection.channel()).get_queue(broker_client.QUEUE_NAME)
    left = [await queue.get(no_ack=True) for _ in range(remaining)]
    await connection.close()
    assert [item.headers.get("x-delivery-count") for item in left] == [None] * remaining


async def test_repository_is_idempotent(postgres_url, clean):
    pool = await asyncpg.create_pool(postgres_url)
    await asyncio.gather(*(tasks_repo.ensure_schema(pool) for _ in range(4)))
    repository = tasks_repo.TaskRepository(pool)
    batch = message("batch:1")

    saved = await asyncio.gather(
        *(repository.save_result(batch, 0.7, "4", 1, f"w{index}") for index in range(10))
    )
    await repository.insert_pending(batch)
    await repository.mark_failed(batch.task_id, batch.source, "boom", 2, "w9")

    assert saved.count(True) == 1
    view = await repository.fetch(batch.task_id)
    assert view.status is TaskStatus.done
    assert view.win_probability == 0.7

    api = message("api-1", TaskSource.api)
    await repository.insert_pending(api)
    assert (await repository.fetch(api.task_id)).status is TaskStatus.pending
    assert await repository.save_result(api, 0.4, "4", 1, "w1")
    assert (await repository.fetch(api.task_id)).status is TaskStatus.done
    assert await pool.fetchval("SELECT count(*) FROM tasks") == 2
    await pool.close()
