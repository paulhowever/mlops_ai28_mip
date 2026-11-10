import asyncio

from dota_winprob.schemas import TaskSource, TaskStatus, TaskView
from dota_winprob.services import tasks as tasks_service
from tests.fakes import (
    MATCH_STATE,
    FakeBroker,
    FakePublisher,
    FakeTasks,
    HealthyPool,
    opendota_ok,
)


async def test_accepted_task_gets_202_and_pending_row(make_client):
    publisher, tasks = FakePublisher(), FakeTasks()
    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, publisher=publisher, tasks=tasks
    )

    response = await client.post(
        "/api/v1/tasks", json=MATCH_STATE, headers={"X-Request-ID": "req-1"}
    )

    assert response.status_code == 202
    body = response.json()
    [message] = publisher.messages
    assert body == {
        "task_id": message.task_id,
        "status": "pending",
        "status_url": f"/api/v1/tasks/{message.task_id}",
    }
    assert response.headers["Location"] == body["status_url"]
    assert message.source is TaskSource.api
    assert message.features.model_dump() == MATCH_STATE
    assert publisher.correlation_ids == ["req-1"]
    assert [pending.task_id for pending in tasks.pending] == [message.task_id]


async def test_full_queue_gives_503_with_retry_after_and_no_row(make_client):
    tasks = FakeTasks()
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        publisher=FakePublisher(accept=False),
        queue_load=(100, 1),
        tasks=tasks,
    )

    response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "5"
    assert "очередь задач заполнена" in response.json()["detail"]
    assert tasks.pending == []


async def test_retry_after_shrinks_with_second_worker(make_client):
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        publisher=FakePublisher(accept=False),
        queue_load=(100, 2),
    )

    response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.headers["Retry-After"] == "3"


async def test_503_comes_back_immediately(make_client, settings):
    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, publisher=FakePublisher(accept=False)
    )

    async with asyncio.timeout(settings.publish_timeout):
        response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.status_code == 503


async def test_broker_unavailable_gives_503(make_client):
    publisher, tasks = FakePublisher(), FakeTasks()
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        broker_connects=False,
        publisher=publisher,
        tasks=tasks,
    )

    response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert "нет связи с брокером" in response.json()["detail"]
    assert publisher.messages == []
    assert tasks.pending == []


async def test_pending_write_failure_still_accepts(make_client):
    tasks = FakeTasks()
    tasks.broken = True
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok, tasks=tasks)

    response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.status_code == 202


async def test_invalid_features_give_422_without_publish(make_client):
    publisher = FakePublisher()
    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, publisher=publisher
    )

    response = await client.post("/api/v1/tasks", json={**MATCH_STATE, "radiant_gold": -1})

    assert response.status_code == 422
    assert publisher.messages == []


async def test_get_task_returns_result(make_client):
    tasks = FakeTasks()
    tasks.views["abc"] = TaskView(
        task_id="abc",
        source=TaskSource.api,
        status=TaskStatus.done,
        win_probability=0.73,
        model_version="4",
        attempts=1,
    )
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok, tasks=tasks)

    response = await client.get("/api/v1/tasks/abc")

    assert response.status_code == 200
    assert response.json()["win_probability"] == 0.73
    assert response.json()["model_version"] == "4"


async def test_get_unknown_task_gives_404(make_client):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/tasks/missing")

    assert response.status_code == 404


async def test_health_reports_broker(make_client):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    rabbitmq = response.json()["components"]["rabbitmq"]
    assert rabbitmq["status"] == "healthy"
    assert rabbitmq["critical"] is True
    assert rabbitmq["detail"] == "очередь tasks.score"


async def test_health_is_503_without_broker(make_client):
    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, broker_connects=False
    )

    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["components"]["rabbitmq"]["detail"] == (
        "ConnectionError: нет подключения к RabbitMQ"
    )


async def test_health_is_503_when_broker_stops_answering(make_client):
    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, broker=FakeBroker(alive=False)
    )

    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["components"]["rabbitmq"]["detail"] == (
        "ConnectionError: RabbitMQ не отвечает"
    )


def test_retry_after_formula(settings):
    assert tasks_service.estimate_retry_after(100, 1, settings) == 5
    assert tasks_service.estimate_retry_after(100, 2, settings) == 3
    assert tasks_service.estimate_retry_after(0, 1, settings) == 1
    assert tasks_service.estimate_retry_after(100, 0, settings) == settings.retry_after_max
    assert tasks_service.estimate_retry_after(10**6, 1, settings) == settings.retry_after_max
