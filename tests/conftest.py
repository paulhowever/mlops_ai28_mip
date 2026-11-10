import asyncio
from contextlib import AsyncExitStack

import httpx
import pytest
from asgi_lifespan import LifespanManager
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dota_winprob.app import create_app
from dota_winprob.clients import broker as broker_client
from dota_winprob.clients import opendota, postgres, tasks_repo
from dota_winprob.config import Settings
from dota_winprob.telemetry import metric_views
from tests import fakes

SPAN_EXPORTER = InMemorySpanExporter()
METRIC_READER = InMemoryMetricReader()


@pytest.fixture(scope="session", autouse=True)
def telemetry_providers():
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(SPAN_EXPORTER))
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(
        MeterProvider(
            metric_readers=[METRIC_READER],
            views=metric_views(Settings(_env_file=None)),
        )
    )
    yield
    tracer_provider.shutdown()


@pytest.fixture
def spans():
    SPAN_EXPORTER.clear()
    yield SPAN_EXPORTER
    SPAN_EXPORTER.clear()


@pytest.fixture
def metric_reader():
    return METRIC_READER


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        environment="test",
        log_level="INFO",
        database_url="postgresql://postgres:postgres@db.invalid:5432/test",
        opendota_base_url="https://opendota.invalid",
        health_probe_timeout=0.2,
        mlflow_tracking_uri="http://mlflow.invalid",
        model_load_timeout=0.5,
        rabbitmq_url="amqp://guest:guest@rabbitmq.invalid:5672/",
        broker_reconnect_interval=0.05,
        worker_name="worker-test",
        otel_sdk_disabled=True,
        git_sha="abc1234",
        built_at="2026-09-24T00:00:00Z",
    )


@pytest.fixture
async def make_client(monkeypatch, settings):
    async with AsyncExitStack() as stack:

        async def factory(
            *,
            pool,
            opendota_handler,
            broker=None,
            broker_connects=True,
            publisher=None,
            queue_load=(0, 1),
            tasks=None,
        ):
            async def fake_create_pool(_):
                return pool

            def fake_create_client(settings):
                return httpx.AsyncClient(
                    base_url=settings.opendota_base_url,
                    transport=httpx.MockTransport(opendota_handler),
                )

            async def fake_connect(broker, settings):
                if not broker_connects:
                    raise ConnectionError("rabbitmq.invalid: connection refused")

            async def fake_queue_load(broker):
                return queue_load

            async def fake_ensure_schema(pool):
                return None

            monkeypatch.setattr(postgres, "create_pool", fake_create_pool)
            monkeypatch.setattr(opendota, "create_client", fake_create_client)
            monkeypatch.setattr(
                broker_client, "create_broker", lambda _: broker or fakes.FakeBroker()
            )
            monkeypatch.setattr(broker_client, "connect_and_declare", fake_connect)
            monkeypatch.setattr(broker_client, "publish_task", publisher or fakes.FakePublisher())
            monkeypatch.setattr(broker_client, "queue_load", fake_queue_load)
            monkeypatch.setattr(tasks_repo, "ensure_schema", fake_ensure_schema)
            monkeypatch.setattr(tasks_repo, "TaskRepository", lambda _: tasks or fakes.FakeTasks())

            application = create_app(settings)
            manager = await stack.enter_async_context(LifespanManager(application))
            if broker_connects:
                async with asyncio.timeout(1):
                    await application.state.broker_ready.wait()
            return await stack.enter_async_context(
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=manager.app),
                    base_url="http://testserver",
                )
            )

        yield factory
