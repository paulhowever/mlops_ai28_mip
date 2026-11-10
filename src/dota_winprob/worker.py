import asyncio
import json
import logging
from dataclasses import dataclass, field
from functools import partial

import asyncpg
from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, AsgiResponse, get
from faststream.asgi.handlers import GetHandler
from faststream.asgi.types import Scope
from faststream.rabbit import Channel, RabbitBroker
from faststream.rabbit.annotations import RabbitMessage

from dota_winprob.clients import broker as broker_client
from dota_winprob.clients import tasks_repo
from dota_winprob.config import Settings, get_settings
from dota_winprob.logging_config import REQUEST_ID, setup_logging
from dota_winprob.services import inference
from dota_winprob.services.loop_lag import LoopLagMonitor
from dota_winprob.services.pipeline import Pipeline
from dota_winprob.services.scoring import Scorer

log = logging.getLogger(__name__)


@dataclass
class WorkerState:
    pipeline: Pipeline | None = None
    pool: asyncpg.Pool | None = None
    monitor: LoopLagMonitor = field(default_factory=LoopLagMonitor)

    def require_pipeline(self) -> Pipeline:
        if self.pipeline is None:
            raise RuntimeError("воркер ещё не инициализирован")
        return self.pipeline


def build_broker(settings: Settings, state: WorkerState) -> RabbitBroker:
    broker = broker_client.create_broker(settings)

    @broker.subscriber(
        broker_client.tasks_queue(settings),
        broker_client.TASKS_EXCHANGE,
        channel=Channel(prefetch_count=settings.prefetch_count or None),
        ack_policy=AckPolicy.REJECT_ON_ERROR,
    )
    async def handle(message: RabbitMessage) -> None:
        token = REQUEST_ID.set(message.correlation_id or message.message_id)
        try:
            await state.require_pipeline().handle(
                message.body,
                message.headers,
                message.correlation_id,
            )
        finally:
            REQUEST_ID.reset(token)

    return broker


def make_health(broker: RabbitBroker, state: WorkerState, settings: Settings) -> GetHandler:
    @get(include_in_schema=False)
    async def health(scope: Scope) -> AsgiResponse:
        connected = await broker.ping(timeout=settings.health_probe_timeout)
        pipeline = state.pipeline
        ready = connected and pipeline is not None
        payload = {
            "status": "ok" if ready else "unavailable",
            "worker": settings.worker_name,
            "broker_connected": connected,
            "model_version": pipeline.scorer.model.version if pipeline else None,
            "prefetch_count": settings.prefetch_count,
            "concurrency": settings.worker_concurrency,
            "in_flight": pipeline.in_flight if pipeline else 0,
            "processed": pipeline.processed if pipeline else 0,
            "duplicates": pipeline.duplicates if pipeline else 0,
            "retried": pipeline.retried if pipeline else 0,
            "failed": pipeline.failed if pipeline else 0,
            "loop_lag_ms": state.monitor.last_ms,
            "loop_lag_max_ms": state.monitor.max_ms,
        }
        return AsgiResponse(
            json.dumps(payload).encode(),
            status_code=200 if ready else 503,
            headers={"content-type": "application/json"},
        )

    return health


def create_app(settings: Settings | None = None) -> AsgiFastStream:
    settings = settings or get_settings()
    setup_logging(settings)
    state = WorkerState()
    broker = build_broker(settings, state)

    async def startup() -> None:
        model = await inference.load_model(settings)
        state.pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=1,
            max_size=settings.worker_concurrency + 1,
        )
        await tasks_repo.ensure_schema(state.pool)
        await broker.connect()
        await broker_client.declare_topology(broker, settings, with_tasks_queue=False)
        state.pipeline = Pipeline(
            repository=tasks_repo.TaskRepository(state.pool),
            scorer=Scorer(model, settings.worker_concurrency, settings.inference_delay),
            publish_retry=partial(
                broker_client.publish_retry,
                broker,
                confirm_timeout=settings.publish_timeout,
            ),
            max_attempts=settings.max_attempts,
            worker=settings.worker_name,
        )

    async def started() -> None:
        state.monitor.start()
        if settings.prefetch_count == 0 or settings.prefetch_count > settings.worker_concurrency:
            log.warning(
                "prefetch_count больше пула потоков: лишние сообщения ждут в памяти воркера",
                extra={
                    "prefetch_count": settings.prefetch_count,
                    "concurrency": settings.worker_concurrency,
                },
            )
        log.info(
            "Воркер запущен",
            extra={
                "worker": settings.worker_name,
                "prefetch_count": settings.prefetch_count,
                "concurrency": settings.worker_concurrency,
                "inference_delay_ms": settings.inference_delay_ms,
            },
        )

    async def stopped() -> None:
        await state.monitor.stop()
        pipeline = state.pipeline
        if pipeline is not None:
            await asyncio.to_thread(pipeline.scorer.close)
        if state.pool is not None:
            await state.pool.close()
        log.info(
            "Воркер остановлен",
            extra={
                "worker": settings.worker_name,
                "processed": pipeline.processed if pipeline else 0,
                "in_flight": pipeline.in_flight if pipeline else 0,
            },
        )

    async def stopping() -> None:
        pipeline = state.pipeline
        log.info(
            "Воркер останавливается",
            extra={"in_flight": pipeline.in_flight if pipeline else 0},
        )
        await broker_client.stop_consuming(broker)

    return AsgiFastStream(
        broker,
        asgi_routes=[("/health", make_health(broker, state, settings))],
        on_startup=[startup],
        after_startup=[started],
        on_shutdown=[stopping],
        after_shutdown=[stopped],
    )


app = create_app()
