import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dota_winprob import __version__
from dota_winprob.api import health, tasks, version
from dota_winprob.clients import broker as broker_client
from dota_winprob.clients import opendota, postgres, tasks_repo
from dota_winprob.config import Settings, get_settings
from dota_winprob.logging_config import REQUEST_ID, setup_logging

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


RECOVERABLE_ERRORS = (*broker_client.PUBLISH_ERRORS, OSError, asyncpg.PostgresError)


async def keep_trying(
    name: str,
    action: Callable[[], Awaitable[None]],
    interval: float,
) -> None:
    while True:
        try:
            await action()
        except RECOVERABLE_ERRORS as error:
            log.warning(
                "Зависимость пока недоступна, повторю попытку",
                extra={"component": name, "error": repr(error), "retry_in_s": interval},
            )
            await asyncio.sleep(interval)
        else:
            log.info("Зависимость готова", extra={"component": name})
            return


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    app.state.db_pool = await postgres.create_pool(settings)
    app.state.tasks = tasks_repo.TaskRepository(app.state.db_pool)
    app.state.opendota = opendota.create_client(settings)
    app.state.broker = broker_client.create_broker(settings)
    app.state.broker_ready = asyncio.Event()

    async def connect_broker() -> None:
        async with asyncio.timeout(settings.broker_reconnect_interval):
            await broker_client.connect_and_declare(app.state.broker, settings)
        app.state.broker_ready.set()

    async def prepare_schema() -> None:
        await tasks_repo.ensure_schema(app.state.db_pool)

    background = [
        asyncio.create_task(
            keep_trying("rabbitmq", connect_broker, settings.broker_reconnect_interval)
        ),
        asyncio.create_task(
            keep_trying("postgres", prepare_schema, settings.broker_reconnect_interval)
        ),
    ]
    log.info("Приложение запущено")
    try:
        yield
    finally:
        for task in background:
            task.cancel()
        await asyncio.gather(*background, return_exceptions=True)
        await app.state.broker.stop()
        await opendota.close_client(app.state.opendota)
        await postgres.close_pool(app.state.db_pool)
        log.info("Приложение остановлено")


async def _call_next_or_500(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    try:
        return await call_next(request)
    except Exception:
        log.exception("Необработанное исключение")
        return JSONResponse({"detail": "Internal Server Error"}, status_code=500)


async def request_context(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    token = REQUEST_ID.set(request_id)
    started = time.perf_counter()
    try:
        response = await _call_next_or_500(request, call_next)
        response.headers[REQUEST_ID_HEADER] = request_id
        level = logging.DEBUG if request.url.path == health.LIVENESS_PATH else logging.INFO
        log.log(
            level,
            "Запрос обработан",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response
    finally:
        REQUEST_ID.reset(token)


async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    detail = [
        {key: value for key, value in item.items() if key != "input"} for item in error.errors()
    ]
    return JSONResponse(
        {"detail": jsonable_encoder(detail)},
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.middleware("http")(request_context)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.include_router(health.router)
    app.include_router(health.v1_router)
    app.include_router(version.router)
    app.include_router(tasks.router)
    return app


app = create_app()
