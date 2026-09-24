import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from dota_winprob import __version__
from dota_winprob.api import health, version
from dota_winprob.clients import opendota, postgres
from dota_winprob.config import Settings, get_settings
from dota_winprob.logging_config import REQUEST_ID, setup_logging

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
LIVENESS_PATH = "/healthz"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    app.state.db_pool = await postgres.create_pool(settings)
    app.state.opendota = opendota.create_client(settings)
    log.info("Приложение запущено")
    try:
        yield
    finally:
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
        level = logging.DEBUG if request.url.path == LIVENESS_PATH else logging.INFO
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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.middleware("http")(request_context)
    app.include_router(health.router)
    app.include_router(health.v1_router)
    app.include_router(version.router)
    return app


app = create_app()
