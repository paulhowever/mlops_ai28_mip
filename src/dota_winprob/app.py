import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from dota_winprob import __version__
from dota_winprob.api import health, version
from dota_winprob.clients import opendota, postgres
from dota_winprob.config import Settings, get_settings

log = logging.getLogger(__name__)


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(health.v1_router)
    app.include_router(version.router)
    return app


app = create_app()
