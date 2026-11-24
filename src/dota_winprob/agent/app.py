import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import httpx2
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from dota_winprob import __version__
from dota_winprob.agent import api
from dota_winprob.agent.config import AgentSettings, base_settings, get_agent_settings
from dota_winprob.agent.errors import AskError
from dota_winprob.agent.llm.client import build_gate, build_llm
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.service import AgentService, build_agent
from dota_winprob.agent.sources import tasks_db
from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.api.middleware import request_context, validation_error
from dota_winprob.config import Settings
from dota_winprob.logging_config import setup_logging
from dota_winprob.telemetry import excluded_urls, instrument_libraries, setup_telemetry

log = logging.getLogger(__name__)

CONTENT_LOGGERS = ("openai", "httpx", "httpx2", "httpcore", "httpcore2", "pydantic_ai")


def silence_content_loggers() -> None:
    for name in CONTENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: AgentSettings = app.state.agent_settings
    telemetry = setup_telemetry(app.state.settings)
    pool = await tasks_db.create_pool(settings)
    app.state.sources = Sources.create(settings, pool, app.state.source_transport)
    llm = build_llm(settings, app.state.gate, app.state.llm_transport)
    app.state.service = AgentService(
        settings=settings,
        redactor=Redactor([settings.api_key]),
        agent=None if llm is None else build_agent(settings, app.state.sources, llm.model),
    )
    log.info(
        "Агент запущен",
        extra={"llm_configured": llm is not None, "llm_model": settings.llm_model},
    )
    try:
        yield
    finally:
        if llm is not None:
            await llm.close()
        await app.state.sources.close()
        await pool.close()
        log.info("Агент остановлен")
        await asyncio.to_thread(telemetry.shutdown, app.state.settings.telemetry_shutdown_timeout)


def create_app(
    agent_settings: AgentSettings | None = None,
    settings: Settings | None = None,
    *,
    source_transport: httpx.AsyncBaseTransport | None = None,
    llm_transport: httpx2.AsyncBaseTransport | None = None,
) -> FastAPI:
    agent_settings = agent_settings or get_agent_settings()
    settings = settings or base_settings()
    setup_logging(settings)
    silence_content_loggers()
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.agent_settings = agent_settings
    app.state.gate = build_gate(agent_settings)
    app.state.source_transport = source_transport
    app.state.llm_transport = llm_transport
    app.middleware("http")(request_context)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(AskError, api.ask_error)
    app.include_router(api.router)
    instrument_libraries()
    FastAPIInstrumentor.instrument_app(app, excluded_urls=excluded_urls())
    return app


app = create_app()
