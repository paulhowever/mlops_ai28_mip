import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import httpx2
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from dota_winprob import __version__
from dota_winprob.agent import api
from dota_winprob.agent.actions import api as actions_api
from dota_winprob.agent.actions.service import ActionService, expired
from dota_winprob.agent.actions.store import ActionConflict, ActionStore, ActionUnknown
from dota_winprob.agent.config import AgentSettings, base_settings, get_agent_settings
from dota_winprob.agent.errors import AskError
from dota_winprob.agent.incidents import api as incidents_api
from dota_winprob.agent.incidents.service import IncidentService
from dota_winprob.agent.incidents.store import IncidentStore
from dota_winprob.agent.llm.client import build_gate, build_llm
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.security import HttpGuard
from dota_winprob.agent.service import AgentService, build_agents
from dota_winprob.agent.sources import tasks_db
from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.agent.sources.dead_letters import DeadLetters
from dota_winprob.agent.web import api as web_api
from dota_winprob.api.middleware import request_context, validation_error
from dota_winprob.config import Settings
from dota_winprob.logging_config import setup_logging
from dota_winprob.telemetry import excluded_urls, instrument_libraries, setup_telemetry

log = logging.getLogger(__name__)

POLLED_URLS = "/agent/actions$,/agent/incidents$,/agent/info$,/static/"
CONTENT_LOGGERS = ("openai", "httpx", "httpx2", "httpcore", "httpcore2", "pydantic_ai")


def silence_content_loggers() -> None:
    for name in CONTENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: AgentSettings = app.state.agent_settings
    telemetry = setup_telemetry(app.state.settings)
    pool = await tasks_db.create_pool(settings)
    sources = app.state.sources = Sources.create(settings, pool, app.state.source_transport)
    actions = app.state.actions = build_actions(settings)
    redactor = Redactor([settings.api_key, settings.alerts_token])
    llm = build_llm(settings, app.state.gate, app.state.llm_transport)
    service = app.state.service = AgentService(
        settings=settings,
        redactor=redactor,
        agents=None if llm is None else build_agents(settings, sources, actions, llm.model),
    )
    incidents = app.state.incidents = build_incidents(settings, sources, redactor, service)
    incidents.start()
    log.info(
        "Агент запущен",
        extra={"llm_configured": llm is not None, "llm_model": settings.llm_model},
    )
    try:
        yield
    finally:
        await incidents.close()
        await actions.close(app.state.settings.graceful_timeout)
        if llm is not None:
            await llm.close()
        await sources.close()
        await pool.close()
        log.info("Агент остановлен")
        await asyncio.to_thread(telemetry.shutdown, app.state.settings.telemetry_shutdown_timeout)


def build_actions(settings: AgentSettings) -> ActionService:
    store = ActionStore(
        ttl=settings.agent_action_ttl,
        capacity=settings.agent_action_capacity,
        on_expired=expired,
    )
    return ActionService(
        store,
        DeadLetters(
            settings.rabbitmq_url,
            scan_limit=settings.agent_dlq_scan_limit,
            timeout=settings.agent_source_timeout,
            publish_timeout=settings.agent_publish_timeout,
            execution_timeout=settings.agent_dlq_timeout,
        ),
    )


def build_incidents(
    settings: AgentSettings, sources: Sources, redactor: Redactor, service: AgentService
) -> IncidentService:
    store = IncidentStore(
        capacity=settings.agent_incident_capacity,
        cooldown=settings.agent_incident_cooldown,
        redactor=redactor,
    )
    return IncidentService(settings, sources, redactor, service, store)


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
    app.add_middleware(
        HttpGuard, hosts=agent_settings.allowed_hosts, alert_hosts=agent_settings.alert_hosts
    )
    app.middleware("http")(request_context)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(AskError, api.ask_error)
    app.add_exception_handler(ActionUnknown, actions_api.unknown_error)
    app.add_exception_handler(ActionConflict, actions_api.conflict_error)
    app.add_exception_handler(incidents_api.AlertsUnauthorized, incidents_api.unauthorized_error)
    for router in (api.router, actions_api.router, incidents_api.router, web_api.router):
        app.include_router(router)
    app.mount("/static", StaticFiles(directory=web_api.STATIC), name="static")
    instrument_libraries()
    FastAPIInstrumentor.instrument_app(app, excluded_urls=f"{excluded_urls()},{POLLED_URLS}")
    return app


app = create_app()
