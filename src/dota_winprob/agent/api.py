import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dota_winprob.agent.errors import AskError
from dota_winprob.agent.health import collect_probes
from dota_winprob.agent.schemas import AskRequest, AskResponse, ErrorResponse
from dota_winprob.agent.service import AgentService, current_trace_id
from dota_winprob.schemas import HealthReport, LivenessResponse
from dota_winprob.services.health import build_report

log = logging.getLogger(__name__)

LIVENESS_PATH = "/healthz"
HEALTH_PATH = "/health"
ASK_PATH = "/agent/ask"

router = APIRouter()


@router.get(LIVENESS_PATH, response_model=LivenessResponse, tags=["health"], summary="Liveness")
async def liveness() -> LivenessResponse:
    return LivenessResponse(status="ok")


@router.get(
    HEALTH_PATH,
    response_model=HealthReport,
    tags=["health"],
    summary="Доступность источников агента",
    description=(
        "Опрашивает источники параллельно. Недоступный источник даёт статус degraded, но не 503: "
        "агент отвечает по остальным. Шлюз языковой модели не опрашивается, чтобы не тратить лимит."
    ),
)
async def health(request: Request) -> HealthReport:
    state = request.app.state
    return await build_report(
        collect_probes(
            state.sources,
            llm_configured=state.service.agent is not None,
            model=state.agent_settings.llm_model,
        ),
        probe_timeout=state.settings.health_probe_timeout,
    )


@router.post(
    ASK_PATH,
    response_model=AskResponse,
    tags=["agent"],
    summary="Задать вопрос агенту",
    responses={
        502: {"model": ErrorResponse, "description": "Шлюз отклонил запрос или ответ непригоден."},
        503: {
            "model": ErrorResponse,
            "description": "Нет ключа, исчерпан лимит или шлюз недоступен.",
            "headers": {
                "Retry-After": {
                    "description": "Через сколько секунд повторить. Нет, если повтор не поможет.",
                    "schema": {"type": "integer"},
                }
            },
        },
        504: {"model": ErrorResponse, "description": "Вопрос не уложился в отведённое время."},
    },
)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    service: AgentService = request.app.state.service
    return await service.ask(body)


async def ask_error(request: Request, error: AskError) -> JSONResponse:
    log.warning(
        "Вопрос агенту без ответа",
        extra={"reason": error.reason, "retry_after": error.retry_after},
    )
    body = ErrorResponse(detail=error.detail, reason=error.reason, trace_id=current_trace_id())
    headers = {} if error.retry_after is None else {"Retry-After": str(error.retry_after)}
    return JSONResponse(body.model_dump(), status_code=error.status_code, headers=headers)
