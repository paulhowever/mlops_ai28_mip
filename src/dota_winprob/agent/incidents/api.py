from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dota_winprob.agent.incidents.models import AlertsAccepted, AlertsPayload, Incident
from dota_winprob.agent.incidents.service import IncidentService
from dota_winprob.agent.schemas import ErrorResponse
from dota_winprob.agent.security import ALERTS_PATH, token_matches

INCIDENTS_PATH = "/agent/incidents"

router = APIRouter(tags=["incidents"])


class AlertsUnauthorized(Exception):
    pass


def _service(request: Request) -> IncidentService:
    return request.app.state.incidents


@router.post(
    ALERTS_PATH,
    response_model=AlertsAccepted,
    status_code=202,
    summary="Webhook алертов Grafana",
    description="Отвечает сразу, разбор идёт в фоне и ничего не меняет в конвейере.",
    responses={401: {"model": ErrorResponse, "description": "Неверный общий секрет."}},
)
async def alerts(request: Request, payload: AlertsPayload) -> AlertsAccepted:
    expected = request.app.state.agent_settings.alerts_token
    if expected and not token_matches(request.headers.get("authorization"), expected):
        raise AlertsUnauthorized
    return _service(request).receive(payload)


@router.get(INCIDENTS_PATH, response_model=list[Incident], summary="Инциденты, новые первыми")
async def list_incidents(request: Request) -> list[Incident]:
    return _service(request).store.all()


async def unauthorized_error(request: Request, error: AlertsUnauthorized) -> JSONResponse:
    body = ErrorResponse(detail="Нужен общий секрет webhook", reason="unauthorized")
    return JSONResponse(body.model_dump(), status_code=401, headers={"WWW-Authenticate": "Bearer"})
