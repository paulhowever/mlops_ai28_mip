from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from dota_winprob.agent.actions.models import ActionStatus, ActionView
from dota_winprob.agent.actions.service import ActionService
from dota_winprob.agent.actions.store import ActionConflict, ActionUnknown
from dota_winprob.agent.schemas import ErrorResponse
from dota_winprob.agent.tracing import current_trace_id

ACTIONS_PATH = "/agent/actions"
DECISIONS = {
    404: {"model": ErrorResponse, "description": "Предложения нет."},
    409: {"model": ErrorResponse, "description": "Решение уже принято."},
    410: {"model": ErrorResponse, "description": "Срок предложения истёк."},
}

router = APIRouter(prefix=ACTIONS_PATH, tags=["actions"])


def _service(request: Request) -> ActionService:
    return request.app.state.actions


@router.get("", response_model=list[ActionView], summary="Предложения, новые первыми")
async def list_actions(request: Request) -> list[ActionView]:
    return [action.view() for action in _service(request).store.all()]


@router.post(
    "/{action_id}/confirm",
    response_model=ActionView,
    summary="Подтвердить и выполнить предложение",
    description=(
        "Решение человека. Выполняется ровно показанный план и только один раз. Статус done — план "
        "выполнен, partial — возврат остановлен на середине, failed — ничего не возвращено."
    ),
    responses=DECISIONS,
)
async def confirm(request: Request, action_id: str) -> ActionView:
    return (await _service(request).confirm(action_id)).view()


@router.post(
    "/{action_id}/reject",
    response_model=ActionView,
    summary="Отклонить предложение",
    responses=DECISIONS,
)
async def reject(request: Request, action_id: str) -> ActionView:
    return _service(request).reject(action_id).view()


def _error(status_code: int, reason: str, detail: str) -> JSONResponse:
    body = ErrorResponse(detail=detail, reason=reason, trace_id=current_trace_id())
    return JSONResponse(body.model_dump(), status_code=status_code)


async def unknown_error(request: Request, error: ActionUnknown) -> JSONResponse:
    return _error(404, "unknown", "Такого предложения нет")


async def conflict_error(request: Request, error: ActionConflict) -> JSONResponse:
    if error.status is ActionStatus.expired:
        return _error(410, "expired", "Срок предложения истёк, попросите агента составить новое")
    return _error(409, error.status.value, "Решение по предложению уже принято")
