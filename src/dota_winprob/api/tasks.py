import logging

from fastapi import APIRouter, HTTPException, Request, Response, status

from dota_winprob.logging_config import REQUEST_ID
from dota_winprob.schemas import MatchState, TaskAccepted, TaskStatus, TaskView
from dota_winprob.services import tasks as tasks_service

log = logging.getLogger(__name__)

TASKS_PATH = "/tasks"

router = APIRouter(prefix="/api/v1", tags=["tasks"])


@router.post(
    TASKS_PATH,
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskAccepted,
    summary="Поставить матч в очередь на расчёт вероятности победы",
    description=(
        "Принимает суммарные золото, опыт и добивания команд на 10-й минуте матча и ставит "
        "задачу в очередь. Результат считает воркер, его видно по адресу из заголовка Location. "
        "Если очередь заполнена, ответ 503 приходит сразу, задача не создаётся, "
        "а Retry-After подсказывает, когда очередь успеет разгрузиться."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Очередь заполнена или брокер недоступен. Задача не создана.",
            "headers": {
                "Retry-After": {
                    "description": "Через сколько секунд имеет смысл повторить запрос.",
                    "schema": {"type": "integer"},
                }
            },
        },
    },
)
async def create_task(request: Request, response: Response, features: MatchState) -> TaskAccepted:
    state = request.app.state
    try:
        message = await tasks_service.submit(
            state.broker,
            state.broker_ready.is_set(),
            state.tasks,
            state.settings,
            features,
            correlation_id=REQUEST_ID.get(),
        )
    except tasks_service.Overloaded as overload:
        log.warning(
            "Задача отклонена: система перегружена",
            extra={"reason": overload.reason, "retry_after": overload.retry_after},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Задача не принята: {overload.reason}",
            headers={"Retry-After": str(overload.retry_after)},
        ) from None

    status_url = f"{router.prefix}{TASKS_PATH}/{message.task_id}"
    response.headers["Location"] = status_url
    log.info("Задача поставлена в очередь", extra={"task_id": message.task_id})
    return TaskAccepted(task_id=message.task_id, status=TaskStatus.pending, status_url=status_url)


@router.get(
    TASKS_PATH + "/{task_id}",
    response_model=TaskView,
    summary="Статус и результат задачи",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Задачи с таким id нет."}},
)
async def get_task(request: Request, task_id: str) -> TaskView:
    task = await request.app.state.tasks.fetch(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена")
    return task
