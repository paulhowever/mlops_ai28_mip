import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from dota_winprob.clients.registry import LoadedModel
from dota_winprob.schemas import MatchState, ProcessResponse
from dota_winprob.services import inference

log = logging.getLogger(__name__)

PROCESS_PATH = "/process"

router = APIRouter(tags=["inference"])


@router.post(
    PROCESS_PATH,
    response_model=ProcessResponse,
    summary="Вероятность победы Radiant",
    description=(
        "Принимает суммарные золото, опыт и добивания команд на 10-й минуте матча и "
        "возвращает вероятность победы Radiant вместе с версией модели, которая её посчитала."
    ),
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Инференс завершился ошибкой."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Модель не загружена на старте."},
    },
)
async def process(request: Request, state: MatchState) -> ProcessResponse:
    model: LoadedModel | None = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Модель не загружена: {request.app.state.model_error}",
        )

    context = {
        "model_name": model.name,
        "model_version": model.version,
        "model_alias": model.alias,
    }
    started = time.perf_counter()
    try:
        probability = await inference.predict(model, state)
    except Exception:
        log.exception("Инференс завершился ошибкой", extra=context)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Инференс завершился ошибкой",
        ) from None

    log.info(
        "Инференс выполнен",
        extra={
            **context,
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
            "win_probability": round(probability, 4),
        },
    )
    return ProcessResponse(win_probability=probability, **context)
