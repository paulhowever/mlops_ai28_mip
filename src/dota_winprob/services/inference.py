import asyncio
import logging
import time

import numpy as np
import pandas as pd

from dota_winprob.clients import registry
from dota_winprob.clients.registry import LoadedModel
from dota_winprob.config import Settings
from dota_winprob.schemas import MatchState

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


def _describe(error: BaseException) -> str:
    message = str(error)
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message.splitlines()[0]}"


async def _fetch(settings: Settings) -> LoadedModel:
    if not settings.mlflow_tracking_uri:
        raise ModelUnavailableError("MLFLOW_TRACKING_URI не задан")

    try:
        async with asyncio.timeout(settings.model_load_timeout):
            model = await asyncio.to_thread(registry.load_model, settings)
    except TimeoutError as error:
        raise ModelUnavailableError(
            f"превышен таймаут загрузки {settings.model_load_timeout} с"
        ) from error
    except Exception as error:
        raise ModelUnavailableError(_describe(error)) from error

    missing = sorted(set(model.features) - set(MatchState.model_fields))
    if missing:
        raise ModelUnavailableError(
            f"модель ждёт признаки, которых нет в запросе: {', '.join(missing)}"
        )
    return model


async def load_model(settings: Settings) -> LoadedModel:
    started = time.perf_counter()
    try:
        model = await _fetch(settings)
    except ModelUnavailableError as error:
        log.error(
            "Модель не загружена, /process будет отвечать 503",
            extra={"model_uri": registry.model_uri(settings), "error": str(error)},
        )
        raise

    log.info(
        "Модель загружена",
        extra={
            "model_name": model.name,
            "model_alias": model.alias,
            "model_version": model.version,
            "model_run_id": model.run_id,
            "load_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return model


async def predict(model: LoadedModel, state: MatchState) -> float:
    frame = pd.DataFrame([state.model_dump()], columns=list(model.features))
    output = await asyncio.to_thread(model.predictor.predict, frame)
    return float(np.asarray(output)[0, 1])
