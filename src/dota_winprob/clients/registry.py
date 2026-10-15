from dataclasses import dataclass
from typing import Protocol

import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient

from dota_winprob.config import Settings


class Predictor(Protocol):
    def predict(self, data: pd.DataFrame) -> object: ...


@dataclass(frozen=True)
class LoadedModel:
    name: str
    alias: str
    version: str
    run_id: str
    features: tuple[str, ...]
    predictor: Predictor


def model_uri(settings: Settings) -> str:
    return f"models:/{settings.model_name}@{settings.model_alias}"


def load_model(settings: Settings) -> LoadedModel:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    predictor = mlflow.pyfunc.load_model(model_uri(settings))
    version = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)

    if version.run_id != predictor.metadata.run_id:
        raise RuntimeError(
            f"алиас {settings.model_alias} переехал на версию {version.version} во время загрузки"
        )

    schema = predictor.metadata.get_input_schema()
    if schema is None:
        raise ValueError("у модели нет сигнатуры, порядок признаков неизвестен")

    return LoadedModel(
        name=settings.model_name,
        alias=settings.model_alias,
        version=str(version.version),
        run_id=version.run_id,
        features=tuple(schema.input_names()),
        predictor=predictor,
    )
