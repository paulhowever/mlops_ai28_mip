import asyncio
import threading

import httpx
import numpy as np
import pandas as pd

from dota_winprob.clients.registry import LoadedModel

POSTGRES_BANNER = "PostgreSQL 16.4 (Debian 16.4-1.pgdg120+1) on x86_64-pc-linux-gnu"


class FakePool:
    def __init__(self) -> None:
        self.terminated = False

    async def acquire(self) -> "FakePool":
        return self

    async def release(self, connection: "FakePool") -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True

    async def close(self) -> None:
        pass


class HealthyPool(FakePool):
    async def fetchval(self, query: str) -> str:
        return POSTGRES_BANNER


class DeadPool(FakePool):
    async def fetchval(self, query: str) -> str:
        raise OSError("connection refused")


class HangingPool(FakePool):
    async def fetchval(self, query: str) -> str:
        return await asyncio.Future()


def opendota_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200)


def opendota_unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503)


MODEL_FEATURES = (
    "lh_advantage",
    "dire_lh",
    "radiant_lh",
    "xp_advantage",
    "dire_xp",
    "radiant_xp",
    "gold_advantage",
    "dire_gold",
    "radiant_gold",
)

MATCH_STATE = {
    "radiant_gold": 13500,
    "dire_gold": 12400,
    "gold_advantage": 1100,
    "radiant_xp": 14200,
    "dire_xp": 13600,
    "xp_advantage": 600,
    "radiant_lh": 125,
    "dire_lh": 116,
    "lh_advantage": 9,
}


class FakePredictor:
    def __init__(self, probability: float = 0.73) -> None:
        self.probability = probability
        self.frames: list[pd.DataFrame] = []
        self.threads: list[threading.Thread] = []

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        self.frames.append(data)
        self.threads.append(threading.current_thread())
        return np.array([[1 - self.probability, self.probability]])


class BrokenPredictor:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        raise ValueError("feature schema mismatch")


def loaded_model(
    predictor: object | None = None,
    version: str = "4",
    features: tuple[str, ...] = MODEL_FEATURES,
) -> LoadedModel:
    return LoadedModel(
        name="dota-winprob",
        alias="champion",
        version=version,
        run_id="run-4",
        features=features,
        predictor=predictor or FakePredictor(),
    )


def loader_for(model: LoadedModel):
    def load(settings) -> LoadedModel:
        return model

    return load


def mlflow_unavailable(settings) -> LoadedModel:
    raise ConnectionError("mlflow.invalid: Name or service not known")
