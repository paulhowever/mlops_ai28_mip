import asyncio
import threading
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd

from dota_winprob.clients.registry import LoadedModel
from dota_winprob.schemas import MatchState, TaskMessage, TaskSource, TaskView

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


class FakeBroker:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.stopped = False

    async def ping(self, **options: object) -> bool:
        return self.alive

    async def stop(self) -> None:
        self.stopped = True


class FakePublisher:
    def __init__(self, accept: bool = True) -> None:
        self.accept = accept
        self.messages: list[TaskMessage] = []
        self.correlation_ids: list[str | None] = []

    async def __call__(self, broker, message, confirm_timeout, correlation_id=None) -> bool:
        self.messages.append(message)
        self.correlation_ids.append(correlation_id)
        return self.accept


class FakeTasks:
    def __init__(self, pool=None) -> None:
        self.pending: list[TaskMessage] = []
        self.results: dict[str, dict] = {}
        self.failed: dict[str, dict] = {}
        self.views: dict[str, TaskView] = {}
        self.broken = False

    async def insert_pending(self, message: TaskMessage) -> None:
        if self.broken:
            raise OSError("connection refused")
        self.pending.append(message)

    async def save_result(self, message, win_probability, model_version, attempt, worker) -> bool:
        if self.broken:
            raise OSError("connection refused")
        if message.task_id in self.results:
            return False
        self.results[message.task_id] = {
            "win_probability": win_probability,
            "model_version": model_version,
            "attempt": attempt,
            "worker": worker,
        }
        return True

    async def mark_failed(self, task_id, source, error, attempt, worker) -> None:
        self.failed[task_id] = {"source": source, "error": error, "attempt": attempt}

    async def fetch(self, task_id: str) -> TaskView | None:
        return self.views.get(task_id)


def task_message(task_id: str = "batch:1") -> TaskMessage:
    return TaskMessage(
        task_id=task_id,
        source=TaskSource.batch,
        features=MatchState.model_validate(MATCH_STATE),
        published_at=datetime(2026, 10, 17, 12, 0, tzinfo=UTC),
    )
