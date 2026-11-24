import math
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field

from dota_winprob.agent import queries
from dota_winprob.agent.envelope import ToolResult
from dota_winprob.agent.sources import matches
from dota_winprob.agent.sources.matches import MatchFile
from dota_winprob.agent.sources.mlflow import MLflow
from dota_winprob.agent.sources.tasks_api import TasksAPI
from dota_winprob.schemas import MatchState

METRIC_KEYS = (
    "roc_auc",
    "cv_roc_auc",
    "cv_roc_auc_std",
    "train_roc_auc",
    "log_loss",
    "brier_score",
    "accuracy",
)
Amount = Annotated[float, Field(ge=0, le=10_000_000, allow_inf_nan=False)]


def build_features(
    radiant_gold: float,
    dire_gold: float,
    radiant_xp: float,
    dire_xp: float,
    radiant_lh: float,
    dire_lh: float,
) -> MatchState:
    return MatchState(
        radiant_gold=radiant_gold,
        dire_gold=dire_gold,
        gold_advantage=radiant_gold - dire_gold,
        radiant_xp=radiant_xp,
        dire_xp=dire_xp,
        xp_advantage=radiant_xp - dire_xp,
        radiant_lh=radiant_lh,
        dire_lh=dire_lh,
        lh_advantage=radiant_lh - dire_lh,
    )


@dataclass(frozen=True)
class MatchTools:
    api: TasksAPI
    mlflow: MLflow
    matches: MatchFile
    model_name: str
    model_alias: str
    predict_timeout: float
    poll_interval: float

    async def model_card(self) -> ToolResult:
        version = await self.mlflow.champion(self.model_name, self.model_alias)
        data: dict[str, Any] = {"model": self.model_name, "alias": self.model_alias}
        if version is None:
            return ToolResult("mlflow", data, f"версии с алиасом {self.model_alias} нет", "empty")
        values = await self.mlflow.run_metrics(version["run_id"])
        data |= {
            "version": str(version["version"]),
            "run_id": version["run_id"],
            "description": version.get("description"),
            "metrics": {
                key: round(values[key], 4)
                for key in METRIC_KEYS
                if key in values and math.isfinite(values[key])
            },
            "inputs": "суммы золота, опыта и добиваний команд на 10-й минуте и их разности",
            "output": "вероятность победы Radiant",
        }
        summary = f"версия {data['version']}, ROC AUC {data['metrics'].get('roc_auc')}"
        return ToolResult("mlflow", data, summary)

    async def match_features(
        self,
        match_id: Annotated[
            int, Field(ge=0, le=10**12, description="Номер пакетного матча, для batch:566 это 566.")
        ],
    ) -> ToolResult:
        state = await self.matches.find(match_id)
        if state is None:
            return ToolResult(
                matches.NAME,
                {"match_id": match_id, "found": False},
                f"матча {match_id} в файле нет",
                "empty",
            )
        data = {
            "match_id": match_id,
            "task_id": f"batch:{match_id}",
            "features": state.model_dump(),
        }
        return ToolResult(matches.NAME, data, f"признаки матча {match_id}")

    async def predict(
        self,
        radiant_gold: Annotated[Amount, Field(description="Золото команды Radiant.")],
        dire_gold: Annotated[Amount, Field(description="Золото команды Dire.")],
        radiant_xp: Annotated[Amount, Field(description="Опыт команды Radiant.")],
        dire_xp: Annotated[Amount, Field(description="Опыт команды Dire.")],
        radiant_lh: Annotated[Amount, Field(description="Добивания крипов Radiant.")],
        dire_lh: Annotated[Amount, Field(description="Добивания крипов Dire.")],
    ) -> ToolResult:
        features = build_features(radiant_gold, dire_gold, radiant_xp, dire_xp, radiant_lh, dire_lh)
        submission = await self.api.submit(features)
        if submission.task_id is None:
            data = {
                "accepted": False,
                "http_status": 503,
                "retry_after_s": submission.retry_after,
                "reason": submission.reason,
            }
            return ToolResult("api", data, "API ответил 503, задача не создана", "rejected")
        return await self._outcome(submission.task_id, features)

    async def _outcome(self, raw_task_id: str, features: MatchState) -> ToolResult:
        task_id = queries.task_id(raw_task_id)
        view = await self.api.wait(task_id, self.predict_timeout, self.poll_interval)
        data: dict[str, Any] = {"task_id": task_id, "features": features.model_dump()}
        if view is not None and view["status"] == "failed":
            data |= {"status": "failed", "error": view.get("error")}
            return ToolResult("api", data, "воркер не смог посчитать задачу", "failed")
        if view is None or view["status"] != "done":
            data |= {
                "status": "pending",
                "waited_s": self.predict_timeout,
                "note": "результата ещё нет, позже его вернёт get_task по этому task_id",
            }
            return ToolResult("api", data, "результат не готов вовремя", "pending")
        data |= {
            "status": "done",
            "win_probability": float(view["win_probability"]),
            "model_version": view.get("model_version"),
            "meaning": "вероятность победы Radiant",
        }
        return ToolResult("api", data, f"вероятность Radiant {data['win_probability']:.4f}")
