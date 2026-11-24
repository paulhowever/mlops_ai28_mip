from dataclasses import dataclass

import asyncpg
import httpx

from dota_winprob.agent.config import AgentSettings
from dota_winprob.agent.sources.loki import Loki
from dota_winprob.agent.sources.matches import MatchFile
from dota_winprob.agent.sources.mlflow import MLflow
from dota_winprob.agent.sources.prometheus import Prometheus
from dota_winprob.agent.sources.tasks_api import TasksAPI
from dota_winprob.agent.sources.tasks_db import TasksReader
from dota_winprob.agent.sources.tempo import Tempo


@dataclass(frozen=True)
class Sources:
    prometheus: Prometheus
    loki: Loki
    tempo: Tempo
    mlflow: MLflow
    api: TasksAPI
    tasks: TasksReader
    matches: MatchFile

    @classmethod
    def create(
        cls,
        settings: AgentSettings,
        pool: asyncpg.Pool,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> "Sources":
        timeout = settings.agent_source_timeout
        return cls(
            prometheus=Prometheus("prometheus", settings.agent_prometheus_url, timeout, transport),
            loki=Loki("loki", settings.agent_loki_url, timeout, transport),
            tempo=Tempo("tempo", settings.agent_tempo_url, timeout, transport),
            mlflow=MLflow("mlflow", settings.mlflow_tracking_uri, timeout, transport),
            api=TasksAPI("api", settings.agent_api_url, timeout, transport),
            tasks=TasksReader(pool),
            matches=MatchFile(settings.agent_tasks_file),
        )

    async def close(self) -> None:
        for source in (self.prometheus, self.loki, self.tempo, self.mlflow, self.api):
            await source.close()
