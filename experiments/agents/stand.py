import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import asyncpg
from pydantic_ai import FunctionToolset
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from dota_winprob.agent.app import build_actions
from dota_winprob.agent.config import AgentSettings
from dota_winprob.agent.llm.client import build_gate, build_llm
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.sources import tasks_db
from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.agent.tools.catalog import build_toolset
from dota_winprob.agent.tools.deps import AgentDeps


class TimedModel(WrapperModel):
    def __init__(self, wrapped: Model) -> None:
        super().__init__(wrapped)
        self.calls = 0
        self.seconds = 0.0

    def reset(self) -> None:
        self.calls, self.seconds = 0, 0.0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        started = time.perf_counter()
        try:
            return await self.wrapped.request(messages, model_settings, model_request_parameters)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - started


@dataclass
class Stand:
    settings: AgentSettings
    sources: Sources
    pool: asyncpg.Pool
    model: Model
    timer: TimedModel
    redactor: Redactor = field(default_factory=Redactor)

    def toolset(self) -> FunctionToolset[AgentDeps]:
        return build_toolset(self.settings, self.sources, build_actions(self.settings))


@asynccontextmanager
async def connect() -> AsyncIterator[Stand]:
    settings = AgentSettings()
    llm = build_llm(settings, build_gate(settings))
    if llm is None:
        raise SystemExit("LLM_API_KEY не задан")
    timer = TimedModel(llm.model.wrapped)
    llm.model.wrapped = timer
    pool = await tasks_db.create_pool(settings)
    sources = Sources.create(settings, pool)
    try:
        yield Stand(settings, sources, pool, llm.model, timer, Redactor([settings.api_key]))
    finally:
        await llm.close()
        await sources.close()
        await pool.close()
