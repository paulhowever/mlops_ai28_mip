from contextlib import AsyncExitStack

import httpx
import pytest
from asgi_lifespan import LifespanManager
from pydantic_ai.models.function import FunctionModel

from dota_winprob.agent import app as agent_app
from dota_winprob.agent.actions.service import ActionService, expired
from dota_winprob.agent.actions.store import ActionStore
from dota_winprob.agent.config import AgentSettings
from dota_winprob.agent.llm.client import LLM, guard
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.service import build_agent
from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.agent.tools.catalog import build_toolset
from dota_winprob.agent.tools.deps import AgentDeps
from tests.agent.fakes import (
    TEST_KEY,
    FakeBackends,
    FakeDeadLetters,
    ReadOnlyPool,
    call,
    payload,
    say,
    tool_outputs,
)

BASE_URL = "http://127.0.0.1:8010"

TASKS_CSV = """match_id,radiant_gold,dire_gold,gold_advantage,radiant_xp,dire_xp,xp_advantage,radiant_lh,dire_lh,lh_advantage
0,13844,14271,-427,14366,13525,841,132,171,-39
566,14321,15902,-1581,13803,15717,-1914,140,128,12
"""


@pytest.fixture
def agent_settings(tmp_path):
    tasks_file = tmp_path / "tasks.csv"
    tasks_file.write_text(TASKS_CSV, encoding="utf-8")
    return AgentSettings(
        _env_file=None,
        llm_api_key=TEST_KEY,
        llm_base_url="http://llm.test/v1",
        llm_model="bonsai2-27b",
        agent_api_url="http://api.test",
        agent_prometheus_url="http://prometheus.test",
        agent_loki_url="http://loki.test",
        agent_tempo_url="http://tempo.test",
        mlflow_tracking_uri="http://mlflow.test",
        agent_tasks_file=tasks_file,
        agent_predict_timeout=0.3,
        agent_predict_poll_interval=0.01,
        agent_tool_timeout=1.0,
        database_url="postgresql://postgres:postgres@db.invalid:5432/test",
        agent_database_url=None,
        rabbitmq_url="amqp://guest:guest@rabbitmq.invalid:5672/",
        model_name="dota-winprob",
        model_alias="champion",
    )


@pytest.fixture
def backends():
    return FakeBackends()


@pytest.fixture
def pool():
    return ReadOnlyPool()


@pytest.fixture
async def sources(agent_settings, backends, pool):
    bundle = Sources.create(agent_settings, pool, httpx.MockTransport(backends))
    yield bundle
    await bundle.close()


@pytest.fixture
def dead_letters():
    return FakeDeadLetters()


@pytest.fixture
def actions(agent_settings, dead_letters):
    store = ActionStore(
        ttl=agent_settings.agent_action_ttl,
        capacity=agent_settings.agent_action_capacity,
        on_expired=expired,
    )
    return ActionService(store, dead_letters)


@pytest.fixture
def deps():
    return AgentDeps(redactor=Redactor([TEST_KEY]))


@pytest.fixture
def run_tool(agent_settings, sources, actions, deps):
    async def run(name, /, **arguments):
        outputs = []

        def respond(messages, info):
            outputs.extend(tool_outputs(messages))
            return say("готово") if outputs else call(name, **arguments)

        agent = build_agent(
            agent_settings,
            build_toolset(agent_settings, sources, actions),
            FunctionModel(respond),
        )
        await agent.run("вопрос", deps=deps)
        return payload(outputs[0]), deps.steps[-1], outputs[0]

    return run


@pytest.fixture
async def make_agent_client(monkeypatch, settings, agent_settings, backends, pool, dead_letters):
    async with AsyncExitStack() as stack:

        async def factory(script=None, **overrides):
            configured = agent_settings.model_copy(update=overrides)

            async def fake_create_pool(_):
                return pool

            monkeypatch.setattr(agent_app.tasks_db, "create_pool", fake_create_pool)
            monkeypatch.setattr(agent_app, "DeadLetters", lambda *_, **__: dead_letters)
            if script is not None:
                monkeypatch.setattr(
                    agent_app,
                    "build_llm",
                    lambda settings, gate, transport: LLM(
                        guard(FunctionModel(script, model_name="bonsai2-27b"), gate, settings)
                    ),
                )
            application = agent_app.create_app(
                configured,
                settings.model_copy(update={"app_name": "dota-agent"}),
                source_transport=httpx.MockTransport(backends),
            )
            factory.app = application
            manager = await stack.enter_async_context(LifespanManager(application))
            return await stack.enter_async_context(
                httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app), base_url=BASE_URL)
            )

        yield factory
