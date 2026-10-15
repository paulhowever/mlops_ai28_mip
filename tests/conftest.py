from contextlib import AsyncExitStack

import httpx
import pytest
from asgi_lifespan import LifespanManager

from dota_winprob.app import create_app
from dota_winprob.clients import opendota, postgres, registry
from dota_winprob.config import Settings
from tests import fakes


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        environment="test",
        log_level="INFO",
        database_url="postgresql://postgres:postgres@db.invalid:5432/test",
        opendota_base_url="https://opendota.invalid",
        health_probe_timeout=0.2,
        mlflow_tracking_uri="http://mlflow.invalid",
        model_load_timeout=0.5,
        git_sha="abc1234",
        built_at="2026-09-24T00:00:00Z",
    )


@pytest.fixture
async def make_client(monkeypatch, settings):
    async with AsyncExitStack() as stack:

        async def factory(*, pool, opendota_handler, model_loader=None):
            async def fake_create_pool(_):
                return pool

            def fake_create_client(settings):
                return httpx.AsyncClient(
                    base_url=settings.opendota_base_url,
                    transport=httpx.MockTransport(opendota_handler),
                )

            monkeypatch.setattr(postgres, "create_pool", fake_create_pool)
            monkeypatch.setattr(opendota, "create_client", fake_create_client)
            monkeypatch.setattr(
                registry, "load_model", model_loader or fakes.loader_for(fakes.loaded_model())
            )

            manager = await stack.enter_async_context(LifespanManager(create_app(settings)))
            return await stack.enter_async_context(
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=manager.app),
                    base_url="http://testserver",
                )
            )

        yield factory
