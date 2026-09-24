from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack

import httpx
import pytest
from asgi_lifespan import LifespanManager

from dota_winprob.app import create_app
from dota_winprob.clients import opendota, postgres
from dota_winprob.config import Settings

OpenDotaHandler = Callable[[httpx.Request], httpx.Response]
ClientFactory = Callable[..., Awaitable[httpx.AsyncClient]]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        log_level="INFO",
        database_url="postgresql://postgres:postgres@db.invalid:5432/test",
        opendota_base_url="https://opendota.invalid",
        health_probe_timeout=0.2,
        git_sha="abc1234",
        built_at="2026-09-24T00:00:00Z",
    )


@pytest.fixture
async def make_client(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> AsyncIterator[ClientFactory]:
    async with AsyncExitStack() as stack:

        async def factory(*, pool: object, opendota_handler: OpenDotaHandler) -> httpx.AsyncClient:
            async def fake_create_pool(_: Settings) -> object:
                return pool

            def fake_create_client(settings: Settings) -> httpx.AsyncClient:
                return httpx.AsyncClient(
                    base_url=settings.opendota_base_url,
                    transport=httpx.MockTransport(opendota_handler),
                )

            monkeypatch.setattr(postgres, "create_pool", fake_create_pool)
            monkeypatch.setattr(opendota, "create_client", fake_create_client)

            manager = await stack.enter_async_context(LifespanManager(create_app(settings)))
            return await stack.enter_async_context(
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=manager.app),
                    base_url="http://testserver",
                )
            )

        yield factory
