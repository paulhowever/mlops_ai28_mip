import asyncio

from dota_winprob.clients import opendota
from tests.fakes import (
    DeadPool,
    HangingPool,
    HealthyPool,
    opendota_ok,
    opendota_unavailable,
)


async def test_all_dependencies_healthy(make_client):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["components"]["postgres"]["status"] == "healthy"
    assert body["components"]["postgres"]["version"] == "16.4"
    assert body["components"]["postgres"]["critical"] is True
    assert body["components"]["postgres"]["detail"] is None
    assert body["components"]["opendota"]["status"] == "healthy"
    assert body["components"]["opendota"]["version"] is None
    assert body["components"]["opendota"]["critical"] is False
    assert body["components"]["opendota"]["detail"] == opendota.NO_VERSION_REASON


async def test_postgres_down_gives_503(make_client):
    client = await make_client(pool=DeadPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["postgres"]["status"] == "unavailable"
    assert body["components"]["postgres"]["version"] is None
    assert body["components"]["postgres"]["detail"] == "OSError: connection refused"
    assert body["components"]["opendota"]["status"] == "healthy"


async def test_opendota_down_keeps_200(make_client):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_unavailable)

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["opendota"]["status"] == "unavailable"
    assert "503" in body["components"]["opendota"]["detail"]
    assert body["components"]["postgres"]["status"] == "healthy"


async def test_hanging_probe_is_cut_by_timeout(make_client, settings):
    pool = HangingPool()
    client = await make_client(pool=pool, opendota_handler=opendota_ok)

    async with asyncio.timeout(settings.health_probe_timeout * 10):
        response = await client.get("/api/v1/health")

    assert response.status_code == 503
    postgres = response.json()["components"]["postgres"]
    assert postgres["detail"] == f"превышен таймаут {settings.health_probe_timeout} с"
    assert postgres["latency_ms"] >= settings.health_probe_timeout * 1000
    assert pool.terminated
