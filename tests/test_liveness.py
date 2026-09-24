from tests.fakes import DeadPool, HealthyPool, opendota_ok, opendota_unavailable


async def test_healthz_returns_ok(make_client):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_does_not_depend_on_database(make_client):
    client = await make_client(pool=DeadPool(), opendota_handler=opendota_ok)

    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_liveness_stays_green_while_health_report_is_red(make_client):
    client = await make_client(pool=DeadPool(), opendota_handler=opendota_unavailable)

    health = await client.get("/api/v1/health")
    liveness = await client.get("/healthz")

    assert health.status_code == 503
    assert liveness.status_code == 200
