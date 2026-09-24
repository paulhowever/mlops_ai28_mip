import asyncpg

from dota_winprob.clients import opendota, postgres
from tests.fakes import POSTGRES_BANNER


def test_parse_version_extracts_number_from_banner():
    assert postgres.parse_version(POSTGRES_BANNER) == "16.4"


def test_parse_version_returns_banner_as_is_on_unexpected_format():
    assert postgres.parse_version("CockroachDB CCL v23.1") == "CockroachDB CCL v23.1"


async def test_create_pool_does_not_connect_to_database(settings):
    pool = await postgres.create_pool(settings)

    assert isinstance(pool, asyncpg.Pool)
    await postgres.close_pool(pool)


async def test_opendota_client_uses_settings(settings):
    client = opendota.create_client(settings)

    assert str(client.base_url).rstrip("/") == settings.opendota_base_url
    assert client.timeout.read == settings.health_probe_timeout
    await opendota.close_client(client)
