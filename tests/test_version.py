import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import dota_winprob
from tests.fakes import HealthyPool, opendota_ok

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


async def test_version_returns_build_metadata_from_settings(make_client, settings):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/version")

    assert response.status_code == 200
    body = response.json()
    assert body["git_sha"] == settings.git_sha
    assert body["built_at"] == settings.built_at
    assert body["environment"] == settings.environment


async def test_version_matches_pyproject(make_client):
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/version")

    assert response.json()["version"] == declared


def test_version_falls_back_when_package_is_not_installed(monkeypatch):
    def not_installed(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(dota_winprob, "_distribution_version", not_installed)

    assert dota_winprob._resolve_version() == dota_winprob.UNKNOWN_VERSION
