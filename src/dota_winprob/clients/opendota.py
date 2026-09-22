import logging

import httpx

from dota_winprob.config import Settings

log = logging.getLogger(__name__)

HEALTH_PATH = "/api/health"
NO_VERSION_REASON = "OpenDota не публикует версию API; проверена доступность"


def create_client(settings: Settings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=settings.opendota_base_url,
        timeout=settings.health_probe_timeout,
    )


async def close_client(client: httpx.AsyncClient) -> None:
    await client.aclose()
    log.info("HTTP-клиент OpenDota закрыт")


async def check_availability(client: httpx.AsyncClient) -> None:
    response = await client.get(HEALTH_PATH)
    response.raise_for_status()
