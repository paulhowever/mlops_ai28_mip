import asyncio
import functools
import os
import re
from urllib.parse import urlsplit

import pytest

from dota_winprob.clients import postgres
from dota_winprob.services.health import Probe, build_report

DATABASE_URL = os.environ.get("INTEGRATION_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="INTEGRATION_DATABASE_URL не задан"),
]


class FreezingProxy:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.thawed = asyncio.Event()
        self.thawed.set()

    async def handle(self, client_reader, client_writer):
        server_reader, server_writer = await asyncio.open_connection(self.host, self.port)
        await asyncio.gather(
            self.forward(client_reader, server_writer),
            self.forward(server_reader, client_writer),
        )

    async def forward(self, reader, writer):
        while data := await reader.read(65536):
            await self.thawed.wait()
            writer.write(data)
            await writer.drain()
        writer.close()


async def test_fetch_version_from_real_postgres(settings):
    pool = await postgres.create_pool(settings.model_copy(update={"database_url": DATABASE_URL}))
    try:
        version = await postgres.fetch_version(pool)
    finally:
        await postgres.close_pool(pool)

    assert re.fullmatch(r"\d+(\.\d+)+", version)


async def test_hung_postgres_is_cut_by_probe_timeout(settings):
    target = urlsplit(DATABASE_URL)
    proxy = FreezingProxy(target.hostname, target.port or 5432)
    server = await asyncio.start_server(proxy.handle, "127.0.0.1", 0)
    proxy_port = server.sockets[0].getsockname()[1]
    credentials = target.netloc.rsplit("@", 1)[0]
    proxied_url = target._replace(netloc=f"{credentials}@127.0.0.1:{proxy_port}").geturl()
    pool = await postgres.create_pool(settings.model_copy(update={"database_url": proxied_url}))
    timeout = settings.health_probe_timeout
    probe = Probe(
        name="postgres", critical=True, run=functools.partial(postgres.fetch_version, pool)
    )
    try:
        await postgres.fetch_version(pool)
        proxy.thawed.clear()
        report = asyncio.create_task(build_report([probe], probe_timeout=timeout))
        done, _ = await asyncio.wait({report}, timeout=timeout * 10)
    finally:
        proxy.thawed.set()
        await postgres.close_pool(pool)
        server.close()

    assert done, "проба не отпустила запрос после таймаута"
    component = report.result().components["postgres"]
    assert component.detail == f"превышен таймаут {timeout} с"
    assert component.latency_ms < timeout * 2 * 1000
