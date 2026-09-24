import asyncio

import httpx

POSTGRES_BANNER = "PostgreSQL 16.4 (Debian 16.4-1.pgdg120+1) on x86_64-pc-linux-gnu"


class HealthyPool:
    def __init__(self) -> None:
        self.closed = False

    async def fetchval(self, query: str) -> str:
        return POSTGRES_BANNER

    async def close(self) -> None:
        self.closed = True


class DeadPool(HealthyPool):
    async def fetchval(self, query: str) -> str:
        raise OSError("connection refused")


class HangingPool(HealthyPool):
    async def fetchval(self, query: str) -> str:
        await asyncio.Event().wait()
        return POSTGRES_BANNER


def opendota_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok"})


def opendota_unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, text="Service Unavailable")
