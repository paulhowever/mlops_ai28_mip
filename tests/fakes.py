import asyncio

import httpx

POSTGRES_BANNER = "PostgreSQL 16.4 (Debian 16.4-1.pgdg120+1) on x86_64-pc-linux-gnu"


class FakePool:
    def __init__(self) -> None:
        self.terminated = False

    async def acquire(self) -> "FakePool":
        return self

    async def release(self, connection: "FakePool") -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True

    async def close(self) -> None:
        pass


class HealthyPool(FakePool):
    async def fetchval(self, query: str) -> str:
        return POSTGRES_BANNER


class DeadPool(FakePool):
    async def fetchval(self, query: str) -> str:
        raise OSError("connection refused")


class HangingPool(FakePool):
    async def fetchval(self, query: str) -> str:
        return await asyncio.Future()


def opendota_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200)


def opendota_unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503)
