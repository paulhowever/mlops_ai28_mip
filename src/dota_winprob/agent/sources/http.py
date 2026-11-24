from typing import Any

import httpx

from dota_winprob.agent.errors import SourceUnavailable


class HTTPSource:
    def __init__(
        self,
        name: str,
        base_url: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def send(
        self,
        method: str,
        path: str,
        **options: object,
    ) -> httpx.Response:
        try:
            return await self._client.request(method, path, **options)
        except httpx.TimeoutException as error:
            raise SourceUnavailable(self.name, "не ответил вовремя") from error
        except httpx.HTTPError as error:
            raise SourceUnavailable(self.name, type(error).__name__) from error

    async def json(
        self,
        method: str,
        path: str,
        **options: object,
    ) -> dict[str, Any]:
        response = await self.send(method, path, **options)
        return self.payload(response)

    def payload(self, response: httpx.Response) -> dict[str, Any]:
        if response.is_error:
            raise SourceUnavailable(self.name, f"HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as error:
            raise SourceUnavailable(self.name, "ответ не JSON") from error
        if not isinstance(body, dict):
            raise SourceUnavailable(self.name, "неожиданный формат ответа")
        return body

    async def check(self, path: str) -> None:
        response = await self.send("GET", path)
        response.raise_for_status()
