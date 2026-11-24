import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.agent.sources.http import HTTPSource
from dota_winprob.schemas import MatchState

TASKS_PATH = "/api/v1/tasks"
FINAL = frozenset({"done", "failed"})


@dataclass(frozen=True)
class Submission:
    task_id: str | None = None
    retry_after: int | None = None
    reason: str | None = None


def _retry_after(response: httpx.Response) -> int | None:
    raw = response.headers.get("Retry-After", "")
    return int(raw) if raw.isdigit() else None


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json()["detail"])
    except (ValueError, LookupError, TypeError):
        return "сервис временно недоступен"


class TasksAPI(HTTPSource):
    ready_path = "/healthz"
    clock: Callable[[], float] = staticmethod(time.monotonic)
    sleep: Callable[[float], Awaitable[None]] = staticmethod(asyncio.sleep)

    async def submit(self, features: MatchState) -> Submission:
        response = await self.send("POST", TASKS_PATH, json=features.model_dump())
        if response.status_code == 503:
            return Submission(retry_after=_retry_after(response), reason=_detail(response))
        return Submission(task_id=str(self.payload(response)["task_id"]))

    async def status(self, task_id: str) -> dict[str, Any] | None:
        response = await self.send("GET", f"{TASKS_PATH}/{quote(task_id, safe='')}")
        if response.status_code == 404:
            return None
        return self.payload(response)

    async def wait(
        self, task_id: str, timeout_s: float, interval_s: float
    ) -> dict[str, Any] | None:
        deadline = self.clock() + timeout_s
        while True:
            try:
                view = await self.status(task_id)
            except SourceUnavailable:
                view = None
            if view is not None and view["status"] in FINAL:
                return view
            if self.clock() + interval_s > deadline:
                return view
            await self.sleep(interval_s)
