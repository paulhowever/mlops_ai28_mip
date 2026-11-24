import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from dota_winprob.agent import metrics
from dota_winprob.agent.errors import LLMBusy

PERIOD = 60.0
DEFAULT_RETRY_AFTER = 5


class LLMGate:
    def __init__(
        self,
        *,
        max_concurrency: int,
        requests_per_minute: int,
        max_waiting: int,
        queue_timeout: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._limit = requests_per_minute
        self._max_waiting = max_waiting
        self._queue_timeout = queue_timeout
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()
        self._not_before = 0.0
        self._waiting = 0

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        deadline = self._clock() + self._queue_timeout
        await self._enter()
        try:
            await self._pace(deadline)
            yield
        finally:
            self._semaphore.release()

    def cool_down(self, delay: float) -> None:
        self._not_before = max(self._not_before, self._clock() + delay)

    def retry_after(self) -> int:
        delay = self._next_start(self._clock()) - self._clock()
        return max(DEFAULT_RETRY_AFTER, math.ceil(delay))

    async def _enter(self) -> None:
        if self._semaphore.locked() and self._waiting >= self._max_waiting:
            raise self._overloaded("slots")
        self._waiting += 1
        try:
            async with asyncio.timeout(self._queue_timeout):
                await self._semaphore.acquire()
        except TimeoutError:
            raise self._overloaded("slots") from None
        finally:
            self._waiting -= 1

    async def _pace(self, deadline: float) -> None:
        now = self._clock()
        start = self._next_start(now)
        if start > deadline:
            raise self._overloaded("rate")
        self._stamps.append(start)
        if start > now:
            await self._sleep(start - now)

    def _next_start(self, now: float) -> float:
        while self._stamps and self._stamps[0] <= now - PERIOD:
            self._stamps.popleft()
        earliest = max(now, self._not_before)
        if len(self._stamps) >= self._limit:
            earliest = max(earliest, self._stamps[-self._limit] + PERIOD)
        return max(earliest, self._stamps[-1]) if self._stamps else earliest

    def _overloaded(self, reason: str) -> LLMBusy:
        metrics.llm_overload.add(1, {"reason": reason})
        return LLMBusy(retry_after=self.retry_after())
