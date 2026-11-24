import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from dota_winprob.agent import metrics
from dota_winprob.agent.errors import LLMBusy

PERIOD = 60.0
DEFAULT_RETRY_AFTER = 5
BACKGROUND_POLL = 1.0
MIN_PAUSE = 0.05

_BACKGROUND: ContextVar[bool] = ContextVar("llm_background", default=False)


@contextmanager
def background() -> Iterator[None]:
    token = _BACKGROUND.set(True)
    try:
        yield
    finally:
        _BACKGROUND.reset(token)


@dataclass(eq=False)
class Ticket:
    start: float
    counted_at: float
    tokens: int
    background: bool
    clock: Callable[[], float]

    def settle(self, tokens: int) -> None:
        if tokens > 0:
            self.tokens = tokens
        self.counted_at = max(self.start, self.clock())


class LLMGate:
    def __init__(
        self,
        *,
        max_concurrency: int,
        requests_per_minute: int,
        max_waiting: int,
        queue_timeout: float,
        tokens_per_minute: int = 100_000,
        background_share: float = 0.25,
        background_timeout: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._limit = requests_per_minute
        self._token_limit = tokens_per_minute
        self._background_limit = max(1, int(requests_per_minute * background_share))
        self._background_tokens = max(1, int(tokens_per_minute * background_share))
        self._background_timeout = background_timeout
        self._background_active = False
        self._max_waiting = max_waiting
        self._queue_timeout = queue_timeout
        self._clock = clock
        self._sleep = sleep
        self._tickets: deque[Ticket] = deque()
        self._not_before = 0.0
        self._waiting = 0

    @asynccontextmanager
    async def slot(self, tokens: int = 0) -> AsyncIterator[Ticket]:
        if not _BACKGROUND.get():
            async with self._admitted(self._clock() + self._queue_timeout, tokens, False) as ticket:
                yield ticket
            return
        deadline = self._clock() + self._background_timeout
        await self._stand_aside(deadline, tokens)
        self._background_active = True
        try:
            async with self._admitted(deadline, tokens, True) as ticket:
                yield ticket
        finally:
            self._background_active = False

    def cool_down(self, delay: float) -> None:
        self._not_before = max(self._not_before, self._clock() + delay)

    def retry_after(self, tokens: int = 0) -> int:
        now = self._clock()
        return max(DEFAULT_RETRY_AFTER, math.ceil(self._next_start(now, tokens, False) - now))

    @asynccontextmanager
    async def _admitted(
        self, deadline: float, tokens: int, is_background: bool
    ) -> AsyncIterator[Ticket]:
        await self._enter(tokens)
        try:
            yield await self._pace(deadline, tokens, is_background)
        finally:
            self._semaphore.release()

    async def _stand_aside(self, deadline: float, tokens: int) -> None:
        while True:
            now = self._clock()
            start = self._next_start(now, tokens, True)
            if max(start, now) > deadline:
                raise self._overloaded("background", tokens)
            occupied = self._background_active or self._waiting or self._semaphore.locked()
            if start <= now and not occupied:
                return
            await self._sleep(min(BACKGROUND_POLL, max(MIN_PAUSE, start - now)))

    async def _enter(self, tokens: int) -> None:
        if self._semaphore.locked() and self._waiting >= self._max_waiting:
            raise self._overloaded("slots", tokens)
        self._waiting += 1
        try:
            async with asyncio.timeout(self._queue_timeout):
                await self._semaphore.acquire()
        except TimeoutError:
            raise self._overloaded("slots", tokens) from None
        finally:
            self._waiting -= 1

    async def _pace(self, deadline: float, tokens: int, is_background: bool) -> Ticket:
        now = self._clock()
        start = self._next_start(now, tokens, is_background)
        if start > deadline:
            raise self._overloaded("rate", tokens)
        ticket = Ticket(start, start, tokens, is_background, self._clock)
        self._tickets.append(ticket)
        if start > now:
            try:
                await self._sleep(start - now)
            except asyncio.CancelledError:
                self._tickets.remove(ticket)
                raise
        return ticket

    def _next_start(self, now: float, tokens: int, is_background: bool) -> float:
        while self._tickets and self._tickets[0].counted_at + PERIOD <= now:
            self._tickets.popleft()
        earliest = max(now, self._not_before)
        if self._tickets:
            earliest = max(earliest, self._tickets[-1].start)
        moments = {earliest}
        for ticket in self._tickets:
            moments |= {
                moment + PERIOD
                for moment in (ticket.start, ticket.counted_at)
                if moment + PERIOD > earliest
            }
        return next(
            moment for moment in sorted(moments) if self._fits(moment, tokens, is_background)
        )

    def _fits(self, moment: float, tokens: int, is_background: bool) -> bool:
        groups = [(list(self._tickets), self._limit, self._token_limit)]
        if is_background:
            own = [ticket for ticket in self._tickets if ticket.background]
            groups.append((own, self._background_limit, self._background_tokens))
        for tickets, limit, token_limit in groups:
            requests = sum(ticket.start + PERIOD > moment for ticket in tickets)
            spent = sum(ticket.tokens for ticket in tickets if ticket.counted_at + PERIOD > moment)
            if requests >= limit or (spent and spent + tokens > token_limit):
                return False
        return True

    def _overloaded(self, reason: str, tokens: int) -> LLMBusy:
        metrics.llm_overload.add(1, {"reason": reason})
        return LLMBusy(retry_after=self.retry_after(tokens))
