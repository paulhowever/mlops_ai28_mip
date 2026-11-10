import asyncio
import contextlib


class LoopLagMonitor:
    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.last_ms = 0.0
        self.max_ms = 0.0
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            await asyncio.sleep(self.interval)
            lag_ms = max(0.0, (loop.time() - started - self.interval) * 1000)
            self.last_ms = round(lag_ms, 1)
            self.max_ms = max(self.max_ms, self.last_ms)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
