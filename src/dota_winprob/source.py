import argparse
import asyncio
import csv
import itertools
import logging
import random
import signal
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from opentelemetry import context as otel_context
from opentelemetry import trace

from dota_winprob import instruments
from dota_winprob.clients import broker as broker_client
from dota_winprob.config import Settings, get_settings
from dota_winprob.logging_config import log_context, setup_logging
from dota_winprob.schemas import MatchState, TaskMessage, TaskSource
from dota_winprob.telemetry import setup_telemetry

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

TASK_SPAN = "source publish"

Publisher = Callable[[TaskMessage], Awaitable[bool]]
Sleep = Callable[[float], Awaitable[None]]


def read_tasks(path: Path) -> Iterator[tuple[str, MatchState]]:
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            match_id = row.pop("match_id")
            yield f"batch:{match_id}", MatchState.model_validate(row)


class Backoff:
    def __init__(self, initial: float, maximum: float, rng: random.Random | None = None) -> None:
        self.initial = initial
        self.maximum = maximum
        self._rng = rng or random.SystemRandom()
        self._current = initial

    def next_delay(self) -> float:
        ceiling = self._current
        self._current = min(self.maximum, self._current * 2)
        return ceiling / 2 + self._rng.uniform(0, ceiling / 2)

    def reset(self) -> None:
        self._current = self.initial


@dataclass
class SourceStats:
    published: int = 0
    rejected: int = 0
    waited: float = 0.0
    started: float = field(default_factory=time.monotonic)

    def rate(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.published / elapsed if elapsed > 0 else 0.0


class RejectionLog:
    def __init__(self, interval: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.interval = interval
        self._clock = clock
        self._last: float | None = None
        self.rejected = 0
        self.waited = 0.0

    def record(self, delay: float) -> None:
        self.rejected += 1
        self.waited += delay
        now = self._clock()
        if self._last is None or now - self._last >= self.interval:
            self._emit(now)

    def flush(self) -> None:
        if self.rejected:
            self._emit(self._clock())

    def _emit(self, now: float) -> None:
        log.warning(
            "Брокер отклоняет задачи, источник замедляется",
            extra={
                "rejected": self.rejected,
                "waited_s": round(self.waited, 3),
                "summary_interval_s": self.interval,
            },
        )
        self._last = now
        self.rejected = 0
        self.waited = 0.0


def _message(task_id: str, features: MatchState) -> TaskMessage:
    return TaskMessage(
        task_id=task_id,
        source=TaskSource.batch,
        features=features,
        published_at=datetime.now(UTC),
    )


async def publish_until_accepted(
    publish: Publisher,
    build: Callable[[], TaskMessage],
    backoff: Backoff,
    stats: SourceStats,
    sleep: Sleep = asyncio.sleep,
    stop: asyncio.Event | None = None,
    rejections: RejectionLog | None = None,
) -> bool:
    span = trace.get_current_span()
    attempt = 1
    while not await publish(build()):
        stats.rejected += 1
        instruments.source_rejected.add(1)
        span.add_event("broker.nack", {"attempt": attempt})
        if stop is not None and stop.is_set():
            span.set_attribute("task.abandoned", True)
            return False
        delay = backoff.next_delay()
        stats.waited += delay
        instruments.source_wait.add(delay)
        span.add_event("backoff", {"attempt": attempt, "delay_s": delay})
        if rejections is not None:
            rejections.record(delay)
        await sleep(delay)
        attempt += 1
    backoff.reset()
    stats.published += 1
    instruments.source_published.add(1)
    span.set_attribute("attempts", attempt)
    log.info("Задача опубликована", extra={"attempts": attempt})
    return True


async def run(
    tasks: Iterable[tuple[str, MatchState]],
    publish: Publisher,
    backoff: Backoff,
    stats: SourceStats,
    sleep: Sleep = asyncio.sleep,
    stop: asyncio.Event | None = None,
    rejections: RejectionLog | None = None,
) -> SourceStats:
    for task_id, features in tasks:
        if stop is not None and stop.is_set():
            break
        with (
            log_context(task_id=task_id),
            tracer.start_as_current_span(
                TASK_SPAN,
                context=otel_context.Context(),
                kind=trace.SpanKind.PRODUCER,
                attributes={"task.id": task_id, "task.source": TaskSource.batch.value},
            ),
        ):
            await publish_until_accepted(
                publish,
                partial(_message, task_id, features),
                backoff,
                stats,
                sleep,
                stop,
                rejections,
            )
    return stats


async def report(stats: SourceStats, interval: float = 1.0) -> None:
    previous = stats.published
    while True:
        await asyncio.sleep(interval)
        log.info(
            "Источник публикует",
            extra={
                "published": stats.published,
                "rejected": stats.rejected,
                "waited_s": round(stats.waited, 3),
                "rate_per_s": round((stats.published - previous) / interval, 1),
            },
        )
        previous = stats.published


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пакетный скоринг файла через очередь задач.")
    parser.add_argument("--file", type=Path, default=Path("data/tasks.csv"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None, settings: Settings | None = None) -> SourceStats:
    args = parse_args(argv)
    settings = settings or get_settings()
    setup_logging(settings)
    telemetry = setup_telemetry(settings)

    broker = broker_client.create_broker(settings)
    await broker.connect()
    await broker_client.declare_topology(broker, settings)

    tasks: Iterable[tuple[str, MatchState]] = read_tasks(args.file)
    if args.limit is not None:
        tasks = itertools.islice(tasks, args.limit)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, request_stop, stop, signum)

    stats = SourceStats()
    rejections = RejectionLog(settings.source_reject_log_interval)
    reporter = asyncio.create_task(report(stats))
    try:
        await run(
            tasks,
            partial(broker_client.publish_task, broker, confirm_timeout=settings.publish_timeout),
            Backoff(settings.source_backoff_min, settings.source_backoff_max),
            stats,
            stop=stop,
            rejections=rejections,
        )
    finally:
        reporter.cancel()
        rejections.flush()
        await broker.stop()
        await asyncio.to_thread(telemetry.shutdown, settings.telemetry_shutdown_timeout)

    log.info(
        "Источник остановлен по сигналу" if stop.is_set() else "Источник закончил",
        extra={
            "published": stats.published,
            "rejected": stats.rejected,
            "waited_s": round(stats.waited, 3),
            "rate_per_s": round(stats.rate(), 1),
        },
    )
    return stats


def request_stop(stop: asyncio.Event, signum: int) -> None:
    if not stop.is_set():
        log.info(
            "Получен сигнал остановки, источник дописывает текущую задачу",
            extra={"signal": signal.Signals(signum).name},
        )
    stop.set()


if __name__ == "__main__":
    asyncio.run(main())
