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

from dota_winprob.clients import broker as broker_client
from dota_winprob.config import Settings, get_settings
from dota_winprob.logging_config import setup_logging
from dota_winprob.schemas import MatchState, TaskMessage, TaskSource

log = logging.getLogger(__name__)

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
    started: float = field(default_factory=time.monotonic)

    def rate(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.published / elapsed if elapsed > 0 else 0.0


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
) -> bool:
    while not await publish(build()):
        stats.rejected += 1
        if stop is not None and stop.is_set():
            return False
        await sleep(backoff.next_delay())
    backoff.reset()
    stats.published += 1
    return True


async def run(
    tasks: Iterable[tuple[str, MatchState]],
    publish: Publisher,
    backoff: Backoff,
    stats: SourceStats,
    sleep: Sleep = asyncio.sleep,
    stop: asyncio.Event | None = None,
) -> SourceStats:
    for task_id, features in tasks:
        if stop is not None and stop.is_set():
            break
        await publish_until_accepted(
            publish,
            partial(_message, task_id, features),
            backoff,
            stats,
            sleep,
            stop,
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

    broker = broker_client.create_broker(settings)
    await broker.connect()
    await broker_client.declare_topology(broker, settings)

    tasks: Iterable[tuple[str, MatchState]] = read_tasks(args.file)
    if args.limit is not None:
        tasks = itertools.islice(tasks, args.limit)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)

    stats = SourceStats()
    reporter = asyncio.create_task(report(stats))
    try:
        await run(
            tasks,
            partial(broker_client.publish_task, broker, confirm_timeout=settings.publish_timeout),
            Backoff(settings.source_backoff_min, settings.source_backoff_max),
            stats,
            stop=stop,
        )
    finally:
        reporter.cancel()
        await broker.stop()

    log.info(
        "Источник остановлен по сигналу" if stop.is_set() else "Источник закончил",
        extra={
            "published": stats.published,
            "rejected": stats.rejected,
            "rate_per_s": round(stats.rate(), 1),
        },
    )
    return stats


if __name__ == "__main__":
    asyncio.run(main())
