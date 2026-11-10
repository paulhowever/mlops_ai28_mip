import asyncio
import random

from dota_winprob import source
from dota_winprob.schemas import MatchState, TaskSource
from tests.fakes import MATCH_STATE


class ScriptedPublisher:
    def __init__(self, answers: list[bool]) -> None:
        self.answers = answers
        self.attempts: list[str] = []
        self.accepted: list[str] = []

    async def __call__(self, message) -> bool:
        self.attempts.append(message.task_id)
        accepted = self.answers.pop(0) if self.answers else True
        if accepted:
            self.accepted.append(message.task_id)
        return accepted


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def features() -> MatchState:
    return MatchState.model_validate(MATCH_STATE)


def backoff() -> source.Backoff:
    return source.Backoff(0.02, 0.5, rng=random.Random(7))


def tracked_tasks(count: int, pulled: list[str]):
    for index in range(count):
        task_id = f"batch:{index}"
        pulled.append(task_id)
        yield task_id, features()


async def test_rejection_holds_the_same_record_and_backs_off():
    publisher = ScriptedPublisher([False, False, False, True])
    sleep = SleepRecorder()
    pulled: list[str] = []
    stats = source.SourceStats()

    await source.run(tracked_tasks(2, pulled), publisher, backoff(), stats, sleep)

    assert publisher.attempts == ["batch:0"] * 4 + ["batch:1"]
    assert publisher.accepted == ["batch:0", "batch:1"]
    assert stats.published == 2
    assert stats.rejected == 3
    assert len(sleep.delays) == 3
    assert sleep.delays[0] < sleep.delays[1] < sleep.delays[2]
    assert all(delay > 0 for delay in sleep.delays)


async def test_source_pulls_next_record_only_after_ack():
    pulled: list[str] = []
    tasks = tracked_tasks(3, pulled)
    states: list[list[str]] = []

    async def publisher(message) -> bool:
        states.append(list(pulled))
        return len(states) % 2 == 0

    await source.run(tasks, publisher, backoff(), source.SourceStats(), SleepRecorder())

    assert states == [
        ["batch:0"],
        ["batch:0"],
        ["batch:0", "batch:1"],
        ["batch:0", "batch:1"],
        ["batch:0", "batch:1", "batch:2"],
        ["batch:0", "batch:1", "batch:2"],
    ]


async def test_every_record_is_delivered_in_order():
    publisher = ScriptedPublisher([False, True, False, False, True, True, False, True])
    pulled: list[str] = []

    await source.run(
        tracked_tasks(5, pulled), publisher, backoff(), source.SourceStats(), SleepRecorder()
    )

    assert publisher.accepted == [f"batch:{index}" for index in range(5)]


async def test_retry_gets_fresh_published_at():
    stamps = []

    async def publisher(message) -> bool:
        stamps.append(message.published_at)
        return len(stamps) == 2

    await source.publish_until_accepted(
        publisher,
        lambda: source._message("batch:1", features()),
        backoff(),
        source.SourceStats(),
        SleepRecorder(),
    )

    assert stamps[0] <= stamps[1]


async def test_stop_signal_ends_run_between_records():
    stop = asyncio.Event()
    pulled: list[str] = []

    async def publisher(message) -> bool:
        stop.set()
        return True

    stats = await source.run(
        tracked_tasks(5, pulled), publisher, backoff(), source.SourceStats(), SleepRecorder(), stop
    )

    assert stats.published == 1
    assert pulled == ["batch:0", "batch:1"]


async def test_stop_signal_abandons_record_that_broker_keeps_refusing():
    stop = asyncio.Event()
    sleep = SleepRecorder()

    async def publisher(message) -> bool:
        stop.set()
        return False

    stats = await source.run(
        tracked_tasks(3, []), publisher, backoff(), source.SourceStats(), sleep, stop
    )

    assert stats.published == 0
    assert stats.rejected == 1
    assert sleep.delays == []


def test_backoff_grows_to_ceiling_and_resets():
    policy = backoff()

    delays = [policy.next_delay() for _ in range(8)]
    policy.reset()

    assert delays[0] >= 0.01
    assert max(delays) <= 0.5
    assert delays[-1] >= 0.25
    assert policy.next_delay() <= 0.02


def test_read_tasks_is_lazy_and_builds_batch_ids(tmp_path):
    path = tmp_path / "tasks.csv"
    header = "match_id," + ",".join(MATCH_STATE)
    rows = [
        f"{match_id}," + ",".join(str(value) for value in MATCH_STATE.values())
        for match_id in (11, 12)
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    tasks = source.read_tasks(path)
    first = next(tasks)

    assert first[0] == "batch:11"
    assert first[1].model_dump() == MATCH_STATE
    assert [task_id for task_id, _ in tasks] == ["batch:12"]


def test_message_is_batch_task():
    message = source._message("batch:3", features())

    assert message.source is TaskSource.batch
    assert message.published_at.tzinfo is not None


def test_args_have_defaults():
    args = source.parse_args([])

    assert str(args.file) == "data/tasks.csv"
    assert args.limit is None
