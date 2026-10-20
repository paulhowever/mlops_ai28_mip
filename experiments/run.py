import asyncio
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import httpx
import pandas as pd

from experiments import plot

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "experiments"
MANAGEMENT_URL = os.environ.get("RABBITMQ_MANAGEMENT_URL", "http://localhost:15672")
MANAGEMENT_AUTH = (
    os.environ.get("RABBITMQ_USER", "dota"),
    os.environ.get("RABBITMQ_PASSWORD", "dota"),
)
DATABASE_URL = os.environ.get(
    "EXPERIMENT_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/dota_winprob",
)
SOURCE_CONTAINER = "experiment-source"
QUEUES = ("tasks.score", "tasks.retry", "tasks.dlq")
WARMUP = 15.0


@dataclass(frozen=True)
class Event:
    at: float
    action: str
    label: str


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    env: dict[str, str]
    workers: int
    duration: float
    events: tuple[Event, ...]
    drain: bool


SECOND_WORKER = Event(45, "scale", "второй воркер")

SCENARIOS = (
    Scenario(
        "limited",
        "Лимит очереди 100, prefetch 4",
        {"QUEUE_MAX_LENGTH": "100", "WORKER_PREFETCH": ""},
        1,
        90,
        (SECOND_WORKER,),
        True,
    ),
    Scenario(
        "unbounded_queue",
        "Очередь без лимита, prefetch 4",
        {"QUEUE_MAX_LENGTH": "0", "WORKER_PREFETCH": ""},
        1,
        90,
        (SECOND_WORKER,),
        False,
    ),
    Scenario(
        "unbounded_prefetch",
        "Лимит очереди 100, prefetch без лимита",
        {"QUEUE_MAX_LENGTH": "100", "WORKER_PREFETCH": "0"},
        1,
        90,
        (SECOND_WORKER,),
        False,
    ),
    Scenario(
        "sigterm",
        "Два воркера, одному SIGTERM под нагрузкой",
        {"QUEUE_MAX_LENGTH": "100", "WORKER_PREFETCH": ""},
        2,
        60,
        (Event(30, "sigterm", "SIGTERM воркеру"),),
        True,
    ),
)


def run(*command: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout


def compose(*args: str, env: dict[str, str] | None = None, check: bool = True) -> str:
    return run("docker", "compose", *args, env=env, check=check)


def docker_socket() -> str:
    host = run("docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
    return host.strip().removeprefix("unix://")


def container_ids(service: str) -> list[str]:
    return compose("ps", "-q", service).split()


def source_id() -> str | None:
    found = run("docker", "ps", "-q", "--filter", f"name=^{SOURCE_CONTAINER}$").split()
    return found[0] if found else None


def healthy_workers() -> int:
    lines = compose("ps", "--format", "json", "worker").strip().splitlines()
    states = [json.loads(line) for line in lines if line.strip()]
    return sum(1 for state in states if state.get("Health") == "healthy")


async def wait_for_workers(count: int, patience: float = 120.0) -> None:
    deadline = time.monotonic() + patience
    while healthy_workers() < count:
        if time.monotonic() > deadline:
            raise RuntimeError(f"воркеры не стали healthy за {patience} с")
        await asyncio.sleep(1)


class Probe:
    def __init__(self, management: httpx.AsyncClient, engine: httpx.AsyncClient) -> None:
        self.management = management
        self.engine = engine

    async def memory_mb(self, container_id: str | None) -> float:
        if container_id is None:
            return 0.0
        response = await self.engine.get(
            f"/containers/{container_id}/stats",
            params={"stream": "false", "one-shot": "true"},
        )
        if response.status_code != 200:
            return 0.0
        memory = response.json().get("memory_stats", {})
        usage = memory.get("usage", 0) - memory.get("stats", {}).get("inactive_file", 0)
        return round(max(usage, 0) / 2**20, 1)

    async def queue(self, name: str = "tasks.score") -> dict:
        response = await self.management.get(f"/api/queues/%2F/{name}")
        return response.json() if response.status_code == 200 else {}

    async def depth(self, name: str) -> int:
        queue = await self.queue(name)
        return queue.get("messages_ready", 0) + queue.get("messages_unacknowledged", 0)

    async def sample(self, elapsed: float, workers: list[str], rabbitmq: str) -> dict:
        queue, rabbitmq_mb, source_mb, *worker_mb = await asyncio.gather(
            self.queue(),
            self.memory_mb(rabbitmq),
            self.memory_mb(source_id()),
            *(self.memory_mb(worker) for worker in workers),
        )
        return {
            "t": round(elapsed, 1),
            "ready": queue.get("messages_ready", 0),
            "unacked": queue.get("messages_unacknowledged", 0),
            "consumers": queue.get("consumers", 0),
            "published_total": queue.get("message_stats", {}).get("publish", 0),
            "rabbitmq_mb": rabbitmq_mb,
            "workers_mb": round(sum(worker_mb), 1),
            "source_mb": source_mb,
        }


async def reset(management: httpx.AsyncClient) -> None:
    run("docker", "rm", "-f", SOURCE_CONTAINER, check=False)
    compose("kill", "worker", check=False)
    compose("rm", "-f", "worker", check=False)
    for name in QUEUES:
        await management.delete(f"/api/queues/%2F/{name}")
    connection = await asyncpg.connect(DATABASE_URL)
    await connection.execute("TRUNCATE tasks")
    await connection.close()


async def apply(event: Event, scenario: Scenario) -> None:
    if event.action == "scale":
        await asyncio.to_thread(
            compose,
            "up",
            "-d",
            "--no-deps",
            "--no-recreate",
            "--scale",
            f"worker={scenario.workers + 1}",
            "worker",
            env=scenario.env,
        )
    elif event.action == "sigterm":
        victim = container_ids("worker")[0]
        await asyncio.to_thread(run, "docker", "stop", "--time", "40", victim)


FINAL_SOURCE_MESSAGES = {"Источник остановлен по сигналу", "Источник закончил"}


def source_records() -> list[dict]:
    logs = run("docker", "logs", SOURCE_CONTAINER, check=False)
    return [json.loads(line) for line in logs.splitlines() if line.startswith("{")]


def source_published(records: list[dict]) -> int | None:
    final = [record for record in records if record.get("message") in FINAL_SOURCE_MESSAGES]
    return final[-1]["published"] if final else None


def source_log(records: list[dict], started_at: datetime) -> pd.DataFrame:
    rows = []
    for record in records:
        if record.get("message") != "Источник публикует":
            continue
        moment = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        rows.append(
            {
                "t": round((moment - started_at).total_seconds()),
                "source_rate": record["rate_per_s"],
                "published": record["published"],
                "rejected": record["rejected"],
            }
        )
    return pd.DataFrame(rows, columns=["t", "source_rate", "published", "rejected"])


async def completed_tasks(started_at: datetime) -> tuple[pd.DataFrame, dict[str, int]]:
    connection = await asyncpg.connect(DATABASE_URL)
    rows = await connection.fetch(
        "SELECT published_at, completed_at, worker FROM tasks "
        "WHERE source = 'batch' AND status = 'done'"
    )
    statuses = await connection.fetch("SELECT status, count(*) AS n FROM tasks GROUP BY status")
    await connection.close()
    total = {"done": 0, "failed": 0, "pending": 0} | {row["status"]: row["n"] for row in statuses}
    frame = pd.DataFrame(
        [dict(row) for row in rows], columns=["published_at", "completed_at", "worker"]
    )
    if frame.empty:
        return frame.assign(t=[], latency_s=[]), total
    frame["t"] = (frame["completed_at"] - started_at).dt.total_seconds()
    frame["latency_s"] = (frame["completed_at"] - frame["published_at"]).dt.total_seconds()
    return frame, total


def per_second(samples: pd.DataFrame, source: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    seconds = pd.DataFrame({"t": range(int(samples["t"].max()) + 1)})
    if tasks.empty:
        seconds = seconds.assign(throughput=0, latency_s=float("nan"))
    else:
        done = tasks.groupby(tasks["t"].astype(int)).agg(
            throughput=("latency_s", "size"),
            latency_s=("latency_s", "mean"),
        )
        seconds = seconds.merge(done, left_on="t", right_index=True, how="left")
    seconds = seconds.merge(source.drop_duplicates("t"), on="t", how="left")
    by_second = samples.assign(t=samples["t"].round().astype(int)).drop_duplicates("t")
    seconds = seconds.merge(by_second, on="t", how="left")
    seconds["throughput"] = seconds["throughput"].fillna(0)
    return seconds


def window(frame: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return frame[(frame["t"] >= start) & (frame["t"] < end)]


def number(value: float, digits: int) -> float | None:
    return None if pd.isna(value) else round(float(value), digits)


def window_stats(part: pd.DataFrame) -> dict:
    return {
        "source_rate": number(part["source_rate"].mean(), 1),
        "throughput": number(part["throughput"].mean(), 1),
        "latency_p50_s": number(part["latency_s"].median(), 2),
        "depth_mean": number((part["ready"] + part["unacked"]).mean(), 1),
    }


def summarize(scenario: Scenario, seconds: pd.DataFrame, final: dict) -> dict:
    split = scenario.events[0].at if scenario.events else scenario.duration
    return {
        "scenario": scenario.name,
        "title": scenario.title,
        "before_event": window_stats(window(seconds, WARMUP, split)),
        "after_event": window_stats(window(seconds, split + WARMUP, scenario.duration)),
        "max_ready": int(seconds["ready"].max()),
        "max_unacked": int(seconds["unacked"].max()),
        "max_rabbitmq_mb": float(seconds["rabbitmq_mb"].max()),
        "max_workers_mb": float(seconds["workers_mb"].max()),
        "max_source_mb": float(seconds["source_mb"].max()),
        "rejected": int(seconds["rejected"].max()) if seconds["rejected"].notna().any() else 0,
        **final,
    }


async def drain(probe: Probe, patience: float = 60.0) -> dict:
    deadline = time.monotonic() + patience
    queue = await probe.queue()
    while queue.get("messages_ready", 0) + queue.get("messages_unacknowledged", 0) > 0:
        if time.monotonic() > deadline:
            break
        await asyncio.sleep(1)
        queue = await probe.queue()
    await asyncio.sleep(2)
    return await probe.queue()


async def record(scenario: Scenario, probe: Probe, rabbitmq: str) -> list[dict]:
    started = time.monotonic()
    pending = list(scenario.events)
    samples = []
    background: list[asyncio.Task[None]] = []
    workers = container_ids("worker")
    while (elapsed := time.monotonic() - started) < scenario.duration:
        while pending and elapsed >= pending[0].at:
            event = pending.pop(0)
            print(f"   t={elapsed:.0f}s: {event.label}")
            background.append(asyncio.create_task(apply(event, scenario)))
        if int(elapsed) % 5 == 0:
            workers = container_ids("worker")
        samples.append(await probe.sample(elapsed, workers, rabbitmq))
        await asyncio.sleep(max(0.0, 1 - (time.monotonic() - started - elapsed)))
    await asyncio.gather(*background)
    return samples


async def run_scenario(scenario: Scenario, probe: Probe) -> dict:
    print(f"== {scenario.name}: {scenario.title}")
    await reset(probe.management)
    compose(
        "up",
        "-d",
        "--no-deps",
        "--scale",
        f"worker={scenario.workers}",
        "worker",
        env=scenario.env,
    )
    await wait_for_workers(scenario.workers)
    rabbitmq = container_ids("rabbitmq")[0]

    started_at = datetime.now(UTC)
    compose(
        "--profile",
        "batch",
        "run",
        "-d",
        "--name",
        SOURCE_CONTAINER,
        "source",
        env=scenario.env,
    )
    samples = await record(scenario, probe, rabbitmq)

    run("docker", "stop", "--time", "10", SOURCE_CONTAINER, check=False)
    records = source_records()
    source = source_log(records, started_at)
    published = source_published(records)
    final_queue = await drain(probe) if scenario.drain else await probe.queue()
    tasks, rows = await completed_tasks(started_at)
    run("docker", "rm", "-f", SOURCE_CONTAINER, check=False)
    redeliveries = compose("logs", "--since", started_at.isoformat(), "worker").count(
        "Повторная доставка"
    )

    seconds = per_second(pd.DataFrame(samples), source, tasks)
    left = final_queue.get("messages_ready", 0) + final_queue.get("messages_unacknowledged", 0)
    retrying = await probe.depth("tasks.retry")
    accounted = rows["done"] + rows["failed"] + left + retrying
    final = {
        "confirmed_publishes": published,
        "rows_done": rows["done"],
        "rows_failed": rows["failed"],
        "rows_pending": rows["pending"],
        "left_in_queue": left,
        "in_retry_queue": retrying,
        "in_dlq": await probe.depth("tasks.dlq"),
        "redeliveries_already_done": redeliveries,
        "lost": published - accounted if scenario.drain and published is not None else None,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    seconds.to_csv(OUTPUT / f"{scenario.name}.csv", index=False)
    plot.scenario_figure(scenario.title, seconds, scenario.events, OUTPUT / f"{scenario.name}.png")
    summary = summarize(scenario, seconds, final)
    print("   ", json.dumps(summary, ensure_ascii=False))
    return summary


async def main() -> None:
    engine = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=docker_socket()),
        base_url="http://docker",
        timeout=10,
    )
    management = httpx.AsyncClient(base_url=MANAGEMENT_URL, auth=MANAGEMENT_AUTH, timeout=10)
    probe = Probe(management, engine)
    selected = os.environ.get("SCENARIOS")
    scenarios = [s for s in SCENARIOS if not selected or s.name in selected.split(",")]
    summaries = []
    try:
        for scenario in scenarios:
            summaries.append(await run_scenario(scenario, probe))
    finally:
        await reset(management)
        compose("up", "-d", "--no-deps", "worker")
        await engine.aclose()
        await management.aclose()

    (OUTPUT / "summary.json").write_text(
        json.dumps(
            {"scenarios": summaries, "config": [asdict(s) for s in scenarios]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    plot.comparison_figure(OUTPUT, [s.name for s in scenarios], OUTPUT / "comparison.png")


if __name__ == "__main__":
    asyncio.run(main())
