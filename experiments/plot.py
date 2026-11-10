from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _mark_events(axes: Iterable[plt.Axes], events: Iterable) -> None:
    for axis in axes:
        for event in events:
            axis.axvline(event.at, color="grey", linestyle="--", linewidth=1)
            axis.annotate(
                event.label,
                xy=(event.at, 1),
                xycoords=("data", "axes fraction"),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color="grey",
            )


def scenario_figure(title: str, seconds: pd.DataFrame, events: Iterable, path: Path) -> None:
    events = list(events)
    figure, (depth, rate, latency, memory) = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    figure.suptitle(title)

    depth.plot(seconds["t"], seconds["ready"], label="ready (ждут в очереди)")
    depth.plot(seconds["t"], seconds["unacked"], label="unacked (взяты воркерами)")
    depth.set_ylabel("сообщений")
    depth.set_title("Глубина очереди tasks.score")
    depth.legend(loc="upper left")

    rate.plot(seconds["t"], seconds["source_rate"], label="источник публикует")
    rate.plot(seconds["t"], seconds["throughput"], label="воркеры выполняют")
    rate.set_ylabel("задач в секунду")
    rate.set_title("Скорость источника и пропускная способность")
    rate.legend(loc="upper left")

    latency.plot(seconds["t"], seconds["latency_s"], color="tab:red")
    latency.set_ylabel("секунд")
    latency.set_title("Время задачи: от публикации до записи результата")

    memory.plot(seconds["t"], seconds["rabbitmq_mb"], label="RabbitMQ")
    memory.plot(seconds["t"], seconds["workers_mb"], label="воркеры, сумма")
    memory.plot(seconds["t"], seconds["source_mb"], label="источник")
    memory.set_ylabel("МБ")
    memory.set_xlabel("секунды от старта источника")
    memory.set_title("Память контейнеров")
    memory.legend(loc="upper left")

    for axis in (depth, rate, latency, memory):
        axis.grid(alpha=0.3)
    _mark_events((depth, rate, latency, memory), events)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def comparison_figure(directory: Path, names: list[str], path: Path) -> None:
    figure, (depth, latency, memory) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    figure.suptitle("Сравнение сценариев")
    for name in names:
        source = directory / f"{name}.csv"
        if not source.exists():
            continue
        seconds = pd.read_csv(source)
        depth.plot(seconds["t"], seconds["ready"] + seconds["unacked"], label=name)
        latency.plot(seconds["t"], seconds["latency_s"], label=name)
        memory.plot(seconds["t"], seconds["rabbitmq_mb"] + seconds["workers_mb"], label=name)

    depth.set_yscale("symlog", linthresh=100)
    depth.set_ylabel("сообщений, ready + unacked")
    depth.set_title("Сообщения в системе (логарифмическая шкала выше 100)")
    latency.set_ylabel("секунд")
    latency.set_title("Время задачи")
    memory.set_ylabel("МБ")
    memory.set_title("Память RabbitMQ и воркеров")
    memory.set_xlabel("секунды от старта источника")
    for axis in (depth, latency, memory):
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
