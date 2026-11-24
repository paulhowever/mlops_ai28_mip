from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from dota_winprob.agent.sources.prometheus import Point

COLOURS = {"single": "tab:blue", "team": "tab:orange"}
LABELS = {"single": "один агент", "team": "координатор и специалисты"}
PANELS = {
    "llm_requests": "Запросов к модели на вопрос",
    "tokens": "Токенов на вопрос",
    "model_s": "Время модели на вопрос, с",
}


def comparison(result: dict[str, Any], path: Path) -> None:
    rows = [row for row in result["rows"] if row["error"] is None]
    for row in rows:
        row["tokens"] = row["input_tokens"] + row["output_tokens"]
    figure, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    figure.suptitle(
        f"Один агент и координатор со специалистами: {len(result['questions'])} вопросов, "
        f"полных прогонов {result['complete_repeats']}, {result['model']}"
    )
    variants = list(LABELS)
    shares = [result["summary"][variant]["accuracy"] * 100 for variant in variants]
    bars = axes[0].bar(
        [LABELS[variant].replace(" и ", "\nи ") for variant in variants],
        shares,
        color=[COLOURS[variant] for variant in variants],
    )
    for bar, variant in zip(bars, variants, strict=True):
        summary = result["summary"][variant]
        axes[0].annotate(
            f"{summary['correct']} из {summary['answered']}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Верных ответов, %")
    for axis, (name, title) in zip(axes[1:], PANELS.items(), strict=True):
        boxes = axis.boxplot(
            [[row[name] for row in rows if row["variant"] == variant] for variant in variants],
            tick_labels=[LABELS[variant].replace(" и ", "\nи ") for variant in variants],
            patch_artist=True,
            widths=0.55,
        )
        for box, variant in zip(boxes["boxes"], variants, strict=True):
            box.set_facecolor(COLOURS[variant])
            box.set_alpha(0.5)
        axis.set_title(title)
        axis.set_ylim(bottom=0)
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def pipeline(series: dict[str, list[Point]], path: Path) -> None:
    figure, (depth, rate, dead) = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    figure.suptitle("Конвейер dota-winprob, время UTC")

    def draw(axis: plt.Axes, name: str, label: str, **style: object) -> None:
        points = series[name]
        moments = [datetime.fromtimestamp(stamp, UTC) for stamp, _ in points]
        axis.plot(moments, [value for _, value in points], label=label, **style)

    draw(depth, "queue_ready", "ready")
    draw(depth, "queue_limit", "лимит", color="tab:red", linestyle="--")
    depth.set_title("Глубина очереди tasks.score, сообщений")
    draw(rate, "source_rate", "источник", color="tab:orange")
    draw(rate, "worker_rate", "воркеры", color="tab:green")
    rate.set_title("Скорость, задач в секунду")
    draw(dead, "dlq_ready", "DLQ", color="tab:red")
    dead.set_title("Сообщений в tasks.dlq")
    for axis in (depth, rate, dead):
        axis.grid(alpha=0.3)
        axis.legend(loc="upper left")
        axis.set_ylim(bottom=0)
        axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=UTC))
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)
