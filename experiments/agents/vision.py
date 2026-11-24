import asyncio
import base64
import json
import logging
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from dota_winprob.agent.errors import AskError
from dota_winprob.agent.images import decode
from dota_winprob.agent.llm.client import model_settings
from dota_winprob.agent.promql import METRICS
from dota_winprob.agent.schemas import AskRequest, ImageInput
from dota_winprob.agent.sources.prometheus import Point
from experiments.agents import plot
from experiments.agents.answers import final_minute, final_number
from experiments.agents.run import OUTPUT
from experiments.agents.stand import Stand, connect
from experiments.agents.variants import single

REPEATS = 2
STEP_S = 10
BUSY = 50
EPISODE_POINTS = 12
MINUTES = int(os.environ.get("VISION_MINUTES", "60"))
PICTURE, NUMBERS, BOTH = "picture", "numbers", "both"
READER = (
    "Ты читаешь график мониторинга. Инструментов нет, отвечай только по картинке, по-русски, одним-двумя "
    "предложениями."
)
FORMAT = " Последняя строка ответа строго в виде «ИТОГ: значение»."
SOURCE_METRICS = ("queue_ready", "queue_limit", "source_rate", "worker_rate", "dlq_ready")


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    expected: float
    check: Callable[[str], bool]


def _near(expected: float, tolerance: float) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        value = final_number(answer)
        return value is not None and abs(value - expected) <= tolerance

    return check


def _minute_near(expected: float, tolerance: float) -> Callable[[str], bool]:
    def check(answer: str) -> bool:
        value = final_minute(answer)
        return value is not None and abs(value - expected) <= tolerance

    return check


def episodes(points: list[Point]) -> list[float]:
    starts, run = [], []
    for stamp, value in [*points, (math.inf, 0.0)]:
        if value > BUSY:
            run.append(stamp)
            continue
        if len(run) >= EPISODE_POINTS:
            starts.append(run[0])
        run = []
    return starts


def questions(series: dict[str, list[Point]]) -> list[Question]:
    window = f"за последние {MINUTES} минут"
    peak = max(value for _, value in series["queue_ready"])
    rate = max(value for _, value in series["worker_rate"])
    dead = max(value for _, value in series["dlq_ready"])
    starts = episodes(series["queue_ready"])
    began = datetime.fromtimestamp(starts[-1], UTC)
    minute = began.hour * 60 + began.minute
    return [
        Question(
            "v1",
            f"Какой была максимальная глубина очереди tasks.score {window}?",
            peak,
            _near(peak, 3),
        ),
        Question(
            "v2",
            f"Какой была максимальная пропускная способность воркеров {window}, задач в секунду?",
            rate,
            _near(rate, 0.1 * rate),
        ),
        Question(
            "v3",
            f"Сколько раз {window} очередь tasks.score держалась выше {BUSY} сообщений дольше двух минут подряд?",
            len(starts),
            _near(len(starts), 0),
        ),
        Question(
            "v4",
            f"Во сколько (ЧЧ:ММ, UTC) начался последний {window} период, когда очередь tasks.score держалась "
            f"выше {BUSY} сообщений?",
            minute,
            _minute_near(minute, 2),
        ),
        Question("v5", f"Каким был максимум сообщений в tasks.dlq {window}?", dead, _near(dead, 1)),
    ]


async def _picture(stand: Stand, question: Question, image: bytes) -> dict[str, Any]:
    agent = Agent(stand.model, instructions=READER, model_settings=model_settings(stand.settings))
    run = await agent.run([question.text + FORMAT, BinaryContent(image, media_type="image/png")])
    usage = run.usage
    return {
        "answer": run.output,
        "tools": [],
        "llm_requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
    }


async def _agent(stand: Stand, question: Question, image: bytes | None) -> dict[str, Any]:
    encoded = None if image is None else ImageInput(data=base64.b64encode(image).decode())
    response = await single(stand).ask(AskRequest(question=question.text + FORMAT, image=encoded))
    return {
        "answer": response.answer,
        "tools": [step.tool for step in response.steps],
        "llm_requests": response.usage.llm_requests,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


async def _measure(stand: Stand, mode: str, question: Question, image: bytes) -> dict[str, Any]:
    stand.timer.reset()
    started = time.perf_counter()
    try:
        if mode == PICTURE:
            row = await _picture(stand, question, image)
        else:
            row = await _agent(stand, question, image if mode == BOTH else None)
    except AskError as error:
        return {"error": error.reason, "ok": None}
    return {
        "error": None,
        "ok": question.check(row["answer"]),
        **row,
        "wall_s": round(time.perf_counter() - started, 2),
        "model_s": round(stand.timer.seconds, 2),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for mode in (PICTURE, NUMBERS, BOTH):
        answered = [row for row in rows if row["mode"] == mode and row["error"] is None]
        tokens = [row["input_tokens"] + row["output_tokens"] for row in answered]
        summary[mode] = {
            "correct": sum(row["ok"] for row in answered),
            "answered": len(answered),
            "errors": sum(row["mode"] == mode and row["error"] is not None for row in rows),
            "median_tokens": statistics.median(tokens),
            "median_model_s": statistics.median(row["model_s"] for row in answered),
            "median_llm_requests": statistics.median(row["llm_requests"] for row in answered),
            "wrong": sorted(
                f"{row['question']}#{row['repeat']}" for row in answered if not row["ok"]
            ),
        }
    return summary


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "pipeline.png"
    async with connect() as stand:
        series = {
            name: await stand.sources.prometheus.range(METRICS[name].query(60), MINUTES, STEP_S)
            for name in SOURCE_METRICS
        }
        plot.pipeline(series, path)
        image = decode(base64.b64encode(path.read_bytes()).decode())
        asked, rows = questions(series), []
        for repeat in range(1, REPEATS + 1):
            for question in asked:
                for mode in (PICTURE, NUMBERS, BOTH):
                    row = await _measure(stand, mode, question, image.data)
                    rows.append({"mode": mode, "repeat": repeat, "question": question.key, **row})
                    print(repeat, question.key, mode, row["ok"], row["error"], flush=True)
        result = {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": stand.settings.llm_model,
            "minutes_window": MINUTES,
            "image": {"width": image.width, "height": image.height, "bytes": len(image.data)},
            "repeats": REPEATS,
            "questions": {item.key: item.text for item in asked},
            "expected": {item.key: item.expected for item in asked},
            "summary": summarize(rows),
            "rows": rows,
        }
        text = stand.redactor.text(json.dumps(result, ensure_ascii=False, indent=1))
    (OUTPUT / "vision.json").write_text(f"{text}\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    logging.disable(logging.ERROR)
    asyncio.run(main())
