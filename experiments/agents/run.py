import asyncio
import json
import logging
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dota_winprob.agent.errors import AskError
from dota_winprob.agent.schemas import AskRequest
from dota_winprob.agent.service import AgentService
from experiments.agents import plot, questions
from experiments.agents.stand import Stand, connect
from experiments.agents.variants import BUILDERS

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "agents"
MIN_REPEATS, MAX_REPEATS = 2, 3
BUDGET_S = 20 * 60
LAST_REPEAT_S = 12 * 60
INFRASTRUCTURE_RETRIES = 1


async def _ask(stand: Stand, service: AgentService, question: questions.Question) -> dict[str, Any]:
    for attempt in range(INFRASTRUCTURE_RETRIES + 1):
        stand.timer.reset()
        started = time.perf_counter()
        try:
            response = await service.ask(AskRequest(question=question.text))
        except AskError as error:
            row = {"error": error.reason, "ok": None, "wall_s": time.perf_counter() - started}
            if attempt < INFRASTRUCTURE_RETRIES and error.retry_after is not None:
                await asyncio.sleep(error.retry_after)
            continue
        return {
            "error": None,
            "ok": question.check(response.answer),
            "status": response.status.value,
            "answer": response.answer,
            "warnings": response.warnings,
            "tools": [step.tool for step in response.steps],
            "llm_requests": response.usage.llm_requests,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "wall_s": round(time.perf_counter() - started, 2),
            "model_s": round(stand.timer.seconds, 2),
        }
    return row


async def _repeat(
    stand: Stand,
    services: dict[str, AgentService],
    asked: list[questions.Question],
    number: int,
    deadline: float,
    rows: list[dict[str, Any]],
) -> bool:
    for index, question in enumerate(asked):
        order = list(services) if (index + number) % 2 else list(reversed(services))
        for variant in order:
            if time.monotonic() > deadline:
                return False
            row = await _ask(stand, services[variant], question)
            rows.append(
                {
                    "variant": variant,
                    "repeat": number,
                    "question": question.key,
                    "area": question.area,
                    **row,
                }
            )
            print(number, question.key, variant, row["ok"], row["error"], flush=True)
    return True


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "total": round(sum(values), 2),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in BUILDERS:
        mine = [row for row in rows if row["variant"] == variant]
        answered = [row for row in mine if row["error"] is None]
        summary[variant] = {
            "runs": len(mine),
            "infrastructure_errors": len(mine) - len(answered),
            "correct": sum(row["ok"] for row in answered),
            "answered": len(answered),
            "accuracy": round(sum(row["ok"] for row in answered) / len(answered), 3),
            "step_limit": sum(row["status"] == "step_limit" for row in answered),
            "with_warnings": sum(bool(row["warnings"]) for row in answered),
            "tokens": _spread([row["input_tokens"] + row["output_tokens"] for row in answered]),
            **{
                name: _spread([row[name] for row in answered])
                for name in ("llm_requests", "wall_s", "model_s")
            },
            "wrong": sorted(
                f"{row['question']}#{row['repeat']}" for row in answered if not row["ok"]
            ),
        }
    return summary


async def main() -> None:
    started, rows = time.monotonic(), []
    deadline = started + BUDGET_S
    async with connect() as stand:
        await questions.prepare(stand)
        truth = await questions.truth(stand)
        asked = questions.build(truth)
        services = {name: build(stand) for name, build in BUILDERS.items()}
        complete = 0
        for number in range(1, MAX_REPEATS + 1):
            late = time.monotonic() - started > LAST_REPEAT_S
            if number > MIN_REPEATS and late:
                break
            if not await _repeat(stand, services, asked, number, deadline, rows):
                break
            complete = number
        after = await questions.truth(stand)
        result = {
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": stand.settings.llm_model,
            "temperature": stand.settings.llm_temperature,
            "step_budget": stand.settings.agent_step_budget,
            "tool_call_budget": stand.settings.agent_tool_call_budget,
            "requests_per_minute": stand.settings.llm_requests_per_minute,
            "minutes": round((time.monotonic() - started) / 60, 1),
            "complete_repeats": complete,
            "questions": {question.key: question.text for question in asked},
            "truth_before": truth,
            "truth_after": after,
            "summary": summarize(rows),
            "rows": rows,
        }
        text = stand.redactor.text(json.dumps(result, ensure_ascii=False, indent=1))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "comparison.json").write_text(f"{text}\n", encoding="utf-8")
    plot.comparison(result, OUTPUT / "comparison.png")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    logging.disable(logging.ERROR)
    asyncio.run(main())
