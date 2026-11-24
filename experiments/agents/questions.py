import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import asyncpg

from dota_winprob.agent.tools.incidents import IncidentTools
from dota_winprob.agent.tools.matches import MatchTools
from dota_winprob.agent.tools.monitoring import MonitoringTools
from experiments.agents.answers import has_any, has_number, has_probability
from experiments.agents.stand import Stand

WINDOW = 120
SLOW_TASK = "batch:15000"
FAILED_TASK = "bench:failed"
FAILED_ERROR = "ValueError: input contains NaN"
MATCH = 566
EXTRA_GOLD = 3000
FEATURES = (14500.0, 13200.0, 15100.0, 14300.0, 128.0, 121.0)
CAUSE = re.compile(r"\b([a-z.]*error|\w+(?:Error|Exception))\b")
MAX_LINES = 5000
RETRIED = "Задача отправлена на повтор"

PREPARE = """
INSERT INTO tasks (task_id, source, status, error, attempts, published_at, completed_at)
VALUES ($1, 'api', 'failed', $2, 3, now(), now())
ON CONFLICT (task_id) DO NOTHING
"""


@dataclass(frozen=True)
class Question:
    key: str
    area: str
    text: str
    check: Callable[[str], bool]


async def prepare(stand: Stand) -> None:
    connection = await asyncpg.connect(stand.settings.database_url)
    try:
        await connection.execute(PREPARE, FAILED_TASK, FAILED_ERROR)
    finally:
        await connection.close()


async def _probability(tools: MatchTools, values: tuple[float, ...]) -> float:
    result = await tools.predict(*values)
    if result.status != "ok":
        raise SystemExit(f"эталонный прогноз не посчитан: {result.summary}")
    return float(result.data["win_probability"])


def _top_exception(entries: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for item in entries:
        text = str(item.get("error") or item.get("reason") or item.get("exception") or "")
        names = CAUSE.findall(text.strip().rsplit("\n", 1)[-1])
        if names:
            counts[names[0].rsplit(".", 1)[-1]] += 1
    return counts.most_common(1)[0][0]


async def truth(stand: Stand) -> dict[str, Any]:
    sources, settings = stand.sources, stand.settings
    monitoring = MonitoringTools(sources.prometheus)
    incidents = IncidentTools(sources.loki, sources.tempo, sources.tasks)
    matches = MatchTools(
        sources.api,
        sources.mlflow,
        sources.matches,
        settings.model_name,
        settings.model_alias,
        settings.agent_predict_timeout,
        settings.agent_predict_poll_interval,
    )
    metrics = (await monitoring.pipeline_status()).data["metrics"]
    status = {item["name"]: item["value"] for item in metrics}
    queue = (await monitoring.metric_history("queue_ready", WINDOW)).data
    rate = (await monitoring.metric_history("worker_rate", WINDOW)).data
    trace = (await incidents.get_trace(task_id=SLOW_TASK)).data
    task = (await incidents.get_task(SLOW_TASK)).data
    card = (await matches.model_card()).data
    totals = await sources.tasks.totals()
    entries = await sources.loki.search(
        '{service="dota-worker", level="WARNING"}', WINDOW, MAX_LINES
    )
    state = await sources.matches.find(MATCH)
    if state is None:
        raise SystemExit("матча для проверки нет в data/tasks.csv")
    radiant_gold, dire_gold = state.radiant_gold, state.dire_gold
    rest = (state.radiant_xp, state.dire_xp, state.radiant_lh, state.dire_lh)
    return {
        "queue_max": queue["max"],
        "rate_max": rate["max"],
        "consumers": status["consumers"],
        "limit": status["queue_limit"],
        "dlq": status["dlq_ready"],
        "wait_s": float(trace["queue_wait_s"][0]),
        "inference_ms": float(trace["inference_ms"][0]),
        "slow_probability": float(task["win_probability"]),
        "exception": _top_exception(entries),
        "roc_auc": card["metrics"]["roc_auc"],
        "version": float(card["version"]),
        "failed": float(totals.get("failed", 0)),
        "predict": await _probability(matches, FEATURES),
        "base": await _probability(matches, (radiant_gold, dire_gold, *rest)),
        "richer": await _probability(matches, (radiant_gold, dire_gold + EXTRA_GOLD, *rest)),
    }


def build(t: dict[str, Any]) -> list[Question]:
    rg, dg, rx, dx, rl, dl = (f"{value:.0f}" for value in FEATURES)
    seconds, millis = (1.0, 1000.0), (1.0, 0.001)
    return [
        Question(
            "m1",
            "метрики",
            f"Какой была максимальная глубина очереди tasks.score за последние {WINDOW} минут?",
            lambda a: has_number(a, t["queue_max"], 1),
        ),
        Question(
            "m2",
            "метрики",
            "Сколько сейчас потребителей у рабочей очереди и какой у неё лимит?",
            lambda a: has_number(a, t["consumers"], 0) and has_number(a, t["limit"], 0),
        ),
        Question(
            "m3",
            "метрики",
            f"До какого максимума доходила пропускная способность воркеров за последние {WINDOW} минут?",
            lambda a: has_number(a, t["rate_max"], 0.03 * t["rate_max"]),
        ),
        Question(
            "l1",
            "логи и трейсы",
            f"Сколько времени задача {SLOW_TASK} ждала в очереди и сколько занял inference?",
            lambda a: (
                has_number(a, t["wait_s"], 0.03 * t["wait_s"], seconds)
                and has_number(a, t["inference_ms"], 0.03 * t["inference_ms"], millis)
            ),
        ),
        Question(
            "l2",
            "логи и трейсы",
            f"Из-за какой ошибки воркеры чаще всего отправляли задачи на повтор за последние {WINDOW} "
            "минут? Назови имя исключения.",
            lambda a: has_any(a, re.escape(t["exception"])),
        ),
        Question(
            "l3",
            "логи и трейсы",
            f"Что с задачей {FAILED_TASK}: статус, сколько было попыток и какая ошибка?",
            lambda a: has_any(a, "failed|упал") and has_number(a, 3, 0) and has_any(a, "NaN"),
        ),
        Question(
            "l4",
            "логи и трейсы",
            "Сколько сообщений сейчас в DLQ и почему они туда попали?",
            lambda a: has_number(a, t["dlq"], 0) and has_any(a, "ValidationError|валид|схем|мусор"),
        ),
        Question(
            "t1",
            "матчи и модель",
            "Какой ROC AUC у рабочей модели и какая у неё версия?",
            lambda a: has_number(a, t["roc_auc"], 0.001) and has_number(a, t["version"], 0),
        ),
        Question(
            "t2",
            "матчи и модель",
            f"Какова вероятность победы Radiant: золото Radiant {rg}, Dire {dg}, опыт {rx} и {dx}, "
            f"добивания {rl} и {dl}?",
            lambda a: has_probability(a, t["predict"]),
        ),
        Question(
            "t3",
            "матчи и модель",
            f"Как изменится вероятность победы Radiant в матче {MATCH}, если у Dire будет на "
            f"{EXTRA_GOLD} золота больше? Назови обе вероятности.",
            lambda a: has_probability(a, t["base"]) and has_probability(a, t["richer"]),
        ),
        Question(
            "x1",
            "сквозные",
            f"Для задачи {SLOW_TASK} назови прогноз, время ожидания в очереди и ROC AUC модели, "
            "которая её посчитала.",
            lambda a: (
                has_probability(a, t["slow_probability"])
                and has_number(a, t["wait_s"], 0.03 * t["wait_s"], seconds)
                and has_number(a, t["roc_auc"], 0.001)
            ),
        ),
        Question(
            "x2",
            "сквозные",
            "Сколько сейчас воркеров, сколько задач со статусом failed и какая версия модели в работе?",
            lambda a: (
                has_number(a, t["consumers"], 0)
                and has_number(a, t["failed"], 0)
                and has_number(a, t["version"], 0)
            ),
        ),
    ]
