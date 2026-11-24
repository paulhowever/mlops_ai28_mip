import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from dota_winprob.agent.schemas import Prediction

TOLERANCE = 0.0051
BEFORE, AFTER = 80, 40
PROBABILITY_TOOLS = frozenset({"predict", "get_task"})
MODEL_TOOL = "model_card"

_NUMBER = re.compile(
    r"(?<![\d.,])(?:(?P<percent>\d{1,3}(?:[.,]\d+)?)\s*%|(?P<share>0[.,]\d+|1[.,]0+))(?![\d%])"
)
_TOPIC = re.compile(r"вероятност|шанс|win_probability", re.IGNORECASE)
_CLAUSE_END = re.compile(r"[\n;!?]|\.(?=\s)")


def _numbers(text: str) -> list[tuple[int, int, float]]:
    found = []
    for match in _NUMBER.finditer(text):
        raw = match["percent"] or match["share"]
        value = float(raw.replace(",", "."))
        found.append((match.start(), match.end(), value / 100 if match["percent"] else value))
    return found


def _context(text: str, start: int, end: int) -> str:
    before = _CLAUSE_END.split(text[max(0, start - BEFORE) : start])[-1]
    after = _CLAUSE_END.split(text[end : end + AFTER])[0]
    return f"{before} {after}"


def claimed_probabilities(text: str) -> list[float]:
    return [
        value for start, end, value in _numbers(text) if _TOPIC.search(_context(text, start, end))
    ]


def mentioned(text: str) -> set[float]:
    return {value for _, _, value in _numbers(text)}


@dataclass
class Evidence:
    predictions: list[Prediction] = field(default_factory=list)
    other: set[float] = field(default_factory=set)
    challenged: bool = False

    def collect(self, tool: str, data: dict[str, Any]) -> None:
        if tool in PROBABILITY_TOOLS and data.get("win_probability") is not None:
            self.predictions.append(
                Prediction(
                    tool=tool,
                    task_id=str(data.get("task_id")),
                    win_probability=float(data["win_probability"]),
                    model_version=data.get("model_version"),
                )
            )
        elif tool == MODEL_TOOL:
            self.other.update((data.get("metrics") or {}).values())

    def allow(self, text: str) -> None:
        self.other |= mentioned(text)

    def ungrounded(self, text: str) -> list[float]:
        known = [prediction.win_probability for prediction in self.predictions]
        allowed = self.other | set(known)
        allowed |= {1 - value for value in known}
        allowed |= {abs(left - right) for left, right in combinations(known, 2)}
        return [
            claim
            for claim in claimed_probabilities(text)
            if not any(abs(claim - value) <= TOLERANCE for value in allowed)
        ]
