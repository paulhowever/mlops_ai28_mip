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
_QUANTITY = re.compile(
    r"(?<![\w.,:/\-])-?(?P<value>(?>\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?![\d])|\d+)(?>[.,]\d+)?)(?P<percent>\s?%)?(?![\dA-Za-z:/])"
)
_CLOCK = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?Z?)?|\d{1,2}:\d{2}(?::\d{2})?Z?"
)
_NUMBER_LIKE = re.compile(r"(?<![A-Za-z\d])\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
TRIVIAL = 10
RELATIVE = 0.01
SCALES = (1.0, 1000.0, 0.001, 1 / 60, 60.0)


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


@dataclass(frozen=True)
class Quantity:
    text: str
    value: float
    step: float
    percent: bool

    def matches(self, known: float) -> bool:
        scales = (100.0,) if self.percent else SCALES
        if self.percent and known > 1:
            scales = (1.0,)
        return any(
            abs(self.value - known * scale) <= max(self.step, RELATIVE * known * scale)
            for scale in scales
        )


def claimed_quantities(text: str) -> list[Quantity]:
    found = []
    for match in _QUANTITY.finditer(_CLOCK.sub(" ", text)):
        raw = re.sub(r"\s", "", match["value"]).replace(",", ".")
        _, _, fraction = raw.partition(".")
        value, percent = float(raw), match["percent"] is not None
        if not fraction and value <= TRIVIAL and not percent:
            continue
        shown = raw + ("%" if percent else "")
        found.append(Quantity(shown, value, 0.51 * 10 ** -len(fraction), percent))
    return found


def all_numbers(text: str) -> set[float]:
    spaceless = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))", "", text)
    return {float(raw) for raw in _NUMBER_LIKE.findall(spaceless.replace(",", "."))} | {
        float(raw) for raw in _NUMBER_LIKE.findall(text)
    }


def mentioned(text: str) -> set[float]:
    return {value for _, _, value in _numbers(text)}


@dataclass
class Evidence:
    predictions: list[Prediction] = field(default_factory=list)
    other: set[float] = field(default_factory=set)
    numbers: set[float] = field(default_factory=set)
    challenged: bool = False

    def collect(self, tool: str, data: dict[str, Any], shown: str = "") -> None:
        self.numbers |= all_numbers(shown)
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
        self.numbers |= all_numbers(text)

    def unknown_numbers(self, text: str) -> list[str]:
        known = self.numbers | {p.win_probability for p in self.predictions}
        known |= {1 - p.win_probability for p in self.predictions}
        known |= {
            abs(left.win_probability - right.win_probability)
            for left, right in combinations(self.predictions, 2)
        }
        missing = [
            claim.text
            for claim in claimed_quantities(text)
            if not any(claim.matches(value) for value in known)
        ]
        return list(dict.fromkeys(missing))

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
