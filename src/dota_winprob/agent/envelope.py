import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from dota_winprob.agent.actions.models import ActionView
from dota_winprob.agent.redaction import Redactor

OPEN_TAG = "untrusted_tool_output"
MAX_RAW_STRING = 2000
MAX_STRING = 600
MAX_SUMMARY = 200
OMITTED = "omitted_items"
FORGED = "[поддельный тег удалён]"
_FORGED_TAG = re.compile(rf"<\s*/?\s*{OPEN_TAG}[^>]{{0,300}}>?", re.IGNORECASE)


@dataclass(frozen=True)
class ToolResult:
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    status: str = "ok"
    action: ActionView | None = None


def without_envelope(text: str) -> str:
    return _FORGED_TAG.sub(FORGED, text)


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… [обрезано]"


def _encode(value: object) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def dump(data: object) -> str:
    text = json.dumps(data, ensure_ascii=False, default=_encode)
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def _clip_value(value: object, limit: int) -> object:
    if isinstance(value, str):
        return clip(value, limit)
    if isinstance(value, Mapping):
        return clip_strings(value, limit)
    if isinstance(value, list | tuple):
        return [_clip_value(item, limit) for item in value]
    return value


def clip_strings(data: Mapping[str, object], limit: int) -> dict[str, Any]:
    return {clip(str(key), limit): _clip_value(value, limit) for key, value in data.items()}


def fit(data: dict[str, Any], limit: int) -> dict[str, Any]:
    result = clip_strings(data, MAX_STRING)
    omitted = 0
    while len(dump(result)) > limit:
        key = max(
            (name for name, value in result.items() if isinstance(value, list) and value),
            key=lambda name: len(dump(result[name])),
            default=None,
        )
        if key is None:
            return {"error": "вывод инструмента не помещается в лимит", "limit_chars": limit}
        keep = len(result[key]) // 2
        omitted += len(result[key]) - keep
        result[key] = result[key][:keep]
        result[OMITTED] = omitted
    return result


def render(tool: str, result: ToolResult, redactor: Redactor, limit: int) -> str:
    masked = redactor.data(clip_strings(result.data, MAX_RAW_STRING))
    body = dump(fit(masked, limit))
    return (
        f'<{OPEN_TAG} tool="{tool}" source="{result.source}" status="{result.status}">\n'
        f"{body}\n"
        f"</{OPEN_TAG}>"
    )
