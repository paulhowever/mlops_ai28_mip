import json
import re

from dota_winprob.agent.errors import ToolInputError

TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._\-]{0,127}")
TRACE_ID = re.compile(r"[0-9a-f]{1,32}")
LABEL = re.compile(r"[A-Za-z0-9_\-]{1,64}")
MAX_TEXT = 200

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_RE2_SPECIAL = re.compile(r"([\\.+*?()|\[\]{}^$])")
_CONTROL = re.compile(r"[\x00-\x1f\x7f\u0085  ]")


def task_id(value: str) -> str:
    value = value.strip()
    if not TASK_ID.fullmatch(value):
        raise ToolInputError(
            "task_id: до 128 символов из латиницы, цифр и знаков : . _ -, например batch:566"
        )
    return value


def trace_id(value: str) -> str:
    value = value.strip().lower()
    if not TRACE_ID.fullmatch(value) or not value.strip("0"):
        raise ToolInputError("trace_id: до 32 шестнадцатеричных символов")
    return value.zfill(32)


def search_text(value: str) -> str:
    cleaned = _CONTROL.sub(" ", value).encode("utf-8", "ignore").decode("utf-8").strip()
    if not cleaned or len(cleaned) > MAX_TEXT:
        raise ToolInputError(f"text: от 1 до {MAX_TEXT} символов")
    return cleaned


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def re2_literal(value: str) -> str:
    return _RE2_SPECIAL.sub(r"\\\1", value)


def levels_from(minimum: str) -> str:
    if minimum not in LEVELS:
        raise ToolInputError(f"level: одно из {', '.join(LEVELS)}")
    return "|".join(LEVELS[LEVELS.index(minimum) :])


def logql(
    service: str | None = None,
    level: str | None = None,
    task: str | None = None,
    text: str | None = None,
) -> str:
    if service is not None and not LABEL.fullmatch(service):
        raise ToolInputError("service: недопустимое имя сервиса")
    matchers = [f"service={quote(service)}" if service else 'service=~".+"']
    if level is not None:
        matchers.append(f"level=~{quote(levels_from(level))}")
    query = "{" + ", ".join(matchers) + "}"
    if text is not None:
        pattern = quote(f"(?i){re2_literal(search_text(text))}")
        query += f" |~ {pattern}"
    if task is not None:
        query += f" | task_id={quote(task_id(task))}"
    return query


def traceql_by_task(task: str) -> str:
    return f"{{ span.task.id = {quote(task_id(task))} }}"
