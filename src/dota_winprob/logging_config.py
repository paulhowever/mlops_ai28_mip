import contextlib
import contextvars
import datetime
import json
import logging
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

from opentelemetry import trace

from dota_winprob.config import Settings
from dota_winprob.telemetry import service_identity

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
LOG_CONTEXT: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "log_context", default=MappingProxyType({})
)

LIBRARY_LOGGERS = (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "faststream",
    "faststream.access",
    "faststream.access.rabbit",
    "mlflow",
)
TELEMETRY_LOGGER = "opentelemetry"
TELEMETRY_LOG_INTERVAL = 60.0

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"asctime", "message", "taskName", "color_message"}
)

_URL_CREDENTIALS = re.compile(
    r"(?P<prefix>[a-zA-Z][a-zA-Z0-9+.\-]*://[^:/?#@\s\"']*):(?P<secret>[^@/\s\"']+)@"
)
_SECRET_PARAMS = re.compile(
    r"(?P<key>\b(?:password|passwd|pwd|secret|token|api_key)=)(?P<value>[^&\s\"']+)",
    re.IGNORECASE,
)


def mask_secrets(text: str) -> str:
    text = _URL_CREDENTIALS.sub(r"\g<prefix>:***@", text)
    return _SECRET_PARAMS.sub(r"\g<key>***", text)


def _merged(fields: Mapping[str, object]) -> Mapping[str, Any]:
    current = dict(LOG_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    return MappingProxyType(current)


@contextlib.contextmanager
def log_context(**fields: object) -> Iterator[None]:
    token = LOG_CONTEXT.set(_merged(fields))
    try:
        yield
    finally:
        LOG_CONTEXT.reset(token)


def bind(**fields: object) -> None:
    LOG_CONTEXT.set(_merged(fields))


def _isoformat(created: float) -> str:
    moment = datetime.datetime.fromtimestamp(created, tz=datetime.UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JsonFormatter(logging.Formatter):
    def __init__(self, settings: Settings, static: Mapping[str, object] | None = None) -> None:
        super().__init__()
        service, version, environment = service_identity(settings)
        self.context = {
            "service": service,
            "version": version,
            "environment": environment,
            "git_sha": settings.git_sha,
            **(static or {}),
        }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _isoformat(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self.context,
        }

        request_id = REQUEST_ID.get()
        if request_id is not None:
            payload["request_id"] = request_id

        payload.update(LOG_CONTEXT.get())

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            payload["trace_id"] = format(span_context.trace_id, "032x")
            payload["span_id"] = format(span_context.span_id, "016x")

        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _RESERVED}
        )

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return mask_secrets(json.dumps(payload, ensure_ascii=False, default=str))


class ThrottleFilter(logging.Filter):
    def __init__(
        self,
        prefix: str,
        interval: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.interval = interval
        self._clock = clock
        self._lock = threading.Lock()
        self._seen: dict[tuple[str, str], tuple[float, int]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith(self.prefix):
            return True
        key = (record.name, str(record.msg))
        now = self._clock()
        with self._lock:
            previous = self._seen.get(key)
            if previous is not None and now - previous[0] < self.interval:
                self._seen[key] = (previous[0], previous[1] + 1)
                return False
            self._seen[key] = (now, 0)
        if previous is not None and previous[1]:
            record.suppressed = previous[1]
        return True


def setup_logging(settings: Settings, **static: object) -> None:
    level = logging.getLevelName(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(settings, static))
    handler.setLevel(level)
    handler.addFilter(ThrottleFilter(TELEMETRY_LOGGER, TELEMETRY_LOG_INTERVAL))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for name in LIBRARY_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger(TELEMETRY_LOGGER).setLevel(max(level, logging.WARNING))
