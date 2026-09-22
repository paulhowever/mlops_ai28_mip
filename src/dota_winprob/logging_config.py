import contextvars
import datetime
import json
import logging
import sys
from typing import Any

from dota_winprob import __version__
from dota_winprob.config import Settings

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"asctime", "message", "taskName", "color_message"}
)


def _isoformat(created: float) -> str:
    moment = datetime.datetime.fromtimestamp(created, tz=datetime.UTC)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class JsonFormatter(logging.Formatter):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.context = {
            "service": settings.app_name,
            "version": __version__,
            "environment": settings.environment,
            "git_sha": settings.git_sha,
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

        payload.update(
            {key: value for key, value in record.__dict__.items() if key not in _RESERVED}
        )

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(settings))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())

    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True

    logging.getLogger("uvicorn.access").disabled = True
