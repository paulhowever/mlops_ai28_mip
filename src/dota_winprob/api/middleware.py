import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from dota_winprob.logging_config import REQUEST_ID

log = logging.getLogger("dota_winprob.app")

REQUEST_ID_HEADER = "X-Request-ID"
QUIET_PATHS = frozenset({"/healthz", "/health", "/api/v1/health"})


async def _call_next_or_500(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    try:
        return await call_next(request)
    except Exception:
        log.exception("Необработанное исключение")
        return JSONResponse({"detail": "Internal Server Error"}, status_code=500)


async def request_context(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    token = REQUEST_ID.set(request_id)
    started = time.perf_counter()
    try:
        response = await _call_next_or_500(request, call_next)
        response.headers[REQUEST_ID_HEADER] = request_id
        level = logging.DEBUG if request.url.path in QUIET_PATHS else logging.INFO
        log.log(
            level,
            "Запрос обработан",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response
    finally:
        REQUEST_ID.reset(token)


async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    detail = [
        {key: value for key, value in item.items() if key != "input"} for item in error.errors()
    ]
    return JSONResponse(
        {"detail": jsonable_encoder(detail)},
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    )
