import hmac
from collections.abc import Awaitable, Callable, Iterable

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ALERTS_PATH = "/agent/alerts"
ASK_PATH = "/agent/ask"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
TRUSTED_SITES = frozenset({"same-origin", "none"})
JSON_TYPE = "application/json"

MAX_ASK_BODY = 6 * 1024 * 1024
MAX_ALERTS_BODY = 512 * 1024
MAX_BODY = 16 * 1024
BODY_LIMITS = {ASK_PATH: MAX_ASK_BODY, ALERTS_PATH: MAX_ALERTS_BODY}

PAGE_PATHS = ("/", "/static/")
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
COMMON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class Refused(Exception):
    def __init__(self, status_code: int, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.reason = reason
        self.detail = detail


def token_matches(authorization: str | None, expected: str) -> bool:
    scheme, _, value = (authorization or "").partition(" ")
    given = value.strip().encode() if scheme.lower() == "bearer" else b""
    return hmac.compare_digest(given, expected.encode())


def host_name(host: str) -> str:
    name, separator, port = host.rpartition(":")
    return (name if separator and port.isdigit() else host).lower()


def _check_origin(headers: Headers, host: str, path: str) -> None:
    origins = headers.getlist("origin")
    expected = {f"http://{host}", f"https://{host}"}
    if path == ALERTS_PATH and origins:
        raise Refused(403, "origin", "Алерты принимаются только от Grafana, не из браузера")
    if any(origin.lower() not in expected for origin in origins) or len(origins) > 1:
        raise Refused(403, "origin", "Запрос пришёл с чужой страницы")
    site = headers.get("sec-fetch-site")
    if site is not None and site not in TRUSTED_SITES:
        raise Refused(403, "origin", "Запрос пришёл с чужой страницы")


def _check_body(headers: Headers, limit: int) -> None:
    media_type = headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != JSON_TYPE:
        raise Refused(415, "content_type", f"Нужен Content-Type: {JSON_TYPE}")
    length = headers.get("content-length")
    if length is not None and (not length.isdigit() or int(length) > limit):
        raise Refused(413, "body_size", "Тело запроса слишком большое")


class HttpGuard:
    def __init__(self, app: ASGIApp, hosts: Iterable[str], alert_hosts: Iterable[str] = ()) -> None:
        self._app = app
        self._hosts = frozenset(hosts)
        self._alert_hosts = self._hosts | frozenset(alert_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        path, secured = scope["path"], _with_headers(send, scope["path"])
        limit = BODY_LIMITS.get(path, MAX_BODY)
        try:
            self._check(scope["method"], path, Headers(scope=scope), limit)
        except Refused as refused:
            body = {"detail": refused.detail, "reason": refused.reason}
            response = JSONResponse(body, status_code=refused.status_code)
            await response(scope, receive, secured)
            return
        await self._app(scope, _limited(receive, limit), secured)

    def _check(self, method: str, path: str, headers: Headers, limit: int) -> None:
        hosts = headers.getlist("host")
        allowed = self._alert_hosts if path == ALERTS_PATH else self._hosts
        if len(hosts) != 1 or host_name(hosts[0]) not in allowed:
            raise Refused(403, "host", "Недопустимый заголовок Host")
        if method in SAFE_METHODS:
            return
        _check_origin(headers, hosts[0].lower(), path)
        _check_body(headers, limit)


def _with_headers(send: Send, path: str) -> Send:
    is_page = path == PAGE_PATHS[0] or path.startswith(PAGE_PATHS[1])

    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = MutableHeaders(scope=message)
            headers.update(COMMON_HEADERS)
            headers["Cache-Control"] = "no-cache" if is_page else "no-store"
            if is_page:
                headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        await send(message)

    return wrapped


def _limited(receive: Receive, limit: int) -> Callable[[], Awaitable[Message]]:
    total = 0

    async def wrapped() -> Message:
        nonlocal total
        message = await receive()
        total += len(message.get("body", b""))
        if total > limit:
            raise HTTPException(413, "Тело запроса слишком большое")
        return message

    return wrapped
