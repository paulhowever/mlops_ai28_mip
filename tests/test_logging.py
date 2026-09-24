import json
import logging
import re
import sys

import httpx

from dota_winprob import __version__
from dota_winprob.app import create_app
from dota_winprob.logging_config import REQUEST_ID, JsonFormatter
from tests.fakes import HealthyPool, opendota_ok

ACCESS_MESSAGE = "Запрос обработан"


def format_record(formatter, message="событие", exc_info=None, extra=None):
    record = logging.getLogger("test").makeRecord(
        "test", logging.INFO, __file__, 1, message, None, exc_info, extra=extra
    )
    return formatter.format(record)


def access_log(captured_out):
    records = [json.loads(line) for line in captured_out.splitlines()]
    return [record for record in records if record["message"] == ACCESS_MESSAGE]


def test_record_is_json_with_service_context(settings):
    payload = json.loads(format_record(JsonFormatter(settings)))

    assert payload["message"] == "событие"
    assert payload["level"] == "INFO"
    assert payload["service"] == settings.app_name
    assert payload["version"] == __version__
    assert payload["environment"] == settings.environment
    assert payload["git_sha"] == settings.git_sha


def test_extra_fields_become_separate_keys(settings):
    line = format_record(
        JsonFormatter(settings), extra={"component": "postgres", "latency_ms": 12.5}
    )

    payload = json.loads(line)
    assert payload["component"] == "postgres"
    assert payload["latency_ms"] == 12.5


def test_request_id_is_taken_from_context(settings):
    formatter = JsonFormatter(settings)
    token = REQUEST_ID.set("req-1")
    try:
        inside = json.loads(format_record(formatter))
    finally:
        REQUEST_ID.reset(token)
    outside = json.loads(format_record(formatter))

    assert inside["request_id"] == "req-1"
    assert "request_id" not in outside


def test_traceback_goes_to_exception_field_and_line_stays_single(settings):
    try:
        raise ValueError("boom")
    except ValueError:
        line = format_record(JsonFormatter(settings), exc_info=sys.exc_info())

    assert "\n" not in line
    payload = json.loads(line)
    assert payload["exception"].startswith("Traceback")
    assert "ValueError: boom" in payload["exception"]


async def test_incoming_request_id_is_echoed_and_logged(make_client, capsys):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/version", headers={"X-Request-ID": "req-42"})

    assert response.headers["X-Request-ID"] == "req-42"
    [record] = access_log(capsys.readouterr().out)
    assert record["request_id"] == "req-42"
    assert record["method"] == "GET"
    assert record["path"] == "/api/v1/version"
    assert record["status_code"] == 200


async def test_request_id_is_generated_when_missing(make_client, capsys):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/version")

    request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", request_id)
    [record] = access_log(capsys.readouterr().out)
    assert record["request_id"] == request_id


async def test_unhandled_error_is_logged_with_request_id(settings, capsys):
    app = create_app(settings)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/boom", headers={"X-Request-ID": "req-500"})

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "req-500"
    out = capsys.readouterr().out
    [error] = [json.loads(line) for line in out.splitlines() if '"exception"' in line]
    assert error["request_id"] == "req-500"
    assert "RuntimeError: kaboom" in error["exception"]
    [record] = access_log(out)
    assert record["status_code"] == 500


async def test_liveness_is_not_logged_at_info(make_client, capsys):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    await client.get("/healthz")

    assert access_log(capsys.readouterr().out) == []
