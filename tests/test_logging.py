import json
import logging
import re
import sys
import warnings

import httpx
import pytest
from opentelemetry import trace

from dota_winprob import __version__
from dota_winprob.app import create_app
from dota_winprob.logging_config import (
    REQUEST_ID,
    JsonFormatter,
    ThrottleFilter,
    bind,
    log_context,
    mask_secrets,
    setup_logging,
)
from tests.fakes import MATCH_STATE, FakePublisher, HealthyPool, opendota_ok

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


def payload_of(formatter, **kwargs):
    return json.loads(format_record(formatter, **kwargs))


def test_contract_fields_are_present(settings):
    payload = payload_of(JsonFormatter(settings))

    required = {"timestamp", "level", "logger", "message", "service", "version", "environment"}
    assert required <= payload.keys()
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z", payload["timestamp"])


def test_service_follows_otel_service_name(settings):
    named = settings.model_copy(update={"otel_service_name": "dota-worker"})

    assert payload_of(JsonFormatter(named))["service"] == "dota-worker"


def test_resource_attributes_override_version_and_environment(settings, monkeypatch):
    monkeypatch.setenv(
        "OTEL_RESOURCE_ATTRIBUTES",
        "service.version=9.9.9,deployment.environment.name=stand,service.instance.id=w-2",
    )

    payload = payload_of(JsonFormatter(settings))

    assert payload["version"] == "9.9.9"
    assert payload["environment"] == "stand"


def test_task_context_is_added_automatically_and_reset(settings):
    formatter = JsonFormatter(settings)

    with log_context(task_id="batch:1", attempt=2):
        bind(task_id="batch:2")
        inside = payload_of(formatter)
    outside = payload_of(formatter)

    assert inside["task_id"] == "batch:2"
    assert inside["attempt"] == 2
    assert "task_id" not in outside
    assert "attempt" not in outside


def test_trace_and_span_ids_come_from_current_span(settings):
    formatter = JsonFormatter(settings)
    tracer = trace.get_tracer("test")

    with tracer.start_as_current_span("work") as span:
        inside = payload_of(formatter)
        context = span.get_span_context()
    outside = payload_of(formatter)

    assert inside["trace_id"] == format(context.trace_id, "032x")
    assert inside["span_id"] == format(context.span_id, "016x")
    assert re.fullmatch(r"[0-9a-f]{32}", inside["trace_id"])
    assert "trace_id" not in outside


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://postgres:s3cr3t-pass@db:5432/dota",
        "amqp://guest:s3cr3t-pass@rabbitmq:5672/",
        "http://mlflow:5000/?password=s3cr3t-pass",
    ],
)
def test_passwords_are_masked_everywhere(settings, dsn):
    formatter = JsonFormatter(settings)
    try:
        raise ConnectionError(f"не подключиться к {dsn}")
    except ConnectionError:
        line = format_record(
            formatter,
            message=f"адрес {dsn}",
            exc_info=sys.exc_info(),
            extra={"url": dsn},
        )

    assert "s3cr3t-pass" not in line
    assert "***" in json.loads(line)["url"]


def test_masking_keeps_urls_without_credentials():
    assert mask_secrets("http://otel-collector:4317/v1") == "http://otel-collector:4317/v1"
    assert mask_secrets("models:/dota-winprob@champion") == "models:/dota-winprob@champion"
    assert mask_secrets("amqp://guest:guest@rabbitmq:5672/") == "amqp://guest:***@rabbitmq:5672/"


def test_settings_repr_never_reaches_log_with_password(settings, capsys):
    setup_logging(settings)
    logging.getLogger("test").info("Настройки", extra={"settings": settings})

    out = capsys.readouterr().out
    assert "postgres:postgres@" not in out
    assert "guest:guest@" not in out


def test_library_loggers_use_the_same_json_format(settings, capsys):
    setup_logging(settings)

    for name in ("uvicorn.error", "faststream", "faststream.access.rabbit", "aio_pika", "mlflow"):
        logging.getLogger(name).warning("сообщение библиотеки")

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["logger"] for record in records] == [
        "uvicorn.error",
        "faststream",
        "faststream.access.rabbit",
        "aio_pika",
        "mlflow",
    ]
    assert {record["service"] for record in records} == {settings.app_name}


def test_level_comes_from_settings_even_for_verbose_libraries(settings, capsys):
    setup_logging(settings.model_copy(update={"log_level": "WARNING"}))
    access = logging.getLogger("faststream.access.rabbit")
    access.setLevel(logging.DEBUG)

    access.debug("Received")
    logging.getLogger("test").info("info")
    logging.getLogger("test").warning("warning")

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["message"] for record in records] == ["warning"]


def test_worker_name_is_a_static_field(settings, capsys):
    setup_logging(settings, worker="worker-2")
    logging.getLogger("test").warning("событие")

    assert json.loads(capsys.readouterr().out)["worker"] == "worker-2"


def test_exporter_errors_are_throttled():
    moments = iter([0.0, 1.0, 2.0, 61.0])
    throttle = ThrottleFilter("opentelemetry", 60.0, clock=lambda: next(moments))

    def record(name="opentelemetry.exporter.otlp", msg="Failed to export %s"):
        return logging.LogRecord(name, logging.ERROR, __file__, 1, msg, ("spans",), None)

    passed = [throttle.filter(record()) for _ in range(3)]
    late = record()
    assert passed == [True, False, False]
    assert throttle.filter(late)
    assert late.suppressed == 2
    assert throttle.filter(record(name="dota_winprob.app"))


async def test_end_to_end_health_is_not_logged_at_info(make_client, capsys):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert access_log(capsys.readouterr().out) == []


async def test_overload_is_warning_with_task_and_trace(make_client, capsys):
    client = await make_client(
        pool=HealthyPool(), opendota_handler=opendota_ok, publisher=FakePublisher(accept=False)
    )

    response = await client.post("/api/v1/tasks", json=MATCH_STATE)

    assert response.status_code == 503
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    [warning] = [record for record in records if record["message"].startswith("Задача отклонена")]
    assert warning["level"] == "WARNING"
    assert re.fullmatch(r"[0-9a-f]{32}", warning["trace_id"])
    assert warning["retry_after"] >= 1


def test_broker_reconnect_errors_are_throttled(settings, capsys):
    setup_logging(settings)
    for _ in range(5):
        logging.getLogger("aiormq.connection").error("error when creating transport: %r", "down")

    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["logger"] == "aiormq.connection"


def test_python_warnings_are_logged_as_json(settings, capsys):
    logging.captureWarnings(False)
    setup_logging(settings)
    warnings.showwarning(RuntimeWarning("coroutine was never awaited"), RuntimeWarning, __file__, 1)

    record = json.loads(capsys.readouterr().out)
    assert record["level"] == "WARNING"
    assert record["logger"] == "py.warnings"
    assert "never awaited" in record["message"]
