import asyncio
import json
from datetime import UTC, datetime

import asyncpg
import httpx
import pytest

from dota_winprob.agent.tools.catalog import DESCRIPTIONS
from tests.agent.fakes import done_view, log_line, stream, task_row

STATE = {
    "radiant_gold": 13500,
    "dire_gold": 12400,
    "radiant_xp": 14200,
    "dire_xp": 13600,
    "radiant_lh": 125,
    "dire_lh": 116,
}

ARGUMENTS = {
    "pipeline_status": ({}, "prometheus"),
    "metric_history": ({"name": "queue_ready"}, "prometheus"),
    "search_logs": ({}, "loki"),
    "get_trace": ({"trace_id": "ab" * 16}, "tempo"),
    "model_card": ({}, "mlflow"),
    "predict": (STATE, "api"),
}


async def test_pipeline_status_reports_queue_and_marks_missing_values(run_tool, backends):
    backends.metrics = {
        'messages_ready{queue="tasks.score"}': "97",
        "pipeline_queue_max_length": "100",
        "queue_consumers": "1",
        'messages_ready{queue="tasks.dlq"}': "0",
        "inference_duration": "NaN",
    }

    data, step, _ = await run_tool("pipeline_status")

    values = {metric["name"]: metric["value"] for metric in data["metrics"]}
    assert values["queue_ready"] == 97
    assert values["consumers"] == 1
    assert values["inference_p95"] is None
    assert values["worker_rate"] is None
    assert data["queue_fill_share"] == 0.97
    assert step.status == "ok"
    assert "97" in step.summary


async def test_pipeline_status_without_series_is_empty(run_tool):
    _, step, _ = await run_tool("pipeline_status")

    assert step.status == "empty"


async def test_metric_history_compresses_series(run_tool, backends):
    backends.ranges = {"tasks.score": [[index, str(index % 100)] for index in range(180)]}

    data, step, _ = await run_tool("metric_history", name="queue_ready", minutes=15)

    assert (data["min"], data["max"], data["last"]) == (0, 99, 79)
    assert data["measured_points"] == 180
    assert len(data["points_unix_time_value"]) <= 20
    assert data["points_unix_time_value"][-1] == [179, 79]
    assert step.status == "ok"
    form = backends.form("prometheus")
    assert form["step"] == "5"
    assert float(form["end"]) - float(form["start"]) == pytest.approx(900, abs=0.01)


async def test_metric_history_window_is_not_shorter_than_step(run_tool, backends):
    await run_tool("metric_history", name="worker_rate", minutes=1440)

    form = backends.form("prometheus")
    assert form["step"] == "432"
    assert "[432s]" in form["query"]


async def test_metric_history_without_points_is_empty(run_tool):
    _, step, _ = await run_tool("metric_history", name="dlq_ready", minutes=5)

    assert step.status == "empty"


async def test_unknown_metric_never_reaches_prometheus(
    agent_settings, sources, actions, deps, backends
):
    from pydantic_ai.models.function import FunctionModel

    from dota_winprob.agent.service import build_agent
    from dota_winprob.agent.tools.catalog import build_toolset
    from tests.agent.fakes import call, say

    turns = [call("metric_history", name="up} or vector(1)"), say("готово")]
    agent = build_agent(
        agent_settings,
        build_toolset(agent_settings, sources, actions),
        FunctionModel(lambda m, i: turns.pop(0)),
    )

    await agent.run("вопрос", deps=deps)

    assert backends.calls("prometheus") == []
    assert deps.steps == []


async def test_search_logs_returns_newest_first_without_unknown_fields(run_tool, backends):
    backends.log_streams = [
        stream(
            log_line("старое", features={"radiant_gold": 1}, git_sha="abc"),
            log_line("новое", task_id="batch:566", exception="x" * 5000),
        )
    ]

    data, step, _ = await run_tool(
        "search_logs", service="dota-worker", level="ERROR", task_id="batch:566", limit=5
    )

    assert [entry["message"] for entry in data["entries"]] == ["новое", "старое"]
    assert "features" not in data["entries"][1]
    assert len(data["entries"][0]["exception"]) == 400
    assert step.status == "ok"
    form = backends.form("loki")
    assert form["query"] == (
        '{service="dota-worker", level=~"ERROR|CRITICAL"} | task_id="batch:566"'
    )
    assert form["limit"] == "5"
    assert backends.calls("loki")[0].url.query == b""


async def test_search_logs_keeps_plain_lines(run_tool, backends):
    backends.log_streams = [stream("plain text line", service="rabbitmq", level="unknown")]

    data, _, _ = await run_tool("search_logs", service="rabbitmq")

    assert data["entries"][0]["line"] == "plain text line"
    assert data["entries"][0]["service"] == "rabbitmq"


async def test_search_logs_without_entries_is_empty(run_tool):
    _, step, _ = await run_tool("search_logs", minutes=5)

    assert step.status == "empty"


async def test_injection_in_log_filter_never_reaches_loki(run_tool, backends):
    data, step, _ = await run_tool("search_logs", task_id='x"} or {service="postgres')

    assert step.status == "invalid_input"
    assert "task_id" in data["error"]
    assert backends.calls("loki") == []


async def test_get_task_reads_row_in_readonly_transaction(run_tool, pool):
    pool.rows = [task_row()]

    data, step, _ = await run_tool("get_task", task_id="batch:566")

    assert data["status"] == "done"
    assert step.status == "ok"
    assert pool.transactions == [{"readonly": True}]
    [(query, args)] = pool.queries
    assert args == ("batch:566",)
    assert "batch:566" not in query
    assert "$1" in query


async def test_get_task_shows_times_in_iso(run_tool, pool):
    pool.rows = [task_row(completed_at=datetime(2026, 11, 24, 12, 0, tzinfo=UTC))]

    data, _, _ = await run_tool("get_task", task_id="batch:566")

    assert data["completed_at"] == "2026-11-24T12:00:00+00:00"


async def test_get_task_missing_is_empty(run_tool):
    data, step, _ = await run_tool("get_task", task_id="batch:1")

    assert step.status == "empty"
    assert data["found"] is False


async def test_sql_injection_is_rejected_before_database(run_tool, pool):
    _, step, _ = await run_tool("get_task", task_id="x'; DROP TABLE tasks; --")

    assert step.status == "invalid_input"
    assert pool.queries == []


@pytest.mark.parametrize(
    "error",
    [
        OSError("connection refused"),
        TimeoutError(),
        asyncpg.PostgresError("boom"),
        asyncpg.InterfaceError("pool is closing"),
    ],
)
@pytest.mark.parametrize("tool", ["get_task", "failed_tasks"])
async def test_database_down_is_a_result(run_tool, pool, error, tool):
    pool.error = error

    data, step, _ = await run_tool(tool, **({"task_id": "batch:1"} if tool == "get_task" else {}))

    assert step.status == "unavailable"
    assert "postgres" in data["error"]


async def test_failed_tasks_lists_errors_and_totals(run_tool, pool):
    pool.rows = [task_row(status="failed", error="OSError: disk", win_probability=None)]
    pool.totals = [{"status": "failed", "total": 3}, {"status": "done", "total": 10}]

    data, step, _ = await run_tool("failed_tasks", limit=5)

    assert data["totals_by_status"] == {"failed": 3, "done": 10}
    assert data["tasks"][0]["error"] == "OSError: disk"
    assert step.summary == "failed всего 3"
    assert pool.queries[0][1] == (5,)


async def test_failed_tasks_empty(run_tool):
    _, step, _ = await run_tool("failed_tasks")

    assert step.status == "empty"


def attribute(key, **value):
    return {"key": key, "value": value}


def trace_payload(root="batches"):
    def spans(service, *items):
        return {
            "resource": {"attributes": [attribute("service.name", stringValue=service)]},
            "scopeSpans": [{"spans": list(items)}],
        }

    def span(span_id, parent, name, start_ms, end_ms, **extra):
        return {
            "spanId": span_id,
            "parentSpanId": parent,
            "name": name,
            "startTimeUnixNano": str(start_ms * 1_000_000),
            "endTimeUnixNano": str(end_ms * 1_000_000),
            **extra,
        }

    return {
        root: [
            spans(
                "dota-worker",
                span(
                    "p2",
                    "p1",
                    "tasks.score process",
                    1500,
                    1800,
                    attributes=[
                        attribute("task.queue_wait_s", doubleValue=1.49),
                        attribute("attempt", intValue="1"),
                        attribute("messaging.message.body", stringValue="секрет"),
                    ],
                    status={"code": "STATUS_CODE_ERROR", "message": "boom"},
                    events=[{"name": "dlq"}],
                ),
                span("p3", "p2", "inference", 1501, 1714),
            ),
            spans("dota-source", span("p1", "", "source publish", 1000, 1010)),
        ]
    }


async def test_get_trace_by_task_builds_tree(run_tool, backends):
    trace_id = "ab" * 16
    backends.found_traces = [
        {"traceID": "cd" * 16, "startTimeUnixNano": "1"},
        {"traceID": trace_id, "startTimeUnixNano": "2"},
    ]
    backends.traces = {trace_id: trace_payload()}

    data, step, _ = await run_tool("get_trace", task_id="batch:566")

    assert [(span["depth"], span["name"]) for span in data["spans"]] == [
        (0, "source publish"),
        (1, "tasks.score process"),
        (2, "inference"),
    ]
    assert data["queue_wait_s"] == [1.49]
    assert data["inference_ms"] == [213.0]
    assert data["total_ms"] == 800.0
    assert data["spans"][1]["error"] == "boom"
    assert data["spans"][1]["events"] == {"dlq": 1}
    assert data["spans"][1]["attributes"] == {"attempt": 1, "task.queue_wait_s": 1.49}
    assert data["other_trace_ids"] == ["cd" * 16]
    assert step.status == "ok"
    [search] = backends.calls("tempo", "/api/search")
    assert search.url.params["q"] == '{ span.task.id = "batch:566" }'


@pytest.mark.parametrize("wrap", [lambda body: body, lambda body: {"trace": body}])
async def test_get_trace_reads_both_tempo_formats(run_tool, backends, wrap):
    backends.traces = {"ab" * 16: wrap(trace_payload("resourceSpans"))}

    data, _, _ = await run_tool("get_trace", trace_id="AB" * 16)

    assert data["span_count"] == 3
    assert data["services"] == ["dota-source", "dota-worker"]


async def test_get_trace_survives_cyclic_parents(run_tool, backends):
    payload = trace_payload()
    payload["batches"][1]["scopeSpans"][0]["spans"][0]["parentSpanId"] = "p3"
    backends.traces = {"ab" * 16: payload}

    data, step, _ = await run_tool("get_trace", trace_id="ab" * 16)

    assert step.status in {"ok", "empty"}
    assert data["span_count"] == 3


@pytest.mark.parametrize("arguments", [{"trace_id": "ab" * 16}, {"task_id": "batch:1"}])
async def test_get_trace_not_found_is_empty(run_tool, arguments):
    _, step, _ = await run_tool("get_trace", **arguments)

    assert step.status == "empty"


@pytest.mark.parametrize("arguments", [{}, {"trace_id": "../search"}, {"task_id": 'x" } || {'}])
async def test_get_trace_rejects_bad_arguments(run_tool, backends, arguments):
    _, step, _ = await run_tool("get_trace", **arguments)

    assert step.status == "invalid_input"
    assert backends.calls("tempo") == []


async def test_model_card_reads_champion(run_tool, backends):
    data, step, _ = await run_tool("model_card")

    assert data["version"] == "4"
    assert data["metrics"] == {"roc_auc": 0.7123, "cv_roc_auc": 0.7101}
    assert step.status == "ok"
    assert backends.calls("mlflow", "alias")[0].url.params["alias"] == "champion"


async def test_model_card_without_champion_is_empty(run_tool, backends):
    backends.champion = None

    _, step, _ = await run_tool("model_card")

    assert step.status == "empty"


async def test_match_features_come_from_file(run_tool):
    data, step, _ = await run_tool("match_features", match_id=566)

    assert data["task_id"] == "batch:566"
    assert data["features"]["gold_advantage"] == -1581
    assert step.status == "ok"


async def test_match_features_unknown_match_is_empty(run_tool):
    _, step, _ = await run_tool("match_features", match_id=5)

    assert step.status == "empty"


async def test_match_features_without_file_is_unavailable(run_tool, agent_settings):
    agent_settings.agent_tasks_file.unlink()

    _, step, _ = await run_tool("match_features", match_id=566)

    assert step.status == "unavailable"


async def test_predict_computes_advantages_and_waits_for_result(run_tool, backends, deps):
    backends.views = [{"task_id": "abc123", "status": "pending"}, done_view()]

    data, step, _ = await run_tool("predict", **STATE)

    assert data["win_probability"] == 0.6234
    assert data["model_version"] == "4"
    assert data["task_id"] == "abc123"
    assert step.status == "ok"
    [post] = [request for request in backends.calls("api") if request.method == "POST"]
    body = json.loads(post.content)
    assert (body["gold_advantage"], body["xp_advantage"], body["lh_advantage"]) == (1100, 600, 9)
    assert [prediction.win_probability for prediction in deps.evidence.predictions] == [0.6234]


async def test_predict_passes_503_and_retry_after(run_tool, backends, deps):
    backends.submit = [
        httpx.Response(
            503,
            json={"detail": "Задача не принята: очередь задач заполнена"},
            headers={"Retry-After": "7"},
        )
    ]

    data, step, _ = await run_tool("predict", **STATE)

    assert step.status == "rejected"
    assert data["retry_after_s"] == 7
    assert "очередь задач заполнена" in data["reason"]
    assert len(backends.calls("api")) == 1
    assert deps.evidence.predictions == []


async def test_predict_timeout_returns_task_id(run_tool, backends):
    backends.views = [{"task_id": "abc123", "status": "pending"}]

    data, step, _ = await run_tool("predict", **STATE)

    assert step.status == "pending"
    assert data["task_id"] == "abc123"
    assert "win_probability" not in data


async def test_predict_waits_through_missing_row(run_tool):
    _, step, _ = await run_tool("predict", **STATE)

    assert step.status == "pending"


async def test_predict_reports_failed_task(run_tool, backends):
    backends.views = [{"task_id": "abc123", "status": "failed", "error": "ValueError: schema"}]

    data, step, _ = await run_tool("predict", **STATE)

    assert step.status == "failed"
    assert data["error"] == "ValueError: schema"


@pytest.mark.parametrize("tool", sorted(ARGUMENTS))
@pytest.mark.parametrize("failure", ["down", "500"])
async def test_source_failure_is_a_result(run_tool, backends, tool, failure):
    arguments, host = ARGUMENTS[tool]
    if failure == "down":
        backends.down.add(host)
    else:
        backends.status[host] = 500

    data, step, text = await run_tool(tool, **arguments)

    assert step.status == "unavailable"
    assert host in data["error"]
    assert "hunter22" not in text


async def test_malformed_answer_is_a_result(run_tool, backends):
    backends.champion = {"unexpected": True}

    _, step, _ = await run_tool("model_card")

    assert step.status == "malformed"


async def test_slow_tool_is_cut(run_tool, sources, agent_settings, monkeypatch):
    async def hang(query, minutes, limit):
        await asyncio.sleep(30)

    monkeypatch.setattr(sources.loki, "search", hang)

    _, step, _ = await run_tool("search_logs")

    assert step.status == "timeout"
    assert step.duration_ms < 5000


async def test_every_tool_is_described(agent_settings, sources, actions):
    from dota_winprob.agent.tools.catalog import build_toolset

    assert set(build_toolset(agent_settings, sources, actions).tools) == set(DESCRIPTIONS)
