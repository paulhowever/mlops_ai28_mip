import json
from pathlib import Path

import pytest

from dota_winprob.agent import queries
from dota_winprob.agent.errors import ToolInputError
from dota_winprob.agent.promql import METRIC_NAMES, METRICS

HOSTILE = [
    'x" } or {service=~".+',
    'a"} | json | drop',
    "batch:1\n{service=~'.+'}",
    "batch:1` or 1=1 --",
    "batch:1 OR 1=1",
    "'; DROP TABLE tasks; --",
    "../../api/v1/admin",
    "batch:1\\",
    "",
    "а" * 10,
    "x" * 129,
]


@pytest.mark.parametrize("value", HOSTILE)
def test_hostile_task_id_is_rejected(value):
    with pytest.raises(ToolInputError):
        queries.task_id(value)


@pytest.mark.parametrize("value", ["batch:566", "0cc9cf11738446f2b122e5aa95ced066", "poison-1_a.b"])
def test_real_task_ids_are_accepted(value):
    assert queries.task_id(f" {value} ") == value


@pytest.mark.parametrize("value", ["", "0" * 32, "xyz", "a" * 33, 'ab" }', "../x", "ab/../cd"])
def test_hostile_trace_id_is_rejected(value):
    with pytest.raises(ToolInputError):
        queries.trace_id(value)


def test_trace_id_is_normalized():
    assert queries.trace_id("ABC") == "abc".zfill(32)


def unquote_go(literal):
    assert literal.startswith('"') and literal.endswith('"')
    return json.loads(literal)


@pytest.mark.parametrize(
    "text",
    [
        '" | json | line_format "{{.}}',
        "\\",
        '\\" or {service="postgres"}',
        "`backtick`",
        ".*",
        "a|b",
        "(unclosed",
        "line\nbreak\r\ttab\x00",
        "Попытки исчерпаны",
    ],
)
def test_text_stays_inside_one_string_literal(text):
    query = queries.logql(service="dota-worker", text=text)

    head = '{service="dota-worker"} |~ '
    assert query.startswith(head)
    literal = unquote_go(query.removeprefix(head))
    assert literal.startswith("(?i)")
    unescaped = literal.removeprefix("(?i)")
    for special in ".*+?()|[]{}^$":
        stripped = unescaped.replace("\\\\", "").replace(f"\\{special}", "")
        assert special not in stripped
    assert "\n" not in query


def test_text_is_bounded():
    for text in ("", "   ", "x" * 201):
        with pytest.raises(ToolInputError):
            queries.logql(text=text)


def test_logql_matches_stack_labels():
    query = queries.logql(service="dota-worker", level="ERROR", task="batch:566")

    assert query == '{service="dota-worker", level=~"ERROR|CRITICAL"} | task_id="batch:566"'


def test_logql_without_filters_covers_all_services():
    assert queries.logql() == '{service=~".+"}'


@pytest.mark.parametrize("field", [{"service": 'a"} or {'}, {"level": "ERROR|.*"}, {"task": 'x"'}])
def test_logql_refuses_hostile_labels(field):
    with pytest.raises(ToolInputError):
        queries.logql(**field)


def test_traceql_uses_span_attribute():
    assert queries.traceql_by_task("batch:566") == '{ span.task.id = "batch:566" }'


def test_every_metric_name_has_a_query():
    assert set(METRIC_NAMES) == set(METRICS)
    assert all("[W]" not in metric.query(60) for metric in METRICS.values())


def test_metric_queries_match_dashboard():
    dashboard = json.loads(
        Path("observability/grafana/dashboards/pipeline.json").read_text(encoding="utf-8")
    )
    expressions = {
        target["expr"].replace("[$__rate_interval]", "[W]").replace('{instance=~"$worker"}', "")
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    }
    own = {
        "consumers",
        "worker_rate",
        "worker_errors",
        "worker_duplicates",
        "in_flight",
        "api_rate",
    }
    for name, metric in METRICS.items():
        if name not in own:
            assert metric.expression in expressions, name
