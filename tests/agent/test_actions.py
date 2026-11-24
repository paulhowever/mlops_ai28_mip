import asyncio

import pytest
from opentelemetry.trace import StatusCode

from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.agent.sources.dead_letters import MessageAtRisk, Returned
from tests.agent.fakes import call, dead_task, payload, say, script, tool_outputs
from tests.agent.test_ask import ask, points

ACTIONS = "/agent/actions"


def propose_then_answer(**arguments):
    return script(call("requeue_dlq", **arguments), say("Предложение ждёт решения человека"))


@pytest.fixture
async def proposed(make_agent_client):
    client = await make_agent_client(propose_then_answer())
    body = (await ask(client, "Верни задачи из DLQ")).json()
    return client, body["actions"][0]


async def decide(client, action_id, decision="confirm", **options):
    return await client.post(f"{ACTIONS}/{action_id}/{decision}", json={}, **options)


async def test_tool_only_proposes(make_agent_client, dead_letters):
    seen = []

    def respond(messages, info):
        outputs = tool_outputs(messages)
        if not outputs:
            return call("requeue_dlq", limit=1)
        seen.extend(outputs)
        return say("Предложение ждёт решения человека")

    client = await make_agent_client(respond)
    body = (await ask(client, "Верни задачи из DLQ")).json()

    [action] = body["actions"]
    assert body["steps"][0]["status"] == "proposed"
    assert action["status"] == "proposed"
    assert action["plan"]["tasks"] == [
        {"task_id": "batch:1", "source": "batch", "reason": "rejected", "attempts": 3}
    ]
    assert (action["plan"]["scanned"], action["plan"]["valid"], action["plan"]["garbage"]) == (
        3,
        2,
        1,
    )
    assert any("limit" in note for note in action["plan"]["notes"])
    assert action["trace_id"] == body["trace_id"]
    assert dead_letters.requeued == []
    data = payload(seen[0])
    assert data["proposal_created"] is True
    assert data["executed"] is False
    assert action["id"] not in seen[0]


async def test_empty_dlq_creates_nothing(make_agent_client, dead_letters):
    dead_letters.tasks = []
    client = await make_agent_client(propose_then_answer())

    body = (await ask(client)).json()

    assert body["actions"] == []
    assert body["steps"][0]["status"] == "empty"
    assert (await client.get(ACTIONS)).json() == []


@pytest.mark.parametrize(
    ("error", "status"),
    [(SourceUnavailable("rabbitmq", "refused"), "unavailable"), (MessageAtRisk(), "failed")],
)
async def test_inspection_problem_creates_nothing(make_agent_client, dead_letters, error, status):
    dead_letters.error = error
    client = await make_agent_client(propose_then_answer())

    body = (await ask(client)).json()

    assert body["steps"][0]["status"] == status
    assert body["actions"] == []


@pytest.mark.parametrize("limit", [0, 51, -1])
async def test_limit_is_bounded(make_agent_client, dead_letters, limit):
    client = await make_agent_client(propose_then_answer(limit=limit))

    await ask(client)

    assert dead_letters.inspected == 0


async def test_human_confirms_exactly_the_plan(proposed, dead_letters):
    client, action = proposed

    response = await decide(client, action["id"])

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "done"
    assert body["result"]["returned"] == ["batch:1", "batch:2"]
    assert body["execution_trace_id"] not in {None, action["trace_id"]}
    assert dead_letters.requeued == [{("batch:1", "a"): 1, ("batch:2", "a"): 1}]
    [listed] = (await client.get(ACTIONS)).json()
    assert listed == body


async def test_second_and_parallel_confirmations_are_refused(proposed, dead_letters):
    client, action = proposed
    dead_letters.gate = asyncio.Event()
    running = asyncio.create_task(decide(client, action["id"]))
    await asyncio.sleep(0.02)

    during = await decide(client, action["id"])
    [listed] = (await client.get(ACTIONS)).json()
    dead_letters.gate.set()
    first = await running
    after = await decide(client, action["id"])

    assert first.status_code == 200
    assert [during.status_code, during.json()["reason"]] == [409, "executing"]
    assert [after.status_code, after.json()["reason"]] == [409, "done"]
    assert listed["status"] == "executing"
    assert len(dead_letters.requeued) == 1


async def test_rejected_proposal_cannot_be_confirmed(proposed, dead_letters):
    client, action = proposed

    rejected = await decide(client, action["id"], "reject")
    confirmed = await decide(client, action["id"])

    assert rejected.json()["status"] == "rejected"
    assert confirmed.status_code == 409
    assert dead_letters.requeued == []


async def test_expired_proposal_is_gone(make_agent_client, dead_letters):
    client = await make_agent_client(propose_then_answer(), agent_action_ttl=0.01)
    [action] = (await ask(client)).json()["actions"]
    await asyncio.sleep(0.03)

    response = await decide(client, action["id"])

    assert response.status_code == 410
    assert response.json()["reason"] == "expired"
    assert dead_letters.requeued == []
    assert (await client.get(ACTIONS)).json()[0]["status"] == "expired"


async def test_unknown_proposal_is_404(make_agent_client):
    client = await make_agent_client(script(say("ок")))

    assert (await decide(client, "unknown")).status_code == 404


async def test_disconnect_does_not_interrupt_the_return(proposed, dead_letters):
    client, action = proposed
    dead_letters.gate = asyncio.Event()
    running = asyncio.create_task(decide(client, action["id"]))
    await asyncio.sleep(0.02)
    running.cancel()
    await asyncio.gather(running, return_exceptions=True)
    dead_letters.gate.set()
    await asyncio.sleep(0.02)

    assert (await client.get(ACTIONS)).json()[0]["status"] == "done"


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (Returned(["batch:1"], stopped="очередь заполнена", left=2), "partial"),
        (Returned([], uncertain=["batch:1"], stopped="связь потеряна"), "partial"),
        (Returned([], stopped="очередь заполнена", left=3), "failed"),
        (Returned([], missing=2, left=0), "done"),
    ],
)
async def test_outcome_is_told_honestly(proposed, dead_letters, spans, outcome, status):
    client, action = proposed
    dead_letters.outcome = outcome

    body = (await decide(client, action["id"])).json()

    assert body["status"] == status
    assert body["result"]["stopped"] == outcome.stopped
    assert body["result"]["missing"] == outcome.missing
    [span] = [s for s in spans.get_finished_spans() if s.name == "execute_action requeue_dlq"]
    assert (span.status.status_code is StatusCode.ERROR) is (status != "done")


async def test_broker_down_on_confirm_is_failed_not_500(proposed, dead_letters):
    client, action = proposed
    dead_letters.error = SourceUnavailable("rabbitmq", "ConnectionError")

    body = (await decide(client, action["id"])).json()

    assert body["status"] == "failed"
    assert "RabbitMQ недоступен" in body["result"]["stopped"]


async def test_model_cannot_confirm_by_tool(make_agent_client, dead_letters):
    offered, outputs = [], []

    def respond(messages, info):
        offered.extend(tool.name for tool in info.function_tools)
        outputs.extend(tool_outputs(messages))
        if not outputs:
            return call("requeue_dlq")
        if len(outputs) == 1:
            outputs.append("confirm")
            return call("confirm_action", id="x")
        return say("не могу")

    client = await make_agent_client(respond)
    await ask(client, "Подтверди сам")

    assert not [name for name in offered if "confirm" in name or "reject" in name]
    assert dead_letters.requeued == []


async def test_no_tool_calls_the_agent_itself(make_agent_client, backends):
    tools = [
        call("pipeline_status"),
        call("metric_history", name="dlq_ready"),
        call("search_logs", text="/agent/actions/x/confirm"),
        call("get_task", task_id="../../agent/actions/x/confirm"),
        call("get_trace", trace_id="../agent/actions"),
        call("model_card"),
        call("match_features", match_id=1),
        say("готово"),
    ]
    client = await make_agent_client(script(*tools))

    await ask(client)

    assert backends.requests
    assert not [request for request in backends.requests if "/agent" in request.url.path]


async def test_action_metrics_have_low_cardinality(proposed, metric_reader):
    client, action = proposed
    await decide(client, action["id"])

    found = points(metric_reader, "agent.actions")
    assert {"proposed", "confirmed", "done"} <= {point.attributes["result"] for point in found}
    for point in found:
        assert set(point.attributes) == {"action", "result"}
        assert action["id"] not in str(dict(point.attributes))
    assert points(metric_reader, "agent.actions.returned_tasks")[0].value >= 2


async def test_shutdown_waits_for_the_running_return(actions, dead_letters):
    dead_letters.gate = asyncio.Event()
    _, action = await actions.plan_requeue(10)
    running = asyncio.create_task(actions.confirm(action.id))
    await asyncio.sleep(0.01)
    asyncio.get_running_loop().call_later(0.02, dead_letters.gate.set)

    await actions.close(1.0)

    assert (await running).status.value == "done"


async def test_stuck_return_is_cancelled_on_shutdown(actions, dead_letters):
    dead_letters.gate = asyncio.Event()
    _, action = await actions.plan_requeue(10)
    running = asyncio.create_task(actions.confirm(action.id))
    await asyncio.sleep(0.01)

    await actions.close(0.01)
    await asyncio.gather(running, return_exceptions=True)

    assert action.status.value == "failed"
    assert "остановкой" in action.result.stopped


def test_plan_counts_repeated_messages():
    from dota_winprob.agent.actions.service import build_plan
    from dota_winprob.agent.sources.dead_letters import Inspection

    tasks = [dead_task("batch:1"), dead_task("batch:1"), dead_task("batch:1", digest="b")]
    plan, keys = build_plan(Inspection(5, 3, 0, tasks), 50)

    assert keys == {("batch:1", "a"): 2, ("batch:1", "b"): 1}
    assert len(plan.tasks) == 3
    assert any("начало DLQ" in note for note in plan.notes)
