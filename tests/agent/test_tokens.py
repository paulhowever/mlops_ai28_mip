import asyncio

import pytest

from dota_winprob.agent.errors import LLMBusy
from dota_winprob.agent.llm.client import model_settings
from dota_winprob.agent.llm.gate import background
from tests.agent.test_llm import (
    MESSAGES,
    Clock,
    Gateway,
    completion,
    failure,
    make_gate,
    make_model,
    send,
)


async def spend(gate, tokens, actual=None):
    async with gate.slot(tokens) as ticket:
        ticket.settle(tokens if actual is None else actual)


async def test_token_window_delays_the_request():
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=1000)
    await spend(gate, 600)
    clock.now += 10

    await spend(gate, 600)

    assert clock.slept == [50.0]


async def test_usage_corrects_the_estimate():
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=1000)
    await spend(gate, 900, actual=100)
    await spend(gate, 800)
    assert clock.slept == []

    await spend(gate, 200)

    assert clock.slept == [60.0]


async def test_underestimated_usage_is_counted():
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=1000)
    await spend(gate, 10, actual=990)

    await spend(gate, 50)

    assert clock.slept == [60.0]


async def test_never_more_than_limit_in_any_minute():
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=1000, queue_timeout=1e6)
    spent = []

    for index in range(40):
        tokens = 100 + index * 17 % 400
        async with gate.slot(tokens):
            spent.append((clock.now, tokens))
        clock.now += 0.5

    for start, _ in spent:
        assert sum(t for moment, t in spent if start <= moment < start + 60) <= 1000


async def test_full_window_survives_float_rounding_of_the_clock():
    for index in range(500):
        clock = Clock()
        clock.now = 32768 - 60 + index * 0.1234567
        gate = make_gate(clock, requests_per_minute=1, queue_timeout=1e6)
        await spend(gate, 1)
        clock.now += 1e-3 * index

        await spend(gate, 1)

        assert len(clock.slept) == 1


async def test_request_above_the_whole_limit_waits_for_empty_window():
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=1000)
    await spend(gate, 10)

    await spend(gate, 5000)

    assert clock.slept == [60.0]


async def test_token_overload_is_busy_with_retry_after():
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=1000, queue_timeout=10.0)
    await spend(gate, 900)

    with pytest.raises(LLMBusy) as busy:
        await spend(gate, 500)

    assert busy.value.retry_after == 60


async def test_cancelled_wait_returns_its_reservation():
    gate = make_gate(tokens_per_minute=1000)
    await spend(gate, 900)
    waiter = asyncio.create_task(spend(gate, 500))
    await asyncio.sleep(0.01)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)

    async with asyncio.timeout(1):
        await spend(gate, 100)


async def test_model_settles_by_usage(agent_settings):
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=3000)
    model = make_model(agent_settings, Gateway(completion()), gate)

    for _ in range(19):
        await send(model)

    assert clock.slept == []
    [first, *_] = gate._tickets
    assert first.tokens == 18


async def test_failed_request_keeps_the_estimate(agent_settings):
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=100_000)
    model = make_model(agent_settings, Gateway(failure(400)), gate)

    with pytest.raises(Exception, match="отклонил"):
        await model.request(MESSAGES, model_settings(agent_settings), _parameters())

    assert gate._tickets[0].tokens > agent_settings.llm_max_output_tokens


def _parameters():
    from pydantic_ai.models import ModelRequestParameters

    return ModelRequestParameters()


async def in_background(gate, tokens=0):
    with background():
        await spend(gate, tokens)


async def test_background_leaves_requests_for_people():
    clock = Clock()
    gate = make_gate(clock, background_share=0.25, background_timeout=30.0)
    for _ in range(5):
        await in_background(gate)

    with pytest.raises(LLMBusy):
        await in_background(gate)
    slept = len(clock.slept)
    for _ in range(15):
        await spend(gate, 0)

    assert len(clock.slept) == slept


async def test_background_leaves_tokens_for_people():
    clock = Clock()
    gate = make_gate(clock, tokens_per_minute=1000, background_timeout=30.0)
    await in_background(gate, 200)

    with pytest.raises(LLMBusy):
        await in_background(gate, 100)
    await spend(gate, 800)

    assert clock.slept == []


async def test_background_never_takes_the_last_slot():
    gate = make_gate(max_concurrency=2)
    release = asyncio.Event()

    async def hold():
        with background():
            async with gate.slot():
                await release.wait()

    first, second = asyncio.create_task(hold()), asyncio.create_task(hold())
    await asyncio.sleep(0.02)
    async with asyncio.timeout(1):
        await spend(gate, 0)
    release.set()
    await asyncio.gather(first, second)


async def test_background_waits_while_people_are_waiting():
    gate = make_gate(max_concurrency=1)
    order = []

    async def person(name):
        async with gate.slot():
            order.append(name)
            await asyncio.sleep(0.03)

    async def incident():
        with background():
            async with gate.slot():
                order.append("incident")

    first = asyncio.create_task(person("first"))
    await asyncio.sleep(0)
    waiting = asyncio.create_task(incident())
    await asyncio.sleep(0.01)
    second = asyncio.create_task(person("second"))
    await asyncio.gather(first, waiting, second)

    assert order == ["first", "second", "incident"]


async def test_background_gives_up_in_time():
    clock = Clock()
    gate = make_gate(clock, max_concurrency=1, background_timeout=5.0)
    async with gate.slot():
        with pytest.raises(LLMBusy):
            await in_background(gate)
