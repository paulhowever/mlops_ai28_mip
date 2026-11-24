import asyncio
import json
import logging

import httpx2
import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from dota_winprob.agent.config import AgentSettings
from dota_winprob.agent.errors import LLMBusy, LLMFailed, LLMUnavailable
from dota_winprob.agent.llm.client import build_gate, build_llm
from dota_winprob.agent.llm.gate import LLMGate
from tests.agent.fakes import TEST_KEY

MESSAGES = [ModelRequest(parts=[UserPromptPart("вопрос")])]


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.slept.append(delay)
        self.now += delay


def make_gate(clock=None, **overrides):
    options = {
        "max_concurrency": 2,
        "requests_per_minute": 20,
        "max_waiting": 8,
        "queue_timeout": 60.0,
    } | overrides
    if clock is not None:
        options |= {"clock": clock, "sleep": clock.sleep}
    return LLMGate(**options)


def completion(text="готово"):
    return httpx2.Response(
        200,
        json={
            "id": "1",
            "object": "chat.completion",
            "created": 1,
            "model": "bonsai2-27b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    )


def failure(status, **headers):
    return httpx2.Response(
        status, json={"error": {"message": f"отказ, ключ {TEST_KEY}"}}, headers=headers
    )


class Gateway:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.active = 0
        self.peak = 0
        self.delay = 0.0

    async def __call__(self, request):
        self.requests.append(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
            if isinstance(response, Exception):
                raise response
            return response
        finally:
            self.active -= 1


def make_model(agent_settings, gateway, gate=None, clock=None, **overrides):
    settings = agent_settings.model_copy(update=overrides)
    llm = build_llm(settings, gate or build_gate(settings), transport=httpx2.MockTransport(gateway))
    if clock is not None:
        llm.model._sleep = clock.sleep
    return llm.model


async def send(model):
    return await model.request(MESSAGES, None, ModelRequestParameters())


async def test_at_most_two_requests_at_once(agent_settings):
    gateway = Gateway(completion())
    gateway.delay = 0.02
    model = make_model(agent_settings, gateway)

    await asyncio.gather(*(send(model) for _ in range(7)))

    assert gateway.peak == 2
    assert len(gateway.requests) == 7


async def test_limits_are_shared_between_models(agent_settings):
    gateway = Gateway(completion())
    gateway.delay = 0.02
    gate = build_gate(agent_settings)
    models = [make_model(agent_settings, gateway, gate) for _ in range(3)]

    await asyncio.gather(*(send(model) for model in models for _ in range(2)))

    assert gateway.peak == 2


async def test_no_more_than_limit_in_any_minute():
    clock = Clock()
    gate = make_gate(clock, queue_timeout=1000.0)
    starts = []

    for _ in range(45):
        async with gate.slot():
            starts.append(clock.now)
        clock.now += 0.1

    for start in starts:
        assert sum(start <= other < start + 60 for other in starts) <= 20
    assert starts[20] - starts[0] >= 60


async def test_rate_limit_beyond_queue_timeout_is_busy_with_retry_after():
    clock = Clock()
    gate = make_gate(clock, requests_per_minute=1, queue_timeout=10.0)
    async with gate.slot():
        pass

    with pytest.raises(LLMBusy) as busy:
        async with gate.slot():
            pass

    assert busy.value.status_code == 503
    assert busy.value.retry_after == 60
    clock.now += 60
    async with gate.slot():
        pass


async def test_full_waiting_room_is_busy_immediately():
    gate = make_gate(max_concurrency=1, max_waiting=0)
    async with gate.slot():
        with pytest.raises(LLMBusy) as busy:
            async with gate.slot():
                pass

    assert busy.value.retry_after >= 1


async def test_waiting_too_long_for_slot_is_busy():
    gate = make_gate(max_concurrency=1, queue_timeout=0.02)
    async with gate.slot():
        with pytest.raises(LLMBusy):
            async with gate.slot():
                pass
    async with gate.slot():
        pass


async def test_cancelled_waiter_does_not_leak_slot():
    gate = make_gate(max_concurrency=1)
    async with gate.slot():
        waiter = asyncio.create_task(gate.slot().__aenter__())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    async with asyncio.timeout(1), gate.slot():
        pass


async def test_429_is_retried_after_requested_pause(agent_settings):
    clock = Clock()
    gateway = Gateway(failure(429, **{"Retry-After": "3"}), completion())
    model = make_model(agent_settings, gateway, make_gate(clock), clock)

    response = await send(model)

    assert response.text == "готово"
    assert len(gateway.requests) == 2
    assert clock.now - 1000 >= 3


async def test_429_slows_down_other_requests(agent_settings):
    clock = Clock()
    gate = make_gate(clock)
    gateway = Gateway(failure(429, **{"Retry-After": "15"}))
    model = make_model(agent_settings, gateway, gate, clock, llm_max_retries=0)

    with pytest.raises(LLMBusy):
        await send(model)

    async with gate.slot():
        assert clock.now - 1000 >= 15


async def test_retries_consume_the_rate_limit(agent_settings):
    clock = Clock()
    gate = make_gate(clock, requests_per_minute=3, queue_timeout=5.0)
    gateway = Gateway(failure(500), failure(500), completion())
    model = make_model(agent_settings, gateway, gate, clock)

    await send(model)

    with pytest.raises(LLMBusy):
        await send(model)
    assert len(gateway.requests) == 3


@pytest.mark.parametrize("status", [500, 502, 503, 504])
async def test_5xx_retries_are_bounded(agent_settings, status):
    clock = Clock()
    gateway = Gateway(failure(status))
    model = make_model(agent_settings, gateway, make_gate(clock), clock)

    with pytest.raises(LLMUnavailable) as error:
        await send(model)

    assert len(gateway.requests) == 3
    assert error.value.retry_after >= 1
    assert all(request.headers["x-stainless-retry-count"] == "0" for request in gateway.requests)


async def test_exhausted_429_is_busy(agent_settings):
    clock = Clock()
    gateway = Gateway(failure(429, **{"Retry-After": "2"}))
    model = make_model(agent_settings, gateway, make_gate(clock), clock)

    with pytest.raises(LLMBusy) as error:
        await send(model)

    assert len(gateway.requests) == 3
    assert error.value.status_code == 503
    assert error.value.retry_after >= 2


async def test_long_retry_after_is_not_waited_out(agent_settings):
    clock = Clock()
    gateway = Gateway(failure(429, **{"Retry-After": "300"}))
    model = make_model(agent_settings, gateway, make_gate(clock), clock)

    with pytest.raises(LLMBusy) as error:
        await send(model)

    assert len(gateway.requests) == 1
    assert error.value.retry_after == 300


async def test_timeout_is_not_retried(agent_settings):
    gateway = Gateway(httpx2.ReadTimeout("timed out"))
    model = make_model(agent_settings, gateway)

    with pytest.raises(LLMUnavailable, match="вовремя"):
        await send(model)

    assert len(gateway.requests) == 1


async def test_connection_error_is_retried(agent_settings):
    clock = Clock()
    gateway = Gateway(httpx2.ConnectError("refused"), completion())
    model = make_model(agent_settings, gateway, make_gate(clock), clock)

    await send(model)

    assert len(gateway.requests) == 2


@pytest.mark.parametrize(
    ("status", "expected"), [(400, LLMFailed), (401, LLMUnavailable), (404, LLMFailed)]
)
async def test_rejected_request_is_not_retried(agent_settings, status, expected, caplog):
    gateway = Gateway(failure(status))
    model = make_model(agent_settings, gateway)

    with caplog.at_level(logging.DEBUG, logger="dota_winprob"), pytest.raises(expected) as error:
        await send(model)

    assert len(gateway.requests) == 1
    assert TEST_KEY not in error.value.detail
    assert TEST_KEY not in caplog.text
    assert error.value.__suppress_context__


async def test_wire_format_fits_openai_compatible_gateway(agent_settings):
    from dota_winprob.agent.llm.client import model_settings

    gateway = Gateway(completion())
    model = make_model(agent_settings, gateway)

    await model.request(MESSAGES, model_settings(agent_settings), ModelRequestParameters())

    [request] = gateway.requests
    body = json.loads(request.content)
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    assert body["model"] == "bonsai2-27b"
    assert body["max_tokens"] == agent_settings.llm_max_output_tokens
    assert "max_completion_tokens" not in body


async def test_streaming_cannot_bypass_limits(agent_settings):
    model = make_model(agent_settings, Gateway(completion()))

    with pytest.raises(NotImplementedError):
        async with model.request_stream(MESSAGES, None, ModelRequestParameters()):
            pass


@pytest.mark.parametrize("key", [None, "", "   "])
def test_blank_key_means_no_llm(key):
    settings = AgentSettings(_env_file=None, llm_api_key=key)

    assert settings.api_key is None
    assert build_llm(settings, build_gate(settings)) is None


def test_key_is_not_shown_in_settings(agent_settings):
    assert TEST_KEY not in repr(agent_settings)
    assert TEST_KEY not in str(agent_settings.model_dump())


def test_defaults_follow_gateway_limits(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    settings = AgentSettings(_env_file=None)

    assert settings.llm_base_url == "https://ftii.llm.actcognitive.org/v1"
    assert settings.llm_model == "bonsai2-27b"
    assert settings.llm_max_concurrency == 2
    assert settings.llm_requests_per_minute == 20
    assert settings.agent_step_budget == 8


def test_api_settings_do_not_need_llm_key(tmp_path):
    from dota_winprob.config import Settings

    env_file = tmp_path / "service.env"
    env_file.write_text("LLM_API_KEY=unit\nAGENT_LOKI_URL=http://x\nLOG_LEVEL=DEBUG\n")

    settings = Settings(_env_file=env_file)

    assert settings.log_level == "DEBUG"
    assert not hasattr(settings, "llm_api_key")
