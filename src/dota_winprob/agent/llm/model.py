import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode
from pydantic_ai import RunContext
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelHTTPError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from dota_winprob.agent import metrics
from dota_winprob.agent.errors import LLMBusy, LLMFailed, LLMUnavailable
from dota_winprob.agent.llm.gate import LLMGate

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

RETRYABLE = (ModelRateLimitError, ModelUnavailableError, ModelServerError, ModelConnectionError)
BACKOFF_BASE = 1.0


def _result_of(error: ModelAPIError) -> str:
    if isinstance(error, ModelRateLimitError):
        return "rate_limited"
    if isinstance(error, ModelUnavailableError | ModelServerError):
        return "server_error"
    if isinstance(error, ModelTimeoutError):
        return "timeout"
    if isinstance(error, ModelConnectionError):
        return "connection"
    return "rejected"


def _status_code(error: ModelAPIError) -> int | None:
    return error.status_code if isinstance(error, ModelHTTPError) else None


class GuardedModel(WrapperModel):
    def __init__(
        self,
        wrapped: Model,
        gate: LLMGate,
        *,
        max_retries: int,
        retry_wait_max: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(wrapped)
        self._gate = gate
        self._max_retries = max_retries
        self._retry_wait_max = retry_wait_max
        self._sleep = sleep

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        attempt = 1
        while True:
            try:
                async with self._gate.slot():
                    return await self._attempt(
                        attempt, messages, model_settings, model_request_parameters
                    )
            except RETRYABLE as error:
                delay = self._retry_delay(error, attempt)
            except ModelAPIError as error:
                raise self._rejected(error) from None
            await self._sleep(delay)
            attempt += 1

    def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AbstractAsyncContextManager[StreamedResponse]:
        raise NotImplementedError("потоковые ответы обошли бы лимиты шлюза")

    async def _attempt(
        self,
        attempt: int,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
    ) -> ModelResponse:
        with tracer.start_as_current_span(
            f"chat {self.model_name}",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": self.system,
                "gen_ai.request.model": self.model_name,
                "agent.llm.attempt": attempt,
            },
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            result = "error"
            started = time.perf_counter()
            metrics.llm_in_flight.add(1)
            try:
                response = await self.wrapped.request(messages, model_settings, parameters)
                result = "ok"
            except ModelAPIError as error:
                result = _result_of(error)
                self._describe_failure(span, error, result, attempt)
                raise
            except asyncio.CancelledError:
                result = "cancelled"
                raise
            finally:
                metrics.llm_in_flight.add(-1)
                metrics.llm_requests.add(1, {"result": result})
                metrics.llm_duration.record(time.perf_counter() - started, {"result": result})
            self._record_usage(span, response)
            return response

    def _describe_failure(
        self, span: Span, error: ModelAPIError, result: str, attempt: int
    ) -> None:
        status_code = _status_code(error)
        span.set_attribute("error.type", type(error).__name__)
        if status_code is not None:
            span.set_attribute("http.response.status_code", status_code)
        span.set_status(Status(StatusCode.ERROR, result))
        log.warning(
            "Запрос к языковой модели не удался",
            extra={
                "result": result,
                "error_type": type(error).__name__,
                "status_code": status_code,
                "llm_attempt": attempt,
            },
        )

    def _record_usage(self, span: Span, response: ModelResponse) -> None:
        usage = response.usage
        span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        if response.model_name:
            span.set_attribute("gen_ai.response.model", response.model_name)
        if response.finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [response.finish_reason])
        metrics.llm_tokens.add(usage.input_tokens, {"type": "input"})
        metrics.llm_tokens.add(usage.output_tokens, {"type": "output"})

    def _retry_delay(self, error: ModelAPIError, attempt: int) -> float:
        delay = error.retry_after
        if delay is None:
            delay = min(self._retry_wait_max, BACKOFF_BASE * 2 ** (attempt - 1))
        if isinstance(error, ModelRateLimitError):
            self._gate.cool_down(delay)
        final = attempt > self._max_retries or isinstance(error, ModelTimeoutError)
        if final or delay > self._retry_wait_max:
            raise self._exhausted(error, delay) from None
        return delay

    def _exhausted(self, error: ModelAPIError, delay: float) -> LLMBusy | LLMUnavailable:
        retry_after = max(self._gate.retry_after(), math.ceil(delay))
        if isinstance(error, ModelRateLimitError):
            return LLMBusy("Шлюз языковой модели ограничил частоту запросов", retry_after)
        if isinstance(error, ModelTimeoutError):
            return LLMUnavailable("Языковая модель не ответила вовремя", retry_after)
        return LLMUnavailable(retry_after=retry_after)

    def _rejected(self, error: ModelAPIError) -> LLMFailed | LLMUnavailable:
        if isinstance(error, ModelAuthenticationError):
            return LLMUnavailable("Шлюз языковой модели не принял ключ LLM_API_KEY")
        status_code = _status_code(error)
        suffix = "" if status_code is None else f" (HTTP {status_code})"
        return LLMFailed(f"Шлюз языковой модели отклонил запрос{suffix}")
