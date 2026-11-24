import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic_ai import RunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset

from dota_winprob.agent import metrics
from dota_winprob.agent.envelope import MAX_SUMMARY, ToolResult, clip, render
from dota_winprob.agent.errors import SourceUnavailable, ToolInputError
from dota_winprob.agent.schemas import Step
from dota_winprob.agent.tools.deps import AgentDeps

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

MALFORMED = (LookupError, ValueError, TypeError, AttributeError)
MAX_ARGUMENT = 200
FAILED_STATUSES = frozenset({"unavailable", "timeout", "malformed"})


@dataclass
class GuardedToolset(WrapperToolset[AgentDeps]):
    timeouts: dict[str, float]
    default_timeout: float
    output_chars: int

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDeps],
        tool: ToolsetTool[AgentDeps],
    ) -> str:
        deps = ctx.deps
        step = Step(tool=name, arguments=_shown(deps, tool_args))
        deps.steps.append(step)
        started = time.perf_counter()
        with tracer.start_as_current_span(
            f"execute_tool {name}",
            attributes={"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name},
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                result = await self._safely(name, tool_args, ctx, tool)
                step.status = result.status
            finally:
                step.duration_ms = round((time.perf_counter() - started) * 1000, 1)
                metrics.tool_calls.add(1, {"tool": name, "result": step.status})
                span.set_attribute("agent.tool.result", step.status)
            if result.status in FAILED_STATUSES:
                span.set_status(Status(StatusCode.ERROR, result.status))
            step.summary = clip(deps.redactor.text(result.summary), MAX_SUMMARY)
            if result.action is not None:
                deps.actions.append(result.action)
            deps.evidence.collect(name, result.data)
            return render(name, result, deps.redactor, self.output_chars)

    async def _safely(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDeps],
        tool: ToolsetTool[AgentDeps],
    ) -> ToolResult:
        return await safely(
            name,
            partial(super().call_tool, name, tool_args, ctx, tool),
            self.timeouts.get(name, self.default_timeout),
        )


async def safely(
    name: str, call: Callable[[], Awaitable[ToolResult]], timeout_s: float
) -> ToolResult:
    try:
        async with asyncio.timeout(timeout_s):
            return await call()
    except ToolInputError as error:
        return _failure("agent", "invalid_input", str(error))
    except SourceUnavailable as error:
        _report(name, "unavailable", error.source, error.reason)
        return _failure(
            error.source, "unavailable", f"источник {error.source} недоступен: {error.reason}"
        )
    except TimeoutError:
        _report(name, "timeout", name, "TimeoutError")
        return _failure("agent", "timeout", f"инструмент не уложился в {timeout_s:g} с")
    except MALFORMED as error:
        _report(name, "malformed", name, type(error).__name__)
        return _failure("agent", "malformed", "источник вернул ответ в неожиданном формате")


def _report(tool: str, status: str, source: str, reason: str) -> None:
    log.warning(
        "Инструмент не получил данные",
        extra={"tool": tool, "result": status, "source": source, "reason": reason},
    )


def _failure(source: str, status: str, message: str) -> ToolResult:
    return ToolResult(source, {"error": message}, message, status)


def _shown(deps: AgentDeps, arguments: dict[str, Any]) -> dict[str, Any]:
    masked = deps.redactor.data(arguments)
    return {
        key: clip(value, MAX_ARGUMENT) if isinstance(value, str) else value
        for key, value in masked.items()
        if value is not None
    }
