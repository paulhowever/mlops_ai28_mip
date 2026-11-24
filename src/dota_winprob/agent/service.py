import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.exceptions import AgentRunError, RunTimedOut, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage

from dota_winprob.agent import metrics
from dota_winprob.agent.config import SERVICE_NAME, AgentSettings
from dota_winprob.agent.errors import AskError, LLMFailed, LLMNotConfigured, QuestionTimeout
from dota_winprob.agent.llm.client import model_settings
from dota_winprob.agent.prompt import (
    INSTRUCTIONS,
    STEP_LIMIT_ANSWER,
    UNGROUNDED_RETRY,
    UNGROUNDED_WARNING,
)
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.schemas import AnswerStatus, AskRequest, AskResponse, Turn, Usage
from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.agent.tools.catalog import ARGUMENT_RETRIES, PREDICT, build_toolset
from dota_winprob.agent.tools.deps import AgentDeps
from dota_winprob.agent.tools.guard import GuardedToolset

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

ASK_SPAN = f"invoke_agent {SERVICE_NAME}"
OUTPUT_RETRIES = 2


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


async def require_grounded(ctx: RunContext[AgentDeps], answer: str) -> str:
    evidence = ctx.deps.evidence
    claims = evidence.ungrounded(answer)
    if claims and not evidence.challenged:
        evidence.challenged = True
        shown = ", ".join(f"{claim:.2f}" for claim in claims)
        raise ModelRetry(UNGROUNDED_RETRY.format(claims=shown))
    return answer


def build_agent(settings: AgentSettings, sources: Sources, model: Model) -> Agent[AgentDeps, str]:
    toolset = GuardedToolset(
        build_toolset(settings, sources),
        timeouts={PREDICT: settings.agent_predict_timeout + settings.agent_tool_timeout},
        default_timeout=settings.agent_tool_timeout,
        output_chars=settings.agent_tool_output_chars,
    )
    agent = Agent(
        model,
        name=SERVICE_NAME,
        instructions=INSTRUCTIONS,
        deps_type=AgentDeps,
        toolsets=[toolset],
        model_settings=model_settings(settings),
        retries={"tools": ARGUMENT_RETRIES, "output": OUTPUT_RETRIES},
    )
    agent.output_validator(require_grounded)
    return agent


def to_messages(history: Sequence[Turn], redactor: Redactor) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for turn in history:
        content = redactor.text(turn.content)
        if turn.role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content)]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content)]))
    return messages


@dataclass(frozen=True)
class AgentService:
    settings: AgentSettings
    redactor: Redactor
    agent: Agent[AgentDeps, str] | None

    async def ask(self, request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        result = "error"
        with tracer.start_as_current_span(
            ASK_SPAN,
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": SERVICE_NAME,
                "gen_ai.request.model": self.settings.llm_model,
            },
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                response = await self._answer(request, span)
                result = response.status.value
                return response
            except AskError as error:
                result = error.reason
                span.set_status(Status(StatusCode.ERROR, error.reason))
                raise
            finally:
                duration = time.perf_counter() - started
                span.set_attribute("agent.question.result", result)
                metrics.questions.add(1, {"result": result})
                metrics.question_duration.record(duration, {"result": result})
                log.info(
                    "Вопрос агенту обработан",
                    extra={"result": result, "duration_s": round(duration, 3)},
                )

    async def _answer(self, request: AskRequest, span: Span) -> AskResponse:
        if self.agent is None:
            raise LLMNotConfigured
        question = self.redactor.text(request.question)
        deps = AgentDeps(redactor=self.redactor)
        for text in (question, *(turn.content for turn in request.history)):
            deps.evidence.allow(text)
        usage = RunUsage()
        status, answer = await self._run(self.agent, question, request.history, deps, usage)
        span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        span.set_attribute("agent.llm.requests", usage.requests)
        span.set_attribute("agent.tool.call_count", len(deps.steps))
        warnings = [UNGROUNDED_WARNING] if deps.evidence.ungrounded(answer) else []
        return AskResponse(
            answer=self.redactor.text(answer),
            status=status,
            steps=deps.steps,
            predictions=deps.evidence.predictions,
            warnings=warnings,
            usage=Usage(
                llm_requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            ),
            model=self.settings.llm_model,
            trace_id=current_trace_id(),
        )

    async def _run(
        self,
        agent: Agent[AgentDeps, str],
        question: str,
        history: Sequence[Turn],
        deps: AgentDeps,
        usage: RunUsage,
    ) -> tuple[AnswerStatus, str]:
        try:
            run = await agent.run(
                question,
                message_history=to_messages(history, self.redactor),
                deps=deps,
                usage=usage,
                usage_limits=UsageLimits(
                    request_limit=self.settings.agent_step_budget,
                    tool_calls_limit=self.settings.agent_tool_call_budget,
                ),
                timeout=self.settings.agent_question_timeout,
            )
        except UsageLimitExceeded:
            return AnswerStatus.step_limit, STEP_LIMIT_ANSWER
        except RunTimedOut:
            raise QuestionTimeout from None
        except AgentRunError as error:
            log.error(
                "Агент не получил пригодный ответ модели",
                extra={"error_type": type(error).__name__},
            )
            raise LLMFailed from None
        return AnswerStatus.ok, run.output
