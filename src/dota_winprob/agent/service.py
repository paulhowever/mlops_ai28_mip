import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.exceptions import AgentRunError, RunTimedOut, UsageLimitExceeded
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from dota_winprob.agent import metrics
from dota_winprob.agent.actions.service import ActionService
from dota_winprob.agent.config import SERVICE_NAME, AgentSettings
from dota_winprob.agent.envelope import without_envelope
from dota_winprob.agent.errors import AskError, LLMFailed, LLMNotConfigured, QuestionTimeout
from dota_winprob.agent.images import Image
from dota_winprob.agent.llm.client import model_settings
from dota_winprob.agent.prompt import (
    BORROWED_RETRY,
    BORROWED_WARNING,
    INSTRUCTIONS,
    LAST_STEP,
    STEP_LIMIT_ANSWER,
    UNGROUNDED_RETRY,
    UNGROUNDED_WARNING,
    UNKNOWN_NUMBERS_RETRY,
    UNKNOWN_NUMBERS_WARNING,
)
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.schemas import AnswerStatus, AskRequest, AskResponse, Step, Turn, Usage
from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.agent.tools.catalog import (
    ARGUMENT_RETRIES,
    PREDICT,
    build_read_toolset,
    build_toolset,
)
from dota_winprob.agent.tools.deps import AgentDeps
from dota_winprob.agent.tools.guard import GuardedToolset, last_step
from dota_winprob.agent.tracing import current_trace_id

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

ASK_SPAN = f"invoke_agent {SERVICE_NAME}"
OUTPUT_RETRIES = 2


@dataclass(frozen=True)
class Agents:
    chat: Agent[AgentDeps, str]
    observer: Agent[AgentDeps, str]


@dataclass(frozen=True)
class Analysis:
    text: str
    steps: list[Step]
    warnings: list[str]


def _exhausted(ctx: RunContext[AgentDeps]) -> bool:
    budget = ctx.deps.step_budget
    return budget is not None and ctx.usage.requests >= budget


def closing_note(ctx: RunContext[AgentDeps]) -> str | None:
    return LAST_STEP if last_step(ctx) and ctx.deps.steps else None


async def require_grounded(ctx: RunContext[AgentDeps], answer: str) -> str:
    evidence = ctx.deps.evidence
    if evidence.challenged or _exhausted(ctx):
        return answer
    complaints = []
    if claims := evidence.ungrounded(answer):
        shown = ", ".join(f"{claim:.2f}" for claim in claims)
        complaints.append(UNGROUNDED_RETRY.format(claims=shown))
    if claims := evidence.borrowed(answer):
        shown = ", ".join(f"{claim:.4g}" for claim in claims)
        complaints.append(BORROWED_RETRY.format(claims=shown))
    if numbers := evidence.unknown_numbers(answer):
        complaints.append(UNKNOWN_NUMBERS_RETRY.format(numbers=", ".join(numbers)))
    if complaints:
        evidence.challenged = True
        raise ModelRetry("\n".join(complaints))
    return answer


def build_agent(
    settings: AgentSettings, toolset: AbstractToolset[AgentDeps], model: Model
) -> Agent[AgentDeps, str]:
    agent = Agent(
        model,
        name=SERVICE_NAME,
        instructions=INSTRUCTIONS,
        deps_type=AgentDeps,
        toolsets=[
            GuardedToolset(
                toolset,
                timeouts={PREDICT: settings.agent_predict_timeout + settings.agent_tool_timeout},
                default_timeout=settings.agent_tool_timeout,
                output_chars=settings.agent_tool_output_chars,
            )
        ],
        model_settings=model_settings(settings),
        retries={"tools": ARGUMENT_RETRIES, "output": OUTPUT_RETRIES},
    )
    agent.instructions(closing_note)
    agent.output_validator(require_grounded)
    return agent


def build_agents(
    settings: AgentSettings, sources: Sources, actions: ActionService, model: Model
) -> Agents:
    return Agents(
        chat=build_agent(settings, build_toolset(settings, sources, actions), model),
        observer=build_agent(settings, build_read_toolset(settings, sources), model),
    )


def to_messages(history: Sequence[Turn], clean: Callable[[str], str]) -> list[ModelMessage]:
    messages: list[ModelMessage] = []
    for turn in history:
        content = clean(turn.content)
        if turn.role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content)]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content)]))
    return messages


@dataclass(frozen=True)
class AgentService:
    settings: AgentSettings
    redactor: Redactor
    agents: Agents | None

    @property
    def configured(self) -> bool:
        return self.agents is not None

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

    def _clean(self, text: str) -> str:
        return without_envelope(self.redactor.text(text))

    async def analyze(self, question: str) -> Analysis:
        if self.agents is None:
            raise LLMNotConfigured
        deps = AgentDeps(redactor=self.redactor)
        for text in (INSTRUCTIONS, question):
            deps.evidence.allow(text)
        status, answer = await self._run(
            self.agents.observer,
            question,
            deps=deps,
            usage=RunUsage(),
            steps=self.settings.agent_incident_step_budget,
            timeout_s=self.settings.agent_incident_timeout,
        )
        warnings = _warnings(deps, answer)
        if status is AnswerStatus.step_limit:
            warnings.append(STEP_LIMIT_ANSWER)
        return Analysis(self.redactor.text(answer), deps.steps, warnings)

    async def _answer(self, request: AskRequest, span: Span) -> AskResponse:
        if self.agents is None:
            raise LLMNotConfigured
        question = self._clean(request.question)
        deps = AgentDeps(redactor=self.redactor)
        for text in (INSTRUCTIONS, question, *(turn.content for turn in request.history)):
            deps.evidence.allow(text)
        usage = RunUsage()
        status, answer = await self._run(
            self.agents.chat,
            question if request.image is None else [question, _binary(request.image.image)],
            request.history,
            deps=deps,
            usage=usage,
            steps=self.settings.agent_step_budget,
            timeout_s=self.settings.agent_question_timeout,
        )
        span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        span.set_attribute("agent.llm.requests", usage.requests)
        span.set_attribute("agent.tool.call_count", len(deps.steps))
        span.set_attribute("agent.question.image", request.image is not None)
        return AskResponse(
            answer=self.redactor.text(answer),
            status=status,
            steps=deps.steps,
            predictions=deps.evidence.predictions,
            actions=deps.actions,
            warnings=_warnings(deps, answer),
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
        prompt: str | list[UserContent],
        history: Sequence[Turn] = (),
        *,
        deps: AgentDeps,
        usage: RunUsage,
        steps: int,
        timeout_s: float,
    ) -> tuple[AnswerStatus, str]:
        deps.step_budget = steps
        try:
            run = await agent.run(
                prompt,
                message_history=to_messages(history, self._clean),
                deps=deps,
                usage=usage,
                usage_limits=UsageLimits(
                    request_limit=steps,
                    tool_calls_limit=self.settings.agent_tool_call_budget,
                ),
                timeout=timeout_s,
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


def _warnings(deps: AgentDeps, answer: str) -> list[str]:
    warnings = [UNGROUNDED_WARNING] if deps.evidence.ungrounded(answer) else []
    if deps.evidence.borrowed(answer):
        warnings.append(BORROWED_WARNING)
    if numbers := deps.evidence.unknown_numbers(answer):
        warnings.append(UNKNOWN_NUMBERS_WARNING.format(numbers=", ".join(numbers)))
    return warnings


def _binary(image: Image) -> BinaryContent:
    return BinaryContent(image.data, media_type=image.media_type)
