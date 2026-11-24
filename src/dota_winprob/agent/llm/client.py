from dataclasses import dataclass

import httpx2
import pydantic_ai
from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider
from pydantic_ai.settings import ModelSettings

from dota_winprob.agent.config import AgentSettings
from dota_winprob.agent.llm.gate import LLMGate
from dota_winprob.agent.llm.model import GuardedModel

PROVIDER_NAME = "llm-gateway"
PROFILE = OpenAIModelProfile(
    openai_supports_strict_tool_definition=False,
    openai_chat_supports_max_completion_tokens=False,
    supports_inline_system_prompts=False,
    openai_chat_supports_multiple_system_messages=False,
)

pydantic_ai.BANNER_ENABLED = False


@dataclass
class LLM:
    model: Model
    client: AsyncOpenAI | None = None

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()


def build_gate(settings: AgentSettings) -> LLMGate:
    return LLMGate(
        max_concurrency=settings.llm_max_concurrency,
        requests_per_minute=settings.llm_requests_per_minute,
        max_waiting=settings.llm_max_waiting,
        queue_timeout=settings.llm_queue_timeout,
        tokens_per_minute=settings.llm_tokens_per_minute,
        background_share=settings.llm_background_share,
        background_timeout=settings.llm_background_timeout,
    )


def guard(model: Model, gate: LLMGate, settings: AgentSettings) -> GuardedModel:
    return GuardedModel(
        model,
        gate,
        max_retries=settings.llm_max_retries,
        retry_wait_max=settings.llm_retry_wait_max,
    )


def build_llm(
    settings: AgentSettings,
    gate: LLMGate,
    transport: httpx2.AsyncBaseTransport | None = None,
) -> LLM | None:
    if settings.api_key is None:
        return None
    client = AsyncOpenAI(
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
        max_retries=0,
        http_client=None if transport is None else httpx2.AsyncClient(transport=transport),
    )
    provider = OpenAICompatibleProvider(PROVIDER_NAME, openai_client=client)
    model = OpenAIChatModel(settings.llm_model, provider=provider, profile=PROFILE)
    return LLM(model=guard(model, gate, settings), client=client)


def model_settings(settings: AgentSettings) -> ModelSettings:
    return ModelSettings(
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_output_tokens,
    )
