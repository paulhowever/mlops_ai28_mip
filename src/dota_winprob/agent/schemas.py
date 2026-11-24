from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_QUESTION = 2000
MAX_TURNS = 10
MAX_TURN = 4000
MAX_HISTORY = 12_000

Question = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION)
]


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=MAX_TURN)]


class AskRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"question": "Что сейчас с очередью?", "history": []}],
        },
    )

    question: Question = Field(description="Вопрос агенту.")
    history: list[Turn] = Field(
        default_factory=list,
        max_length=MAX_TURNS,
        description="Последние реплики диалога, сервер их не хранит.",
    )

    @model_validator(mode="after")
    def _history_is_short(self) -> "AskRequest":
        if sum(len(turn.content) for turn in self.history) > MAX_HISTORY:
            raise ValueError(f"история длиннее {MAX_HISTORY} символов")
        return self


class AnswerStatus(StrEnum):
    ok = "ok"
    step_limit = "step_limit"


class Step(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(
        default="error",
        description="ok, empty, rejected, pending, failed, invalid_input, unavailable, timeout, malformed, error.",
    )
    summary: str = ""
    duration_ms: float = 0.0


class Prediction(BaseModel):
    tool: str
    task_id: str
    win_probability: float = Field(description="Вероятность победы Radiant.")
    model_version: str | None = None


class Usage(BaseModel):
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AskResponse(BaseModel):
    answer: str
    status: AnswerStatus
    steps: list[Step]
    predictions: list[Prediction] = Field(
        description="Вероятности модели проекта из predict и get_task. Заполняет код, не языковая модель.",
    )
    warnings: list[str]
    usage: Usage
    model: str
    trace_id: str | None


class ErrorResponse(BaseModel):
    detail: str
    reason: str
    trace_id: str | None = None
