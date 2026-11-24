from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dota_winprob.config import Settings

SERVICE_NAME = "dota-agent"

Seconds = Annotated[float, Field(gt=0)]


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    llm_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
        description="Ключ шлюза. Без него сервис стартует, но на вопросы отвечает 503.",
    )
    llm_base_url: str = Field(
        default="https://ftii.llm.actcognitive.org/v1",
        description="Адрес OpenAI-совместимого шлюза.",
    )
    llm_model: str = Field(default="bonsai2-27b", description="Имя модели на шлюзе.")
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    llm_max_output_tokens: int = Field(
        default=1500, ge=1, description="Потолок длины одного ответа модели."
    )
    llm_timeout: Seconds = Field(default=60.0, description="Тайм-аут одного запроса к шлюзу.")
    llm_max_concurrency: int = Field(
        default=2, ge=1, description="Одновременных запросов к шлюзу на процесс."
    )
    llm_requests_per_minute: int = Field(
        default=20, ge=1, description="Запросов к шлюзу за любые 60 секунд, вместе с повторами."
    )
    llm_max_waiting: int = Field(
        default=8, ge=0, description="Сколько запросов могут ждать свободного слота."
    )
    llm_queue_timeout: Seconds = Field(
        default=60.0,
        description="Сколько запрос ждёт слота и ограничителя, прежде чем получить 503.",
    )
    llm_max_retries: int = Field(
        default=2, ge=0, description="Повторов на 429, 5xx и обрыв соединения."
    )
    llm_retry_wait_max: Seconds = Field(
        default=20.0, description="Retry-After длиннее этого не ждём, сразу отвечаем 503."
    )

    agent_step_budget: int = Field(default=8, ge=1, description="Запросов к модели на один вопрос.")
    agent_tool_call_budget: int = Field(
        default=16, ge=1, description="Вызовов инструментов на один вопрос."
    )
    agent_question_timeout: Seconds = Field(
        default=240.0, description="Потолок времени на один вопрос."
    )
    agent_tool_timeout: Seconds = Field(
        default=10.0, description="Тайм-аут одного инструмента, кроме predict."
    )
    agent_source_timeout: Seconds = Field(
        default=5.0, description="Тайм-аут одного запроса к источнику данных."
    )
    agent_tool_output_chars: int = Field(
        default=4000, ge=500, description="Потолок вывода одного инструмента в символах."
    )
    agent_predict_timeout: Seconds = Field(
        default=30.0, description="Сколько predict ждёт результата задачи."
    )
    agent_predict_poll_interval: Seconds = Field(
        default=0.5, description="Пауза между опросами статуса задачи."
    )

    agent_api_url: str = Field(default="http://localhost:8000", description="Адрес API задач.")
    agent_prometheus_url: str = Field(default="http://localhost:9090")
    agent_loki_url: str = Field(default="http://localhost:3100")
    agent_tempo_url: str = Field(default="http://localhost:3200")
    mlflow_tracking_uri: str = Field(default="http://localhost:5001")
    agent_tasks_file: Path = Field(
        default=Path("data/tasks.csv"), description="Признаки пакетных матчей."
    )
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5433/dota_winprob")
    agent_database_url: str | None = Field(
        default=None,
        description="Отдельная строка подключения агента, например под ролью только для чтения.",
    )
    model_name: str = "dota-winprob"
    model_alias: str = "champion"

    @field_validator("llm_api_key", mode="before")
    @classmethod
    def _blank_key_is_missing(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("agent_database_url", mode="before")
    @classmethod
    def _blank_url_is_missing(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def readonly_database_url(self) -> str:
        return self.agent_database_url or self.database_url

    @property
    def api_key(self) -> str | None:
        return None if self.llm_api_key is None else self.llm_api_key.get_secret_value().strip()


def get_agent_settings() -> AgentSettings:
    return AgentSettings()


def base_settings() -> Settings:
    return Settings(app_name=SERVICE_NAME, queue_max_length=0)
