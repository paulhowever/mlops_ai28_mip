from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LivenessResponse(BaseModel):
    status: str = Field(description='Всегда "ok": процесс жив и обрабатывает запросы.')


class VersionResponse(BaseModel):
    version: str = Field(description="Версия из метаданных пакета (источник — pyproject.toml).")
    git_sha: str = Field(description="Коммит, из которого собран образ.")
    built_at: str = Field(description="Время сборки образа, ISO-8601 UTC.")
    environment: str = Field(description="Окружение, в котором запущен экземпляр.")


class ComponentStatus(StrEnum):
    healthy = "healthy"
    unavailable = "unavailable"


class ReportStatus(StrEnum):
    ok = "ok"
    degraded = "degraded"


class ComponentHealth(BaseModel):
    status: ComponentStatus
    version: str | None = Field(
        default=None,
        description="Версия компонента, если он её публикует.",
    )
    latency_ms: float = Field(description="Время ответа компонента в миллисекундах.")
    critical: bool = Field(
        description="Влияет ли недоступность компонента на работоспособность сервиса.",
    )
    detail: str | None = Field(
        default=None,
        description="Пояснение: причина отказа или отсутствия версии.",
    )


class HealthReport(BaseModel):
    status: ReportStatus
    total_latency_ms: float = Field(
        description=(
            "Время сбора всего отчёта. Пробы выполняются параллельно, поэтому "
            "это величина по часам, а не сумма времён отдельных компонентов."
        ),
    )
    components: dict[str, ComponentHealth]


class MatchState(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "radiant_gold": 13500,
                    "dire_gold": 12400,
                    "gold_advantage": 1100,
                    "radiant_xp": 14200,
                    "dire_xp": 13600,
                    "xp_advantage": 600,
                    "radiant_lh": 125,
                    "dire_lh": 116,
                    "lh_advantage": 9,
                }
            ]
        },
    )

    radiant_gold: float = Field(ge=0, description="Суммарное золото пяти игроков Radiant.")
    dire_gold: float = Field(ge=0, description="Суммарное золото пяти игроков Dire.")
    gold_advantage: float = Field(description="Преимущество по золоту: Radiant минус Dire.")
    radiant_xp: float = Field(ge=0, description="Суммарный опыт игроков Radiant.")
    dire_xp: float = Field(ge=0, description="Суммарный опыт игроков Dire.")
    xp_advantage: float = Field(description="Преимущество по опыту: Radiant минус Dire.")
    radiant_lh: float = Field(ge=0, description="Суммарные добивания крипов у Radiant.")
    dire_lh: float = Field(ge=0, description="Суммарные добивания крипов у Dire.")
    lh_advantage: float = Field(description="Преимущество по добиваниям: Radiant минус Dire.")


class ProcessResponse(BaseModel):
    win_probability: float = Field(
        ge=0,
        le=1,
        description="Вероятность победы Radiant.",
    )
    model_name: str = Field(description="Имя модели в MLflow Model Registry.")
    model_version: str = Field(
        description="Версия из Registry, на которую указывал алиас при загрузке модели.",
    )
    model_alias: str = Field(description="Алиас, по которому модель загружена на старте.")
