from enum import StrEnum

from pydantic import BaseModel, Field


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
