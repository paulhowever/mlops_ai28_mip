from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "dota-winprob"
    environment: str = "development"
    log_level: str = "INFO"

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5433/dota_winprob",
        description="Строка подключения к Postgres в формате asyncpg.",
    )
    opendota_base_url: str = Field(
        default="https://api.opendota.com",
        description="Базовый URL OpenDota API — внешнего источника данных о матчах.",
    )
    health_probe_timeout: float = Field(
        default=3.0,
        gt=0,
        description="Таймаут одной health-пробы в секундах.",
    )

    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="Адрес MLflow Tracking Server. Без него модель не загружается.",
    )
    model_name: str = Field(
        default="dota-winprob",
        description="Имя модели в MLflow Model Registry.",
    )
    model_alias: str = Field(
        default="champion",
        description="Алиас версии, которая загружается на старте.",
    )
    model_load_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Сколько секунд ждать загрузки модели на старте.",
    )

    git_sha: str = "unknown"
    built_at: str = "unknown"


def get_settings() -> Settings:
    return Settings()
