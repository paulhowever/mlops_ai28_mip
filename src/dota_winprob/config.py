import socket

from pydantic import Field, field_validator
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

    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672/",
        description="Адрес RabbitMQ.",
    )
    queue_max_length: int = Field(
        default=100,
        ge=0,
        description="Ёмкость очереди задач. 0 — без ограничения, только для эксперимента.",
    )
    delivery_limit: int = Field(
        default=3,
        ge=1,
        description="Сколько раз сообщение может пережить падение воркера, прежде чем уйти в DLQ.",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Сколько раз обработчик пробует задачу, прежде чем отправить её в DLQ.",
    )
    retry_delay_ms: int = Field(
        default=1000,
        gt=0,
        description="Пауза перед повторной попыткой, держит сообщение в очереди tasks.retry.",
    )
    worker_concurrency: int = Field(
        default=4,
        ge=1,
        description="Размер пула потоков для инференса в одном воркере.",
    )
    worker_prefetch: int | None = Field(
        default=None,
        ge=0,
        description="prefetch_count воркера. По умолчанию равен worker_concurrency, 0 — без ограничения.",
    )
    inference_delay_ms: int = Field(
        default=200,
        ge=0,
        description="Искусственная длительность инференса одной задачи.",
    )
    publish_timeout: float = Field(
        default=1.0,
        gt=0,
        description="Сколько ждать подтверждения публикации от брокера.",
    )
    broker_reconnect_interval: float = Field(
        default=5.0,
        gt=0,
        description="Пауза между попытками API подключиться к брокеру.",
    )
    graceful_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Сколько воркер ждёт завершения взятых задач при остановке.",
    )
    source_backoff_min: float = Field(
        default=0.02,
        gt=0,
        description="Начальная пауза источника после отказа брокера.",
    )
    source_backoff_max: float = Field(
        default=0.5,
        gt=0,
        description="Потолок паузы источника после отказа брокера.",
    )
    retry_after_max: int = Field(
        default=60,
        ge=1,
        description="Верхняя граница заголовка Retry-After.",
    )
    worker_name: str = Field(
        default_factory=socket.gethostname,
        description="Имя экземпляра воркера в логах и в таблице задач.",
    )

    git_sha: str = "unknown"
    built_at: str = "unknown"

    @field_validator("worker_prefetch", mode="before")
    @classmethod
    def _empty_prefetch_is_default(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def prefetch_count(self) -> int:
        if self.worker_prefetch is None:
            return self.worker_concurrency
        return self.worker_prefetch

    @property
    def inference_delay(self) -> float:
        return self.inference_delay_ms / 1000


def get_settings() -> Settings:
    return Settings()
