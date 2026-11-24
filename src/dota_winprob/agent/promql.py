from dataclasses import dataclass
from typing import Literal, get_args

MetricName = Literal[
    "queue_ready",
    "queue_unacked",
    "queue_retry",
    "queue_limit",
    "consumers",
    "dlq_ready",
    "dlq_transfers",
    "source_rate",
    "worker_rate",
    "worker_errors",
    "in_flight",
    "source_rejections",
    "source_wait_share",
    "api_rate",
    "api_503_share",
    "api_p95",
    "inference_p95",
    "queue_wait_p95",
]
METRIC_NAMES: tuple[str, ...] = get_args(MetricName)


@dataclass(frozen=True)
class Metric:
    expression: str
    unit: str
    meaning: str

    def query(self, window_s: int) -> str:
        return self.expression.replace("[W]", f"[{int(window_s)}s]")


def _ready(queue: str) -> str:
    return f'max(rabbitmq_detailed_queue_messages_ready{{queue="{queue}"}})'


def _p95(histogram: str, matcher: str = "") -> str:
    return f"histogram_quantile(0.95, sum by (le) (rate({histogram}_bucket{matcher}[W])))"


_API = "http_server_request_duration_seconds"
_API_JOB = '{job="dota-api"}'
_API_503 = '{job="dota-api",http_response_status_code="503"}'

METRICS: dict[str, Metric] = {
    "queue_ready": Metric(_ready("tasks.score"), "сообщений", "готовые задачи в tasks.score"),
    "queue_unacked": Metric(
        'max(rabbitmq_detailed_queue_messages_unacked{queue="tasks.score"})',
        "сообщений",
        "задачи в работе у воркеров (unacked)",
    ),
    "queue_retry": Metric(_ready("tasks.retry"), "сообщений", "очередь повторов tasks.retry"),
    "queue_limit": Metric(
        "max(last_over_time(pipeline_queue_max_length[10m]))",
        "сообщений",
        "лимит x-max-length рабочей очереди",
    ),
    "consumers": Metric(
        'max(rabbitmq_detailed_queue_consumers{queue="tasks.score"})',
        "штук",
        "потребители tasks.score, то есть живые воркеры",
    ),
    "dlq_ready": Metric(_ready("tasks.dlq"), "сообщений", "мёртвые письма в tasks.dlq"),
    "dlq_transfers": Metric(
        'sum(increase(pipeline_worker_tasks_total{result="dlq"}[W]))',
        "задач за окно",
        "переводы в DLQ за окно",
    ),
    "source_rate": Metric(
        "sum(rate(pipeline_source_published_total[W]))", "задач/с", "скорость источника"
    ),
    "worker_rate": Metric(
        'sum(rate(pipeline_worker_tasks_total{result="ok"}[W]))',
        "задач/с",
        "пропускная способность воркеров",
    ),
    "worker_errors": Metric(
        'sum(rate(pipeline_worker_tasks_total{result=~"error|dlq"}[W]))',
        "задач/с",
        "ошибки обработки и переводы в DLQ",
    ),
    "in_flight": Metric(
        "sum(pipeline_worker_in_flight)", "задач", "задачи в обработке по данным воркеров"
    ),
    "source_rejections": Metric(
        "sum(rate(pipeline_source_rejected_total[W]))", "отказов/с", "отказы брокера источнику"
    ),
    "source_wait_share": Metric(
        "sum(rate(pipeline_source_wait_seconds_total[W]))",
        "доля",
        "доля времени источника в паузах",
    ),
    "api_rate": Metric(f"sum(rate({_API}_count{_API_JOB}[W]))", "запросов/с", "запросы к API"),
    "api_503_share": Metric(
        f"sum(rate({_API}_count{_API_503}[W])) / sum(rate({_API}_count{_API_JOB}[W]))",
        "доля",
        "доля ответов 503",
    ),
    "api_p95": Metric(_p95(_API, _API_JOB), "с", "p95 ответа API"),
    "inference_p95": Metric(
        _p95("pipeline_worker_inference_duration_seconds"), "с", "p95 длительности inference"
    ),
    "queue_wait_p95": Metric(
        _p95("pipeline_worker_queue_wait_seconds"), "с", "p95 ожидания задачи в очереди"
    ),
}
