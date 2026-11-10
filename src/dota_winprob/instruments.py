from opentelemetry import metrics

meter = metrics.get_meter("dota_winprob")

source_published = meter.create_counter(
    "pipeline.source.published",
    description="Задачи источника, которые брокер подтвердил.",
)
source_rejected = meter.create_counter(
    "pipeline.source.rejected",
    description="Отказы брокера источнику: nack при reject-publish и таймаут подтверждения.",
)
source_wait = meter.create_counter(
    "pipeline.source.wait",
    unit="s",
    description="Суммарное время пауз источника перед повторной публикацией.",
)

worker_tasks = meter.create_counter(
    "pipeline.worker.tasks",
    description="Задачи, обработанные воркером, по результату: ok, duplicate, error, dlq.",
)
inference_duration = meter.create_histogram(
    "pipeline.worker.inference.duration",
    unit="s",
    description="Длительность инференса одной задачи в пуле потоков.",
)
queue_wait = meter.create_histogram(
    "pipeline.worker.queue_wait",
    unit="s",
    description="Сколько задача ждала в очереди: от публикации до начала обработки.",
)
in_flight = meter.create_up_down_counter(
    "pipeline.worker.in_flight",
    description="Задачи, которые воркер обрабатывает прямо сейчас.",
)
redeliveries = meter.create_counter(
    "pipeline.worker.redeliveries",
    description="Повторные доставки: redelivered после падения воркера, retry из очереди повторов.",
)


def observe_queue_limit(limit: int) -> None:
    meter.create_observable_gauge(
        "pipeline.queue.max_length",
        callbacks=[lambda options: [metrics.Observation(limit)]],
        description="Лимит x-max-length рабочей очереди из QUEUE_MAX_LENGTH.",
    )
