from opentelemetry import metrics

meter = metrics.get_meter("dota_winprob.agent")

questions = meter.create_counter(
    "agent.questions",
    description="Вопросы агенту по результату: ok, step_limit, no_key, busy, llm_unavailable, llm_error, timeout, error.",
)
question_duration = meter.create_histogram(
    "agent.question.duration",
    unit="s",
    description="Длительность ответа на вопрос целиком.",
)
tool_calls = meter.create_counter(
    "agent.tool.calls",
    description="Вызовы инструментов по имени и результату.",
)
llm_requests = meter.create_counter(
    "agent.llm.requests",
    description="Запросы к шлюзу по результату: ok, rate_limited (429), server_error, timeout, connection, rejected, cancelled, error.",
)
llm_duration = meter.create_histogram(
    "agent.llm.duration",
    unit="s",
    description="Длительность одного запроса к шлюзу без ожидания слота.",
)
llm_tokens = meter.create_counter(
    "agent.llm.tokens",
    description="Токены по типу: input, output.",
)
llm_overload = meter.create_counter(
    "agent.llm.overload",
    description="Запросы, которые не дождались своего лимита: slots, rate.",
)
llm_in_flight = meter.create_up_down_counter(
    "agent.llm.in_flight",
    description="Запросы к шлюзу прямо сейчас.",
)

actions = meter.create_counter(
    "agent.actions",
    description="Предложения по действию и результату: proposed, confirmed, rejected, expired, done, partial, failed.",
)
returned_tasks = meter.create_counter(
    "agent.actions.returned_tasks",
    description="Задачи, возвращённые из DLQ в рабочую очередь.",
)
alerts = meter.create_counter(
    "agent.alerts",
    description="Алерты из webhook по результату: opened, repeated, cooldown, resolved, unknown, overflow.",
)
incidents = meter.create_counter(
    "agent.incidents",
    description="Разборы инцидентов по результату: done, facts_only, failed, skipped.",
)
incident_duration = meter.create_histogram(
    "agent.incident.duration",
    unit="s",
    description="Длительность фонового разбора инцидента.",
)
