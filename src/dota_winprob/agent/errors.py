class AskError(Exception):
    reason = "error"
    status_code = 500
    detail = "Не удалось ответить на вопрос"

    def __init__(self, detail: str | None = None, retry_after: int | None = None) -> None:
        super().__init__(detail or self.detail)
        self.detail = detail or self.detail
        self.retry_after = retry_after


class LLMNotConfigured(AskError):
    reason = "no_key"
    status_code = 503
    detail = "Агент не настроен: переменная LLM_API_KEY не задана. Конвейер работает без агента."


class LLMBusy(AskError):
    reason = "busy"
    status_code = 503
    detail = "Агент перегружен: лимит запросов к языковой модели исчерпан"


class LLMUnavailable(AskError):
    reason = "llm_unavailable"
    status_code = 503
    detail = "Языковая модель недоступна"


class LLMFailed(AskError):
    reason = "llm_error"
    status_code = 502
    detail = "Языковая модель вернула непригодный ответ"


class QuestionTimeout(AskError):
    reason = "timeout"
    status_code = 504
    detail = "Время на вопрос вышло"


class SourceUnavailable(Exception):
    def __init__(self, source: str, reason: str) -> None:
        super().__init__(f"{source}: {reason}")
        self.source = source
        self.reason = reason


class ToolInputError(Exception):
    pass
