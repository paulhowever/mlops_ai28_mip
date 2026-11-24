from opentelemetry import trace
from opentelemetry.trace import SpanContext


def current_span_context() -> SpanContext | None:
    context = trace.get_current_span().get_span_context()
    return context if context.is_valid else None


def trace_id_of(context: SpanContext | None) -> str | None:
    return None if context is None else format(context.trace_id, "032x")


def current_trace_id() -> str | None:
    return trace_id_of(current_span_context())
