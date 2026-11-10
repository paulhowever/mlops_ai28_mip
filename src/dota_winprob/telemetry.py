import logging
import os
import threading
from dataclasses import dataclass

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from dota_winprob import __version__
from dota_winprob.config import Settings

log = logging.getLogger(__name__)

DEPLOYMENT_ENVIRONMENT = "deployment.environment.name"
SEMCONV_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
FASTAPI_EXCLUDED_URLS = "OTEL_PYTHON_FASTAPI_EXCLUDED_URLS"
DEFAULT_EXCLUDED_URLS = "healthz"

HTTP_SERVER_DURATION = "http.server.request.duration"
MESSAGING_DURATION = "messaging.*.duration"
INFERENCE_DURATION = "pipeline.worker.inference.duration"
QUEUE_WAIT = "pipeline.worker.queue_wait"

LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
QUEUE_WAIT_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
INFERENCE_OFFSETS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
INFERENCE_FACTORS = (1.0, 2.0, 5.0)

PROCESS_METRICS: dict[str, list[str] | None] = {"process.memory.usage": None}


def build_resource(settings: Settings) -> Resource:
    detected = Resource.create()
    attributes = dict(detected.attributes)
    if str(attributes.get(SERVICE_NAME, "")).startswith("unknown_service"):
        attributes[SERVICE_NAME] = settings.service_name
    attributes.setdefault(SERVICE_VERSION, __version__)
    attributes.setdefault(DEPLOYMENT_ENVIRONMENT, settings.environment)
    return Resource(attributes, detected.schema_url)


def service_identity(settings: Settings) -> tuple[str, str, str]:
    attributes = build_resource(settings).attributes
    return (
        str(attributes[SERVICE_NAME]),
        str(attributes[SERVICE_VERSION]),
        str(attributes[DEPLOYMENT_ENVIRONMENT]),
    )


def inference_buckets(delay: float) -> tuple[float, ...]:
    bounds = {delay + offset for offset in INFERENCE_OFFSETS}
    if delay > 0:
        bounds |= {delay * factor for factor in INFERENCE_FACTORS}
    return tuple(sorted({round(bound, 6) for bound in bounds}))


def metric_views(settings: Settings) -> list[View]:
    return [
        View(
            instrument_name=HTTP_SERVER_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS),
        ),
        View(
            instrument_name=MESSAGING_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(LATENCY_BUCKETS),
        ),
        View(
            instrument_name=INFERENCE_DURATION,
            aggregation=ExplicitBucketHistogramAggregation(
                inference_buckets(settings.inference_delay)
            ),
        ),
        View(
            instrument_name=QUEUE_WAIT,
            aggregation=ExplicitBucketHistogramAggregation(QUEUE_WAIT_BUCKETS),
        ),
    ]


def instrument_libraries() -> None:
    os.environ.setdefault(SEMCONV_OPT_IN, "http")
    for instrumentor in (AsyncPGInstrumentor(), HTTPXClientInstrumentor()):
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()


def excluded_urls() -> str:
    return os.environ.get(FASTAPI_EXCLUDED_URLS, DEFAULT_EXCLUDED_URLS)


@dataclass
class Telemetry:
    tracer_provider: TracerProvider | None = None
    meter_provider: MeterProvider | None = None

    @property
    def enabled(self) -> bool:
        return self.tracer_provider is not None

    def _flush(self) -> None:
        if self.tracer_provider is not None:
            self.tracer_provider.shutdown()
        if self.meter_provider is not None:
            self.meter_provider.shutdown()

    def shutdown(self, timeout: float) -> bool:
        if not self.enabled:
            return True
        flusher = threading.Thread(target=self._flush, name="telemetry-shutdown", daemon=True)
        flusher.start()
        flusher.join(timeout)
        if flusher.is_alive():
            log.warning(
                "Телеметрия не отправлена за отведённое время, остаток отброшен",
                extra={"timeout_s": timeout},
            )
            return False
        return True


_ACTIVE: dict[str, Telemetry] = {}


def setup_telemetry(settings: Settings) -> Telemetry:
    if "process" in _ACTIVE:
        return _ACTIVE["process"]
    if settings.otel_sdk_disabled:
        telemetry = Telemetry()
    else:
        resource = build_resource(settings)
        tracer_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
            views=metric_views(settings),
            shutdown_on_exit=False,
        )
        trace.set_tracer_provider(tracer_provider)
        metrics.set_meter_provider(meter_provider)
        SystemMetricsInstrumentor(config=PROCESS_METRICS).instrument(meter_provider=meter_provider)
        telemetry = Telemetry(tracer_provider, meter_provider)
    _ACTIVE["process"] = telemetry
    log.info(
        "Телеметрия настроена" if telemetry.enabled else "Телеметрия выключена",
        extra={"otel_sdk_disabled": settings.otel_sdk_disabled},
    )
    return telemetry
