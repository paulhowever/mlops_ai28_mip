import logging

from fastapi import APIRouter, Request, Response, status

from dota_winprob.clients import opendota, postgres
from dota_winprob.config import Settings
from dota_winprob.schemas import HealthReport, LivenessResponse
from dota_winprob.services.health import Probe, build_report, has_critical_failure

log = logging.getLogger(__name__)

LIVENESS_PATH = "/healthz"

router = APIRouter(tags=["health"])
v1_router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get(
    LIVENESS_PATH,
    response_model=LivenessResponse,
    summary="Liveness-проба",
    description="Быстрая проверка, что процесс жив. Внешних вызовов не делает.",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse(status="ok")


def _collect_probes(request: Request) -> list[Probe]:
    pool = request.app.state.db_pool
    opendota_client = request.app.state.opendota

    async def probe_postgres() -> str | None:
        return await postgres.fetch_version(pool)

    async def probe_opendota() -> str | None:
        await opendota.check_availability(opendota_client)
        return None

    return [
        Probe(name="postgres", critical=True, run=probe_postgres),
        Probe(
            name="opendota",
            critical=False,
            run=probe_opendota,
            note=opendota.NO_VERSION_REASON,
        ),
    ]


@v1_router.get(
    "/health",
    response_model=HealthReport,
    summary="End-to-end health-check",
    description=(
        "Опрашивает внешние зависимости параллельно и возвращает по каждой "
        "статус, версию и время ответа. Код 503 — только если недоступна "
        "зависимость, критичная для работы сервиса."
    ),
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": HealthReport,
            "description": "Недоступна критичная зависимость.",
        },
    },
)
async def health(request: Request, response: Response) -> HealthReport:
    settings: Settings = request.app.state.settings

    report = await build_report(
        _collect_probes(request),
        probe_timeout=settings.health_probe_timeout,
    )

    if has_critical_failure(report):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        log.warning("Сервис деградировал: недоступна критичная зависимость")

    return report
