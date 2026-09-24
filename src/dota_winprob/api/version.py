from fastapi import APIRouter, Request

from dota_winprob import __version__
from dota_winprob.config import Settings
from dota_winprob.schemas import VersionResponse

router = APIRouter(prefix="/api/v1", tags=["service"])


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Версия приложения",
    description="Версия из метаданных пакета и метаданные сборки образа.",
)
async def version(request: Request) -> VersionResponse:
    settings: Settings = request.app.state.settings
    return VersionResponse(
        version=__version__,
        git_sha=settings.git_sha,
        built_at=settings.built_at,
        environment=settings.environment,
    )
