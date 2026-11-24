from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from dota_winprob.agent.schemas import AgentInfo

STATIC = Path(__file__).parent / "static"
INDEX = STATIC / "index.html"
INFO_PATH = "/agent/info"

router = APIRouter()


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX, media_type="text/html; charset=utf-8")


@router.get(INFO_PATH, response_model=AgentInfo, tags=["agent"], summary="Настройки для страницы")
async def info(request: Request) -> AgentInfo:
    settings = request.app.state.agent_settings
    return AgentInfo(
        model=settings.llm_model,
        llm_configured=request.app.state.service.configured,
        grafana_url=settings.agent_grafana_url.rstrip("/"),
    )
