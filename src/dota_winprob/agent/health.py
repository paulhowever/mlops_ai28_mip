from collections.abc import Awaitable, Callable

from dota_winprob.agent.sources.bundle import Sources
from dota_winprob.services.health import Probe

Check = Callable[[], Awaitable[None]]


def _probe(name: str, check: Check, note: str | None = None) -> Probe:
    async def run() -> str | None:
        await check()
        return None

    return Probe(name=name, critical=False, run=run, note=note)


def collect_probes(sources: Sources, llm_configured: bool, model: str) -> list[Probe]:
    async def llm() -> None:
        if not llm_configured:
            raise LookupError("переменная LLM_API_KEY не задана")

    remote = (sources.api, sources.prometheus, sources.loki, sources.tempo, sources.mlflow)
    return [
        _probe("llm", llm, note=f"модель {model}, ключ задан, шлюз не опрашивается"),
        _probe("postgres", sources.tasks.check),
        *(
            _probe(source.name, lambda source=source: source.check(source.ready_path))
            for source in remote
        ),
    ]
