import asyncio
import csv
from pathlib import Path

from dota_winprob.agent.errors import SourceUnavailable
from dota_winprob.schemas import MatchState

NAME = "data/tasks.csv"


class MatchFile:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _find(self, match_id: str) -> MatchState | None:
        with self._path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row.pop("match_id") == match_id:
                    return MatchState.model_validate(row)
        return None

    async def find(self, match_id: int) -> MatchState | None:
        try:
            return await asyncio.to_thread(self._find, str(match_id))
        except OSError as error:
            raise SourceUnavailable(NAME, type(error).__name__) from error
