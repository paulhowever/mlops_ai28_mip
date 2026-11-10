import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from dota_winprob.clients.registry import LoadedModel
from dota_winprob.schemas import MatchState


class Scorer:
    def __init__(self, model: LoadedModel, concurrency: int, delay: float) -> None:
        self.model = model
        self.concurrency = concurrency
        self.delay = delay
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="inference")

    def _score(self, features: MatchState) -> float:
        if self.delay:
            time.sleep(self.delay)
        frame = pd.DataFrame([features.model_dump()], columns=list(self.model.features))
        output = self.model.predictor.predict(frame)
        return float(np.asarray(output)[0, 1])

    async def score(self, features: MatchState) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._score, features)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
