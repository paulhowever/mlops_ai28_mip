import asyncio
import contextvars
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from opentelemetry import trace

from dota_winprob import instruments
from dota_winprob.clients.registry import LoadedModel
from dota_winprob.schemas import MatchState

tracer = trace.get_tracer(__name__)

INFERENCE_SPAN = "inference"
ATTEMPT_ATTRIBUTE = "attempt"


class Scorer:
    def __init__(self, model: LoadedModel, concurrency: int, delay: float) -> None:
        self.model = model
        self.concurrency = concurrency
        self.delay = delay
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="inference")

    def _predict(self, features: MatchState) -> float:
        if self.delay:
            time.sleep(self.delay)
        frame = pd.DataFrame([features.model_dump()], columns=list(self.model.features))
        output = self.model.predictor.predict(frame)
        return float(np.asarray(output)[0, 1])

    def _score(self, features: MatchState, attempt: int) -> float:
        attributes = {
            "model.name": self.model.name,
            "model.alias": self.model.alias,
            "model.version": self.model.version,
            ATTEMPT_ATTRIBUTE: attempt,
        }
        with tracer.start_as_current_span(INFERENCE_SPAN, attributes=attributes):
            started = time.perf_counter()
            try:
                return self._predict(features)
            finally:
                instruments.inference_duration.record(time.perf_counter() - started)

    async def score(self, features: MatchState, attempt: int = 1) -> float:
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        return await loop.run_in_executor(self._pool, context.run, self._score, features, attempt)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
