import contextlib
import json
import re
from urllib.parse import parse_qs

import httpx
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from dota_winprob.agent.sources.dead_letters import DeadTask, Inspection, Returned

TEST_KEY = "unit-test-key-0000"


class FakeBackends:
    def __init__(self):
        self.requests = []
        self.down = set()
        self.status = {}
        self.metrics = {}
        self.ranges = {}
        self.log_streams = []
        self.found_traces = []
        self.traces = {}
        self.champion = {
            "version": "4",
            "run_id": "run-4",
            "description": "gradient-boosting-tuned",
        }
        self.run_metrics = {"roc_auc": 0.7123456, "cv_roc_auc": 0.7101, "n_iter": 80.0}
        self.submit = [httpx.Response(202, json={"task_id": "abc123", "status": "pending"})]
        self.views = []

    def __call__(self, request):
        host = request.url.host.split(".")[0]
        self.requests.append(request)
        if host in self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path in {"/healthz", "/ready", "/-/ready", "/health"}:
            return httpx.Response(200)
        if host in self.status:
            return httpx.Response(self.status[host], text="boom password=hunter22")
        return getattr(self, f"_{host}")(request)

    def calls(self, host, path=""):
        return [
            request
            for request in self.requests
            if request.url.host == f"{host}.test" and path in request.url.path
        ]

    def form(self, host):
        return {
            key: value[0] for key, value in parse_qs(self.calls(host)[-1].content.decode()).items()
        }

    def _prometheus(self, request):
        query = parse_qs(request.content.decode())["query"][0]
        if request.url.path.endswith("query_range"):
            values = next((v for key, v in self.ranges.items() if key in query), None)
            result = [] if values is None else [{"metric": {}, "values": values}]
        else:
            value = next((v for key, v in self.metrics.items() if key in query), None)
            result = [] if value is None else [{"metric": {}, "value": [1, value]}]
        return httpx.Response(200, json={"status": "success", "data": {"result": result}})

    def _loki(self, request):
        return httpx.Response(200, json={"data": {"result": self.log_streams}})

    def _tempo(self, request):
        if request.url.path == "/api/search":
            return httpx.Response(200, json={"traces": self.found_traces})
        trace = self.traces.get(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(404) if trace is None else httpx.Response(200, json=trace)

    def _mlflow(self, request):
        if request.url.path.endswith("/alias"):
            if self.champion is None:
                return httpx.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST"})
            return httpx.Response(200, json={"model_version": self.champion})
        metrics = [{"key": key, "value": value} for key, value in self.run_metrics.items()]
        return httpx.Response(200, json={"run": {"data": {"metrics": metrics}}})

    def _api(self, request):
        if request.method == "POST":
            return self.submit.pop(0) if len(self.submit) > 1 else self.submit[0]
        if not self.views:
            return httpx.Response(404, json={"detail": "Задача не найдена"})
        view = self.views.pop(0) if len(self.views) > 1 else self.views[0]
        return httpx.Response(200, json=view)


def dead_task(task_id, digest="a", source="batch", reason="rejected", attempts=3):
    return DeadTask((task_id, digest), source, reason, attempts)


class FakeDeadLetters:
    def __init__(self):
        self.tasks = [dead_task("batch:1"), dead_task("batch:2")]
        self.garbage = 1
        self.error = None
        self.inspected = 0
        self.requeued = []
        self.outcome = None
        self.gate = None

    async def inspect(self):
        self.inspected += 1
        if self.error is not None:
            raise self.error
        depth = len(self.tasks) + self.garbage
        return Inspection(depth, depth, self.garbage, list(self.tasks))

    async def requeue(self, planned):
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        self.requeued.append(planned)
        if self.outcome is not None:
            return self.outcome
        return Returned([task_id for task_id, _ in planned.elements()], left=self.garbage)


class Connection:
    def __init__(self, pool):
        self.pool = pool

    @contextlib.asynccontextmanager
    async def transaction(self, **options):
        self.pool.transactions.append(options)
        yield

    async def fetch(self, query, *args):
        if self.pool.error is not None:
            raise self.pool.error
        self.pool.queries.append((query, args))
        if "count(*)" in query:
            return self.pool.totals
        return self.pool.rows


class ReadOnlyPool:
    def __init__(self):
        self.transactions = []
        self.queries = []
        self.rows = []
        self.totals = []
        self.error = None
        self.closed = False

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield Connection(self)

    async def close(self):
        self.closed = True


def task_row(**overrides):
    row = {
        "task_id": "batch:566",
        "source": "batch",
        "status": "done",
        "win_probability": 0.6234,
        "model_version": "4",
        "error": None,
        "attempts": 1,
        "worker": "worker-1",
        "published_at": None,
        "completed_at": None,
        "created_at": None,
    }
    return row | overrides


def done_view(probability=0.6234, task_id="abc123"):
    return {
        "task_id": task_id,
        "status": "done",
        "win_probability": probability,
        "model_version": "4",
        "attempts": 1,
    }


def stream(*lines, service="dota-worker", level="ERROR"):
    return {
        "stream": {"service": service, "level": level},
        "values": [[str(1_000 + index), line] for index, line in enumerate(lines)],
    }


def log_line(message, **fields):
    return json.dumps({"level": "ERROR", "service": "dota-worker", "message": message, **fields})


def tool_outputs(messages):
    return [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def payload(text):
    body = re.fullmatch(
        r"<untrusted_tool_output [^>]*>\n(.*)\n</untrusted_tool_output>", text, re.DOTALL
    )
    return json.loads(body[1])


def call(tool, /, **arguments):
    return ModelResponse(parts=[ToolCallPart(tool, arguments)])


def say(text):
    return ModelResponse(parts=[TextPart(text)])


def script(*turns):
    remaining = list(turns)

    def respond(messages, info):
        turn = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return turn(messages, info) if callable(turn) else turn

    return respond
