from typing import Any

from dota_winprob.agent.sources.http import HTTPSource


class MLflow(HTTPSource):
    ready_path = "/health"

    async def champion(self, name: str, alias: str) -> dict[str, Any] | None:
        response = await self.send(
            "GET",
            "/api/2.0/mlflow/registered-models/alias",
            params={"name": name, "alias": alias},
        )
        if response.status_code == 404:
            return None
        return self.payload(response)["model_version"]

    async def run_metrics(self, run_id: str) -> dict[str, float]:
        body = await self.json("GET", "/api/2.0/mlflow/runs/get", params={"run_id": run_id})
        metrics = body["run"]["data"].get("metrics") or []
        return {metric["key"]: float(metric["value"]) for metric in metrics}
