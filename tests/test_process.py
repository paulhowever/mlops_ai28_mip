import json
import math
import threading

import pytest
from ml.data import FEATURES

from dota_winprob.schemas import MatchState
from tests.fakes import (
    MATCH_STATE,
    MODEL_FEATURES,
    BrokenPredictor,
    FakePredictor,
    HealthyPool,
    loaded_model,
    loader_for,
    mlflow_unavailable,
    opendota_ok,
)


def records(captured_out, message):
    lines = [json.loads(line) for line in captured_out.splitlines()]
    return [line for line in lines if line["message"] == message]


async def test_returns_probability_and_model_version(make_client):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    response = await client.post("/process", json=MATCH_STATE)

    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["win_probability"] <= 1
    assert body["win_probability"] == pytest.approx(0.73)
    assert body["model_name"] == "dota-winprob"
    assert body["model_version"] == "4"
    assert body["model_alias"] == "champion"


async def test_features_are_ordered_by_model_signature(make_client):
    predictor = FakePredictor()
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        model_loader=loader_for(loaded_model(predictor)),
    )

    await client.post("/process", json=MATCH_STATE)

    [frame] = predictor.frames
    assert tuple(frame.columns) == MODEL_FEATURES
    assert (frame.dtypes == "float64").all()
    assert frame.iloc[0].to_dict() == {name: float(MATCH_STATE[name]) for name in MODEL_FEATURES}


async def test_inference_runs_outside_event_loop_thread(make_client):
    predictor = FakePredictor()
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        model_loader=loader_for(loaded_model(predictor)),
    )

    await client.post("/process", json=MATCH_STATE)

    [thread] = predictor.threads
    assert thread is not threading.current_thread()


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        (
            {key: value for key, value in MATCH_STATE.items() if key != "gold_advantage"},
            "gold_advantage",
        ),
        ({**MATCH_STATE, "radiant_gold": -1}, "radiant_gold"),
        ({**MATCH_STATE, "radiant_gold": "много"}, "radiant_gold"),
        ({**MATCH_STATE, "duration": 2400}, "duration"),
        ({**MATCH_STATE, "xp_advantage": math.nan}, "xp_advantage"),
    ],
    ids=["missing-feature", "negative-gold", "not-a-number", "unknown-field", "nan"],
)
async def test_invalid_payload_gives_422(make_client, payload, field):
    predictor = FakePredictor()
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        model_loader=loader_for(loaded_model(predictor)),
    )

    response = await client.post(
        "/process",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    [error] = response.json()["detail"]
    assert error["loc"] == ["body", field]
    assert "input" not in error
    assert predictor.frames == []


async def test_model_not_loaded_gives_503_and_liveness_stays_green(make_client):
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        model_loader=mlflow_unavailable,
    )

    process = await client.post("/process", json=MATCH_STATE)
    health = await client.get("/api/v1/health")
    liveness = await client.get("/healthz")

    assert process.status_code == 503
    assert process.json()["detail"] == (
        "Модель не загружена: ConnectionError: mlflow.invalid: Name or service not known"
    )
    assert health.status_code == 503
    model = health.json()["components"]["model"]
    assert model["status"] == "unavailable"
    assert model["critical"] is True
    assert "ConnectionError" in model["detail"]
    assert liveness.status_code == 200


async def test_inference_error_gives_500_and_service_stays_alive(make_client, capsys):
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        model_loader=loader_for(loaded_model(BrokenPredictor())),
    )

    response = await client.post("/process", json=MATCH_STATE, headers={"X-Request-ID": "req-7"})
    liveness = await client.get("/healthz")

    assert response.status_code == 500
    assert response.json() == {"detail": "Инференс завершился ошибкой"}
    assert liveness.status_code == 200
    [error] = records(capsys.readouterr().out, "Инференс завершился ошибкой")
    assert error["request_id"] == "req-7"
    assert error["model_version"] == "4"
    assert "ValueError: feature schema mismatch" in error["exception"]


async def test_inference_is_logged_with_model_version_and_latency(make_client, capsys):
    client = await make_client(pool=HealthyPool(), opendota_handler=opendota_ok)

    await client.post("/process", json=MATCH_STATE, headers={"X-Request-ID": "req-8"})

    [record] = records(capsys.readouterr().out, "Инференс выполнен")
    assert record["request_id"] == "req-8"
    assert record["model_name"] == "dota-winprob"
    assert record["model_version"] == "4"
    assert record["model_alias"] == "champion"
    assert record["inference_ms"] >= 0
    assert record["win_probability"] == pytest.approx(0.73)


async def test_health_reports_loaded_model_version(make_client):
    client = await make_client(
        pool=HealthyPool(),
        opendota_handler=opendota_ok,
        model_loader=loader_for(loaded_model(version="7")),
    )

    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    model = response.json()["components"]["model"]
    assert model["status"] == "healthy"
    assert model["version"] == "7"
    assert model["detail"] == "models:/dota-winprob@champion"


def test_request_schema_matches_training_features():
    assert tuple(MatchState.model_fields) == FEATURES
