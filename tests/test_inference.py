import logging
import re
import time

import pytest

from dota_winprob.clients import registry
from dota_winprob.services import inference
from tests.fakes import loaded_model, loader_for


async def test_load_model_logs_version(monkeypatch, settings, caplog):
    monkeypatch.setattr(registry, "load_model", loader_for(loaded_model(version="9")))

    with caplog.at_level(logging.INFO, logger=inference.__name__):
        model = await inference.load_model(settings)

    assert model.version == "9"
    [record] = caplog.records
    assert record.getMessage() == "Модель загружена"
    assert record.model_version == "9"
    assert record.model_run_id == "run-4"
    assert record.load_ms >= 0


async def test_load_model_without_tracking_uri(settings, caplog):
    settings = settings.model_copy(update={"mlflow_tracking_uri": None})

    with pytest.raises(inference.ModelUnavailableError, match="MLFLOW_TRACKING_URI не задан"):
        await inference.load_model(settings)

    [record] = caplog.records
    assert record.levelno == logging.ERROR
    assert record.model_uri == "models:/dota-winprob@champion"
    assert record.error == "MLFLOW_TRACKING_URI не задан"


async def test_load_model_is_cut_by_timeout(monkeypatch, settings):
    def hanging(_):
        time.sleep(0.3)
        return loaded_model()

    monkeypatch.setattr(registry, "load_model", hanging)
    settings = settings.model_copy(update={"model_load_timeout": 0.05})

    with pytest.raises(
        inference.ModelUnavailableError, match=re.escape("превышен таймаут загрузки 0.05 с")
    ):
        await inference.load_model(settings)


async def test_load_model_keeps_first_line_of_error(monkeypatch, settings):
    def broken(_):
        raise ConnectionError("mlflow:5000 refused\nдлинный стек ретраев")

    monkeypatch.setattr(registry, "load_model", broken)

    with pytest.raises(inference.ModelUnavailableError) as caught:
        await inference.load_model(settings)

    assert str(caught.value) == "ConnectionError: mlflow:5000 refused"


async def test_load_model_names_error_type_when_message_is_empty(monkeypatch, settings):
    def broken(_):
        raise ConnectionError

    monkeypatch.setattr(registry, "load_model", broken)

    with pytest.raises(inference.ModelUnavailableError, match=r"^ConnectionError$"):
        await inference.load_model(settings)


async def test_load_model_rejects_features_missing_from_request(monkeypatch, settings):
    model = loaded_model(features=("gold_advantage", "towers_destroyed"))
    monkeypatch.setattr(registry, "load_model", loader_for(model))

    with pytest.raises(inference.ModelUnavailableError, match="towers_destroyed"):
        await inference.load_model(settings)
