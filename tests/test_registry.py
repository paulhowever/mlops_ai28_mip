from types import SimpleNamespace

import mlflow.pyfunc
import pytest

from dota_winprob.clients import registry
from tests.fakes import MODEL_FEATURES, FakePredictor


class FakeSchema:
    def input_names(self):
        return list(MODEL_FEATURES)


class FakePyFuncModel(FakePredictor):
    def __init__(self, run_id, schema):
        super().__init__()
        self.metadata = SimpleNamespace(run_id=run_id, get_input_schema=lambda: schema)


@pytest.fixture
def fake_mlflow(monkeypatch):
    state = SimpleNamespace(
        tracking_uri=None,
        loaded_uri=None,
        model=FakePyFuncModel("run-4", FakeSchema()),
        version=SimpleNamespace(version="4", run_id="run-4"),
    )

    def set_tracking_uri(uri):
        state.tracking_uri = uri

    def load_model(uri):
        state.loaded_uri = uri
        return state.model

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            state.alias = (name, alias)
            return state.version

    monkeypatch.setattr(mlflow, "set_tracking_uri", set_tracking_uri)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", load_model)
    monkeypatch.setattr(registry, "MlflowClient", FakeClient)
    return state


def test_loads_model_by_alias_and_takes_version_from_registry(fake_mlflow, settings):
    model = registry.load_model(settings)

    assert fake_mlflow.tracking_uri == settings.mlflow_tracking_uri
    assert fake_mlflow.loaded_uri == "models:/dota-winprob@champion"
    assert fake_mlflow.alias == ("dota-winprob", "champion")
    assert model.name == "dota-winprob"
    assert model.alias == "champion"
    assert model.version == "4"
    assert model.run_id == "run-4"
    assert model.features == MODEL_FEATURES
    assert model.predictor is fake_mlflow.model


def test_alias_from_settings_is_used(fake_mlflow, settings):
    settings = settings.model_copy(update={"model_alias": "challenger"})

    model = registry.load_model(settings)

    assert fake_mlflow.loaded_uri == "models:/dota-winprob@challenger"
    assert model.alias == "challenger"


def test_alias_moved_during_loading_is_rejected(fake_mlflow, settings):
    fake_mlflow.version = SimpleNamespace(version="5", run_id="run-5")

    with pytest.raises(RuntimeError, match="переехал на версию 5"):
        registry.load_model(settings)


def test_model_without_signature_is_rejected(fake_mlflow, settings):
    fake_mlflow.model = FakePyFuncModel("run-4", None)

    with pytest.raises(ValueError, match="нет сигнатуры"):
        registry.load_model(settings)
