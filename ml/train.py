import os
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data import DEFAULT_MINUTE, FEATURES, TARGET, build_dataset, log_dataset

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "dota-winprob")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MODEL_NAME = "dota-winprob"
CHAMPION_ALIAS = "champion"
PRIMARY_METRIC = "roc_auc"
SELECTION_METRIC = "cv_roc_auc"
ARTIFACT_DIR = "diagnostics"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
THRESHOLD = 0.5
CALIBRATION_BINS = 10

SEARCH_SPACE = {
    "model__learning_rate": [0.03, 0.1],
    "model__max_leaf_nodes": [2, 3, 4, 8, 15],
    "model__min_samples_leaf": [200, 1000],
    "model__max_iter": [1000],
}

SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]
CLASS_LABELS = ("dire win", "radiant win")


@dataclass(frozen=True)
class Outcome:
    run_name: str
    run_id: str
    model_uri: str
    metrics: dict[str, float]


def baseline() -> Pipeline:
    return Pipeline([("model", LogisticRegression())])


def logreg_scaled() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )


def gradient_boosting() -> Pipeline:
    return Pipeline([("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE))])


def folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def gradient_boosting_tuned() -> GridSearchCV:
    return GridSearchCV(
        gradient_boosting(),
        SEARCH_SPACE,
        scoring="roc_auc",
        cv=folds(),
        n_jobs=-1,
    )


CANDIDATES: tuple[tuple[str, Callable[[], Pipeline | GridSearchCV]], ...] = (
    ("baseline", baseline),
    ("logreg-scaled", logreg_scaled),
    ("gradient-boosting", gradient_boosting),
    ("gradient-boosting-tuned", gradient_boosting_tuned),
)


def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[list(FEATURES)].astype("float64")


def describe(model: Pipeline) -> dict[str, object]:
    estimator = model[-1]
    preprocessing = [name for name, _ in model.steps[:-1]]
    return {
        "model_class": type(estimator).__name__,
        "preprocessing": "+".join(preprocessing) or "none",
        "minute": DEFAULT_MINUTE,
        "features": list(FEATURES),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        **{f"model__{key}": value for key, value in estimator.get_params().items()},
    }


def score(target: pd.Series, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(target, proba)),
        "log_loss": float(log_loss(target, proba)),
        "brier_score": float(brier_score_loss(target, proba)),
        "accuracy": float(accuracy_score(target, proba >= THRESHOLD)),
    }


def roc_figure(target: pd.Series, proba: np.ndarray, run_name: str) -> plt.Figure:
    fpr, tpr, _ = roc_curve(target, proba)
    figure, axes = plt.subplots(figsize=(5.5, 5))
    axes.plot(
        fpr,
        tpr,
        color="#1d4ed8",
        linewidth=2,
        label=f"{run_name}, ROC-AUC {roc_auc_score(target, proba):.4f}",
    )
    axes.plot(
        [0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1, label="случайный прогноз"
    )
    axes.set_title("ROC-кривая на тестовой выборке")
    axes.set_xlabel("доля ложных срабатываний, FPR")
    axes.set_ylabel("доля верных срабатываний, TPR")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend(loc="lower right")
    figure.tight_layout()
    return figure


def confusion_figure(target: pd.Series, proba: np.ndarray) -> plt.Figure:
    matrix = confusion_matrix(target, proba >= THRESHOLD)
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    figure, axes = plt.subplots(figsize=(5.5, 4.5))
    image = axes.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for row in range(2):
        for column in range(2):
            axes.text(
                column,
                row,
                f"{matrix[row, column]:,}\n{shares[row, column]:.1%}",
                ha="center",
                va="center",
                color="white" if shares[row, column] > 0.5 else "#111827",
            )
    axes.set_xticks([0, 1], CLASS_LABELS)
    axes.set_yticks([0, 1], CLASS_LABELS)
    axes.set_xlabel("предсказание")
    axes.set_ylabel("факт")
    axes.set_title(f"Матрица ошибок при пороге {THRESHOLD}")
    figure.colorbar(image, ax=axes, shrink=0.8, label="доля от фактического класса")
    figure.tight_layout()
    return figure


def calibration_figure(target: pd.Series, proba: np.ndarray) -> plt.Figure:
    observed, predicted = calibration_curve(
        target, proba, n_bins=CALIBRATION_BINS, strategy="quantile"
    )
    figure, axes = plt.subplots(figsize=(5.5, 5))
    axes.plot(
        [0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1, label="идеальная калибровка"
    )
    axes.plot(
        predicted,
        observed,
        marker="o",
        color="#047857",
        linewidth=2,
        label=f"модель, Brier {brier_score_loss(target, proba):.4f}",
    )
    axes.set_title(f"Калибровочная кривая, {CALIBRATION_BINS} квантильных бинов")
    axes.set_xlabel("предсказанная вероятность победы Radiant")
    axes.set_ylabel("фактическая доля побед Radiant")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.legend(loc="upper left")
    figure.tight_layout()
    return figure


def probability_figure(target: pd.Series, proba: np.ndarray) -> plt.Figure:
    bins = np.linspace(0, 1, 41)
    actual = target.to_numpy()
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for value, color, style in ((1, "#047857", "-"), (0, "#b45309", "--")):
        axes.hist(
            proba[actual == value],
            bins=bins,
            histtype="step",
            linewidth=2,
            linestyle=style,
            color=color,
            label=f"факт: {CLASS_LABELS[value]}",
        )
    axes.axvline(THRESHOLD, color="#991b1b", linestyle=":", linewidth=1, label=f"порог {THRESHOLD}")
    axes.set_title("Предсказанные вероятности по фактическому исходу")
    axes.set_xlabel("предсказанная вероятность победы Radiant")
    axes.set_ylabel("матчей")
    axes.legend()
    figure.tight_layout()
    return figure


def permutation_figure(model: Pipeline, frame: pd.DataFrame, target: pd.Series) -> plt.Figure:
    result = permutation_importance(
        model,
        frame,
        target,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    order = np.argsort(result.importances_mean)
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.barh(
        np.array(FEATURES)[order],
        result.importances_mean[order],
        xerr=result.importances_std[order],
        color="#1d4ed8",
    )
    axes.set_title("Перестановочная важность признаков на тестовой выборке")
    axes.set_xlabel("падение ROC-AUC при перемешивании признака")
    figure.tight_layout()
    return figure


def coefficient_figure(model: Pipeline) -> plt.Figure:
    coefficients = model[-1].coef_[0]
    order = np.argsort(np.abs(coefficients))
    scale = "стандартизованных" if "scaler" in model.named_steps else "сырых"
    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.barh(
        np.array(FEATURES)[order],
        coefficients[order],
        color=["#047857" if value > 0 else "#b45309" for value in coefficients[order]],
    )
    axes.axvline(0, color="#6b7280", linewidth=1)
    axes.set_title(f"Коэффициенты логистической регрессии на {scale} признаках")
    axes.set_xlabel("вес в логите: больше нуля — за Radiant, меньше — за Dire")
    figure.tight_layout()
    return figure


def log_diagnostics(
    model: Pipeline, frame: pd.DataFrame, target: pd.Series, proba: np.ndarray, run_name: str
) -> None:
    mlflow.log_figure(roc_figure(target, proba, run_name), f"{ARTIFACT_DIR}/roc_curve.png")
    mlflow.log_figure(confusion_figure(target, proba), f"{ARTIFACT_DIR}/confusion_matrix.png")
    mlflow.log_figure(calibration_figure(target, proba), f"{ARTIFACT_DIR}/calibration_curve.png")
    mlflow.log_figure(
        probability_figure(target, proba), f"{ARTIFACT_DIR}/probability_distribution.png"
    )
    mlflow.log_figure(
        permutation_figure(model, frame, target), f"{ARTIFACT_DIR}/permutation_importance.png"
    )
    if isinstance(model[-1], LogisticRegression):
        mlflow.log_figure(coefficient_figure(model), f"{ARTIFACT_DIR}/logreg_coefficients.png")
    plt.close("all")


def cross_validate(
    estimator: Pipeline | GridSearchCV, frame: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    scores = cross_val_score(estimator, frame, target, scoring="roc_auc", cv=folds(), n_jobs=-1)
    return {"cv_roc_auc": float(scores.mean()), "cv_roc_auc_std": float(scores.std())}


def log_search(search: GridSearchCV) -> None:
    results = search.cv_results_
    for index, params in enumerate(results["params"]):
        with mlflow.start_run(run_name=f"trial-{index:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "cv_roc_auc": float(results["mean_test_score"][index]),
                    "cv_roc_auc_std": float(results["std_test_score"][index]),
                }
            )
    mlflow.log_params({"search": "grid", "search_space": SEARCH_SPACE})
    mlflow.log_metric("search_best_cv_roc_auc", float(search.best_score_))


def train_candidate(
    run_name: str,
    factory: Callable[[], Pipeline | GridSearchCV],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Outcome:
    train_features, train_target = features(train), train[TARGET]
    test_features, test_target = features(test), test[TARGET]

    with mlflow.start_run(run_name=run_name) as run:
        training = log_dataset(train, context="training")
        evaluation = log_dataset(test, context="evaluation")

        validation = cross_validate(factory(), train_features, train_target)
        estimator = factory().fit(train_features, train_target)
        if isinstance(estimator, GridSearchCV):
            log_search(estimator)
            model = estimator.best_estimator_
        else:
            model = estimator
        mlflow.log_params({**describe(model), "cv_folds": CV_FOLDS})

        train_proba = model.predict_proba(train_features)
        test_proba = model.predict_proba(test_features)[:, 1]

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(train_features, train_proba),
            input_example=train_features.head(5),
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        test_metrics = score(test_target, test_proba)
        train_metrics = {
            **validation,
            "train_roc_auc": float(roc_auc_score(train_target, train_proba[:, 1])),
        }
        mlflow.log_metrics(test_metrics, model_id=model_info.model_id, dataset=evaluation)
        mlflow.log_metrics(train_metrics, model_id=model_info.model_id, dataset=training)
        mlflow.log_metric("n_iter", float(np.max(model[-1].n_iter_)))

        log_diagnostics(model, test_features, test_target, test_proba, run_name)

    return Outcome(
        run_name, run.info.run_id, model_info.model_uri, {**train_metrics, **test_metrics}
    )


def register(client: MlflowClient, outcome: Outcome) -> str:
    version = mlflow.register_model(
        outcome.model_uri,
        MODEL_NAME,
        tags={
            "run_name": outcome.run_name,
            SELECTION_METRIC: f"{outcome.metrics[SELECTION_METRIC]:.4f}",
            PRIMARY_METRIC: f"{outcome.metrics[PRIMARY_METRIC]:.4f}",
        },
    )
    client.update_model_version(
        MODEL_NAME,
        version.version,
        description=(
            f"{outcome.run_name}: ROC-AUC {outcome.metrics['cv_roc_auc']:.4f} "
            f"± {outcome.metrics['cv_roc_auc_std']:.4f} на кросс-валидации, "
            f"на отложенной выборке ROC-AUC {outcome.metrics['roc_auc']:.4f}, "
            f"Brier {outcome.metrics['brier_score']:.4f}"
        ),
    )
    return version.version


def publish(outcomes: list[Outcome]) -> str:
    client = MlflowClient()
    versions = [register(client, outcome) for outcome in outcomes]
    best = max(range(len(outcomes)), key=lambda index: outcomes[index].metrics[SELECTION_METRIC])
    client.update_registered_model(
        MODEL_NAME,
        description=(
            f"Вероятность победы Radiant по состоянию матча на {DEFAULT_MINUTE}-й минуте. "
            f"Рабочая версия помечена алиасом {CHAMPION_ALIAS}, она выбирается по ROC-AUC "
            "на кросс-валидации внутри обучающей выборки."
        ),
    )
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, versions[best])
    return versions[best]


def check_champion(test: pd.DataFrame) -> float:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    proba = np.asarray(model.predict(features(test)))[:, 1]
    return float(roc_auc_score(test[TARGET], proba))


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    train, test = split(build_dataset())
    outcomes = [train_candidate(name, factory, train, test) for name, factory in CANDIDATES]
    version = publish(outcomes)

    summary = pd.DataFrame([{"run": outcome.run_name, **outcome.metrics} for outcome in outcomes])
    print(summary.round(4).to_markdown(index=False))
    print(
        f"{MODEL_NAME}@{CHAMPION_ALIAS} -> версия {version}, "
        f"ROC-AUC при загрузке по алиасу {check_champion(test):.4f}"
    )


if __name__ == "__main__":
    main()
