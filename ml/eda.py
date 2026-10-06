import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from ml.data import (
    DEFAULT_MINUTE,
    FEATURES,
    MINUTES,
    TARGET,
    build_panel,
    log_dataset,
    select_minute,
)

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "dota-winprob")
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
ARTIFACT_DIR = "eda"


def class_balance(panel: pd.DataFrame) -> plt.Figure:
    counts = panel[panel["minute"] == DEFAULT_MINUTE][TARGET].value_counts().sort_index()
    figure, axes = plt.subplots(figsize=(5, 4))
    axes.bar(["dire win", "radiant win"], counts.to_numpy(), color=["#b45309", "#047857"])
    axes.set_title(f"Баланс классов на {DEFAULT_MINUTE}-й минуте")
    axes.set_ylabel("матчей")
    for index, value in enumerate(counts.to_numpy()):
        axes.text(index, value, f"{value:,}", ha="center", va="bottom")
    figure.tight_layout()
    return figure


def advantage_distribution(panel: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    subset = panel[panel["minute"] == DEFAULT_MINUTE]
    for ax, column, title in zip(
        axes,
        ("gold_advantage", "xp_advantage"),
        ("Преимущество по золоту", "Преимущество по опыту"),
        strict=True,
    ):
        ax.hist(subset[column], bins=80, color="#1d4ed8", alpha=0.85)
        ax.axvline(0, color="#991b1b", linestyle="--", linewidth=1)
        ax.set_title(f"{title}, минута {DEFAULT_MINUTE}")
        ax.set_xlabel("radiant минус dire")
        ax.set_ylabel("матчей")
    figure.tight_layout()
    return figure


def winrate_by_advantage(panel: pd.DataFrame, bins: int = 20) -> plt.Figure:
    subset = panel[panel["minute"] == DEFAULT_MINUTE].copy()
    subset["bucket"] = pd.qcut(subset["gold_advantage"], q=bins, duplicates="drop")
    grouped = subset.groupby("bucket", observed=True).agg(
        centre=("gold_advantage", "median"),
        winrate=(TARGET, "mean"),
        size=(TARGET, "size"),
    )
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.plot(grouped["centre"], grouped["winrate"], marker="o", color="#047857")
    axes.axhline(0.5, color="#991b1b", linestyle="--", linewidth=1)
    axes.axvline(0, color="#6b7280", linestyle=":", linewidth=1)
    axes.set_title(f"Доля побед Radiant от преимущества по золоту, минута {DEFAULT_MINUTE}")
    axes.set_xlabel("преимущество по золоту")
    axes.set_ylabel("доля побед Radiant")
    axes.set_ylim(0, 1)
    figure.tight_layout()
    return figure


def signal_growth(panel: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    rows = []
    for minute in sorted(panel["minute"].unique()):
        subset = panel[panel["minute"] == minute]
        rows.append(
            {
                "minute": int(minute),
                "matches": len(subset),
                "gold_corr": subset["gold_advantage"].corr(subset[TARGET]),
                "xp_corr": subset["xp_advantage"].corr(subset[TARGET]),
                "lh_corr": subset["lh_advantage"].corr(subset[TARGET]),
            }
        )
    table = pd.DataFrame(rows)

    figure, axes = plt.subplots(figsize=(7, 4.5))
    for column, label in (
        ("gold_corr", "золото"),
        ("xp_corr", "опыт"),
        ("lh_corr", "добивания"),
    ):
        axes.plot(table["minute"], table[column], marker="o", label=label)
    axes.set_title("Связь преимущества с исходом по ходу матча")
    axes.set_xlabel("минута")
    axes.set_ylabel("корреляция с победой Radiant")
    axes.set_xticks(table["minute"].tolist())
    axes.legend()
    figure.tight_layout()
    return figure, table


def correlation_heatmap(panel: pd.DataFrame) -> plt.Figure:
    subset = panel[panel["minute"] == DEFAULT_MINUTE][[*FEATURES, TARGET]]
    matrix = subset.corr()
    figure, axes = plt.subplots(figsize=(8, 7))
    image = axes.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axes.set_xticks(range(len(matrix.columns)))
    axes.set_xticklabels(matrix.columns, rotation=45, ha="right")
    axes.set_yticks(range(len(matrix.columns)))
    axes.set_yticklabels(matrix.columns)
    axes.set_title(f"Корреляции признаков, минута {DEFAULT_MINUTE}")
    figure.colorbar(image, ax=axes, shrink=0.8)
    figure.tight_layout()
    return figure


def survival_table(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = panel["match_id"].nunique()
    for minute in sorted(panel["minute"].unique()):
        subset = panel[panel["minute"] == minute]
        rows.append(
            {
                "minute": int(minute),
                "matches": len(subset),
                "share_of_total": round(len(subset) / total, 4),
                "radiant_winrate": round(float(subset[TARGET].mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def build_summary(panel: pd.DataFrame, survival: pd.DataFrame, signal: pd.DataFrame) -> str:
    primary = panel[panel["minute"] == DEFAULT_MINUTE]
    lines = [
        "# EDA: вероятность победы по состоянию матча",
        "",
        f"Источник: датасет Kaggle `devinanzelmo/dota-2-matches`, срезы на минутах {list(MINUTES)}.",
        f"Основной срез для обучения: минута {DEFAULT_MINUTE}, {len(primary):,} матчей.",
        "",
        "## Сколько матчей доживает до среза",
        "",
        survival.to_markdown(index=False),
        "",
        "## Насколько преимущество связано с исходом",
        "",
        signal.round(4).to_markdown(index=False),
        "",
        "## Описательная статистика основного среза",
        "",
        primary[list(FEATURES)].describe().round(2).to_markdown(),
        "",
        "## Выводы",
        "",
        "- Классы сбалансированы, доля побед Radiant близка к половине, поэтому accuracy не вводит в заблуждение, а ROC-AUC применим напрямую.",
        "- Преимущество по золоту и опыту монотонно связано с исходом: чем позже срез, тем сильнее связь.",
        "- Матчи, закончившиеся раньше среза, исключаются, поэтому выборка на поздних минутах смещена в сторону затяжных игр.",
        "- Признаки по золоту, опыту и добиваниям сильно скоррелированы между собой, что ограничивает пользу линейной модели без регуляризации.",
    ]
    return "\n".join(lines)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    panel = build_panel()
    survival = survival_table(panel)
    signal_figure, signal = signal_growth(panel)

    with mlflow.start_run(run_name="eda"):
        log_dataset(select_minute(panel, DEFAULT_MINUTE), context="eda")

        mlflow.log_params(
            {
                "minutes": list(MINUTES),
                "primary_minute": DEFAULT_MINUTE,
                "features": list(FEATURES),
            }
        )
        mlflow.log_metrics(
            {
                "matches_total": float(panel["match_id"].nunique()),
                "matches_primary_minute": float(len(panel[panel["minute"] == DEFAULT_MINUTE])),
                "radiant_winrate": float(panel[panel["minute"] == DEFAULT_MINUTE][TARGET].mean()),
                "gold_corr_primary": float(
                    signal.loc[signal["minute"] == DEFAULT_MINUTE, "gold_corr"].iloc[0]
                ),
            }
        )

        mlflow.log_figure(class_balance(panel), f"{ARTIFACT_DIR}/class_balance.png")
        mlflow.log_figure(
            advantage_distribution(panel), f"{ARTIFACT_DIR}/advantage_distribution.png"
        )
        mlflow.log_figure(
            winrate_by_advantage(panel), f"{ARTIFACT_DIR}/winrate_by_gold_advantage.png"
        )
        mlflow.log_figure(signal_figure, f"{ARTIFACT_DIR}/signal_growth.png")
        mlflow.log_figure(correlation_heatmap(panel), f"{ARTIFACT_DIR}/correlation_heatmap.png")

        mlflow.log_text(build_summary(panel, survival, signal), f"{ARTIFACT_DIR}/eda_summary.md")

        plt.close("all")


if __name__ == "__main__":
    main()
