import os
from pathlib import Path

import mlflow
import pandas as pd

RAW_DIR = Path(os.environ.get("DATA_RAW_DIR", "data/raw"))
MATCH_FILE = RAW_DIR / "match.csv"
PLAYER_TIME_FILE = RAW_DIR / "player_time.csv"

DATASET_NAME = "dota-2-matches"
DATASET_SOURCE = "https://www.kaggle.com/datasets/devinanzelmo/dota-2-matches"

TARGET = "radiant_win"
MINUTES = (5, 10, 15, 20)
DEFAULT_MINUTE = 10

RADIANT_SLOTS = (0, 1, 2, 3, 4)
DIRE_SLOTS = (128, 129, 130, 131, 132)
STATS = ("gold", "xp", "lh")

FEATURES = (
    "radiant_gold",
    "dire_gold",
    "gold_advantage",
    "radiant_xp",
    "dire_xp",
    "xp_advantage",
    "radiant_lh",
    "dire_lh",
    "lh_advantage",
)


def _slot_columns(stat: str, slots: tuple[int, ...]) -> list[str]:
    return [f"{stat}_t_{slot}" for slot in slots]


def read_snapshots(minutes: tuple[int, ...] = MINUTES, chunksize: int = 200_000) -> pd.DataFrame:
    wanted = {minute * 60 for minute in minutes}
    needed = ["match_id", "times"]
    for stat in STATS:
        needed += _slot_columns(stat, RADIANT_SLOTS) + _slot_columns(stat, DIRE_SLOTS)

    parts = []
    for chunk in pd.read_csv(PLAYER_TIME_FILE, usecols=needed, chunksize=chunksize):
        parts.append(chunk[chunk["times"].isin(wanted)])
    return pd.concat(parts, ignore_index=True)


def read_matches() -> pd.DataFrame:
    return pd.read_csv(MATCH_FILE, usecols=["match_id", "radiant_win", "duration"])


def aggregate_teams(snapshots: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "match_id": snapshots["match_id"],
            "times": snapshots["times"],
        }
    )
    for stat in STATS:
        radiant = snapshots[_slot_columns(stat, RADIANT_SLOTS)].sum(axis=1)
        dire = snapshots[_slot_columns(stat, DIRE_SLOTS)].sum(axis=1)
        frame[f"radiant_{stat}"] = radiant
        frame[f"dire_{stat}"] = dire
        frame[f"{stat}_advantage"] = radiant - dire
    frame["minute"] = frame["times"] // 60
    return frame


def build_panel(minutes: tuple[int, ...] = MINUTES) -> pd.DataFrame:
    matches = read_matches()
    snapshots = read_snapshots(minutes)
    teams = aggregate_teams(snapshots)
    merged = teams.merge(matches, on="match_id", how="inner")
    merged = merged[merged["duration"] >= merged["minute"] * 60]
    merged[TARGET] = merged[TARGET].astype(int)
    return merged[["match_id", "minute", *FEATURES, TARGET]].reset_index(drop=True)


def select_minute(panel: pd.DataFrame, minute: int) -> pd.DataFrame:
    rows = panel["minute"] == minute
    return panel.loc[rows, ["match_id", *FEATURES, TARGET]].reset_index(drop=True)


def build_dataset(minute: int = DEFAULT_MINUTE) -> pd.DataFrame:
    return select_minute(build_panel((minute,)), minute)


def log_dataset(frame: pd.DataFrame, context: str, minute: int = DEFAULT_MINUTE) -> None:
    dataset = mlflow.data.from_pandas(
        frame,
        source=DATASET_SOURCE,
        name=f"{DATASET_NAME}-minute-{minute}",
        targets=TARGET,
    )
    mlflow.log_input(dataset, context=context)
