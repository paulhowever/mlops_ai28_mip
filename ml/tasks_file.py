from pathlib import Path

from ml.data import DEFAULT_MINUTE, FEATURES, build_dataset

OUTPUT = Path("data/tasks.csv")


def main() -> None:
    frame = build_dataset(DEFAULT_MINUTE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame[["match_id", *FEATURES]].to_csv(OUTPUT, index=False)
    print(f"{OUTPUT}: {len(frame)} матчей, минута {DEFAULT_MINUTE}")


if __name__ == "__main__":
    main()
