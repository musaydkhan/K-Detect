import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "raid_abstracts.csv"
)


def inspect_conditions():

    df = pd.read_csv(INPUT_FILE)

    print("===== DECODING DISTRIBUTION =====")

    print(
        df.groupby(["model", "decoding"], dropna=False)
        .size()
        .to_string()
    )

    print("\n===== REPETITION PENALTY DISTRIBUTION =====")

    print(
        df.groupby(
            ["model", "repetition_penalty"],
            dropna=False
        )
        .size()
        .to_string()
    )

    print("\n===== GENERATION CONDITIONS PER MODEL =====")

    conditions = (
        df.groupby(
            ["model", "decoding", "repetition_penalty"],
            dropna=False
        )
        .size()
        .reset_index(name="records")
    )

    print(conditions.to_string(index=False))

    print("\n===== RECORDS PER SOURCE-MODEL =====")

    counts = (
        df.groupby(["source_id", "model"])
        .size()
    )

    print(counts.value_counts().sort_index().to_string())

    print("\n===== COMPLETE =====")


if __name__ == "__main__":
    inspect_conditions()