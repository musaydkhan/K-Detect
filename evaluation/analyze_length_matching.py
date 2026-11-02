
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "validation.csv"
)


def analyze_length_matching():

    df = pd.read_csv(VALIDATION_FILE)

    # Calculate word count
    df["word_count"] = df["text"].fillna("").apply(
        lambda text: len(text.split())
    )

    # Group human and AI answers by their original question
    paired = df.pivot(
        index="split_group",
        columns="label",
        values="word_count"
    )

    # Keep groups containing both human and AI answers
    paired = paired.dropna()

    paired.columns = ["human_words", "ai_words"]

    # Calculate length similarity
    paired["length_ratio"] = (
        paired[["human_words", "ai_words"]].min(axis=1)
        / paired[["human_words", "ai_words"]].max(axis=1)
    )

    print("\n===== K-Detect: Length Matching Analysis =====")

    print("Original validation samples:", len(df))
    print("Complete human-AI pairs:", len(paired))

    print("\nWord count statistics:")
    print(
        df.groupby("label")["word_count"]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )

    print("\nMatched pairs at different similarity thresholds:")

    for threshold in [0.90, 0.80, 0.70, 0.60]:

        matched = paired[
            paired["length_ratio"] >= threshold
        ]

        print(
            f"Threshold {threshold:.2f}: "
            f"{len(matched)} pairs "
            f"({len(matched) * 2} samples)"
        )

    print("\nLength ratio summary:")
    print(paired["length_ratio"].describe().round(3))


if __name__ == "__main__":
    analyze_length_matching()
