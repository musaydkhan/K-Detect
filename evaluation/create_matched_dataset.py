
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "datasets" / "processed" / "validation.csv"
)

OUTPUT_DIR = (
    BASE_DIR / "datasets" / "processed" / "length_matching"
)

OUTPUT_FILE = OUTPUT_DIR / "validation_060.csv"

THRESHOLD = 0.60


def create_matched_dataset():

    df = pd.read_csv(INPUT_FILE)

    # Calculate word count
    df["word_count"] = df["text"].fillna("").apply(
        lambda text: len(text.split())
    )

    # Pair human and AI answers from the same question
    paired = df.pivot(
        index="split_group",
        columns="label",
        values="word_count"
    )

    paired.columns = ["human_words", "ai_words"]

    # Calculate relative length similarity
    paired["length_ratio"] = (
        paired[["human_words", "ai_words"]].min(axis=1)
        / paired[["human_words", "ai_words"]].max(axis=1)
    )

    # Select sufficiently similar pairs
    matched_groups = paired[
        paired["length_ratio"] >= THRESHOLD
    ].index

    matched_df = df[
        df["split_group"].isin(matched_groups)
    ].copy()

    # Remove temporary analysis column
    matched_df = matched_df.drop(columns=["word_count"])

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matched_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== Matched Dataset Created =====")

    print("Similarity threshold:", THRESHOLD)
    print("Matched question pairs:", len(matched_groups))
    print("Total samples:", len(matched_df))

    print("\nClass distribution:")
    print(matched_df["label"].value_counts().sort_index())

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    create_matched_dataset()
