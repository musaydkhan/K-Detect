import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "processed" / "hc3_wiki_processed.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "hc3_wiki_100plus.csv"

MIN_WORDS = 100


def filter_dataset():

    df = pd.read_csv(INPUT_FILE)

    # Calculate word count
    df["word_count"] = df["text"].fillna("").apply(
        lambda text: len(text.split())
    )

    # Keep samples with at least 100 words
    filtered_df = df[
        df["word_count"] >= MIN_WORDS
    ].copy()

    # Remove temporary word-count column
    filtered_df.drop(columns=["word_count"], inplace=True)

    # Save separate dataset
    filtered_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== Dataset Filtering Completed =====")

    print("Original samples:", len(df))
    print("Filtered samples:", len(filtered_df))
    print("Removed samples:", len(df) - len(filtered_df))

    print("\nClass distribution:")
    print(filtered_df["label"].value_counts().sort_index())

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    filter_dataset()