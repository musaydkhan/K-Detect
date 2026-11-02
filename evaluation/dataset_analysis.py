import pandas as pd


DATASET_FILE = "datasets/processed/hc3_wiki_processed.csv"


def analyze_dataset():

    df = pd.read_csv(DATASET_FILE)

    # Calculate word count for every sample
    df["word_count"] = df["text"].fillna("").apply(
        lambda text: len(text.split())
    )

    print("\n===== K-Detect Dataset Analysis =====")

    print("\nOverall dataset:")
    print("Total samples:", len(df))

    print("\nWord count statistics by class:")

    statistics = df.groupby("label")["word_count"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )

    statistics.index = ["Human-written", "AI-generated"]

    print(statistics.round(2))

    print("\nAverage word count by class:")
    print(statistics["mean"])

    print("\nSamples under 100 words:")
    print(
        df.groupby("label")["word_count"]
        .apply(lambda values: (values < 100).sum())
    )


if __name__ == "__main__":
    analyze_dataset()