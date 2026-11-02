import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "processed"
OUTPUT_DIR = SOURCE_DIR / "length_experiment"

MIN_WORDS = 100


def split_dataset():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {}

    # Filter each original split separately
    for split_name in ["train", "validation", "test"]:

        input_file = SOURCE_DIR / f"{split_name}.csv"
        df = pd.read_csv(input_file)

        df["word_count"] = df["text"].fillna("").apply(
            lambda text: len(text.split())
        )

        filtered_df = df[
            df["word_count"] >= MIN_WORDS
        ].copy()

        filtered_df.drop(columns=["word_count"], inplace=True)

        output_file = OUTPUT_DIR / f"{split_name}.csv"
        filtered_df.to_csv(output_file, index=False)

        datasets[split_name] = filtered_df

        print(f"\n{split_name.capitalize()}:")
        print("Samples:", len(filtered_df))
        print(filtered_df["label"].value_counts().sort_index())

    # Verify source-group separation
    train_groups = set(datasets["train"]["split_group"])
    val_groups = set(datasets["validation"]["split_group"])
    test_groups = set(datasets["test"]["split_group"])

    print("\n===== Group Overlap Check =====")
    print("Train/validation:", len(train_groups & val_groups))
    print("Train/test:", len(train_groups & test_groups))
    print("Validation/test:", len(val_groups & test_groups))

    print("\nExperimental datasets saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    split_dataset()