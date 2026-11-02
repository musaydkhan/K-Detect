
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "processed" / "scihra_processed.csv"
OUTPUT_DIR = BASE_DIR / "processed" / "scihra_splits"

RANDOM_STATE = 42


def split_scihra():

    df = pd.read_csv(INPUT_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # First: separate 20% for validation + testing
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    train_idx, remaining_idx = next(
        first_split.split(
            df,
            groups=df["split_group"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    remaining_df = df.iloc[remaining_idx].copy()

    # Second: divide remaining 20% equally
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    val_idx, test_idx = next(
        second_split.split(
            remaining_df,
            groups=remaining_df["split_group"]
        )
    )

    val_df = remaining_df.iloc[val_idx].copy()
    test_df = remaining_df.iloc[test_idx].copy()

    datasets = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    # Save and report
    for name, split_df in datasets.items():

        output_file = OUTPUT_DIR / f"{name}.csv"

        split_df.to_csv(
            output_file,
            index=False
        )

        print(f"\n{name.upper()}")
        print("Samples:", len(split_df))
        print("Articles:", split_df["article_id"].nunique())

        print("\nClass distribution:")
        print(
            split_df["class_name"]
            .value_counts()
            .to_string()
        )

    # Verify article separation
    train_groups = set(train_df["split_group"])
    val_groups = set(val_df["split_group"])
    test_groups = set(test_df["split_group"])

    print("\n===== GROUP OVERLAP CHECK =====")
    print("Train/validation:", len(train_groups & val_groups))
    print("Train/test:", len(train_groups & test_groups))
    print("Validation/test:", len(val_groups & test_groups))

    print("\nSplits saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    split_scihra()
