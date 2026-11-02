
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_dataset(
    input_file="datasets/processed/hc3_wiki_processed.csv",
    random_state=42
):
    """
    Split data into training, validation, and test sets.

    Groups are kept intact to prevent source leakage.
    """

    df = pd.read_csv(input_file)

    # First: separate training data (80%) from temporary data (20%)
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state
    )

    train_idx, temp_idx = next(
        first_split.split(df, groups=df["split_group"])
    )

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # Second: divide temporary data equally
    # Result: 80% train, 10% validation, 10% test
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state
    )

    val_idx, test_idx = next(
        second_split.split(
            temp_df,
            groups=temp_df["split_group"]
        )
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Save all three datasets
    output_dir = "datasets/processed"

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print("Dataset splitting completed!")
    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

    # Verify that no groups overlap
    train_groups = set(train_df["split_group"])
    val_groups = set(val_df["split_group"])
    test_groups = set(test_df["split_group"])

    print("Train/validation overlap:", len(train_groups & val_groups))
    print("Train/test overlap:", len(train_groups & test_groups))
    print("Validation/test overlap:", len(val_groups & test_groups))


if __name__ == "__main__":
    split_dataset()
