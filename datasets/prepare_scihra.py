
import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "raw" / "scihra_detect.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "scihra_processed.csv"


def prepare_scihra():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "article_id",
        "category",
        "source",
        "hgt",
        "art",
        "agt",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    samples = []

    abstract_types = {
        "hgt": (0, "Human-written"),
        "art": (1, "AI-revised"),
        "agt": (2, "AI-generated"),
    }

    for _, row in df.iterrows():

        for column, (label, editing_status) in abstract_types.items():

            text = str(row[column]).strip()

            if not text:
                continue

            samples.append({
                "text": text,
                "label": label,
                "class_name": editing_status,
                "article_id": str(row["article_id"]),
                "domain": row["category"],
                "source": row["source"],
                "split_group": str(row["article_id"]),
            })

    processed_df = pd.DataFrame(samples)

    processed_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("SciHRA preprocessing completed!")
    print("Total samples:", len(processed_df))

    print("\nClass distribution:")
    print(
        processed_df["class_name"]
        .value_counts()
        .to_string()
    )

    print("\nUnique articles:", processed_df["article_id"].nunique())

    print("\nSamples per class:")
    print(
        processed_df.groupby("label")
        .size()
        .to_string()
    )

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    prepare_scihra()
