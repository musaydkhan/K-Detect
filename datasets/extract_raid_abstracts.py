
import pandas as pd
from pathlib import Path
from collections import Counter


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "raw"
    / "raid"
    / "train_none.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "raid_abstracts.csv"
)

CHUNK_SIZE = 25000

REQUIRED_COLUMNS = [
    "id",
    "adv_source_id",
    "source_id",
    "model",
    "decoding",
    "repetition_penalty",
    "attack",
    "domain",
    "title",
    "prompt",
    "generation",
]


def extract_abstracts():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_rows = 0
    extracted_rows = 0

    model_counts = Counter()

    first_chunk = True

    print("Extracting RAID abstracts with metadata...")

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        total_rows += len(chunk)

        domain = (
            chunk["domain"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        attack = (
            chunk["attack"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = (
            (domain == "abstracts")
            & (attack == "none")
            & (chunk["model"].notna())
            & (chunk["generation"].notna())
            & (chunk["generation"].astype(str).str.strip() != "")
        )

        filtered = chunk.loc[mask].copy()

        if not filtered.empty:

            extracted_rows += len(filtered)

            model_counts.update(
                filtered["model"].astype(str)
            )

            filtered.to_csv(
                OUTPUT_FILE,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )

            first_chunk = False

        print(
            f"Processed {total_rows:,} rows | "
            f"Extracted {extracted_rows:,} abstracts"
        )

    print("\n===== EXTRACTION COMPLETE =====")
    print("Total source rows:", total_rows)
    print("Extracted abstracts:", extracted_rows)

    print("\nModel distribution:")

    for model, count in model_counts.most_common():
        print(f"{model}: {count:,}")

    print("\nOutput:", OUTPUT_FILE)


if __name__ == "__main__":
    extract_abstracts()
