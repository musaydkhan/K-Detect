
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR / "datasets" / "raw" / "raid" / "train_none.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "datasets" / "processed" / "raid_abstracts.csv"
)

CHUNK_SIZE = 25000

COLUMNS = [
    "id",
    "adv_source_id",
    "source_id",
    "model",
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

    first_chunk = True
    total_rows = 0
    total_extracted = 0

    print("Extracting RAID abstracts...")

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=COLUMNS,
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        filtered = chunk[
            (chunk["domain"] == "abstracts")
            & (chunk["attack"] == "none")
            & (chunk["model"].notna())
            & (chunk["generation"].notna())
        ].copy()

        filtered.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        total_extracted += len(filtered)
        first_chunk = False

        print(
            f"Processed {total_rows:,} rows | "
            f"Extracted {total_extracted:,} abstracts"
        )

    print("\nExtraction complete!")
    print("Total source rows:", total_rows)
    print("Total extracted abstracts:", total_extracted)
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    extract_abstracts()
