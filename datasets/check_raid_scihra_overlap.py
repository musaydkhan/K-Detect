import pandas as pd
import re
import unicodedata
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

RAID_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "raid_abstracts.csv"
)

SCIHRA_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_processed.csv"
)


def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    print("Loading RAID human abstracts...")

    raid = pd.read_csv(
        RAID_FILE,
        usecols=["model", "generation", "source_id"]
    )

    raid_human = raid[
        raid["model"] == "human"
    ].copy()

    print("RAID human records:", len(raid_human))

    print("\nLoading SciHRA abstracts...")

    scihra = pd.read_csv(
        SCIHRA_FILE,
        usecols=[
            "text",
            "class_name",
            "article_id",
            "domain",
            "source"
        ]
    )

    print("SciHRA records:", len(scihra))

    # Normalize abstract text
    raid_human["normalized"] = (
        raid_human["generation"].apply(normalize_text)
    )

    scihra["normalized"] = (
        scihra["text"].apply(normalize_text)
    )

    # Remove empty text
    raid_human = raid_human[
        raid_human["normalized"] != ""
    ]

    scihra = scihra[
        scihra["normalized"] != ""
    ]

    # Build lookup sets
    raid_texts = set(raid_human["normalized"])

    human_scihra = scihra[
        scihra["class_name"] == "Human-written"
    ]

    scihra_human_texts = set(
        human_scihra["normalized"]
    )

    all_scihra_texts = set(
        scihra["normalized"]
    )

    # Exact normalized matches
    human_matches = raid_texts & scihra_human_texts
    all_matches = raid_texts & all_scihra_texts

    print("\n===== EXACT TEXT OVERLAP =====")

    print("Unique RAID human abstracts:", len(raid_texts))
    print("Unique SciHRA human abstracts:", len(scihra_human_texts))

    print(
        "Matches against SciHRA human abstracts:",
        len(human_matches)
    )

    print(
        "Matches against all SciHRA classes:",
        len(all_matches)
    )

    # Show matching records if any
    if all_matches:

        print("\n===== MATCHING RECORDS =====")

        matched = scihra[
            scihra["normalized"].isin(all_matches)
        ]

        print(
            matched[
                ["class_name", "article_id", "domain", "source"]
            ].drop_duplicates().to_string(index=False)
        )

    else:

        print("\nNo exact normalized-text matches found.")

    print("\n===== CHECK COMPLETE =====")


if __name__ == "__main__":
    main()