import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "raid_abstracts.csv"
)


def validate_pairs():

    df = pd.read_csv(INPUT_FILE)

    # Select standard AI generations
    ai = df[
        (df["model"] != "human")
        & (df["attack"] == "none")
        & (df["repetition_penalty"] == "no")
    ].copy()

    # Human records have no decoding or repetition penalty
    humans = df[
        (df["model"] == "human")
        & (df["attack"] == "none")
    ].copy()

    human_sources = set(humans["source_id"])

    print("===== STANDARD RAID SUBSET =====")
    print("Human records:", len(humans))
    print("AI records:", len(ai))
    print("Unique human sources:", len(human_sources))

    print("\n===== SOURCE MATCHING =====")

    all_passed = True

    for model in sorted(ai["model"].unique()):

        for decoding in ["greedy", "sampling"]:

            subset = ai[
                (ai["model"] == model)
                & (ai["decoding"] == decoding)
            ]

            source_counts = subset["source_id"].value_counts()

            unique_sources = set(subset["source_id"])

            missing = human_sources - unique_sources
            extra = unique_sources - human_sources

            duplicates = source_counts[
                source_counts > 1
            ]

            passed = (
                len(subset) == 1766
                and len(unique_sources) == 1766
                and len(missing) == 0
                and len(extra) == 0
                and len(duplicates) == 0
            )

            if not passed:
                all_passed = False

            print(
                f"\n{model} | {decoding}"
            )

            print("Records:", len(subset))
            print("Unique sources:", len(unique_sources))
            print("Missing sources:", len(missing))
            print("Extra sources:", len(extra))
            print("Duplicate sources:", len(duplicates))
            print("Status:", "PASS" if passed else "FAIL")

    print("\n===== FINAL RESULT =====")

    print("Human records:", len(humans))
    print("Expected human records: 1766")

    print(
        "Overall:",
        "PASS" if all_passed and len(humans) == 1766 else "FAIL"
    )


if __name__ == "__main__":
    validate_pairs()