
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "raid_abstracts.csv"
)


def inspect_prompts():

    print("Loading RAID abstracts...")

    df = pd.read_csv(INPUT_FILE)

    # Find source IDs with all 12 model labels
    source_model_counts = (
        df.groupby("source_id")["model"]
        .nunique()
    )

    complete_sources = source_model_counts[
        source_model_counts == 12
    ].index

    if len(complete_sources) == 0:
        print("No complete source groups found.")
        return

    # Select one complete source
    selected_source = complete_sources[0]

    sample = df[
        df["source_id"] == selected_source
    ].copy()

    print("\n" + "=" * 80)
    print("SOURCE ID:", selected_source)
    print("TOTAL RECORDS:", len(sample))
    print("=" * 80)

    # Show original human abstract first
    human = sample[sample["model"] == "human"]

    if not human.empty:

        row = human.iloc[0]

        print("\n\n========== ORIGINAL HUMAN ABSTRACT ==========")

        print("\nTITLE:")
        print(row["title"])

        print("\nPROMPT:")
        print(row["prompt"])

        print("\nHUMAN ABSTRACT:")
        print(row["generation"])

    # Show one example from every AI model
    ai_models = sorted(
        sample.loc[
            sample["model"] != "human",
            "model"
        ].unique()
    )

    for model in ai_models:

        model_samples = sample[
            sample["model"] == model
        ]

        print("\n\n" + "=" * 80)
        print("AI MODEL:", model)
        print("AVAILABLE GENERATIONS:", len(model_samples))
        print("=" * 80)

        # Display first generation only
        row = model_samples.iloc[0]

        print("\nTITLE:")
        print(row["title"])

        print("\nPROMPT:")
        print(row["prompt"])

        print("\nGENERATED ABSTRACT:")
        print(row["generation"])

    print("\n\nInspection complete.")


if __name__ == "__main__":
    inspect_prompts()
