
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "raid_abstracts.csv"
)


def audit_prompts():

    df = pd.read_csv(INPUT_FILE)

    source_id = "005e74ce-4f9e-4d73-b800-390cfdfd25fd"

    sample = df[df["source_id"] == source_id]

    print("SOURCE:", source_id)

    for model in sorted(sample["model"].unique()):

        rows = sample[sample["model"] == model]

        print("\n" + "=" * 70)
        print("MODEL:", model)
        print("RECORDS:", len(rows))

        for index, (_, row) in enumerate(rows.iterrows(), start=1):

            prompt = row["prompt"]

            if pd.isna(prompt):
                prompt = ""

            prompt = str(prompt)

            print(f"\nGeneration {index}")
            print("Prompt length:", len(prompt))
            print("Prompt representation:")
            print(repr(prompt))

            print("\nPrompt ending:")
            print(repr(prompt[-500:]))

    print("\nAudit complete.")


if __name__ == "__main__":
    audit_prompts()
