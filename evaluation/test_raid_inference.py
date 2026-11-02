import sys
import joblib
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


DATA_FILE = (
    BASE_DIR / "datasets" / "processed" / "raid_abstracts.csv"
)

ARTIFACT_DIR = (
    BASE_DIR / "models" / "artifacts" / "scihra"
)

MODELS = {
    "TF-IDF Logistic Regression":
        "tfidf_logistic_3class.joblib",

    "Calibrated Character SVM":
        "calibrated_char_svm.joblib",

    "Ensemble Word Model":
        "ensemble_word_model.joblib",
}


def main():

    print("Loading RAID sample...")

    df = pd.read_csv(DATA_FILE)

    human = df[
        df["model"] == "human"
    ].iloc[0]

    ai = df[
        (df["model"] == "gpt4")
        & (df["decoding"] == "greedy")
        & (df["repetition_penalty"] == "no")
    ].iloc[0]

    samples = [
        ("Human reference", human["generation"]),
        ("GPT-4 generation", ai["generation"]),
    ]

    print("\nLoading models...")

    for model_name, filename in MODELS.items():

        model = joblib.load(ARTIFACT_DIR / filename)

        print(f"\n===== {model_name} =====")

        for sample_name, text in samples:

            probabilities = model.predict_proba([text])[0]

            human_probability = probabilities[0]
            revised_probability = probabilities[1]
            generated_probability = probabilities[2]

            ai_probability = (
                revised_probability + generated_probability
            )

            print(f"\n{sample_name}")
            print("Class probabilities:", probabilities.round(4))
            print("Probability sum:", round(probabilities.sum(), 6))
            print("Human score:", round(human_probability, 4))
            print("AI score:", round(ai_probability, 4))


if __name__ == "__main__":
    main()