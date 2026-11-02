import pandas as pd
import joblib

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "length_experiment"
    / "validation.csv"
)

ORIGINAL_DIR = BASE_DIR / "models" / "artifacts"

EXPERIMENT_DIR = (
    ORIGINAL_DIR / "length_experiment"
)


def compare_length_models():

    df = pd.read_csv(VALIDATION_FILE)

    X_val = df["text"].fillna("")
    y_true = df["label"]

    models = {
        "Word TF-IDF": (
            ORIGINAL_DIR / "tfidf_logistic.joblib",
            EXPERIMENT_DIR / "tfidf_logistic.joblib",
        ),
        "Character SVM": (
            ORIGINAL_DIR / "char_svm.joblib",
            EXPERIMENT_DIR / "char_svm.joblib",
        ),
    }

    print("\n===== Original vs 100+ Word Models =====")
    print("Shared validation samples:", len(df))

    for name, (original_file, experiment_file) in models.items():

        original_model = joblib.load(original_file)
        experiment_model = joblib.load(experiment_file)

        original_predictions = original_model.predict(X_val)
        experiment_predictions = experiment_model.predict(X_val)

        original_correct = original_predictions == y_true
        experiment_correct = experiment_predictions == y_true

        changed = original_predictions != experiment_predictions

        print(f"\n===== {name} =====")

        print(
            "Original model accuracy:",
            round(original_correct.mean(), 4)
        )

        print(
            "100+ word model accuracy:",
            round(experiment_correct.mean(), 4)
        )

        print("Predictions changed:", int(changed.sum()))

        print(
            "Original correct, experiment wrong:",
            int((original_correct & ~experiment_correct).sum())
        )

        print(
            "Original wrong, experiment correct:",
            int((~original_correct & experiment_correct).sum())
        )

        # Export samples where predictions changed
        changed_df = df.loc[changed].copy()

        changed_df["original_prediction"] = (
            original_predictions[changed]
        )

        changed_df["experiment_prediction"] = (
            experiment_predictions[changed]
        )

        changed_df["actual_label"] = y_true[changed]

        output_file = (
            BASE_DIR
            / "datasets"
            / "processed"
            / "length_experiment"
            / f"{name.lower().replace(' ', '_')}_changes.csv"
        )

        changed_df.to_csv(output_file, index=False)

        print("Changed samples exported:", output_file)


if __name__ == "__main__":
    compare_length_models()