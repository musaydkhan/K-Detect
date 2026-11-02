
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[1]

WORD_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "ensemble_word_model.joblib"
)

CHAR_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "calibrated_char_svm.joblib"
)

VALIDATION_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "validation.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_ensemble_results.csv"
)

CLASS_NAMES = [
    "Human-written",
    "AI-revised",
    "AI-generated",
]


def evaluate_predictions(name, y_true, predictions):

    return {
        "experiment": name,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_precision": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_ensemble():

    print("Loading models...")

    word_model = joblib.load(WORD_MODEL_FILE)
    char_model = joblib.load(CHAR_MODEL_FILE)

    df = pd.read_csv(VALIDATION_FILE)

    X_val = df["text"].fillna("")
    y_true = df["label"].to_numpy()

    print("Validation samples:", len(df))

    print("\nGenerating probability predictions...")

    word_probabilities = word_model.predict_proba(X_val)
    char_probabilities = char_model.predict_proba(X_val)

    # Check that both models use the same class ordering.
    expected_classes = np.array([0, 1, 2])

    if not np.array_equal(
        word_model.classes_,
        expected_classes
    ):
        raise ValueError("Unexpected word model class ordering.")

    if not np.array_equal(
        char_model.classes_,
        expected_classes
    ):
        raise ValueError("Unexpected character model class ordering.")

    experiments = {
        "Word-only": (1.0, 0.0),
        "Character-only": (0.0, 1.0),
        "Equal voting": (0.5, 0.5),
        "Character-heavy 70/30": (0.3, 0.7),
        "Character-heavy 80/20": (0.2, 0.8),
    }

    results = []

    print("\n===== ENSEMBLE EXPERIMENTS =====")

    for name, (word_weight, char_weight) in experiments.items():

        combined_probabilities = (
            word_weight * word_probabilities
            + char_weight * char_probabilities
        )

        predictions = np.argmax(
            combined_probabilities,
            axis=1
        )

        result = evaluate_predictions(
            name,
            y_true,
            predictions
        )

        result["word_weight"] = word_weight
        result["character_weight"] = char_weight

        results.append(result)

        print(f"\n{name}")

        print(
            "Accuracy:",
            round(result["accuracy"], 4)
        )

        print(
            "Macro precision:",
            round(result["macro_precision"], 4)
        )

        print(
            "Macro recall:",
            round(result["macro_recall"], 4)
        )

        print(
            "Macro F1:",
            round(result["macro_f1"], 4)
        )

        if name == "Character-heavy 70/30":

            print("\nClassification Report:")

            print(
                classification_report(
                    y_true,
                    predictions,
                    labels=[0, 1, 2],
                    target_names=CLASS_NAMES,
                    zero_division=0,
                )
            )

            print("Confusion Matrix:")

            print(
                confusion_matrix(
                    y_true,
                    predictions,
                    labels=[0, 1, 2],
                )
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== FINAL COMPARISON =====")

    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate_ensemble()
