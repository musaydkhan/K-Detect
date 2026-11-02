
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

TEST_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "test.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_final_test_predictions.csv"
)

CLASS_NAMES = [
    "Human-written",
    "AI-revised",
    "AI-generated",
]


def test_ensemble():

    print("Loading trained models...")

    word_model = joblib.load(WORD_MODEL_FILE)
    char_model = joblib.load(CHAR_MODEL_FILE)

    print("Loading untouched test dataset...")

    df = pd.read_csv(TEST_FILE)

    X_test = df["text"].fillna("")
    y_true = df["label"].to_numpy()

    print("Test samples:", len(df))
    print("Unique articles:", df["article_id"].nunique())

    print("\nGenerating predictions...")

    word_probabilities = word_model.predict_proba(X_test)
    char_probabilities = char_model.predict_proba(X_test)

    # Equal-weight probability ensemble
    ensemble_probabilities = (
        0.5 * word_probabilities
        + 0.5 * char_probabilities
    )

    word_predictions = np.argmax(
        word_probabilities,
        axis=1
    )

    char_predictions = np.argmax(
        char_probabilities,
        axis=1
    )

    ensemble_predictions = np.argmax(
        ensemble_probabilities,
        axis=1
    )

    # Evaluate all three models
    models = {
        "Word Logistic Regression": word_predictions,
        "Calibrated Character SVM": char_predictions,
        "Equal-weight Ensemble": ensemble_predictions,
    }

    print("\n===== FINAL TEST RESULTS =====")

    summary = []

    for name, predictions in models.items():

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        macro_precision = precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )

        summary.append({
            "model": name,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        })

        print(f"\n{name}")

        print("Accuracy:", round(accuracy, 4))
        print("Macro precision:", round(macro_precision, 4))
        print("Macro recall:", round(macro_recall, 4))
        print("Macro F1:", round(macro_f1, 4))

        print("\nClassification Report:")

        print(
            classification_report(
                y_true,
                predictions,
                labels=[0, 1, 2],
                target_names=CLASS_NAMES,
                zero_division=0
            )
        )

        print("Confusion Matrix:")

        print(
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1, 2]
            )
        )

    # Save model comparison
    summary_df = pd.DataFrame(summary)

    summary_file = (
        BASE_DIR
        / "evaluation"
        / "scihra_final_test_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # Save detailed ensemble predictions
    results = df.copy()

    results["word_prediction"] = word_predictions
    results["char_prediction"] = char_predictions
    results["ensemble_prediction"] = ensemble_predictions

    results["ensemble_correct"] = (
        results["label"] == ensemble_predictions
    )

    for class_index, class_name in enumerate(CLASS_NAMES):

        results[
            f"ensemble_probability_{class_index}"
        ] = ensemble_probabilities[:, class_index]

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== TEST COMPLETED =====")

    print("Summary saved:", summary_file)
    print("Predictions saved:", OUTPUT_FILE)


if __name__ == "__main__":
    test_ensemble()
