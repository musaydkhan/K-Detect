
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "tfidf_logistic_3class.joblib"
)

VALIDATION_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "validation.csv"
)


def evaluate_scihra_tfidf():

    model = joblib.load(MODEL_FILE)

    df = pd.read_csv(VALIDATION_FILE)

    X_val = df["text"].fillna("")
    y_true = df["label"]

    y_pred = model.predict(X_val)

    print("\n===== SciHRA TF-IDF Evaluation =====")

    print("Validation samples:", len(X_val))

    print(
        "Accuracy:",
        round(accuracy_score(y_true, y_pred), 4)
    )

    print(
        "Precision (macro):",
        round(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ), 4
        )
    )

    print(
        "Recall (macro):",
        round(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ), 4
        )
    )

    print(
        "F1 (macro):",
        round(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            ), 4
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true, y_pred,
            labels=[0, 1, 2]
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[
                "Human-written",
                "AI-revised",
                "AI-generated"
            ],
            zero_division=0
        )
    )


if __name__ == "__main__":
    evaluate_scihra_tfidf()
