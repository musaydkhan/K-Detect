
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR / "datasets" / "processed"
    / "scihra_splits" / "train.csv"
)

VALIDATION_FILE = (
    BASE_DIR / "datasets" / "processed"
    / "scihra_splits" / "validation.csv"
)


def calculate_word_count(df):

    return df["text"].fillna("").apply(
        lambda text: len(text.split())
    ).to_frame(name="word_count")


def run_length_baseline():

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)

    X_train = calculate_word_count(train_df)
    X_val = calculate_word_count(val_df)

    y_train = train_df["label"]
    y_val = val_df["label"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    print("\n===== SciHRA Length-Only Baseline =====")

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    print(
        "Accuracy:",
        round(accuracy_score(y_val, predictions), 4)
    )

    print(
        "Macro precision:",
        round(
            precision_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            ), 4
        )
    )

    print(
        "Macro recall:",
        round(
            recall_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            ), 4
        )
    )

    print(
        "Macro F1:",
        round(
            f1_score(
                y_val, predictions,
                average="macro",
                zero_division=0
            ), 4
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1, 2]
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_val,
            predictions,
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
    run_length_baseline()
