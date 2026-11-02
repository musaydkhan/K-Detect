
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import evaluate_classifier


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR / "datasets" / "processed" / "train.csv"
)

VALIDATION_FILE = (
    BASE_DIR / "datasets" / "processed" / "validation.csv"
)


def calculate_word_count(df):

    return df["text"].fillna("").apply(
        lambda text: len(text.split())
    ).to_frame(name="word_count")


def run_length_baseline():

    # Load original training and validation datasets
    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)

    # Extract ONLY word count
    X_train = calculate_word_count(train_df)
    X_val = calculate_word_count(val_df)

    y_train = train_df["label"]
    y_val = val_df["label"]

    # Build a simple length-only classifier
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_val)

    # Evaluate
    results = evaluate_classifier(y_val, predictions)

    print("\n===== K-Detect: Length-Only Baseline =====")

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))

    print("\nPerformance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")

    print(
        f"False Positive Rate: "
        f"{results['false_positive_rate']:.4f}"
    )

    print(
        f"False Negative Rate: "
        f"{results['false_negative_rate']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nLearned coefficient:")
    print(model.named_steps["classifier"].coef_[0][0])


if __name__ == "__main__":
    run_length_baseline()
