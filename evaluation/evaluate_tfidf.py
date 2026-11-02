import pandas as pd
import joblib

from evaluation.metrics import evaluate_classifier


# File locations
MODEL_FILE = "models/artifacts/tfidf_logistic.joblib"
VALIDATION_FILE = "datasets/processed/validation.csv"


def evaluate_tfidf():
    """
    Evaluate the saved TF-IDF model on validation data.
    """

    # Load trained model
    model = joblib.load(MODEL_FILE)

    # Load validation dataset
    df = pd.read_csv(VALIDATION_FILE)

    X_val = df["text"].fillna("")
    y_val = df["label"]

    # Generate predictions
    predictions = model.predict(X_val)

    # Calculate evaluation metrics
    results = evaluate_classifier(y_val, predictions)

    print("\n===== K-Detect: TF-IDF Model Evaluation =====")

    print(f"Validation samples: {len(X_val)}")
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


if __name__ == "__main__":
    evaluate_tfidf()