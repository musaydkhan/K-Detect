import pandas as pd
import joblib

from pathlib import Path
from evaluation.metrics import evaluate_classifier


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    BASE_DIR / "models" / "artifacts" / "length_experiment"
)

VALIDATION_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "length_experiment"
    / "validation.csv"
)


def evaluate_length_experiment():

    df = pd.read_csv(VALIDATION_FILE)

    X_val = df["text"].fillna("")
    y_true = df["label"]

    models = {
        "TF-IDF Logistic Regression": (
            MODEL_DIR / "tfidf_logistic.joblib"
        ),
        "Character SVM": (
            MODEL_DIR / "char_svm.joblib"
        ),
    }

    print("\n===== K-Detect: Length Experiment Evaluation =====")
    print("Validation samples:", len(df))

    for name, model_file in models.items():

        model = joblib.load(model_file)

        predictions = model.predict(X_val)

        results = evaluate_classifier(y_true, predictions)

        print(f"\n===== {name} =====")

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

        print("Confusion Matrix:")
        print(results["confusion_matrix"])


if __name__ == "__main__":
    evaluate_length_experiment()