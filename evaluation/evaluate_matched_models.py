
import pandas as pd
import joblib

from pathlib import Path
from evaluation.metrics import evaluate_classifier


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "length_matching"
    / "validation_060.csv"
)

MODEL_DIR = BASE_DIR / "models" / "artifacts"


def evaluate_matched_models():

    df = pd.read_csv(DATA_FILE)

    X = df["text"].fillna("")
    y_true = df["label"]

    models = {
        "Word TF-IDF + Logistic Regression":
            MODEL_DIR / "tfidf_logistic.joblib",

        "Character TF-IDF + SVM":
            MODEL_DIR / "char_svm.joblib",
    }

    print("\n===== K-Detect: Length-Matched Evaluation =====")

    print("Matched samples:", len(df))
    print("Human samples:", (y_true == 0).sum())
    print("AI samples:", (y_true == 1).sum())

    for name, model_file in models.items():

        model = joblib.load(model_file)

        predictions = model.predict(X)

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
    evaluate_matched_models()
