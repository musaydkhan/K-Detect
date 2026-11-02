
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import recall_score, precision_score, f1_score


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR / "models" / "artifacts" / "scihra"
    / "tfidf_logistic_3class.joblib"
)

VALIDATION_FILE = (
    BASE_DIR / "datasets" / "processed"
    / "scihra_splits" / "validation.csv"
)

CLASS_NAMES = {
    0: "Human",
    1: "AI-revised",
    2: "AI-generated",
}


def analyze_domain_classes():

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(VALIDATION_FILE)

    df["prediction"] = model.predict(
        df["text"].fillna("")
    )

    print("\n===== Per-Class Performance by Domain =====")

    for domain, group in df.groupby("domain"):

        print(f"\n{domain}")

        for label, class_name in CLASS_NAMES.items():

            y_true = (
                group["label"] == label
            ).astype(int)

            y_pred = (
                group["prediction"] == label
            ).astype(int)

            recall = recall_score(
                y_true, y_pred,
                zero_division=0
            )

            precision = precision_score(
                y_true, y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true, y_pred,
                zero_division=0
            )

            print(
                f"{class_name}: "
                f"Precision={precision:.3f}, "
                f"Recall={recall:.3f}, "
                f"F1={f1:.3f}"
            )


if __name__ == "__main__":
    analyze_domain_classes()
