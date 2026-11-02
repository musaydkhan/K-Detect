
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR / "models" / "artifacts" / "scihra"
    / "tfidf_logistic_3class.joblib"
)

VALIDATION_FILE = (
    BASE_DIR / "datasets" / "processed"
    / "scihra_splits" / "validation.csv"
)


def analyze_domains():

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(VALIDATION_FILE)

    df["prediction"] = model.predict(
        df["text"].fillna("")
    )

    print("\n===== Performance by Scientific Domain =====")

    for domain, group in df.groupby("domain"):

        accuracy = accuracy_score(
            group["label"],
            group["prediction"]
        )

        macro_f1 = f1_score(
            group["label"],
            group["prediction"],
            average="macro",
            zero_division=0
        )

        print(f"\n{domain}")
        print("Samples:", len(group))
        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))


if __name__ == "__main__":
    analyze_domains()
