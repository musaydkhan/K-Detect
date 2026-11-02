import pandas as pd
import joblib
from pathlib import Path

from models.traditional.tfidf_logistic import (
    build_tfidf_logistic_model
)

from models.traditional.char_svm import (
    build_char_svm_model
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "length_experiment"
    / "train.csv"
)

ARTIFACT_DIR = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "length_experiment"
)


def train_length_experiment():

    # Load experimental training data
    df = pd.read_csv(DATA_FILE)

    X_train = df["text"].fillna("")
    y_train = df["label"]

    # Create separate models
    models = {
        "tfidf_logistic": build_tfidf_logistic_model(),
        "char_svm": build_char_svm_model(),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n===== K-Detect: Length Experiment =====")

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        output_file = ARTIFACT_DIR / f"{name}.joblib"

        joblib.dump(model, output_file)

        print("Training samples:", len(X_train))
        print("Saved:", output_file)

    print("\nBoth experimental models trained successfully!")


if __name__ == "__main__":
    train_length_experiment()