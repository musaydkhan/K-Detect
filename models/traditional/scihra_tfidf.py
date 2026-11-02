
import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from core.model_preprocessor import clean_text_batch

from config.model_config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER,
    RANDOM_STATE,
)


# Project paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "train.csv"
)

ARTIFACT_DIR = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
)

ARTIFACT_FILE = (
    ARTIFACT_DIR
    / "tfidf_logistic_3class.joblib"
)


def build_scihra_tfidf_model():

    model = Pipeline([
        (
            "preprocessor",
            FunctionTransformer(
                clean_text_batch,
                validate=False
            ),
        ),
        (
            "tfidf",
            TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE,
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=LOGISTIC_REGRESSION_MAX_ITER,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
    ])

    return model


def train_scihra_tfidf():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    X_train = df["text"].fillna("")
    y_train = df["label"]

    model = build_scihra_tfidf_model()

    model.fit(X_train, y_train)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, ARTIFACT_FILE)

    print("SciHRA TF-IDF training completed!")
    print("Training samples:", len(X_train))
    print(
        "Vocabulary size:",
        len(model.named_steps["tfidf"].vocabulary_)
    )
    print(
        "Classes:",
        model.named_steps["classifier"].classes_.tolist()
    )
    print("Saved model:", ARTIFACT_FILE)

    return model


if __name__ == "__main__":
    train_scihra_tfidf()
