
import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from core.model_preprocessor import clean_text_batch

from config.model_config import (
    CHAR_NGRAM_RANGE,
    SVM_C,
    RANDOM_STATE,
)


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
    / "char_svm_3class.joblib"
)


def build_scihra_char_model():

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
                analyzer="char",
                ngram_range=CHAR_NGRAM_RANGE,
                max_features=50000,
                lowercase=True,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=SVM_C,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return model


def train_scihra_char_model():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    X_train = df["text"].fillna("")
    y_train = df["label"]

    model = build_scihra_char_model()

    model.fit(X_train, y_train)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        ARTIFACT_FILE
    )

    print("SciHRA Character SVM training completed!")

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
    train_scihra_char_model()
