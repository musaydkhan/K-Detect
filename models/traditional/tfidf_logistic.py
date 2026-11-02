
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
# Model storage location
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "models" / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "tfidf_logistic.joblib"

def build_tfidf_logistic_model():
    """
    Build a word-level TF-IDF + Logistic Regression classifier.
    """

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


def train_tfidf_logistic(
    train_file="datasets/processed/train.csv",
):
    """
    Train the classifier using training data only.
    """

    train_df = pd.read_csv(train_file)

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"]

    model = build_tfidf_logistic_model()

    model.fit(X_train, y_train)

    # Save trained model
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_FILE)

    print("TF-IDF + Logistic Regression training completed!")
    print("Training samples:", len(X_train))
    print("Vocabulary size:", len(model.named_steps["tfidf"].vocabulary_))
    print("Model saved to:", ARTIFACT_FILE)

    return model


if __name__ == "__main__":
    train_tfidf_logistic()
