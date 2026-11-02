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


# Model storage location
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = BASE_DIR / "models" / "artifacts"
ARTIFACT_FILE = ARTIFACT_DIR / "char_svm.joblib"

def build_char_svm_model():
    """
    Build a character-level TF-IDF + Linear SVM classifier.
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
            analyzer="char",
            ngram_range=CHAR_NGRAM_RANGE,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            max_features=50000,
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

def train_char_svm(
    train_file="datasets/processed/train.csv",
):
    """
    Train the character-level SVM classifier.
    """

    train_df = pd.read_csv(train_file)

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"]

    model = build_char_svm_model()

    model.fit(X_train, y_train)

    # Save trained model
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_FILE)

    print("Character TF-IDF + Linear SVM training completed!")
    print("Training samples:", len(X_train))
    print(
        "Character vocabulary size:",
        len(model.named_steps["tfidf"].vocabulary_)
    )
    print("Model saved to:", ARTIFACT_FILE)

    return model


if __name__ == "__main__":
    train_char_svm()