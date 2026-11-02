
import pandas as pd
import joblib

from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from core.model_preprocessor import clean_text_batch

from config.model_config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    CHAR_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER,
    SVM_C,
    RANDOM_STATE,
)


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
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

OUTPUT_FILE = (
    ARTIFACT_DIR
    / "calibrated_char_svm.joblib"
)


def build_word_model():

    return Pipeline([
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


def build_char_model():

    return Pipeline([
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


def train_ensemble_components():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_FILE}"
        )

    df = pd.read_csv(TRAIN_FILE)

    X_train = df["text"].fillna("")
    y_train = df["label"]
    groups = df["article_id"]

    print("Training samples:", len(df))
    print("Unique articles:", groups.nunique())

    # Group-aware stratified calibration folds.
    # All three versions of an article remain together.
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("\nChecking calibration fold separation...")

    splits = list(
        cv.split(X_train, y_train, groups)
    )

    for fold_number, (train_idx, calibration_idx) in enumerate(
        splits,
        start=1
    ):

        train_groups = set(groups.iloc[train_idx])
        calibration_groups = set(groups.iloc[calibration_idx])

        overlap = train_groups & calibration_groups

        if overlap:
            raise RuntimeError(
                f"Article leakage detected in fold {fold_number}!"
            )

        print(
            f"Fold {fold_number}: "
            f"training articles={len(train_groups)}, "
            f"calibration articles={len(calibration_groups)}, "
            f"overlap={len(overlap)}"
        )

    # Train word-level model
    print("\nTraining word-level Logistic Regression...")

    word_model = build_word_model()
    word_model.fit(X_train, y_train)

    # Train calibrated character-level SVM
    print("\nTraining calibrated character-level SVM...")

    char_model = CalibratedClassifierCV(
        estimator=build_char_model(),
        method="sigmoid",
        cv=splits,
        ensemble=False,
    )

    char_model.fit(X_train, y_train)

    # Save both components
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        word_model,
        ARTIFACT_DIR / "ensemble_word_model.joblib"
    )

    joblib.dump(
        char_model,
        OUTPUT_FILE
    )

    print("\n===== TRAINING COMPLETED =====")

    print("Word model saved.")

    print(
        "Calibrated character model saved:",
        OUTPUT_FILE
    )

    print(
        "Classes:",
        char_model.classes_.tolist()
    )


if __name__ == "__main__":
    train_ensemble_components()
