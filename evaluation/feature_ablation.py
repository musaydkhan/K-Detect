
import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

from core.model_preprocessor import clean_text_batch

from config.model_config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER,
    RANDOM_STATE,
)

from evaluation.metrics import evaluate_classifier


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "datasets" / "processed" / "train.csv"
VALIDATION_FILE = BASE_DIR / "datasets" / "processed" / "validation.csv"

FEATURE_FILE = BASE_DIR / "evaluation" / "tfidf_feature_importance.csv"

OUTPUT_DIR = BASE_DIR / "models" / "artifacts" / "feature_ablation"
RESULT_FILE = BASE_DIR / "evaluation" / "feature_ablation_results.csv"


def build_reduced_model(removed_features):

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        stop_words=None,
        vocabulary=None,
    )

    model = Pipeline([
        (
            "preprocessor",
            FunctionTransformer(
                clean_text_batch,
                validate=False
            ),
        ),
        ("tfidf", vectorizer),
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


def run_feature_ablation():

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)
    importance_df = pd.read_csv(FEATURE_FILE)

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"]

    X_val = val_df["text"].fillna("")
    y_val = val_df["label"]

    ai_features = importance_df.nlargest(
        30, "coefficient"
    )["feature"].tolist()

    human_features = importance_df.nsmallest(
        30, "coefficient"
    )["feature"].tolist()

    experiments = {
        "baseline": [],
        "remove_ai_features": ai_features,
        "remove_human_features": human_features,
        "remove_both_groups": list(
            set(ai_features + human_features)
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n===== K-Detect: Feature Ablation Experiment =====")

    for experiment_name, removed_features in experiments.items():

        print(f"\nRunning: {experiment_name}")

        # Fit a fresh vectorizer on training data only
        preprocessor = FunctionTransformer(
            clean_text_batch,
            validate=False
        )

        cleaned_train = preprocessor.transform(X_train)
        cleaned_val = preprocessor.transform(X_val)

        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )

        X_train_vectorized = vectorizer.fit_transform(cleaned_train)
        X_val_vectorized = vectorizer.transform(cleaned_val)

        # Remove selected feature columns before classifier training
        vocabulary = vectorizer.vocabulary_

        removed_indices = {
            vocabulary[feature]
            for feature in removed_features
            if feature in vocabulary
        }

        if removed_indices:

            keep_indices = [
                index
                for index in range(X_train_vectorized.shape[1])
                if index not in removed_indices
            ]

            X_train_vectorized = X_train_vectorized[:, keep_indices]
            X_val_vectorized = X_val_vectorized[:, keep_indices]

        # Train a fresh classifier on the reduced feature matrix
        classifier = LogisticRegression(
            max_iter=LOGISTIC_REGRESSION_MAX_ITER,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )

        classifier.fit(X_train_vectorized, y_train)

        predictions = classifier.predict(X_val_vectorized)

        metrics = evaluate_classifier(y_val, predictions)

        results.append({
            "experiment": experiment_name,
            "removed_features": len(removed_indices),
            "remaining_features": X_train_vectorized.shape[1],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "false_positive_rate": metrics["false_positive_rate"],
            "false_negative_rate": metrics["false_negative_rate"],
        })

        print("Removed features:", len(removed_indices))
        print("Remaining features:", X_train_vectorized.shape[1])
        print("Accuracy:", round(metrics["accuracy"], 4))
        print("F1:", round(metrics["f1_score"], 4))
        print("Confusion matrix:", metrics["confusion_matrix"])

        # Save classifier and feature names for this experiment
        artifact = {
            "classifier": classifier,
            "feature_names": (
                vectorizer.get_feature_names_out()[keep_indices]
                if removed_indices
                else vectorizer.get_feature_names_out()
            ),
        }

        joblib.dump(
            artifact,
            OUTPUT_DIR / f"{experiment_name}.joblib"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_FILE, index=False)

    print("\n===== Final Comparison =====")
    print(results_df.to_string(index=False))

    print("\nResults saved to:", RESULT_FILE)


if __name__ == "__main__":
    run_feature_ablation()
