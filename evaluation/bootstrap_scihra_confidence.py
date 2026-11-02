
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]

WORD_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "ensemble_word_model.joblib"
)

CHAR_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "calibrated_char_svm.joblib"
)

TEST_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "test.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_bootstrap_results.csv"
)

N_BOOTSTRAPS = 2000
RANDOM_STATE = 42


def calculate_metrics(y_true, predictions):

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),
    }


def bootstrap_confidence_interval(
    y_true,
    predictions,
    article_ids,
    n_bootstraps=N_BOOTSTRAPS,
):

    rng = np.random.default_rng(RANDOM_STATE)

    unique_articles = np.unique(article_ids)

    # Preserve all three abstracts from each sampled article.
    article_indices = {
        article: np.flatnonzero(article_ids == article)
        for article in unique_articles
    }

    bootstrap_accuracy = []
    bootstrap_f1 = []

    for _ in range(n_bootstraps):

        sampled_articles = rng.choice(
            unique_articles,
            size=len(unique_articles),
            replace=True
        )

        sampled_indices = np.concatenate([
            article_indices[article]
            for article in sampled_articles
        ])

        metrics = calculate_metrics(
            y_true[sampled_indices],
            predictions[sampled_indices]
        )

        bootstrap_accuracy.append(metrics["accuracy"])
        bootstrap_f1.append(metrics["macro_f1"])

    return {
        "accuracy_lower": np.percentile(
            bootstrap_accuracy, 2.5
        ),
        "accuracy_upper": np.percentile(
            bootstrap_accuracy, 97.5
        ),
        "f1_lower": np.percentile(
            bootstrap_f1, 2.5
        ),
        "f1_upper": np.percentile(
            bootstrap_f1, 97.5
        ),
    }


def run_bootstrap():

    print("Loading models...")

    word_model = joblib.load(WORD_MODEL_FILE)
    char_model = joblib.load(CHAR_MODEL_FILE)

    df = pd.read_csv(TEST_FILE)

    X_test = df["text"].fillna("")
    y_true = df["label"].to_numpy()
    article_ids = df["article_id"].to_numpy()

    print("Test samples:", len(df))
    print("Unique articles:", len(np.unique(article_ids)))
    print("Bootstrap iterations:", N_BOOTSTRAPS)

    print("\nGenerating predictions...")

    word_probabilities = word_model.predict_proba(X_test)
    char_probabilities = char_model.predict_proba(X_test)

    predictions = {
        "Word Logistic Regression": np.argmax(
            word_probabilities, axis=1
        ),
        "Calibrated Character SVM": np.argmax(
            char_probabilities, axis=1
        ),
        "Equal-weight Ensemble": np.argmax(
            0.5 * word_probabilities
            + 0.5 * char_probabilities,
            axis=1
        ),
    }

    results = []

    print("\n===== ARTICLE-LEVEL BOOTSTRAP =====")

    for model_name, model_predictions in predictions.items():

        intervals = bootstrap_confidence_interval(
            y_true,
            model_predictions,
            article_ids
        )

        point_metrics = calculate_metrics(
            y_true,
            model_predictions
        )

        result = {
            "model": model_name,
            "accuracy": point_metrics["accuracy"],
            "accuracy_lower_95": intervals["accuracy_lower"],
            "accuracy_upper_95": intervals["accuracy_upper"],
            "macro_f1": point_metrics["macro_f1"],
            "f1_lower_95": intervals["f1_lower"],
            "f1_upper_95": intervals["f1_upper"],
        }

        results.append(result)

        print(f"\n{model_name}")

        print(
            "Accuracy:",
            f"{point_metrics['accuracy']:.4f}"
        )

        print(
            "Accuracy 95% CI:",
            f"[{intervals['accuracy_lower']:.4f}, "
            f"{intervals['accuracy_upper']:.4f}]"
        )

        print(
            "Macro F1:",
            f"{point_metrics['macro_f1']:.4f}"
        )

        print(
            "Macro F1 95% CI:",
            f"[{intervals['f1_lower']:.4f}, "
            f"{intervals['f1_upper']:.4f}]"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== SUMMARY =====")

    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_bootstrap()
