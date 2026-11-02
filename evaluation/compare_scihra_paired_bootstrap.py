
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_final_test_predictions.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_paired_bootstrap.csv"
)

N_BOOTSTRAPS = 5000
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


def run_paired_bootstrap():

    df = pd.read_csv(PREDICTIONS_FILE)

    y_true = df["label"].to_numpy()

    char_predictions = (
        df["char_prediction"].to_numpy()
    )

    ensemble_predictions = (
        df["ensemble_prediction"].to_numpy()
    )

    article_ids = df["article_id"].to_numpy()

    unique_articles = np.unique(article_ids)

    print("Test samples:", len(df))
    print("Unique articles:", len(unique_articles))
    print("Bootstrap iterations:", N_BOOTSTRAPS)

    # Observed differences on the original test set
    char_metrics = calculate_metrics(
        y_true,
        char_predictions
    )

    ensemble_metrics = calculate_metrics(
        y_true,
        ensemble_predictions
    )

    observed_accuracy_difference = (
        ensemble_metrics["accuracy"]
        - char_metrics["accuracy"]
    )

    observed_f1_difference = (
        ensemble_metrics["macro_f1"]
        - char_metrics["macro_f1"]
    )

    # Prepare article-level indices
    article_indices = {
        article: np.flatnonzero(article_ids == article)
        for article in unique_articles
    }

    rng = np.random.default_rng(RANDOM_STATE)

    accuracy_differences = []
    f1_differences = []

    print("\nRunning paired article-level bootstrap...")

    for _ in range(N_BOOTSTRAPS):

        sampled_articles = rng.choice(
            unique_articles,
            size=len(unique_articles),
            replace=True
        )

        sampled_indices = np.concatenate([
            article_indices[article]
            for article in sampled_articles
        ])

        sampled_y = y_true[sampled_indices]

        sampled_char = char_predictions[sampled_indices]

        sampled_ensemble = (
            ensemble_predictions[sampled_indices]
        )

        char_result = calculate_metrics(
            sampled_y,
            sampled_char
        )

        ensemble_result = calculate_metrics(
            sampled_y,
            sampled_ensemble
        )

        accuracy_differences.append(
            ensemble_result["accuracy"]
            - char_result["accuracy"]
        )

        f1_differences.append(
            ensemble_result["macro_f1"]
            - char_result["macro_f1"]
        )

    accuracy_differences = np.array(
        accuracy_differences
    )

    f1_differences = np.array(
        f1_differences
    )

    # Percentile confidence intervals
    accuracy_ci = np.percentile(
        accuracy_differences,
        [2.5, 97.5]
    )

    f1_ci = np.percentile(
        f1_differences,
        [2.5, 97.5]
    )

    print("\n===== PAIRED BOOTSTRAP RESULTS =====")

    print("\nEnsemble minus Character SVM")

    print(
        "Observed accuracy difference:",
        f"{observed_accuracy_difference * 100:.4f}",
        "percentage points"
    )

    print(
        "Accuracy difference 95% CI:",
        f"[{accuracy_ci[0] * 100:.4f}, "
        f"{accuracy_ci[1] * 100:.4f}]"
    )

    print(
        "Observed macro F1 difference:",
        f"{observed_f1_difference * 100:.4f}",
        "percentage points"
    )

    print(
        "Macro F1 difference 95% CI:",
        f"[{f1_ci[0] * 100:.4f}, "
        f"{f1_ci[1] * 100:.4f}]"
    )

    print(
        "\nBootstrap proportion with positive F1 difference:",
        f"{np.mean(f1_differences > 0) * 100:.2f}%"
    )

    print(
        "Bootstrap proportion with positive accuracy difference:",
        f"{np.mean(accuracy_differences > 0) * 100:.2f}%"
    )

    # Save results
    results = pd.DataFrame([{
        "comparison": "Ensemble minus Character SVM",
        "observed_accuracy_difference": (
            observed_accuracy_difference
        ),
        "accuracy_difference_lower_95": accuracy_ci[0],
        "accuracy_difference_upper_95": accuracy_ci[1],
        "observed_macro_f1_difference": (
            observed_f1_difference
        ),
        "f1_difference_lower_95": f1_ci[0],
        "f1_difference_upper_95": f1_ci[1],
        "bootstrap_iterations": N_BOOTSTRAPS,
    }])

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_paired_bootstrap()
