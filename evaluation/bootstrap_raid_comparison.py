
import numpy as np
import pandas as pd

from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    BASE_DIR / "evaluation" / "raid_external_predictions.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "evaluation" / "raid_paired_bootstrap_results.csv"
)

N_BOOTSTRAP = 5000
RANDOM_STATE = 42
THRESHOLD = 0.50

CLASSIFIERS = [
    "TF-IDF Logistic Regression",
    "Character SVM",
    "Equal-weight Ensemble",
]

COMPARISONS = [
    ("TF-IDF Logistic Regression", "Character SVM"),
    ("TF-IDF Logistic Regression", "Equal-weight Ensemble"),
    ("Character SVM", "Equal-weight Ensemble"),
]


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(human_scores, ai_scores):
    """
    Calculate binary metrics from paired human and AI scores.

    Human = 0
    AI = 1
    """

    human_predictions = human_scores >= THRESHOLD
    ai_predictions = ai_scores >= THRESHOLD

    tn = np.sum(~human_predictions)
    fp = np.sum(human_predictions)

    fn = np.sum(~ai_predictions)
    tp = np.sum(ai_predictions)

    total = tn + fp + fn + tp

    accuracy = (tn + tp) / max(total, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    specificity = tn / max(tn + fp, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
    }


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_prediction_matrices():

    print("Loading RAID predictions...")

    df = pd.read_csv(PREDICTIONS_FILE)

    required_columns = [
        "source_id",
        "ai_model",
        "decoding",
        "classifier",
        "human_ai_score",
        "ai_generation_score",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df[
        df["classifier"].isin(CLASSIFIERS)
    ].copy()

    df["condition"] = (
        df["ai_model"].astype(str)
        + "__"
        + df["decoding"].astype(str)
    )

    matrices = {}

    expected_conditions = 22

    for classifier in CLASSIFIERS:

        subset = df[
            df["classifier"] == classifier
        ].copy()

        if subset.duplicated(
            ["source_id", "condition"]
        ).any():

            raise ValueError(
                f"Duplicate source-condition records: {classifier}"
            )

        human_matrix = subset.pivot(
            index="source_id",
            columns="condition",
            values="human_ai_score",
        )

        ai_matrix = subset.pivot(
            index="source_id",
            columns="condition",
            values="ai_generation_score",
        )

        if human_matrix.shape[1] != expected_conditions:
            raise ValueError(
                f"Unexpected condition count for {classifier}: "
                f"{human_matrix.shape[1]}"
            )

        if human_matrix.isna().any().any():
            raise ValueError(
                f"Missing human scores for {classifier}"
            )

        if ai_matrix.isna().any().any():
            raise ValueError(
                f"Missing AI scores for {classifier}"
            )

        if not human_matrix.index.equals(ai_matrix.index):
            raise ValueError(
                f"Human and AI source IDs do not align: {classifier}"
            )

        matrices[classifier] = {
            "human": human_matrix.to_numpy(),
            "ai": ai_matrix.to_numpy(),
            "source_ids": human_matrix.index.to_numpy(),
            "conditions": human_matrix.columns.tolist(),
        }

        print(
            f"{classifier}: "
            f"{len(human_matrix)} sources, "
            f"{human_matrix.shape[1]} conditions"
        )

    # Verify that all classifiers use identical source ordering
    reference_ids = matrices[CLASSIFIERS[0]]["source_ids"]

    for classifier in CLASSIFIERS[1:]:

        current_ids = matrices[classifier]["source_ids"]

        if not np.array_equal(reference_ids, current_ids):
            raise ValueError(
                "Source IDs differ between classifiers."
            )

    return matrices


# --------------------------------------------------
# PAIRED BOOTSTRAP
# --------------------------------------------------

def run_bootstrap(matrices):

    rng = np.random.default_rng(RANDOM_STATE)

    n_sources = len(
        matrices[CLASSIFIERS[0]]["source_ids"]
    )

    print("\nNumber of independent source groups:", n_sources)
    print("Bootstrap iterations:", N_BOOTSTRAP)

    # Observed metrics
    observed = {}

    for classifier in CLASSIFIERS:

        human = matrices[classifier]["human"].ravel()
        ai = matrices[classifier]["ai"].ravel()

        observed[classifier] = calculate_metrics(
            human,
            ai
        )

    # Store bootstrap metric differences
    bootstrap_results = {
        (first, second, metric): []
        for first, second in COMPARISONS
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1",
        ]
    }

    print("\nRunning paired article-level bootstrap...")

    for iteration in range(N_BOOTSTRAP):

        # One shared sample of source IDs for every classifier.
        sampled_indices = rng.integers(
            0,
            n_sources,
            size=n_sources
        )

        iteration_metrics = {}

        for classifier in CLASSIFIERS:

            human = matrices[classifier]["human"][
                sampled_indices
            ].ravel()

            ai = matrices[classifier]["ai"][
                sampled_indices
            ].ravel()

            iteration_metrics[classifier] = (
                calculate_metrics(human, ai)
            )

        # Paired differences
        for first, second in COMPARISONS:

            for metric in [
                "accuracy",
                "balanced_accuracy",
                "f1",
            ]:

                difference = (
                    iteration_metrics[first][metric]
                    - iteration_metrics[second][metric]
                )

                bootstrap_results[
                    (first, second, metric)
                ].append(difference)

        if (iteration + 1) % 500 == 0:

            print(
                f"Completed {iteration + 1}/"
                f"{N_BOOTSTRAP} iterations"
            )

    return observed, bootstrap_results


# --------------------------------------------------
# SUMMARIZE RESULTS
# --------------------------------------------------

def summarize_results(observed, bootstrap_results):

    rows = []

    for first, second in COMPARISONS:

        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1",
        ]:

            differences = np.array(
                bootstrap_results[
                    (first, second, metric)
                ]
            )

            observed_difference = (
                observed[first][metric]
                - observed[second][metric]
            )

            ci_lower, ci_upper = np.percentile(
                differences,
                [2.5, 97.5]
            )

            positive_percentage = (
                np.mean(differences > 0) * 100
            )

            rows.append({
                "classifier_1": first,
                "classifier_2": second,
                "metric": metric,
                "observed_difference": observed_difference,
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "bootstrap_positive_percentage": positive_percentage,
                "ci_excludes_zero": (
                    ci_lower > 0 or ci_upper < 0
                ),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("RAID: PAIRED ARTICLE-LEVEL BOOTSTRAP")
    print("=" * 65)

    matrices = load_prediction_matrices()

    observed, bootstrap_results = run_bootstrap(
        matrices
    )

    results = summarize_results(
        observed,
        bootstrap_results
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("OBSERVED CLASSIFIER PERFORMANCE")
    print("=" * 65)

    for classifier in CLASSIFIERS:

        print(f"\n{classifier}")

        for metric, value in observed[classifier].items():

            print(f"{metric}: {value:.4f}")

    print("\n" + "=" * 65)
    print("PAIRED BOOTSTRAP RESULTS")
    print("=" * 65)

    for _, row in results.iterrows():

        print(
            f"\n{row['classifier_1']} "
            f"MINUS "
            f"{row['classifier_2']}"
        )

        print("Metric:", row["metric"])

        print(
            "Observed difference:",
            f"{row['observed_difference']:.6f}"
        )

        print(
            "95% CI:",
            f"[{row['ci_lower_95']:.6f}, "
            f"{row['ci_upper_95']:.6f}]"
        )

        print(
            "Bootstrap differences above zero:",
            f"{row['bootstrap_positive_percentage']:.2f}%"
        )

        print(
            "95% CI excludes zero:",
            row["ci_excludes_zero"]
        )

    print("\nSaved results:")
    print(OUTPUT_FILE)

    print("\nBootstrap analysis complete.")


if __name__ == "__main__":
    main()
