
import sys
import joblib
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_FILE = (
    BASE_DIR / "datasets" / "processed" / "raid_abstracts.csv"
)

ARTIFACT_DIR = (
    BASE_DIR / "models" / "artifacts" / "scihra"
)

OUTPUT_DIR = BASE_DIR / "evaluation"

# --------------------------------------------------
# MODEL FILES
# --------------------------------------------------

MODEL_FILES = {
    "TF-IDF Logistic Regression": "tfidf_logistic_3class.joblib",
    "Character SVM": "calibrated_char_svm.joblib",
    "Ensemble Word Model": "ensemble_word_model.joblib",
}

# --------------------------------------------------
# PREDICTION FUNCTIONS
# --------------------------------------------------

def get_ai_probability(model, texts):
    """
    Convert three-class probabilities into binary AI scores.

    Class 0 = Human-written
    Class 1 = AI-revised
    Class 2 = AI-generated

    Binary AI score = P(class 1) + P(class 2)
    """

    probabilities = model.predict_proba(texts)

    return probabilities[:, 1] + probabilities[:, 2]


def calculate_metrics(y_true, ai_scores):
    """
    Calculate binary classification metrics.
    Threshold = 0.50
    """

    y_pred = (ai_scores >= 0.50).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, ai_scores),
        "false_positive_rate": fp / max(tn + fp, 1),
        "false_negative_rate": fn / max(fn + tp, 1),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("K-DETECT: RAID EXTERNAL EVALUATION")
    print("=" * 65)

    # Load dataset
    print("\nLoading RAID dataset...")

    df = pd.read_csv(DATA_FILE)

    print("Total records:", len(df))

    # Separate human and AI records
    human_df = df[df["model"] == "human"].copy()

    ai_df = df[
        (df["model"] != "human")
        & (df["repetition_penalty"] == "no")
        & (df["decoding"].isin(["greedy", "sampling"]))
    ].copy()

    print("Human records:", len(human_df))
    print("Standard AI records:", len(ai_df))

    # Validate unique human source IDs
    if human_df["source_id"].duplicated().any():
        raise ValueError(
            "Duplicate human source IDs detected."
        )

    human_lookup = human_df.set_index("source_id")["generation"]

    # Validate AI source matching
    missing_sources = (
        set(ai_df["source_id"]) - set(human_lookup.index)
    )

    if missing_sources:
        raise ValueError(
            f"AI records without matching human sources: "
            f"{len(missing_sources)}"
        )

    # Load models
    print("\nLoading frozen SciHRA models...")

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = ARTIFACT_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(model_path)

        models[model_name] = joblib.load(model_path)

        print("Loaded:", model_name)

    # Cache human predictions once per model
    print("\nCalculating human reference scores...")

    human_texts = human_df["generation"].tolist()

    human_scores = {}

    for model_name, model in models.items():

        human_scores[model_name] = get_ai_probability(
            model,
            human_texts
        )

    # Prepare evaluation
    summary_results = []
    prediction_results = []

    ai_models = sorted(ai_df["model"].unique())
    decoding_conditions = ["greedy", "sampling"]

    print("\nStarting external evaluation...")

    total_conditions = len(ai_models) * len(decoding_conditions)

    condition_number = 0

    for ai_model_name in ai_models:

        for decoding in decoding_conditions:

            condition_number += 1

            subset = ai_df[
                (ai_df["model"] == ai_model_name)
                & (ai_df["decoding"] == decoding)
            ].copy()

            subset = subset.sort_values("source_id")

            matched_sources = subset["source_id"].tolist()

            matched_human_texts = [
                human_lookup.loc[source_id]
                for source_id in matched_sources
            ]

            ai_texts = subset["generation"].tolist()

            print(
                f"\n[{condition_number}/{total_conditions}] "
                f"{ai_model_name} | {decoding} | "
                f"{len(subset)} samples"
            )

            # Evaluate each classifier
            for classifier_name, model in models.items():

                # Human scores aligned to source IDs
                human_score_map = dict(
                    zip(
                        human_df["source_id"],
                        human_scores[classifier_name]
                    )
                )

                current_human_scores = [
                    human_score_map[source_id]
                    for source_id in matched_sources
                ]

                # AI generation scores
                current_ai_scores = get_ai_probability(
                    model,
                    ai_texts
                )

                # Construct paired binary evaluation arrays
                y_true = [0] * len(subset) + [1] * len(subset)

                scores = (
                    current_human_scores
                    + current_ai_scores.tolist()
                )

                metrics = calculate_metrics(
                    y_true,
                    pd.Series(scores).to_numpy()
                )

                summary_results.append({
                    "ai_model": ai_model_name,
                    "decoding": decoding,
                    "classifier": classifier_name,
                    "sample_pairs": len(subset),
                    **metrics,
                })

                # Save paired predictions
                for index, source_id in enumerate(matched_sources):

                    prediction_results.append({
                        "source_id": source_id,
                        "ai_model": ai_model_name,
                        "decoding": decoding,
                        "classifier": classifier_name,
                        "human_ai_score": current_human_scores[index],
                        "ai_generation_score": current_ai_scores[index],
                        "human_prediction": int(
                            current_human_scores[index] >= 0.50
                        ),
                        "ai_prediction": int(
                            current_ai_scores[index] >= 0.50
                        ),
                    })

                print(
                    f"  {classifier_name}: "
                    f"F1={metrics['f1']:.4f}, "
                    f"FPR={metrics['false_positive_rate']:.4f}, "
                    f"FNR={metrics['false_negative_rate']:.4f}"
                )

    # --------------------------------------------------
    # CREATE TRUE EQUAL-WEIGHT ENSEMBLE
    # --------------------------------------------------

    print("\nCalculating equal-weight ensemble...")

    word_name = "Ensemble Word Model"
    char_name = "Character SVM"

    ensemble_rows = []

    summary_df = pd.DataFrame(summary_results)
    predictions_df = pd.DataFrame(prediction_results)

    for ai_model_name in ai_models:

        for decoding in decoding_conditions:

            word_rows = predictions_df[
                (predictions_df["ai_model"] == ai_model_name)
                & (predictions_df["decoding"] == decoding)
                & (predictions_df["classifier"] == word_name)
            ].sort_values("source_id")

            char_rows = predictions_df[
                (predictions_df["ai_model"] == ai_model_name)
                & (predictions_df["decoding"] == decoding)
                & (predictions_df["classifier"] == char_name)
            ].sort_values("source_id")

            if (
                len(word_rows) != len(char_rows)
                or word_rows["source_id"].tolist()
                != char_rows["source_id"].tolist()
            ):
                raise ValueError(
                    "Word and character predictions are not aligned."
                )

            human_ensemble_scores = (
                word_rows["human_ai_score"].to_numpy()
                + char_rows["human_ai_score"].to_numpy()
            ) / 2

            ai_ensemble_scores = (
                word_rows["ai_generation_score"].to_numpy()
                + char_rows["ai_generation_score"].to_numpy()
            ) / 2

            y_true = (
                [0] * len(word_rows)
                + [1] * len(word_rows)
            )

            ensemble_scores = (
                human_ensemble_scores.tolist()
                + ai_ensemble_scores.tolist()
            )

            metrics = calculate_metrics(
                y_true,
                pd.Series(ensemble_scores).to_numpy()
            )

            ensemble_rows.append({
                "ai_model": ai_model_name,
                "decoding": decoding,
                "classifier": "Equal-weight Ensemble",
                "sample_pairs": len(word_rows),
                **metrics,
            })

            for index, source_id in enumerate(
                word_rows["source_id"].tolist()
            ):

                prediction_results.append({
                    "source_id": source_id,
                    "ai_model": ai_model_name,
                    "decoding": decoding,
                    "classifier": "Equal-weight Ensemble",
                    "human_ai_score": human_ensemble_scores[index],
                    "ai_generation_score": ai_ensemble_scores[index],
                    "human_prediction": int(
                        human_ensemble_scores[index] >= 0.50
                    ),
                    "ai_prediction": int(
                        ai_ensemble_scores[index] >= 0.50
                    ),
                })

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    final_summary = pd.concat(
        [summary_df, pd.DataFrame(ensemble_rows)],
        ignore_index=True
    )

    final_predictions = pd.DataFrame(prediction_results)

    summary_path = (
        OUTPUT_DIR / "raid_external_evaluation_summary.csv"
    )

    predictions_path = (
        OUTPUT_DIR / "raid_external_predictions.csv"
    )

    final_summary.to_csv(summary_path, index=False)

    final_predictions.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("RAID EXTERNAL EVALUATION COMPLETE")
    print("=" * 65)

    print("\nTotal evaluation conditions:", len(final_summary))
    print("Total paired prediction records:", len(final_predictions))

    print("\nAverage performance by classifier:")

    average_results = (
        final_summary
        .groupby("classifier")[
            [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "false_positive_rate",
                "false_negative_rate",
            ]
        ]
        .mean()
        .round(4)
    )

    print(average_results.to_string())

    print("\nSaved files:")
    print(summary_path)
    print(predictions_path)

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
