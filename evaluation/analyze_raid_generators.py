
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "evaluation" / "raid_external_predictions.csv"
OUTPUT_FILE = BASE_DIR / "evaluation" / "raid_generator_analysis.csv"


def calculate_metrics(group):
    """
    Calculate generator-specific binary classification metrics.

    Human = 0
    AI-generated = 1
    """

    y_true = np.concatenate([
        np.zeros(len(group), dtype=int),
        np.ones(len(group), dtype=int),
    ])

    y_pred = np.concatenate([
        group["human_prediction"].to_numpy(),
        group["ai_prediction"].to_numpy(),
    ])

    scores = np.concatenate([
        group["human_ai_score"].to_numpy(),
        group["ai_generation_score"].to_numpy(),
    ])

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "samples": len(group),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "fpr": fp / (fp + tn) if (fp + tn) else 0,
        "fnr": fn / (fn + tp) if (fn + tp) else 0,
        "roc_auc": roc_auc_score(y_true, scores),
        "human_mean_ai_score": group["human_ai_score"].mean(),
        "generated_mean_ai_score": group["ai_generation_score"].mean(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def main():
    print("Loading RAID predictions...")

    df = pd.read_csv(INPUT_FILE)

    print("Total prediction rows:", len(df))
    print("Classifiers:", sorted(df["classifier"].unique()))
    print("Generators:", sorted(df["ai_model"].unique()))
    print("Decoding methods:", sorted(df["decoding"].unique()))

    results = []

    grouped = df.groupby(
        ["classifier", "ai_model", "decoding"],
        sort=True
    )

    for (classifier, model, decoding), group in grouped:

        metrics = calculate_metrics(group)

        results.append({
            "classifier": classifier,
            "ai_model": model,
            "decoding": decoding,
            **metrics,
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["classifier", "ai_model", "decoding"]
    )

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\n===== RAID GENERATOR-SPECIFIC ANALYSIS =====")

    display_columns = [
        "classifier",
        "ai_model",
        "decoding",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "roc_auc",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nSaved results to:")
    print(OUTPUT_FILE)

    print("\n===== ANALYSIS COMPLETE =====")


if __name__ == "__main__":
    main()
