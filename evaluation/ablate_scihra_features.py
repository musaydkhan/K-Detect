
import pandas as pd

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from core.model_preprocessor import clean_text_batch

from config.model_config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER,
    RANDOM_STATE,
)


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "train.csv"
)

VALIDATION_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "validation.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_ablation_results.csv"
)


CLASS_NAMES = {
    0: "Human-written",
    1: "AI-revised",
    2: "AI-generated",
}


def build_model(vocabulary=None):

    vectorizer_args = {
        "ngram_range": TFIDF_NGRAM_RANGE,
        "lowercase": True,
        "strip_accents": "unicode",
        "sublinear_tf": True,
    }

    if vocabulary is None:
        vectorizer_args["max_features"] = TFIDF_MAX_FEATURES
    else:
        vectorizer_args["vocabulary"] = vocabulary

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
            TfidfVectorizer(**vectorizer_args),
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


def get_top_features(model, class_index, count=30):

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    features = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[class_index]

    indices = coefficients.argsort()[-count:]

    return set(features[indices])


def evaluate_predictions(name, y_true, predictions):

    return {
        "experiment": name,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_precision": precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),
    }


def run_ablation():

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VALIDATION_FILE)

    X_train = train_df["text"].fillna("")
    y_train = train_df["label"]

    X_val = val_df["text"].fillna("")
    y_val = val_df["label"]

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    # Train the baseline model
    print("\nTraining baseline model...")

    baseline_model = build_model()

    baseline_model.fit(X_train, y_train)

    baseline_predictions = baseline_model.predict(X_val)

    baseline_result = evaluate_predictions(
        "Baseline",
        y_val,
        baseline_predictions
    )

    print(
        "Baseline Macro F1:",
        round(baseline_result["macro_f1"], 4)
    )

    # Identify top features from training data only
    human_features = get_top_features(
        baseline_model, 0
    )

    revised_features = get_top_features(
        baseline_model, 1
    )

    generated_features = get_top_features(
        baseline_model, 2
    )

    feature_groups = {
        "Remove Human Features": human_features,
        "Remove AI-revised Features": revised_features,
        "Remove AI-generated Features": generated_features,
        "Remove All Top Features": (
            human_features
            | revised_features
            | generated_features
        ),
    }

    results = [baseline_result]

    # Run ablation experiments
    for experiment_name, removed_features in feature_groups.items():

        print(f"\nRunning: {experiment_name}")
        print(
            "Features selected for removal:",
            len(removed_features)
        )

        # Keep only features not selected for removal.
        # Re-index vocabulary consecutively for scikit-learn.
        original_vocabulary = (
            baseline_model
            .named_steps["tfidf"]
            .vocabulary_
        )

        remaining_features = sorted(
            feature
            for feature in original_vocabulary
            if feature not in removed_features
        )

        filtered_vocabulary = {
            feature: index
            for index, feature in enumerate(remaining_features)
        }

        if not filtered_vocabulary:
            raise ValueError(
                "No vocabulary features remain after ablation."
            )

        ablated_model = build_model(
            vocabulary=filtered_vocabulary
        )

        ablated_model.fit(X_train, y_train)

        predictions = ablated_model.predict(X_val)

        result = evaluate_predictions(
            experiment_name,
            y_val,
            predictions
        )

        results.append(result)

        print(
            "Accuracy:",
            round(result["accuracy"], 4)
        )

        print(
            "Macro F1:",
            round(result["macro_f1"], 4)
        )

        print(
            "Remaining vocabulary:",
            len(filtered_vocabulary)
        )

    # Save results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== FINAL ABLATION RESULTS =====")

    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    run_ablation()
