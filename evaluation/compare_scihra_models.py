
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parents[1]

WORD_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "tfidf_logistic_3class.joblib"
)

CHAR_MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "char_svm_3class.joblib"
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
    / "scihra_model_disagreement.csv"
)

CLASS_NAMES = {
    0: "Human-written",
    1: "AI-revised",
    2: "AI-generated",
}


def compare_models():

    print("Loading models...")

    word_model = joblib.load(WORD_MODEL_FILE)
    char_model = joblib.load(CHAR_MODEL_FILE)

    df = pd.read_csv(VALIDATION_FILE)

    X_val = df["text"].fillna("")
    y_true = df["label"]

    print("Generating predictions...")

    word_predictions = word_model.predict(X_val)
    char_predictions = char_model.predict(X_val)

    # Add predictions to the validation data
    results = df.copy()

    results["word_prediction"] = word_predictions
    results["char_prediction"] = char_predictions

    results["word_correct"] = (
        results["word_prediction"] == results["label"]
    )

    results["char_correct"] = (
        results["char_prediction"] == results["label"]
    )

    results["models_agree"] = (
        results["word_prediction"]
        == results["char_prediction"]
    )

    # Overall agreement
    agreement = results["models_agree"].mean()

    both_correct = (
        results["word_correct"]
        & results["char_correct"]
    ).sum()

    both_wrong = (
        ~results["word_correct"]
        & ~results["char_correct"]
    ).sum()

    only_word_correct = (
        results["word_correct"]
        & ~results["char_correct"]
    ).sum()

    only_char_correct = (
        ~results["word_correct"]
        & results["char_correct"]
    ).sum()

    print("\n===== MODEL AGREEMENT =====")

    print("Validation samples:", len(results))

    print(
        "Agreement:",
        round(agreement * 100, 2),
        "%"
    )

    print(
        "Disagreement:",
        int((~results["models_agree"]).sum())
    )

    print("\n===== CORRECTNESS COMPARISON =====")

    print("Both correct:", int(both_correct))
    print("Both wrong:", int(both_wrong))
    print("Only word model correct:", int(only_word_correct))
    print("Only character model correct:", int(only_char_correct))

    print("\n===== ACCURACY COMPARISON =====")

    print(
        "Word TF-IDF:",
        round(
            accuracy_score(y_true, word_predictions),
            4
        )
    )

    print(
        "Character SVM:",
        round(
            accuracy_score(y_true, char_predictions),
            4
        )
    )

    # Analyze disagreements by actual class
    print("\n===== DISAGREEMENT BY CLASS =====")

    for label, class_name in CLASS_NAMES.items():

        subset = results[results["label"] == label]

        disagreement_count = (
            ~subset["models_agree"]
        ).sum()

        print(
            f"{class_name}: "
            f"{disagreement_count}/{len(subset)} "
            f"({disagreement_count / len(subset) * 100:.2f}%)"
        )

    # Analyze the direction of disagreements
    print("\n===== DISAGREEMENT DETAILS =====")

    disagreements = results[
        ~results["models_agree"]
    ]

    if disagreements.empty:

        print("No disagreements found.")

    else:

        for (word_pred, char_pred), group in disagreements.groupby(
            ["word_prediction", "char_prediction"]
        ):

            print(
                f"Word model: {CLASS_NAMES[word_pred]} | "
                f"Character model: {CLASS_NAMES[char_pred]} "
                f"| Samples: {len(group)}"
            )

    # Save detailed results
    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nComparison completed!")
    print("Detailed results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    compare_models()
