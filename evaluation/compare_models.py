import pandas as pd
import joblib

from sklearn.metrics import accuracy_score


# Model locations
WORD_MODEL_FILE = "models/artifacts/tfidf_logistic.joblib"
CHAR_MODEL_FILE = "models/artifacts/char_svm.joblib"

# Validation dataset
VALIDATION_FILE = "datasets/processed/validation.csv"


def compare_models():

    # Load both trained models
    word_model = joblib.load(WORD_MODEL_FILE)
    char_model = joblib.load(CHAR_MODEL_FILE)

    # Load validation data
    df = pd.read_csv(VALIDATION_FILE)

    X_val = df["text"].fillna("")
    y_true = df["label"]

    # Generate predictions
    word_predictions = word_model.predict(X_val)
    char_predictions = char_model.predict(X_val)

    # Compare predictions
    word_correct = word_predictions == y_true
    char_correct = char_predictions == y_true

    both_correct = word_correct & char_correct
    both_wrong = (~word_correct) & (~char_correct)

    word_only_correct = word_correct & (~char_correct)
    char_only_correct = char_correct & (~word_correct)

    # Display results
    print("\n===== K-Detect: Model Disagreement Analysis =====")

    print("Total validation samples:", len(df))

    print(
        "Word model accuracy:",
        round(accuracy_score(y_true, word_predictions), 4)
    )

    print(
        "Character model accuracy:",
        round(accuracy_score(y_true, char_predictions), 4)
    )

    print("\n===== Prediction Agreement =====")

    print("Both models correct:", both_correct.sum())
    print("Both models wrong:", both_wrong.sum())

    print("Only word model correct:", word_only_correct.sum())
    print("Only character model correct:", char_only_correct.sum())

    print(
        "Models disagree:",
        (word_predictions != char_predictions).sum()
    )
    # Export disagreement samples
    disagreement_mask = word_predictions != char_predictions

    disagreement_df = df.loc[disagreement_mask].copy()

    disagreement_df["word_prediction"] = word_predictions[disagreement_mask]
    disagreement_df["char_prediction"] = char_predictions[disagreement_mask]

    disagreement_df["actual_label"] = y_true[disagreement_mask]

    output_file = "datasets/processed/model_disagreements.csv"

    disagreement_df.to_csv(output_file, index=False)

    print("\nDisagreement samples exported!")
    print("Output:", output_file)


if __name__ == "__main__":
    compare_models()