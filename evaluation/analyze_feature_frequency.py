
import numpy as np
import pandas as pd
import joblib

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = BASE_DIR / "models" / "artifacts" / "tfidf_logistic.joblib"
TRAIN_FILE = BASE_DIR / "datasets" / "processed" / "train.csv"

OUTPUT_FILE = BASE_DIR / "evaluation" / "tfidf_feature_frequency.csv"

MIN_DOCUMENT_FREQUENCY = 5


def analyze_feature_frequency():

    # Load model and original training data
    model = joblib.load(MODEL_FILE)
    train_df = pd.read_csv(TRAIN_FILE)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    if list(classifier.classes_) != [0, 1]:
        raise ValueError("Unexpected class labels.")

    # Apply the model's original preprocessing
    texts = model.named_steps["preprocessor"].transform(
        train_df["text"].fillna("")
    )

    # Transform training documents into TF-IDF features
    X = vectorizer.transform(texts)

    # Convert feature values to binary presence/absence
    X_binary = X.copy()
    X_binary.data[:] = 1

    y = train_df["label"].to_numpy()

    # Count documents containing each feature in each class
    human_counts = np.asarray(
        X_binary[y == 0].sum(axis=0)
    ).ravel()

    ai_counts = np.asarray(
        X_binary[y == 1].sum(axis=0)
    ).ravel()

    # Feature coefficients
    coefficients = classifier.coef_[0]
    features = vectorizer.get_feature_names_out()

    results = pd.DataFrame({
        "feature": features,
        "coefficient": coefficients,
        "human_documents": human_counts,
        "ai_documents": ai_counts,
    })

    results["total_documents"] = (
        results["human_documents"] + results["ai_documents"]
    )

    # Calculate percentage of documents in each class
    human_total = (y == 0).sum()
    ai_total = (y == 1).sum()

    results["human_percentage"] = (
        results["human_documents"] / human_total * 100
    ).round(2)

    results["ai_percentage"] = (
        results["ai_documents"] / ai_total * 100
    ).round(2)

    results["frequency_difference"] = (
        results["ai_percentage"] - results["human_percentage"]
    ).round(2)

    # Export all features
    results.to_csv(OUTPUT_FILE, index=False)

    # Exclude extremely rare features from the printed rankings
    reliable_candidates = results[
        results["total_documents"] >= MIN_DOCUMENT_FREQUENCY
    ]

    ai_features = reliable_candidates.nlargest(
        30, "coefficient"
    )

    human_features = reliable_candidates.nsmallest(
        30, "coefficient"
    )

    columns = [
        "feature",
        "coefficient",
        "human_documents",
        "ai_documents",
        "frequency_difference",
    ]

    print("\n===== K-Detect: Feature Frequency Analysis =====")

    print("\nTop 30 AI-associated features:")
    print(ai_features[columns].to_string(index=False))

    print("\nTop 30 Human-associated features:")
    print(human_features[columns].to_string(index=False))

    print("\nTotal vocabulary:", len(features))
    print("Features appearing in at least 5 documents:",
          len(reliable_candidates))
    print("Training documents:", len(train_df))
    print("Human documents:", human_total)
    print("AI documents:", ai_total)

    print("\nFull results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    analyze_feature_frequency()
