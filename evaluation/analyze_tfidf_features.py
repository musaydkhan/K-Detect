
import pandas as pd
import joblib

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "tfidf_logistic.joblib"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "tfidf_feature_importance.csv"
)


def analyze_tfidf_features():

    # Load trained pipeline
    model = joblib.load(MODEL_FILE)

    # Extract TF-IDF vectorizer and classifier
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    # Get vocabulary and learned coefficients
    features = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    # Create feature importance table
    importance_df = pd.DataFrame({
        "feature": features,
        "coefficient": coefficients
    })

    # Sort by coefficient
    importance_df = importance_df.sort_values(
        "coefficient",
        ascending=False
    )

    # Most AI-associated features
    ai_features = importance_df.head(30)

    # Most human-associated features
    human_features = importance_df.tail(30).sort_values(
        "coefficient"
    )

    print("\n===== K-Detect: TF-IDF Feature Analysis =====")

    print("\nTop 30 AI-associated features:")
    print(ai_features.to_string(index=False))

    print("\nTop 30 Human-associated features:")
    print(human_features.to_string(index=False))

    # Export all features
    importance_df.to_csv(OUTPUT_FILE, index=False)

    print("\nTotal vocabulary features:", len(features))
    print("Feature analysis exported to:", OUTPUT_FILE)


if __name__ == "__main__":
    analyze_tfidf_features()
