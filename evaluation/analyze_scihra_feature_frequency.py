
import pandas as pd
import joblib

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "tfidf_logistic_3class.joblib"
)

TRAIN_FILE = (
    BASE_DIR
    / "datasets"
    / "processed"
    / "scihra_splits"
    / "train.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_feature_comparison.csv"
)

CLASS_NAMES = {
    0: "Human-written",
    1: "AI-revised",
    2: "AI-generated",
}


def analyze_feature_frequency():

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(TRAIN_FILE)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    features = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_

    results = []

    for class_index, class_label in enumerate(
        classifier.classes_
    ):

        class_name = CLASS_NAMES[class_label]

        top_indices = coefficients[
            class_index
        ].argsort()[-30:][::-1]

        top_features = [
            features[index]
            for index in top_indices
        ]

        for feature in top_features:

            row = {
                "target_class": class_name,
                "feature": feature,
            }

            for label, name in CLASS_NAMES.items():

                class_texts = df.loc[
                    df["label"] == label,
                    "text"
                ].fillna("").str.lower()

                count = class_texts.str.contains(
                    feature,
                    regex=False
                ).sum()

                percentage = (
                    count / len(class_texts) * 100
                )

                row[name + "_percentage"] = round(
                    percentage, 2
                )

            results.append(row)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== Cross-Class Feature Comparison =====")

    for class_name in CLASS_NAMES.values():

        print(f"\nTop features associated with: {class_name}")

        subset = results_df[
            results_df["target_class"] == class_name
        ]

        print(
            subset.to_string(index=False)
        )

    print("\nAnalysis completed!")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    analyze_feature_frequency()
