
import joblib
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "artifacts"
    / "scihra"
    / "tfidf_logistic_3class.joblib"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "scihra_feature_importance.csv"
)

CLASS_NAMES = {
    0: "Human-written",
    1: "AI-revised",
    2: "AI-generated",
}


def analyze_features():

    model = joblib.load(MODEL_FILE)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    features = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_

    results = []

    for class_index, class_label in enumerate(
        classifier.classes_
    ):

        class_name = CLASS_NAMES[class_label]
        class_coefficients = coefficients[class_index]

        top_indices = class_coefficients.argsort()[-30:][::-1]

        print(f"\n===== Top Features: {class_name} =====")

        for index in top_indices:

            feature = features[index]
            coefficient = class_coefficients[index]

            results.append({
                "class": class_name,
                "feature": feature,
                "coefficient": coefficient,
            })

            print(
                f"{feature:<30} "
                f"{coefficient:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature analysis completed!")
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    analyze_features()
