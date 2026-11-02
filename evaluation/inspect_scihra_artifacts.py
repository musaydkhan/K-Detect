import sys
import joblib
from pathlib import Path


# Locate the project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Add project root to Python import path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


ARTIFACT_DIR = (
    BASE_DIR / "models" / "artifacts" / "scihra"
)

MODEL_FILES = [
    "tfidf_logistic_3class.joblib",
    "calibrated_char_svm.joblib",
    "ensemble_word_model.joblib",
]


def inspect_models():

    print("===== SCIHRA MODEL ARTIFACT INSPECTION =====")

    print("Project root:", BASE_DIR)

    for filename in MODEL_FILES:

        path = ARTIFACT_DIR / filename

        print(f"\n--- {filename} ---")

        if not path.exists():
            print("STATUS: FILE NOT FOUND")
            continue

        try:
            model = joblib.load(path)

            print("STATUS: LOADED")
            print("Model type:", type(model).__name__)

            print(
                "Classes:",
                getattr(model, "classes_", "Not directly exposed")
            )

            print(
                "Has predict:",
                callable(getattr(model, "predict", None))
            )

            print(
                "Has predict_proba:",
                callable(getattr(model, "predict_proba", None))
            )

            print(
                "Has decision_function:",
                callable(getattr(model, "decision_function", None))
            )

        except Exception as error:

            print("STATUS: LOAD FAILED")
            print("Error:", repr(error))

    print("\n===== INSPECTION COMPLETE =====")


if __name__ == "__main__":
    inspect_models()