
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_classifier(y_true, y_pred):
    """
    Evaluate a binary human-written vs AI-generated classifier.

    Expected labels:
        0 = Human-written
        1 = AI-generated
    """

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
                "false_positive_rate": (
            confusion_matrix(y_true, y_pred, labels=[0, 1])[0][1]
            / max(sum(y == 0 for y in y_true), 1)
        ),

        "false_negative_rate": (
            confusion_matrix(y_true, y_pred, labels=[0, 1])[1][0]
            / max(sum(y == 1 for y in y_true), 1)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Human-written", "AI-generated"],
            zero_division=0,
            output_dict=True,
        ),
    }

    return results
