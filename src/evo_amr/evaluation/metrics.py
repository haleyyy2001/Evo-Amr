"""Small metric helpers used by lightweight examples."""

from __future__ import annotations

from math import sqrt


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Compute accuracy without requiring scikit-learn."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def confusion_counts(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    """Return binary confusion-matrix counts."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            counts["tp"] += 1
        elif actual == 0 and predicted == 0:
            counts["tn"] += 1
        elif actual == 0 and predicted == 1:
            counts["fp"] += 1
        elif actual == 1 and predicted == 0:
            counts["fn"] += 1
        else:
            raise ValueError("Binary metrics expect labels in {0, 1}")
    return counts


def precision(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary precision."""
    counts = confusion_counts(y_true, y_pred)
    denom = counts["tp"] + counts["fp"]
    return counts["tp"] / denom if denom else 0.0


def recall(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary recall."""
    counts = confusion_counts(y_true, y_pred)
    denom = counts["tp"] + counts["fn"]
    return counts["tp"] / denom if denom else 0.0


def f1_score(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary F1."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def matthews_corrcoef(y_true: list[int], y_pred: list[int]) -> float:
    """Compute binary Matthews correlation coefficient."""
    c = confusion_counts(y_true, y_pred)
    denom = sqrt(
        (c["tp"] + c["fp"])
        * (c["tp"] + c["fn"])
        * (c["tn"] + c["fp"])
        * (c["tn"] + c["fn"])
    )
    return ((c["tp"] * c["tn"]) - (c["fp"] * c["fn"])) / denom if denom else 0.0


def binary_classification_summary(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Compute a compact binary classification metric summary."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
