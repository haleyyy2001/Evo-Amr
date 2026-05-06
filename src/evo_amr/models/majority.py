"""Simple baseline classifiers used by examples and sanity checks."""

from __future__ import annotations


class MajorityClassifier:
    """Predict the majority class observed during fitting."""

    def __init__(self) -> None:
        self.label_: int | None = None

    def fit(self, y: list[int]) -> "MajorityClassifier":
        """Fit the majority label."""
        if not y:
            raise ValueError("Cannot fit MajorityClassifier on empty labels")
        self.label_ = int(sum(y) >= (len(y) / 2))
        return self

    def predict(self, n: int) -> list[int]:
        """Predict the fitted majority label n times."""
        if self.label_ is None:
            raise RuntimeError("MajorityClassifier must be fit before predict")
        return [self.label_] * n
