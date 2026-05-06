"""Abstract interface for FeatureTransform → Model stage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AMRPredictor(Protocol):
    """Fits and predicts binary AMR phenotype from feature vectors."""

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        """Fit the predictor on feature matrix X and binary labels y."""
        ...

    def predict(self, X: list[list[float]]) -> list[float]:
        """Return predicted probabilities for each sample."""
        ...

    def describe(self) -> str:
        """Return a one-line description for dry-run logs."""
        ...


@runtime_checkable
class MultiDrugPredictor(Protocol):
    """Fits and predicts across multiple antibiotics simultaneously."""

    def fit(self, X: list[list[float]], Y: list[list[float | None]]) -> None:
        """Fit on feature matrix X and multi-drug label matrix Y (NaN = missing)."""
        ...

    def predict(self, X: list[list[float]]) -> list[list[float]]:
        """Return per-drug probabilities for each sample."""
        ...

    def describe(self) -> str:
        """Return a one-line description for dry-run logs."""
        ...
