"""Abstract interface for Representation → FeatureTransform stage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FeatureTransform(Protocol):
    """Converts ordered embedding windows into fixed-size feature vectors."""

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        """Fit and transform the input embeddings."""
        ...

    def describe(self) -> str:
        """Return a one-line description for dry-run logs."""
        ...


@runtime_checkable
class FeatureTransformPlan(Protocol):
    """Dry-run description of a feature transform without data access."""

    def describe(self) -> str:
        """Return a human-readable plan for the transform."""
        ...
