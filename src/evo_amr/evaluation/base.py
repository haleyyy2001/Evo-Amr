"""Abstract interface for Model → Evaluation stage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Evaluator(Protocol):
    """Computes metrics from a collection of prediction records."""

    def evaluate(self, records: list[dict[str, object]]) -> dict[str, float]:
        """Return metric name → value for the given prediction records."""
        ...

    def describe(self) -> str:
        """Return a one-line description for dry-run logs."""
        ...


@runtime_checkable
class GroupedEvaluator(Protocol):
    """Evaluates per-group (species, drug, partition) breakdowns."""

    def evaluate_grouped(
        self,
        records: list[dict[str, object]],
        group_key: str,
    ) -> dict[str, dict[str, float]]:
        """Return group → metric dict for each group in the key."""
        ...
