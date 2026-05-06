"""Abstract interface for Manifest → Split stage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SplitStrategy(Protocol):
    """Assigns a partition_label to each manifest row."""

    def assign(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Return rows with partition_label populated."""
        ...

    def describe(self) -> str:
        """Return a one-line description for dry-run logs."""
        ...


@runtime_checkable
class LeakageChecker(Protocol):
    """Audits a split assignment for cross-partition species leakage."""

    def check(self, rows: list[dict[str, str]]) -> dict[str, set[str]]:
        """Return species that appear in both train and outside partitions."""
        ...
