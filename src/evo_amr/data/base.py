"""Abstract interface for Dataset → Manifest stage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ManifestSource(Protocol):
    """Produces manifest rows from a raw data source."""

    def load(self) -> list[dict[str, str]]:
        """Load rows in the shared manifest schema."""
        ...

    def describe(self) -> str:
        """Return a one-line description for dry-run logs."""
        ...


@runtime_checkable
class PhenotypeNormalizer(Protocol):
    """Maps raw phenotype labels to binary 0/1."""

    def normalize(self, value: object) -> int | None:
        """Return 0, 1, or None (intermediate/unknown)."""
        ...
