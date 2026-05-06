"""Shared result schemas for AMR benchmarking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultRecord:
    """Normalized per-partition result row."""

    experiment: str
    antibiotic: str
    partition: str
    representation: str
    aggregation: str
    model: str
    metric: str
    value: float
    split_replicate: str | None = None
    species: str | None = None

    def as_dict(self) -> dict[str, str | float | None]:
        """Serialize the result record."""
        return {
            "experiment": self.experiment,
            "antibiotic": self.antibiotic,
            "partition": self.partition,
            "representation": self.representation,
            "aggregation": self.aggregation,
            "model": self.model,
            "metric": self.metric,
            "value": self.value,
            "split_replicate": self.split_replicate,
            "species": self.species,
        }
