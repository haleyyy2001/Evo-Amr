"""Dataset filtering policies mined from the legacy AMR preprocessing scripts.

The original AMR_benchmarking preprocessing keeps only species-drug tasks with
enough genomes, applies a loose/strict phenotype quality level, and normalizes
phenotype labels before fold construction. This module captures those rules as
inspectable configuration objects that can later wrap the legacy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


RESISTANT_LABELS = {"1", "r", "resistant", "resistance", "non-susceptible", "nonsusceptible"}
SUSCEPTIBLE_LABELS = {"0", "s", "susceptible", "sensitive"}


@dataclass(frozen=True)
class DatasetFilterConfig:
    """Filtering policy for species-antibiotic AMR tasks."""

    min_genomes_per_species_antibiotic: int = 500
    qc_level: str = "loose"
    drop_intermediate: bool = True

    def describe(self) -> str:
        """Render the policy for dry-run reports."""
        intermediate = "drop" if self.drop_intermediate else "keep"
        return (
            "DatasetFilterConfig("
            f"min_genomes_per_species_antibiotic={self.min_genomes_per_species_antibiotic}, "
            f"qc_level={self.qc_level}, intermediate={intermediate})"
        )


def normalize_binary_phenotype(value: object) -> int | None:
    """Normalize common AMR phenotype labels to 0/1.

    Returns None for intermediate, missing, or unknown labels so callers can
    decide whether to drop or audit them.
    """
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in RESISTANT_LABELS:
        return 1
    if label in SUSCEPTIBLE_LABELS:
        return 0
    return None


def species_antibiotic_counts(
    rows: Iterable[Mapping[str, object]],
    species_key: str = "species",
    antibiotic_key: str = "antibiotic",
) -> dict[tuple[str, str], int]:
    """Count genomes by species-antibiotic task."""
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        species = str(row.get(species_key, "")).strip()
        antibiotic = str(row.get(antibiotic_key, "")).strip()
        if not species or not antibiotic:
            continue
        key = (species, antibiotic)
        counts[key] = counts.get(key, 0) + 1
    return counts


def eligible_species_antibiotic_tasks(
    rows: Iterable[Mapping[str, object]],
    config: DatasetFilterConfig = DatasetFilterConfig(),
) -> dict[tuple[str, str], int]:
    """Return species-drug tasks meeting the configured minimum sample size."""
    counts = species_antibiotic_counts(rows)
    return {
        task: count
        for task, count in counts.items()
        if count >= config.min_genomes_per_species_antibiotic
    }
