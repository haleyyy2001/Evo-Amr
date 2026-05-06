"""Leakage checks for species-aware AMR splits."""

from __future__ import annotations


def species_by_partition(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """Return species sets grouped by partition label."""
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(row["partition_label"], set()).add(row["species"])
    return grouped


def outside_species_leakage(
    rows: list[dict[str, str]],
    train_partition: str = "train",
    outside_partitions: tuple[str, ...] = ("val_outside", "test_outside"),
) -> dict[str, set[str]]:
    """Find species that appear in both train and outside partitions."""
    grouped = species_by_partition(rows)
    train_species = grouped.get(train_partition, set())
    return {
        partition: train_species & grouped.get(partition, set())
        for partition in outside_partitions
        if train_species & grouped.get(partition, set())
    }
