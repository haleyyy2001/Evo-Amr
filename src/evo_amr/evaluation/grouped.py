"""Grouped evaluation utilities for species, drug, and cluster audits."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from .metrics import binary_classification_summary


def grouped_binary_classification_summary(
    rows: Iterable[Mapping[str, object]],
    group_key: str,
    truth_key: str = "phenotype",
    pred_key: str = "prediction",
) -> dict[str, dict[str, float]]:
    """Compute binary metrics separately for each species/drug/cluster group."""
    grouped_truth: dict[str, list[int]] = defaultdict(list)
    grouped_pred: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        group = str(row[group_key])
        grouped_truth[group].append(int(row[truth_key]))
        grouped_pred[group].append(int(row[pred_key]))

    return {
        group: binary_classification_summary(grouped_truth[group], grouped_pred[group])
        for group in sorted(grouped_truth)
    }


def inverse_frequency_weights(groups: Sequence[str]) -> list[float]:
    """Return sample weights giving each group equal total mass."""
    if not groups:
        return []
    counts: dict[str, int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    group_mass = 1.0 / len(counts)
    return [group_mass / counts[group] for group in groups]
