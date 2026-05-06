"""Split summary utilities."""

from __future__ import annotations

from collections import Counter, defaultdict


def summarize_partitions(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    """Count rows and unique species per partition."""
    counts = Counter(row["partition_label"] for row in rows)
    species = defaultdict(set)
    for row in rows:
        species[row["partition_label"]].add(row["species"])

    return {
        partition: {
            "n_genomes": count,
            "n_species": len(species[partition]),
        }
        for partition, count in sorted(counts.items())
    }
