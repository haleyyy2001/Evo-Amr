"""Small feature transforms for lightweight examples."""

from __future__ import annotations


def mean_pool(vectors: list[list[float]]) -> list[float]:
    """Mean-pool a sequence of equal-length vectors."""
    if not vectors:
        return []
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("All vectors must have the same length")
    return [sum(vector[idx] for vector in vectors) / len(vectors) for idx in range(width)]
