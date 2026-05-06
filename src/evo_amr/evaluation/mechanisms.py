"""Mechanism-aware aggregation guidance.

The thesis argues that AMR representation quality depends on the dominant
resistance mechanism in held-out species. This module captures that design
principle as lightweight metadata for reports and future experiment planners.
"""

from __future__ import annotations

from dataclasses import dataclass


CASSETTE_MEDIATED = "cassette_mediated"
CHROMOSOMAL_DIFFUSE = "chromosomal_diffuse"
MIXED_OR_UNKNOWN = "mixed_or_unknown"


@dataclass(frozen=True)
class AggregationRecommendation:
    """Recommended aggregation strategy for a mechanism regime."""

    mechanism_regime: str
    preferred_aggregation: str
    rationale: str


def recommend_aggregation(mechanism_regime: str) -> AggregationRecommendation:
    """Return the thesis-guided aggregation recommendation."""
    if mechanism_regime == CASSETTE_MEDIATED:
        return AggregationRecommendation(
            mechanism_regime=mechanism_regime,
            preferred_aggregation="minirocket",
            rationale=(
                "Local pattern preservation is well matched to plasmids, "
                "transposons, integrons, and cassette-scale resistance loci."
            ),
        )
    if mechanism_regime == CHROMOSOMAL_DIFFUSE:
        return AggregationRecommendation(
            mechanism_regime=mechanism_regime,
            preferred_aggregation="global_pooling",
            rationale=(
                "Genome-wide pooling is better aligned with diffuse, lineage-coupled, "
                "or chromosomal resistance signals."
            ),
        )
    return AggregationRecommendation(
        mechanism_regime=MIXED_OR_UNKNOWN,
        preferred_aggregation="ensemble_or_compare",
        rationale=(
            "When mechanism composition is unknown, compare local-pattern and "
            "global-summary representations rather than assuming one dominates."
        ),
    )
