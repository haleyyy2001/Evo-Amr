"""Layer-selection diagnostics from the thesis.

The thesis identifies Evo-1-8k-base Layer 10 as the deepest stable extraction
layer before a sharp Layer 11 stability boundary. This module records that
decision as framework metadata so configs and dry-runs can validate layer
choices.
"""

from __future__ import annotations

from dataclasses import dataclass


RECOMMENDED_EVO_LAYER = 10
STABILITY_BOUNDARY_LAYER = 11


@dataclass(frozen=True)
class LayerDiagnosticConfig:
    """Configuration for Evo layer diagnostic sweeps."""

    model: str = "evo-1-8k-base"
    layers: tuple[int, ...] = tuple(range(32))
    dtype: str = "bfloat16"
    metrics: tuple[str, ...] = (
        "activation_scale",
        "isotropy",
        "effective_rank",
        "singular_spectrum",
        "token_norm_concentration",
        "cross_seed_stability",
    )
    recommended_layer: int = RECOMMENDED_EVO_LAYER
    stability_boundary: int = STABILITY_BOUNDARY_LAYER


def validate_layer_choice(layer: int) -> tuple[bool, str]:
    """Validate an Evo extraction layer against thesis diagnostics."""
    if layer == RECOMMENDED_EVO_LAYER:
        return True, "Layer 10 is the thesis-selected deepest stable extraction layer."
    if layer >= STABILITY_BOUNDARY_LAYER:
        return (
            False,
            "Layer choice is at or beyond the Layer 11 stability boundary.",
        )
    return (
        True,
        "Layer is before the stability boundary but is shallower than the thesis default.",
    )
