"""Registry of Representation → FeatureTransform source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureScript:
    """A legacy feature transform or aggregation script."""

    name: str
    source: Path
    transform: str
    action: str
    notes: str = ""


FEATURE_SCRIPTS: dict[str, FeatureScript] = {
    "minirocket_pipeline": FeatureScript(
        name="minirocket_pipeline",
        source=Path("minirocket/minirocket_pipeline/amr_classification_pipeline.py"),
        transform="minirocket",
        action="MIGRATE",
        notes=(
            "~2000 lines; primary ML experiment driver. "
            "MiniRocket (2000 kernels → PPV), global mean pool, PCA (41-dim), SRP, MIL. "
            "Migrate transform logic into features/; classifier grid into models/."
        ),
    ),
    "pca_analysis": FeatureScript(
        name="pca_analysis",
        source=Path("scripts/Z_stats_pca_analysis.py"),
        transform="pca",
        action="MIGRATE",
        notes="Loads *_pca_stats_dim41.npz files, computes per-partition statistics.",
    ),
    "amr_pred_representations": FeatureScript(
        name="amr_pred_representations",
        source=Path("prediction/lib/data.py"),
        transform="mean_pool / sum_pool",
        action="MIGRATE",
        notes=(
            "compute_representations: mean/sum reduction over protein embeddings. "
            "Generalise to Evo window embeddings during migration."
        ),
    ),
    "amr_pred_cached_reps": FeatureScript(
        name="amr_pred_cached_reps",
        source=Path("prediction/utils/data_utils.py"),
        transform="precomputed",
        action="MIGRATE",
        notes="load_cached_simple_reps: loads pre-cached numpy arrays from disk.",
    ),
}


def get_feature_script(name: str) -> FeatureScript:
    try:
        return FEATURE_SCRIPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown feature script: {name}") from exc


def list_feature_scripts(transform: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, s in sorted(FEATURE_SCRIPTS.items())
        if transform is None or s.transform == transform
    )
