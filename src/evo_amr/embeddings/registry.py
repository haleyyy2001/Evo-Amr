"""Registry of Split → Representation source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingScript:
    """A legacy embedding extraction or diagnostic script."""

    name: str
    source: Path
    model_family: str
    action: str
    notes: str = ""


EMBEDDING_SCRIPTS: dict[str, EmbeddingScript] = {
    "evo_embedding_generator": EmbeddingScript(
        name="evo_embedding_generator",
        source=Path("embedding_pipeline/embedding_generator.py"),
        model_family="evo",
        action="ADAPT",
        notes=(
            "Production GPU embedding generator (~700 lines). "
            "Multi-GPU, sliding window (8192/4096), HDF5 output. "
            "Wrap in EvoEmbeddingAdapter; do not reimplement GPU logic."
        ),
    ),
    "evo_diagnostics_sbatch": EmbeddingScript(
        name="evo_diagnostics_sbatch",
        source=Path("diagnostics/run_diagnostics.sbatch"),
        model_family="evo",
        action="ADAPT",
        notes="SLURM job running 32-layer diagnostic sweep. Wrap in SlurmCommand.",
    ),
    "evo_diagnostic_analysis_sbatch": EmbeddingScript(
        name="evo_diagnostic_analysis_sbatch",
        source=Path("diagnostics/diagnostic_analysis.sbatch"),
        model_family="evo",
        action="ADAPT",
        notes="Post-hoc analysis of layer sweep results.",
    ),
    "amr_pred_esm_loader": EmbeddingScript(
        name="amr_pred_esm_loader",
        source=Path("prediction/lib/data.py"),
        model_family="esm",
        action="MIGRATE",
        notes=(
            "Loads ESM1b/ESM2 .pt files, aggregates protein embeddings to genome level. "
            "Generalise to also support Evo HDF5 during migration."
        ),
    ),
    "amr_pred_simple_reps": EmbeddingScript(
        name="amr_pred_simple_reps",
        source=Path("prediction/simple_model/simple_model.py"),
        model_family="esm",
        action="MIGRATE",
        notes="compute_or_load_X: mean/sum over per-protein ESM embeddings.",
    ),
}


def get_embedding_script(name: str) -> EmbeddingScript:
    try:
        return EMBEDDING_SCRIPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown embedding script: {name}") from exc


def list_embedding_scripts(model_family: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, s in sorted(EMBEDDING_SCRIPTS.items())
        if model_family is None or s.model_family == model_family
    )
