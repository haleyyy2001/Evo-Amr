"""Registry of Manifest → Split source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitScript:
    """A legacy split construction or fold generation script."""

    name: str
    source: Path
    strategy: str
    action: str
    notes: str = ""


SPLIT_SCRIPTS: dict[str, SplitScript] = {
    "amr_pred_cv": SplitScript(
        name="amr_pred_cv",
        source=Path("prediction/lib/cv.py"),
        strategy="kfold / logo / stratified / group",
        action="MIGRATE",
        notes=(
            "Most complete CV implementation across all three systems. "
            "Supports KFold, StratifiedKFold, GroupKFold by species/genus, LOGO. "
            "Use as canonical implementation in src/evo_amr/splits/."
        ),
    ),
    "amr_benchmarking_prepare_folds": SplitScript(
        name="amr_benchmarking_prepare_folds",
        source=Path("benchmarking/src/cv_folds/prepare_folds.py"),
        strategy="random / kma_cluster / phylogeny",
        action="MIGRATE",
        notes="Central fold logic for AMR_benchmarking; three CV strategies.",
    ),
    "amr_benchmarking_cluster2folds": SplitScript(
        name="amr_benchmarking_cluster2folds",
        source=Path("benchmarking/src/cv_folds/cluster2folds.py"),
        strategy="kma_cluster",
        action="MIGRATE",
        notes="KMA cluster membership → balanced fold assignment.",
    ),
    "amr_benchmarking_random_folds": SplitScript(
        name="amr_benchmarking_random_folds",
        source=Path("benchmarking/src/cv_folds/generate_random_folds.py"),
        strategy="random",
        action="MIGRATE",
    ),
    "amr_pred_data_splits": SplitScript(
        name="amr_pred_data_splits",
        source=Path("prediction/data_prep/"),
        strategy="clustered / random / stratified",
        action="MIGRATE",
        notes="config-driven split construction; reads configs.json per split design.",
    ),
    "compute_drug_split_stats": SplitScript(
        name="compute_drug_split_stats",
        source=Path("prediction/compute_drug_split_stats.py"),
        strategy="reporting",
        action="MIGRATE",
        notes="Per-drug summary statistics across splits; maps to Split → Report.",
    ),
}


def get_split_script(name: str) -> SplitScript:
    try:
        return SPLIT_SCRIPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown split script: {name}") from exc


def list_split_scripts(strategy: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, s in sorted(SPLIT_SCRIPTS.items())
        if strategy is None or s.strategy == strategy
    )
