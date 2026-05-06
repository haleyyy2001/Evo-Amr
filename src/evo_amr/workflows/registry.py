"""Registry of HPC Orchestration source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkflowScript:
    """A legacy HPC launcher, SLURM job, or orchestration script."""

    name: str
    source: Path
    backend: str
    action: str
    notes: str = ""


WORKFLOW_SCRIPTS: dict[str, WorkflowScript] = {
    "amr_pred_launch_training": WorkflowScript(
        name="amr_pred_launch_training",
        source=Path("prediction/workflows/launch_training.py"),
        backend="slurm",
        action="ADAPT",
        notes=(
            "Config merge + submit_dynamic_sbatch for single/multi-drug training. "
            "Hardcoded server paths; wrap sbatch submission, migrate config merging."
        ),
    ),
    "amr_pred_launch_baseline": WorkflowScript(
        name="amr_pred_launch_baseline",
        source=Path("prediction/workflows/launch_baseline_training.py"),
        backend="slurm",
        action="ADAPT",
        notes="Same pattern as launch_training but for simple model baselines.",
    ),
    "amr_pred_launch_sweep": WorkflowScript(
        name="amr_pred_launch_sweep",
        source=Path("prediction/workflows/launch_sweep.py"),
        backend="slurm + wandb",
        action="ADAPT",
        notes="W&B sweep config + SLURM array job submission.",
    ),
    "amr_pred_workflow_utils": WorkflowScript(
        name="amr_pred_workflow_utils",
        source=Path("prediction/utils/workflow_utils.py"),
        backend="slurm",
        action="ADAPT",
        notes="submit_dynamic_sbatch: generates and submits sbatch from template.",
    ),
    "evo_diagnostics_sbatch": WorkflowScript(
        name="evo_diagnostics_sbatch",
        source=Path("diagnostics/run_diagnostics.sbatch"),
        backend="slurm",
        action="ADAPT",
        notes="32-layer Evo diagnostic sweep; GPU job on Insomnia/Manitou.",
    ),
    "evo_diagnostic_analysis_sbatch": WorkflowScript(
        name="evo_diagnostic_analysis_sbatch",
        source=Path("diagnostics/diagnostic_analysis.sbatch"),
        backend="slurm",
        action="ADAPT",
        notes="Post-hoc analysis of layer sweep results.",
    ),
    "amr_benchmarking_main": WorkflowScript(
        name="amr_benchmarking_main",
        source=Path("benchmarking/main.sh"),
        backend="bash",
        action="ADAPT",
        notes="Top-level orchestration: preprocessing → folds → baselines → analysis.",
    ),
    "amr_benchmarking_model_scripts": WorkflowScript(
        name="amr_benchmarking_model_scripts",
        source=Path("benchmarking/scripts/model/"),
        backend="bash",
        action="ADAPT",
        notes="Per-baseline shell launchers; invoked by ShellBaselineAdapter.",
    ),
}


def get_workflow_script(name: str) -> WorkflowScript:
    try:
        return WORKFLOW_SCRIPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown workflow script: {name}") from exc


def list_workflow_scripts(backend: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, s in sorted(WORKFLOW_SCRIPTS.items())
        if backend is None or backend in s.backend
    )
