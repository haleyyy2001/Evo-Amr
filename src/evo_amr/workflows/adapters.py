"""Adapter stubs for HPC Orchestration source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .slurm import SlurmConfig, SlurmCommand, render_sbatch_dry_run


@dataclass(frozen=True)
class AMRPredLaunchAdapter:
    """Adapter for prediction/workflows/launch_training.py."""

    project_root: Path
    env_name: str
    slurm_out_dir: Path
    config: SlurmConfig

    def plan(self, experiment_config: object) -> str:
        """Render a dry-run sbatch script for a training job."""
        command = (
            f"python {self.project_root}/train.py <config>"
        )
        script = render_sbatch_dry_run(self.config, command)
        return script.script()

    def describe(self) -> str:
        return (
            f"AMRPredLaunchAdapter(env={self.env_name}, "
            f"job={self.config.job_name})"
        )


@dataclass(frozen=True)
class DiagnosticsSBatchAdapter:
    """Adapter for diagnostics/run_diagnostics.sbatch."""

    base_dir: Path
    model_dir: Path
    output_dir: Path
    config: SlurmConfig = SlurmConfig(
        job_name="evo_amr_diagnostics",
        partition="gpu",
        gres="gpu:1",
        cpus_per_task=8,
        memory="100G",
        time="120:00:00",
    )

    def plan(self, experiment_config: object) -> str:
        """Render a dry-run sbatch for the 32-layer diagnostic sweep."""
        command = (
            f"python {self.base_dir}/evo-hidden/examples/layer_sweep.py "
            f"--model_dir {self.model_dir} --sequence <DNA_SEQUENCE>"
        )
        script = render_sbatch_dry_run(self.config, command)
        return script.script()

    def describe(self) -> str:
        return f"DiagnosticsSBatchAdapter(model_dir={self.model_dir})"


@dataclass(frozen=True)
class AMRBenchmarkingOrchestrationAdapter:
    """Adapter for benchmarking/main.sh."""

    benchmarking_root: Path
    config_yaml: Path

    def plan(self, experiment_config: object) -> str:
        raise NotImplementedError(
            "Wrap benchmarking/main.sh behind the ShellBaselineAdapter. "
            "See src/evo_amr/baselines/adapters.py."
        )

    def describe(self) -> str:
        return f"AMRBenchmarkingOrchestrationAdapter(root={self.benchmarking_root})"
