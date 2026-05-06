"""SLURM command rendering for server-dependent AMR workflows.

This is a non-executing counterpart to amr_pred's dynamic sbatch launchers. It
keeps HPC experience visible in the project while making dry-runs deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlurmConfig:
    """Resources for a single SLURM job."""

    job_name: str
    partition: str | None = None
    account: str | None = None
    time: str = "04:00:00"
    cpus_per_task: int = 4
    memory: str = "16G"
    gres: str | None = None
    output: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SlurmCommand:
    """Rendered sbatch script and command."""

    config: SlurmConfig
    body: str

    def header(self) -> str:
        """Render SBATCH directives."""
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={self.config.job_name}",
            f"#SBATCH --time={self.config.time}",
            f"#SBATCH --cpus-per-task={self.config.cpus_per_task}",
            f"#SBATCH --mem={self.config.memory}",
        ]
        optional = {
            "partition": self.config.partition,
            "account": self.config.account,
            "gres": self.config.gres,
            "output": self.config.output,
            "error": self.config.error,
        }
        for key, value in optional.items():
            if value:
                lines.append(f"#SBATCH --{key}={value}")
        return "\n".join(lines)

    def script(self) -> str:
        """Render the full sbatch script."""
        return f"{self.header()}\n\nset -euo pipefail\n{self.body.rstrip()}\n"


def render_sbatch_dry_run(config: SlurmConfig, command: str) -> SlurmCommand:
    """Create a dry-run sbatch script for a backend command."""
    return SlurmCommand(config=config, body=command)
