"""Adapters for external AMR baseline backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendCommand:
    """A backend command plus working directory metadata."""

    argv: tuple[str, ...]
    cwd: Path | None = None

    def render(self) -> str:
        """Render the command for dry-run logs."""
        prefix = f"(cd {self.cwd} && " if self.cwd else ""
        suffix = ")" if self.cwd else ""
        return prefix + " ".join(self.argv) + suffix


@dataclass(frozen=True)
class ShellBaselineAdapter:
    """Thin wrapper around an existing shell-based baseline backend."""

    name: str
    script: Path
    backend_root: Path | None = None

    def dry_run_command(self, config_path: str | Path) -> BackendCommand:
        """Return the command that would be launched for this baseline."""
        return BackendCommand(
            argv=("bash", str(self.script), "--config", str(config_path)),
            cwd=self.backend_root,
        )
