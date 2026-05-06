"""Abstract interface for HPC Orchestration stage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkflowStep(Protocol):
    """Plans or executes one step in an HPC-aware research workflow."""

    def plan(self, config: object) -> str:
        """Return a dry-run description of what would be executed."""
        ...

    def describe(self) -> str:
        """Return a one-line description for logs."""
        ...
