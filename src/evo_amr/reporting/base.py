"""Abstract interface for Evaluation → Report stage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reporter(Protocol):
    """Renders a human-readable report from evaluation results."""

    def render(self, results: list[dict[str, object]]) -> str:
        """Return the report as a string (markdown, TSV, or plain text)."""
        ...

    def describe(self) -> str:
        """Return a one-line description for dry-run logs."""
        ...
