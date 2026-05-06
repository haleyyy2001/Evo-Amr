"""Abstract interface for Split → Representation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Produces genome embeddings from a manifest."""

    def dry_run_command(self, manifest: str | Path) -> str:
        """Return the backend command string without launching it."""
        ...

    def describe(self) -> str:
        """Return a multi-line description of the backend configuration."""
        ...
