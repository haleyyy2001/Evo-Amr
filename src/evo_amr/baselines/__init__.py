"""Wrappers for classical AMR baseline backends."""

from .adapters import BackendCommand, ShellBaselineAdapter
from .registry import (
    BaselineBackend,
    DEFAULT_BASELINE_BACKENDS,
    get_baseline_backend,
    list_baseline_backends,
)

__all__ = [
    "BackendCommand",
    "BaselineBackend",
    "DEFAULT_BASELINE_BACKENDS",
    "ShellBaselineAdapter",
    "get_baseline_backend",
    "list_baseline_backends",
]
