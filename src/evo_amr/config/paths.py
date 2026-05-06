"""Path profile abstraction for local and server-dependent research runs.

The old amr_pred code reads many paths from environment variables. This module
keeps that useful idea but makes it config-driven and dry-run friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Mapping


@dataclass(frozen=True)
class PathProfile:
    """Named collection of roots used to resolve experiment paths."""

    name: str
    roots: Mapping[str, str]

    def require(self, key: str) -> str:
        """Return a root path or raise a clear error."""
        if key not in self.roots:
            raise KeyError(f"path profile '{self.name}' missing root '{key}'")
        return self.roots[key]

    def resolve(self, template: str) -> str:
        """Resolve a Template-style path using this profile's roots."""
        return Template(template).safe_substitute(dict(self.roots))

    def describe(self) -> str:
        """Render roots without checking filesystem availability."""
        roots = ", ".join(f"{key}={value}" for key, value in sorted(self.roots.items()))
        return f"PathProfile(name={self.name}, {roots})"


def path_profile_from_config(name: str, config: Mapping[str, object]) -> PathProfile:
    """Build a profile from a YAML mapping."""
    roots = config.get("roots", config)
    if not isinstance(roots, Mapping):
        raise TypeError("path profile config must contain a mapping of roots")
    return PathProfile(name=name, roots={str(key): str(value) for key, value in roots.items()})
