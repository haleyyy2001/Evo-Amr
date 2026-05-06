"""Adapter stubs for Manifest → Split source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AMRPredCVAdapter:
    """Adapter for prediction/lib/cv.py — KFold, GroupKFold, LOGO."""

    design: str
    k_folds: int = 5
    random_state: int = 112

    def assign(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError(
            "Migrate prediction/lib/cv.py::define_cv_kfold and define_cv_logo "
            "into src/evo_amr/splits/."
        )

    def describe(self) -> str:
        return f"AMRPredCVAdapter(design={self.design}, k={self.k_folds})"


@dataclass(frozen=True)
class AMRBenchmarkingFoldsAdapter:
    """Adapter for benchmarking/src/cv_folds/prepare_folds.py."""

    strategy: str
    cv: int = 10
    temp_path: Path = Path("temp")

    def assign(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError(
            "Migrate benchmarking/src/cv_folds/prepare_folds.py "
            "into src/evo_amr/splits/."
        )

    def describe(self) -> str:
        return f"AMRBenchmarkingFoldsAdapter(strategy={self.strategy}, cv={self.cv})"


@dataclass(frozen=True)
class KMAClusterFoldsAdapter:
    """Adapter for benchmarking/src/cv_folds/cluster2folds.py."""

    cluster_file: Path
    k_folds: int = 10

    def assign(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        raise NotImplementedError(
            "Migrate benchmarking/src/cv_folds/cluster2folds.py "
            "into src/evo_amr/splits/."
        )

    def describe(self) -> str:
        return f"KMAClusterFoldsAdapter(cluster_file={self.cluster_file})"
