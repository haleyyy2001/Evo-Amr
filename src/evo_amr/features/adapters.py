"""Adapter stubs for Representation → FeatureTransform source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MiniRocketAdapter:
    """Adapter for the MiniRocket transform in amr_classification_pipeline.py."""

    num_kernels: int = 2000
    window_size: int = 8192
    stride: int = 4096

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        raise NotImplementedError(
            "Migrate MiniRocket transform from "
            "minirocket/minirocket_pipeline/amr_classification_pipeline.py "
            "into src/evo_amr/features/."
        )

    def describe(self) -> str:
        return (
            f"MiniRocketAdapter(num_kernels={self.num_kernels}, "
            f"window={self.window_size}, stride={self.stride})"
        )


@dataclass(frozen=True)
class PCATransformAdapter:
    """Adapter for PCA dimensionality reduction (41-dim default from thesis)."""

    n_components: int = 41
    embedding_dir: Path = Path("embeddings")

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        raise NotImplementedError(
            "Migrate PCA transform from "
            "scripts/Z_stats_pca_analysis.py and "
            "minirocket/minirocket_pipeline/amr_classification_pipeline.py."
        )

    def describe(self) -> str:
        return f"PCATransformAdapter(n_components={self.n_components})"


@dataclass(frozen=True)
class SRPTransformAdapter:
    """Adapter for Sparse Random Projection (baseline to PCA)."""

    n_components: int = 41
    density: str = "auto"

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        raise NotImplementedError(
            "Migrate SRP transform from "
            "minirocket/minirocket_pipeline/amr_classification_pipeline.py."
        )

    def describe(self) -> str:
        return f"SRPTransformAdapter(n_components={self.n_components})"


@dataclass(frozen=True)
class MeanPoolAdapter:
    """Mean pooling over ordered window embeddings (already implemented)."""

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        from .transforms import mean_pool
        return [mean_pool(X)]

    def describe(self) -> str:
        return "MeanPoolAdapter(global mean over all windows)"
