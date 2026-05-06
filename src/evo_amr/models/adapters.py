"""Adapter stubs for FeatureTransform → Model source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LightningTrainerAdapter:
    """Adapter for prediction/train.py — PyTorch Lightning training."""

    config_path: Path
    run_name: str
    dataset: str
    use_gpu: bool = True

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        raise NotImplementedError(
            "Migrate prediction/train.py and prediction/lib/lightning_modules.py "
            "into src/evo_amr/models/."
        )

    def predict(self, X: list[list[float]]) -> list[float]:
        raise NotImplementedError(
            "Use prediction/evaluate.py pending migration."
        )

    def describe(self) -> str:
        return (
            f"LightningTrainerAdapter(run={self.run_name}, "
            f"dataset={self.dataset}, gpu={self.use_gpu})"
        )


@dataclass(frozen=True)
class SimpleModelAdapter:
    """Adapter for prediction/simple_model/ — LR over precomputed embeddings."""

    dataset: str
    embedding_model: str = "esm2_t33_650M_UR50D"
    cv_design: str = "kfold"
    k_folds: int = 10

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        raise NotImplementedError(
            "Migrate prediction/simple_model/simple_model.py "
            "into src/evo_amr/models/."
        )

    def predict(self, X: list[list[float]]) -> list[float]:
        raise NotImplementedError

    def describe(self) -> str:
        return (
            f"SimpleModelAdapter(dataset={self.dataset}, "
            f"model={self.embedding_model}, cv={self.cv_design})"
        )


@dataclass(frozen=True)
class MiniRocketClassifierAdapter:
    """Adapter for the classifier grid in amr_classification_pipeline.py."""

    classifiers: tuple[str, ...] = ("lr", "svm", "rf")
    cv_folds: int = 5

    def fit(self, X: list[list[float]], y: list[int]) -> None:
        raise NotImplementedError(
            "Migrate classifier grid from "
            "minirocket/minirocket_pipeline/amr_classification_pipeline.py "
            "into src/evo_amr/models/."
        )

    def predict(self, X: list[list[float]]) -> list[float]:
        raise NotImplementedError

    def describe(self) -> str:
        classifiers = ", ".join(self.classifiers)
        return f"MiniRocketClassifierAdapter(classifiers=[{classifiers}])"
