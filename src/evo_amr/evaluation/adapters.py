"""Adapter stubs for Model → Evaluation source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LightningEvaluatorAdapter:
    """Adapter for prediction/evaluate.py — checkpoint-based evaluation."""

    checkpoint_path: Path
    run_name: str
    dataset: str
    data_split: str

    def evaluate(self, records: list[dict[str, object]]) -> dict[str, float]:
        raise NotImplementedError(
            "Migrate prediction/evaluate.py into src/evo_amr/evaluation/."
        )

    def describe(self) -> str:
        return (
            f"LightningEvaluatorAdapter(run={self.run_name}, "
            f"split={self.data_split}, ckpt={self.checkpoint_path.name})"
        )


@dataclass(frozen=True)
class AMRBenchmarkingAnalyzerAdapter:
    """Adapter for benchmarking/src/analysis_utility/result_analysis.py."""

    software_name: str
    level: str
    cv: int = 10
    temp_path: Path = Path("temp")

    def evaluate(self, records: list[dict[str, object]]) -> dict[str, float]:
        raise NotImplementedError(
            "Migrate benchmarking/src/analysis_utility/result_analysis.py "
            "into src/evo_amr/evaluation/."
        )

    def describe(self) -> str:
        return (
            f"AMRBenchmarkingAnalyzerAdapter("
            f"software={self.software_name}, level={self.level})"
        )


@dataclass(frozen=True)
class MiniRocketEvaluatorAdapter:
    """Adapter for per-partition evaluation in amr_classification_pipeline.py."""

    partitions: tuple[str, ...] = (
        "val_overlapped",
        "val_outside",
        "test_overlapped",
        "test_outside",
    )

    def evaluate(self, records: list[dict[str, object]]) -> dict[str, float]:
        raise NotImplementedError(
            "Migrate per-partition evaluation from "
            "minirocket/minirocket_pipeline/amr_classification_pipeline.py."
        )

    def describe(self) -> str:
        return f"MiniRocketEvaluatorAdapter(partitions={self.partitions})"
