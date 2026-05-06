"""Adapter stubs for Evaluation → Report source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AMRBenchmarkingReportAdapter:
    """Adapter for benchmarking/src/analysis_utility/result_analysis.py."""

    output_path: Path
    cv: int = 10
    score_set: tuple[str, ...] = (
        "f1_macro", "f1_positive", "f1_negative",
        "accuracy", "mcc", "auc",
    )

    def render(self, results: list[dict[str, object]]) -> str:
        raise NotImplementedError(
            "Migrate result_analysis.py into src/evo_amr/reporting/."
        )

    def describe(self) -> str:
        return f"AMRBenchmarkingReportAdapter(output={self.output_path})"


@dataclass(frozen=True)
class DatasetAnalyzerReportAdapter:
    """Adapter for scripts/dataset_analyzer.py."""

    output_path: Path
    taxonomy: str = "gtdb"

    def render(self, results: list[dict[str, object]]) -> str:
        raise NotImplementedError(
            "Migrate scripts/dataset_analyzer.py into src/evo_amr/reporting/."
        )

    def describe(self) -> str:
        return f"DatasetAnalyzerReportAdapter(taxonomy={self.taxonomy})"


@dataclass(frozen=True)
class MiniRocketReportAdapter:
    """Adapter for the plot/report section of amr_classification_pipeline.py."""

    output_dir: Path
    antibiotic: str

    def render(self, results: list[dict[str, object]]) -> str:
        raise NotImplementedError(
            "Migrate visualization and metric reporting from "
            "minirocket/minirocket_pipeline/amr_classification_pipeline.py."
        )

    def describe(self) -> str:
        return f"MiniRocketReportAdapter(antibiotic={self.antibiotic})"
