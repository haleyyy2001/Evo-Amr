"""Registry of Evaluation → Report source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportScript:
    """A legacy reporting, visualization, or analysis script."""

    name: str
    source: Path
    output_kind: str
    action: str
    notes: str = ""


REPORT_SCRIPTS: dict[str, ReportScript] = {
    "amr_benchmarking_result_analysis": ReportScript(
        name="amr_benchmarking_result_analysis",
        source=Path("benchmarking/src/analysis_utility/result_analysis.py"),
        output_kind="nested_cv_table",
        action="MIGRATE",
        notes="Mean±std summary tables over CV folds; clinical and standard thresholds.",
    ),
    "amr_benchmarking_visualization": ReportScript(
        name="amr_benchmarking_visualization",
        source=Path("benchmarking/scripts/analysis_visualization/"),
        output_kind="multi_model_comparison",
        action="ADAPT",
        notes="Shell scripts orchestrating comparison/supplement figure generation.",
    ),
    "amr_pred_plotting": ReportScript(
        name="amr_pred_plotting",
        source=Path("prediction/analysis/plotting.py"),
        output_kind="sweep_visualization",
        action="MIGRATE",
        notes="Seaborn boxplot/swarmplot for hyperparameter sweep results.",
    ),
    "pca_stats_analysis": ReportScript(
        name="pca_stats_analysis",
        source=Path("scripts/Z_stats_pca_analysis.py"),
        output_kind="partition_distribution",
        action="MIGRATE",
        notes="Per-partition PCA statistics; reads *_pca_stats_dim41.npz files.",
    ),
    "dataset_analyzer_report": ReportScript(
        name="dataset_analyzer_report",
        source=Path("scripts/dataset_analyzer.py"),
        output_kind="dataset_statistics",
        action="MIGRATE",
        notes="Partition-wise species/R-S statistics log; stable pandas logic.",
    ),
    "compute_drug_split_stats": ReportScript(
        name="compute_drug_split_stats",
        source=Path("prediction/compute_drug_split_stats.py"),
        output_kind="split_statistics",
        action="MIGRATE",
        notes="Per-drug genome counts and R/S ratios across splits.",
    ),
}


def get_report_script(name: str) -> ReportScript:
    try:
        return REPORT_SCRIPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown report script: {name}") from exc


def list_report_scripts(output_kind: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, s in sorted(REPORT_SCRIPTS.items())
        if output_kind is None or s.output_kind == output_kind
    )
