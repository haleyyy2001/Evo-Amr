"""Registry of Model → Evaluation source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalScript:
    """A legacy evaluation or metric aggregation script."""

    name: str
    source: Path
    scope: str
    action: str
    notes: str = ""


EVAL_SCRIPTS: dict[str, EvalScript] = {
    "amr_pred_evaluate": EvalScript(
        name="amr_pred_evaluate",
        source=Path("prediction/evaluate.py"),
        scope="single_drug",
        action="MIGRATE",
        notes="PyTorch Lightning evaluation entry point; loads checkpoint, runs predict.",
    ),
    "amr_pred_metrics": EvalScript(
        name="amr_pred_metrics",
        source=Path("prediction/analysis/metrics.py"),
        scope="binary_classification",
        action="MIGRATE",
        notes=(
            "compute_classification_metrics: AUROC, AUPRC, accuracy, "
            "sensitivity, specificity, F1; handles all-one-class edge case."
        ),
    ),
    "amr_pred_metric_utils": EvalScript(
        name="amr_pred_metric_utils",
        source=Path("prediction/utils/metric_utils.py"),
        scope="multi_drug",
        action="MIGRATE",
        notes="torchmetrics.MetricCollection init for single/multi-drug Lightning runs.",
    ),
    "amr_benchmarking_result_analysis": EvalScript(
        name="amr_benchmarking_result_analysis",
        source=Path("benchmarking/src/analysis_utility/result_analysis.py"),
        scope="nested_cv",
        action="MIGRATE",
        notes=(
            "Aggregates nested-CV JSON score files into mean±std tables. "
            "Primary reporting logic for the baseline suite."
        ),
    ),
    "amr_benchmarking_benchmark": EvalScript(
        name="amr_benchmarking_benchmark",
        source=Path("benchmarking/src/benchmark_utility/benchmark.py"),
        scope="baseline_suite",
        action="MIGRATE",
        notes="Orchestrates species × antibiotic × classifier evaluation loops.",
    ),
}


def get_eval_script(name: str) -> EvalScript:
    try:
        return EVAL_SCRIPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown eval script: {name}") from exc


def list_eval_scripts(scope: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, s in sorted(EVAL_SCRIPTS.items())
        if scope is None or s.scope == scope
    )
