"""A tiny end-to-end workflow used as an executable architecture example."""

from __future__ import annotations

from pathlib import Path

from evo_amr.config import load_yaml
from evo_amr.data.manifest import load_manifest, validate_manifest_rows
from evo_amr.evaluation.metrics import binary_classification_summary
from evo_amr.models import MajorityClassifier
from evo_amr.reporting import render_tiny_report
from evo_amr.splits import outside_species_leakage, summarize_partitions


def run_tiny_workflow(config_path: str | Path) -> str:
    """Run a deterministic toy workflow from manifest to markdown report."""
    config = load_yaml(config_path)
    manifest_path = config["dataset"]["manifest"]
    rows = load_manifest(manifest_path)
    validate_manifest_rows(rows)
    leakage = outside_species_leakage(rows)
    if leakage:
        raise ValueError(f"Species leakage detected in outside partitions: {leakage}")

    # Deterministic toy baseline: predict the majority label in the manifest.
    labels = [int(row["phenotype"]) for row in rows]
    predictions = MajorityClassifier().fit(labels).predict(len(labels))

    return render_tiny_report(
        experiment_name=config["experiment"]["name"],
        partition_summary=summarize_partitions(rows),
        metrics=binary_classification_summary(labels, predictions),
    )
