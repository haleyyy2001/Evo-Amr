#!/usr/bin/env python3
"""
Basic Evo-AMR Workflow Example
=============================

Demonstrates the species-holdout evaluation protocol that is the core
methodological contribution of this project. Works without GPU or real
genomic data using the bundled tiny fixture manifest.

Usage:
    # Run with the bundled demo manifest (no data required):
    python examples/scripts/basic_workflow.py

    # Run with your own manifest:
    python examples/scripts/basic_workflow.py --manifest path/to/manifest.csv

The tiny demo shows:
  1. Manifest loading and schema validation
  2. Species-leakage checking (the key evaluation safeguard)
  3. Partition summary (train / val_outside / test_outside split)
  4. Majority-class baseline evaluation
  5. Markdown report generation

For full embedding extraction and MiniRocket classification, see:
  docs/project_overview.md  — methodology
  embedding_pipeline/       — Evo Layer 10 extraction (requires GPU + data)
  minirocket/               — MiniRocket downstream pipeline
"""

import argparse
import sys
from pathlib import Path

# Ensure the package is importable when running from the repo root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from evo_amr.data.manifest import load_manifest, validate_manifest_rows
from evo_amr.evaluation.metrics import binary_classification_summary
from evo_amr.models import MajorityClassifier
from evo_amr.reporting import render_tiny_report
from evo_amr.splits import outside_species_leakage, summarize_partitions


DEFAULT_MANIFEST = repo_root / "examples" / "tiny_manifest.csv"


def run(manifest_path: Path) -> None:
    print(f"Loading manifest: {manifest_path}")
    rows = load_manifest(manifest_path)
    validate_manifest_rows(rows)
    print(f"  {len(rows)} genomes loaded and validated\n")

    # Check for species leakage between train and outside partitions.
    # This is the core safeguard of the species-holdout evaluation design.
    leakage = outside_species_leakage(rows)
    if leakage:
        print(f"WARNING: species leakage detected in outside partitions: {leakage}")
    else:
        print("Species leakage check passed — no train species appear in outside partitions\n")

    # Summarise partition sizes.
    summary = summarize_partitions(rows)
    print("Partition summary:")
    for partition, stats in summary.items():
        print(f"  {partition}: {stats['n_genomes']} genomes, {stats['n_species']} species")
    print()

    # Majority-class baseline (lower bound — real results use Evo embeddings).
    labels = [int(row["phenotype"]) for row in rows]
    predictions = MajorityClassifier().fit(labels).predict(len(labels))
    metrics = binary_classification_summary(labels, predictions)

    print("Majority-class baseline metrics (lower bound):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print()

    # Generate markdown report.
    report = render_tiny_report(
        experiment_name="demo_ampicillin_species_holdout",
        partition_summary=summary,
        metrics=metrics,
    )
    out_path = Path("workflow_demo_report.md")
    out_path.write_text(report)
    print(f"Report written to: {out_path}")
    print("\nDone. To run real experiments, see docs/project_overview.md.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evo-AMR demo: species-holdout evaluation on a tiny manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to manifest CSV (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found at {args.manifest}")
        return 1

    run(args.manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
