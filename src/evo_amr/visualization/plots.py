"""Dry-run descriptors for visualization outputs.

The heavy plotting work — actual figure rendering — lives in::

    minirocket/minirocket_pipeline/visualization/

This module provides lightweight :class:`PlotSpec` descriptors that describe
*what would be plotted* without importing matplotlib or seaborn.  They are
used for dry-run CLI output and testing.

Plot types produced by real experiments
----------------------------------------

neighbour_audit
    For each genome in a held-out partition, retrieve its k nearest neighbours
    in the embedding space (by cosine distance) and record whether each
    neighbour shares the same species and phenotype.  Reveals how MiniRocket
    reorganises the neighbourhood structure relative to global pooling.

phylogenetic_curve
    Per-species AUROC or MCC as a function of phylogenetic distance (average
    amino-acid identity, AAI) from the training species set.  Tests whether
    cross-phylum transfer degrades monotonically with evolutionary distance.

method_comparison
    Heatmap of AUROC / MCC across all (aggregation × classifier) combinations
    for a given antibiotic and evaluation partition.  Identifies which
    aggregation–classifier pairs perform best on cross-species genomes.

species_heatmap
    Per-species accuracy or MCC for each held-out species, shown as a grid
    with aggregation methods on one axis and species on the other.  Rows are
    sorted by the MiniRocket − GlobalPool delta, highlighting species where
    local pattern preservation matters most.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotSpec:
    """Lightweight description of a planned figure for dry-run output.

    Attributes
    ----------
    kind:
        Figure type — one of ``neighbour_audit``, ``phylogenetic_curve``,
        ``method_comparison``, ``species_heatmap``.
    title:
        Figure title string.
    x_label:
        Horizontal axis label.
    y_label:
        Vertical axis label.
    output_path:
        Intended filesystem path for the saved ``.png`` or ``.pdf``.
    note:
        Optional extra context shown in dry-run output.
    """

    kind: str
    title: str
    x_label: str
    y_label: str
    output_path: str
    note: str = ""

    def describe(self) -> str:
        """Render a human-readable dry-run description."""
        lines = [
            f"[plot] kind={self.kind}",
            f"[plot] title={self.title}",
            f"[plot] axes=({self.x_label}, {self.y_label})",
            f"[plot] output={self.output_path}",
        ]
        if self.note:
            lines.append(f"[plot] note={self.note}")
        return "\n".join(lines)


def neighbour_audit_spec(
    genome_id: str,
    aggregation: str,
    output_path: str,
    k: int = 5,
) -> PlotSpec:
    """Describe a k-NN neighbour audit figure."""
    return PlotSpec(
        kind="neighbour_audit",
        title=f"k-NN neighbours ({aggregation}) — {genome_id}",
        x_label="neighbour rank",
        y_label="phenotype match (0/1)",
        output_path=output_path,
        note=f"k={k}; checks whether neighbours share species and phenotype label",
    )


def phylogenetic_curve_spec(
    antibiotic: str,
    metric: str,
    output_path: str,
) -> PlotSpec:
    """Describe a phylogenetic generalisation curve figure."""
    return PlotSpec(
        kind="phylogenetic_curve",
        title=f"Phylogenetic generalisation — {antibiotic}",
        x_label="phylogenetic distance from training species (AAI)",
        y_label=metric.upper(),
        output_path=output_path,
        note="x-axis is GTDB-derived average amino-acid identity distance",
    )


def method_comparison_spec(
    antibiotic: str,
    partition: str,
    metric: str,
    output_path: str,
) -> PlotSpec:
    """Describe an aggregation × classifier comparison heatmap figure."""
    return PlotSpec(
        kind="method_comparison",
        title=f"Aggregation × classifier — {antibiotic} / {partition}",
        x_label="classifier",
        y_label="aggregation method",
        output_path=output_path,
        note=f"colour = {metric.upper()}; rows ordered by mean performance",
    )


def species_heatmap_spec(
    antibiotic: str,
    partition: str,
    metric: str,
    output_path: str,
) -> PlotSpec:
    """Describe a per-species accuracy heatmap figure."""
    return PlotSpec(
        kind="species_heatmap",
        title=f"Per-species {metric.upper()} — {antibiotic} / {partition}",
        x_label="aggregation method",
        y_label="held-out species",
        output_path=output_path,
        note="rows sorted by MiniRocket − GlobalPool delta",
    )
