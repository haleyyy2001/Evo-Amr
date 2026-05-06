"""Markdown report rendering."""

from __future__ import annotations


def render_tiny_report(
    experiment_name: str,
    partition_summary: dict[str, dict[str, int]],
    metrics: dict[str, float],
) -> str:
    """Render a compact markdown report for the tiny example workflow."""
    lines = [
        f"# {experiment_name}",
        "",
        "## Partition Summary",
        "",
        "| Partition | Genomes | Species |",
        "| --- | ---: | ---: |",
    ]
    for partition, values in partition_summary.items():
        lines.append(
            f"| {partition} | {values['n_genomes']} | {values['n_species']} |"
        )

    lines.extend(
        [
            "",
            "## Example Metric",
            "",
        ]
    )
    for name, value in metrics.items():
        lines.append(f"- {name}: {value:.3f}")
    lines.extend(
        [
            "",
            "This report is generated from a tiny fixture and is not a biological result.",
        ]
    )
    return "\n".join(lines) + "\n"
