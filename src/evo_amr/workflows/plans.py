"""Dry-run planning for CLI stages."""

from __future__ import annotations

from dataclasses import dataclass

from evo_amr.config.experiment import ExperimentConfig


@dataclass(frozen=True)
class StagePlan:
    """Human-readable description of a planned workflow stage."""

    stage: str
    experiment_name: str
    steps: tuple[str, ...]
    inputs: dict[str, str]
    outputs: dict[str, str]
    notes: tuple[str, ...] = ()

    def render(self) -> str:
        """Render the plan as CLI-friendly text."""
        lines = [
            f"[evo-amr] planned stage: {self.stage}",
            f"[evo-amr] experiment: {self.experiment_name}",
        ]
        if self.inputs:
            lines.append("[evo-amr] inputs:")
            lines.extend(
                f"[evo-amr]   - {name}: {value}"
                for name, value in sorted(self.inputs.items())
            )
        if self.outputs:
            lines.append("[evo-amr] outputs:")
            lines.extend(
                f"[evo-amr]   - {name}: {value}"
                for name, value in sorted(self.outputs.items())
            )
        if self.notes:
            lines.append("[evo-amr] notes:")
            lines.extend(f"[evo-amr]   - {note}" for note in self.notes)
        lines.append("[evo-amr] steps:")
        lines.extend(f"[evo-amr]   {idx}. {step}" for idx, step in enumerate(self.steps, 1))
        return "\n".join(lines)


def build_stage_plan(stage: str, config: ExperimentConfig) -> StagePlan:
    """Build a dry-run plan for a named CLI stage."""
    common: dict[str, tuple[str, ...]] = {
        "prepare-data": (
            "load source metadata and phenotype tables",
            "normalize genome/species/antibiotic identifiers",
            "write or validate a manifest",
        ),
        "create-splits": (
            "load manifest",
            "construct or validate random/species-holdout partitions",
            "check species leakage across outside partitions",
        ),
        "embed": (
            "load manifest sequence paths",
            "extract configured foundation-model representation",
            "write embedding artifacts and update manifest",
        ),
        "run-baseline": (
            "load baseline adapter config",
            "prepare backend-specific inputs",
            "launch or print external baseline command",
        ),
        "train": (
            "load manifest and representation artifacts",
            "apply configured feature transformation",
            "fit configured model/classifier set",
        ),
        "evaluate": (
            "load predictions",
            "compute per-partition and per-species metrics",
            "write normalized result tables",
        ),
        "report": (
            "load result tables",
            "render markdown summary",
            "link artifacts and reproducibility metadata",
        ),
    }
    notes = []
    if config.dataset.restricted:
        notes.append("dataset may depend on restricted/local data; dry-run is inspectable")
    if config.evaluation:
        notes.append(f"evaluation={config.evaluation}")
    if config.antibiotic:
        notes.append(f"antibiotic={config.antibiotic}")
    if config.features.methods:
        notes.append(f"feature_methods={', '.join(config.features.methods)}")
    if config.model.classifiers:
        notes.append(f"classifiers={', '.join(config.model.classifiers)}")

    return StagePlan(
        stage=stage,
        experiment_name=config.name,
        steps=common.get(stage, ("validate config",)),
        inputs=config.planned_inputs(),
        outputs=config.planned_outputs(),
        notes=tuple(notes),
    )
