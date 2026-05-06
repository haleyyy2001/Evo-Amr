"""Experiment configuration models.

These lightweight dataclasses are intentionally dependency-free. They provide
enough structure for dry-runs, validation, and tests while leaving room for a
future Pydantic/Hydra/OmegaConf migration if the project needs richer config
composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_yaml


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-level inputs used by all experiment stages."""

    manifest: str | None
    config: str | None = None
    split_design: str | None = None
    partitions: dict[str, str] | None = None
    restricted: bool = True


@dataclass(frozen=True)
class RepresentationConfig:
    """Foundation-model representation settings."""

    model: str | None = None
    extraction_layer: int | None = None
    dtype: str | None = None
    output: str | None = None


@dataclass(frozen=True)
class FeatureConfig:
    """Feature transformation settings."""

    methods: tuple[str, ...] = ()
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Downstream classifier or trainable model settings."""

    classifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation metric settings."""

    metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    """Output locations for run artifacts."""

    run_dir: str | None = None
    report: str | None = None


@dataclass(frozen=True)
class BaselineConfig:
    """External baseline backend settings."""

    name: str | None = None
    backend: str | None = None
    mode: str | None = None
    script: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment description."""

    name: str
    task: str
    antibiotic: str | None
    evaluation: str | None
    dataset: DatasetConfig
    representation: RepresentationConfig
    features: FeatureConfig
    model: ModelConfig
    evaluation_config: EvaluationConfig
    outputs: OutputConfig
    baseline: BaselineConfig
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate an experiment YAML file."""
        return cls.from_dict(load_yaml(path))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ExperimentConfig":
        """Create a config object from a parsed YAML dictionary."""
        experiment = config.get("experiment", {})
        baseline = config.get("baseline", {})
        dataset = config.get("dataset", {})
        inputs = config.get("inputs", {})
        representation = config.get("representation", {})
        aggregation = config.get("aggregation", {})
        models = config.get("models", {})
        outputs = config.get("outputs", {})
        execution = config.get("execution", {})
        name = experiment.get("name") or baseline.get("name")
        task = experiment.get("task") or ("baseline" if baseline else None)

        missing = []
        if not name:
            missing.append("experiment.name")
        if not task:
            missing.append("experiment.task")
        if missing:
            raise ValueError(f"Experiment config missing required fields: {missing}")

        return cls(
            name=name,
            task=task,
            antibiotic=experiment.get("antibiotic"),
            evaluation=experiment.get("evaluation") or baseline.get("mode"),
            dataset=DatasetConfig(
                config=dataset.get("config"),
                manifest=dataset.get("manifest") or inputs.get("manifest"),
                split_design=dataset.get("split_design"),
                partitions=dataset.get("partitions"),
                restricted=bool(dataset.get("restricted", True)),
            ),
            representation=RepresentationConfig(
                model=representation.get("model"),
                extraction_layer=representation.get("extraction_layer"),
                dtype=representation.get("dtype"),
                output=representation.get("output"),
            ),
            features=FeatureConfig(
                methods=tuple(aggregation.get("methods", ())),
                params={k: v for k, v in aggregation.items() if k != "methods"} or None,
            ),
            model=ModelConfig(classifiers=tuple(models.get("classifiers", ()))),
            evaluation_config=EvaluationConfig(metrics=tuple(config.get("metrics", ()))),
            outputs=OutputConfig(
                run_dir=outputs.get("run_dir"),
                report=outputs.get("report"),
            ),
            baseline=BaselineConfig(
                name=baseline.get("name"),
                backend=baseline.get("backend"),
                mode=baseline.get("mode"),
                script=execution.get("script"),
            ),
            raw=config,
        )

    def planned_inputs(self) -> dict[str, str]:
        """Return stable, human-readable input paths for dry-run output."""
        inputs = {}
        if self.dataset.manifest:
            inputs["manifest"] = self.dataset.manifest
        if self.dataset.config:
            inputs["dataset_config"] = self.dataset.config
        if self.representation.output:
            inputs["representation_artifacts"] = self.representation.output
        if self.baseline.backend:
            inputs["baseline_backend"] = self.baseline.backend
        return inputs

    def planned_outputs(self) -> dict[str, str]:
        """Return stable, human-readable output paths for dry-run output."""
        outputs = {}
        if self.outputs.run_dir:
            outputs["run_dir"] = self.outputs.run_dir
        if self.outputs.report:
            outputs["report"] = self.outputs.report
        if self.representation.output:
            outputs["embedding_output"] = self.representation.output
        return outputs
