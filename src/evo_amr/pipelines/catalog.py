"""Catalog of rebuilt research pipelines in the Evo-AMR project.

The old `AMR_benchmarking` and `amr_pred` codebases are treated as source
pipelines to rebuild into one coherent framework. This catalog describes the
target public pipeline surface.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineSpec:
    """One pipeline in the rebuilt Evo-AMR framework."""

    name: str
    stage: str
    purpose: str
    source_systems: tuple[str, ...]
    clean_modules: tuple[str, ...]
    entrypoint: str
    status: str
    capabilities: tuple[str, ...]


pipeline_specs: tuple[PipelineSpec, ...] = (
    PipelineSpec(
        name="dataset-manifest",
        stage="Dataset -> Manifest",
        purpose="Normalize AMR phenotype metadata and build inspectable manifests.",
        source_systems=(
            "benchmarking/src/data_preprocess",
            "prediction/data_prep/metadata",
            "prediction/data_prep/download",
        ),
        clean_modules=("evo_amr.data.manifest", "evo_amr.data.filters"),
        entrypoint="evo-amr prepare-data",
        status="SCAFFOLD_READY",
        capabilities=(
            "BV-BRC/PATRIC metadata",
            "phenotype normalization",
            "QC policy descriptors",
            "species-antibiotic task eligibility",
        ),
    ),
    PipelineSpec(
        name="split-construction",
        stage="Manifest -> Split",
        purpose="Create random, species-holdout, clustered, KMA, and phylogeny-aware splits.",
        source_systems=(
            "benchmarking/src/cv_folds",
            "prediction/data_prep/data_splits",
        ),
        clean_modules=("evo_amr.splits.designs", "evo_amr.splits.leakage"),
        entrypoint="evo-amr create-splits",
        status="SCAFFOLD_READY",
        capabilities=(
            "random splits",
            "species holdout",
            "cluster-aware splits",
            "KMA/phylogeny design descriptors",
            "leakage checks",
        ),
    ),
    PipelineSpec(
        name="genome-representation",
        stage="Split -> Representation",
        purpose="Extract Evo and ESM representations and register embedding artifacts.",
        source_systems=(
            "embedding_pipeline",
            "models/evo_1_131k_base/modified_model.py",
            "prediction/data_prep/embeddings",
            "prediction/lib/data.py",
        ),
        clean_modules=("evo_amr.embeddings.evo", "evo_amr.diagnostics.layers"),
        entrypoint="evo-amr embed",
        status="ADAPTER_READY",
        capabilities=(
            "Evo layer-10 extraction",
            "layer diagnostics",
            "ESM/proteome extraction plan",
            "dry-run artifact registration",
        ),
    ),
    PipelineSpec(
        name="feature-transforms",
        stage="Representation -> Feature Transformation",
        purpose="Turn ordered embeddings into downstream classifier features.",
        source_systems=("minirocket/minirocket_pipeline",),
        clean_modules=("evo_amr.features.transforms",),
        entrypoint="evo-amr train",
        status="SCAFFOLD_READY",
        capabilities=(
            "mean pooling",
            "PCA plan",
            "sparse random projection plan",
            "MiniRocket plan",
        ),
    ),
    PipelineSpec(
        name="classical-benchmarking",
        stage="Feature/Genome -> Baseline",
        purpose="Run classical AMR tools through registered backend adapters.",
        source_systems=(
            "benchmarking/scripts/model",
            "benchmarking/AMR_software",
        ),
        clean_modules=("evo_amr.baselines.registry", "evo_amr.baselines.adapters"),
        entrypoint="evo-amr run-baseline",
        status="ADAPTER_READY",
        capabilities=(
            "Kover",
            "ResFinder/PointFinder",
            "PhenotypeSeeker",
            "Seq2Geno2Pheno",
            "Aytan-Aktug SSSA/SSMA/MSMA",
            "majority baseline",
        ),
    ),
    PipelineSpec(
        name="trainable-amr-models",
        stage="Feature -> Model",
        purpose="Train single-drug and multi-drug neural/linear AMR predictors.",
        source_systems=(
            "prediction/lib/model",
            "prediction/lib/dataset",
            "prediction/lib/data_modules",
            "prediction/lib/lightning_modules",
        ),
        clean_modules=("evo_amr.models.families", "evo_amr.models.majority"),
        entrypoint="evo-amr train",
        status="MODEL_FAMILIES_REGISTERED",
        capabilities=(
            "single-drug linear probes",
            "single-drug MLP probes",
            "bag-of-proteins models",
            "attention encoders",
            "multi-drug shared trunk with antibiotic heads",
        ),
    ),
    PipelineSpec(
        name="evaluation-and-audits",
        stage="Model -> Evaluation",
        purpose="Compute standard, species-aware, drug-aware, and mechanism-aware metrics.",
        source_systems=(
            "prediction/lib/metrics",
            "prediction/analysis",
            "benchmarking/src/analysis_utility",
        ),
        clean_modules=(
            "evo_amr.evaluation.metrics",
            "evo_amr.evaluation.grouped",
            "evo_amr.evaluation.mechanisms",
        ),
        entrypoint="evo-amr evaluate",
        status="SCAFFOLD_READY",
        capabilities=(
            "accuracy/F1/MCC",
            "grouped species/drug summaries",
            "inverse-frequency weighting",
            "mechanism-aware aggregation guidance",
        ),
    ),
    PipelineSpec(
        name="reporting-and-visualization",
        stage="Evaluation -> Report",
        purpose="Create research tables, markdown reports, and visualization specs.",
        source_systems=(
            "benchmarking/src/benchmark_utility",
            "benchmarking/scripts/analysis_visualization",
            "minirocket/minirocket_pipeline/visualization",
        ),
        clean_modules=("evo_amr.reporting.markdown", "evo_amr.visualization.plots"),
        entrypoint="evo-amr report",
        status="SCAFFOLD_READY",
        capabilities=(
            "method comparison reports",
            "species heatmaps",
            "neighbor audits",
            "phylogenetic performance curves",
        ),
    ),
    PipelineSpec(
        name="hpc-orchestration",
        stage="Workflow -> HPC Execution",
        purpose="Render reproducible local/server/SLURM execution plans.",
        source_systems=(
            "prediction/workflows",
            "prediction/utils/workflow_utils.py",
            "diagnostics/*.sbatch",
        ),
        clean_modules=("evo_amr.workflows.plans", "evo_amr.workflows.slurm"),
        entrypoint="evo-amr <stage> --profile server",
        status="DRY_RUN_READY",
        capabilities=(
            "SLURM dry-run rendering",
            "server path profiles",
            "stage plans",
            "GPU diagnostic launch design",
        ),
    ),
)


def list_pipelines(stage: str | None = None, status: str | None = None) -> tuple[PipelineSpec, ...]:
    """Return pipeline specs, optionally filtered by stage or status."""
    return tuple(
        pipeline
        for pipeline in pipeline_specs
        if (stage is None or pipeline.stage == stage)
        and (status is None or pipeline.status == status)
    )
