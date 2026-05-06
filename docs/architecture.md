# Architecture

Evo-AMR is organized as a research framework around three historical systems:

1. The current Evo/MiniRocket code in the repository root.
2. The imported `AMR_benchmarking` baseline suite.
3. The `amr_pred` training and data-preparation codebase.

The goal is not to hide that history. The goal is to present it as a coherent AMR benchmarking framework with a stable public interface and documented legacy backends.

## Conceptual Layers

```text
Raw genomes + AMR phenotypes
        |
        v
Dataset normalization and manifest construction
        |
        v
Species-aware split construction
        |
        v
Representation extraction
  - Evo genome embeddings
  - ESM/protein embeddings
  - k-mer features
  - AMR determinant calls
        |
        v
Feature aggregation
  - global pooling
  - PCA / sparse random projection
  - MiniRocket local pattern features
        |
        v
Model training and baseline execution
        |
        v
Evaluation, neighbor analysis, reports, figures
```

## Package Boundary

`src/evo_amr/` is the clean framework boundary. It should contain stable APIs, config loading, adapters, schemas, and CLIs.

Existing research code should be wrapped before it is rewritten:

- `embedding_pipeline/` becomes an Evo embedding backend.
- `minirocket/minirocket_pipeline/` becomes a feature aggregation and classifier backend.
- `benchmarking/` becomes an external baseline backend for Kover, ResFinder, PhenotypeSeeker, Seq2Geno, Aytan-Aktug, and majority models.
- `prediction/` becomes a legacy source for data prep, Lightning models, ESM baselines, SLURM launchers, and analysis utilities.

The first migrated primitives now live behind this boundary:

- `evo_amr.data.filters` captures phenotype normalization and species-drug eligibility rules from `benchmarking/src/data_preprocess`.
- `evo_amr.splits.designs` captures random, clustered, KMA/phylogeny, and species-holdout split designs from both source systems.
- `evo_amr.config.paths` captures server/local path profiles without depending on environment variables.
- `evo_amr.baselines.registry` catalogs Kover, ResFinder, PhenotypeSeeker, Seq2Geno, majority, and Aytan-Aktug backends.
- `evo_amr.workflows.slurm` renders dry-run SLURM scripts without submitting jobs.

## Target Repository Layout

```text
Evo-Amr/
├── src/evo_amr/
│   ├── config/          # path profiles and experiment config loading
│   ├── data/            # manifests, phenotype normalization, split utilities
│   ├── splits/          # split generation and leakage checks
│   ├── embeddings/      # Evo/ESM embedding adapters
│   ├── diagnostics/     # layer diagnostics and representation stability
│   ├── features/        # pooling, PCA, SRP, MiniRocket
│   ├── baselines/       # wrappers around Kover/ResFinder/etc.
│   ├── models/          # trainable AMR models
│   ├── evaluation/      # metrics and result schemas
│   ├── reporting/       # markdown/table reports
│   ├── visualization/   # reports and plots
│   ├── workflows/       # local/HPC orchestration
│   └── cli.py
├── configs/             # inspectable example configs
├── docs/                # research and architecture documentation
├── examples/            # tiny fixtures and fake outputs
├── external/            # vendored/adapted third-party benchmark systems
├── legacy/              # older internal research systems
└── tests/               # lightweight sanity checks
```

The current repository has not fully migrated to this tree yet. The migration should be non-destructive: keep old code available while moving stable workflows behind adapters.

## Backend Responsibilities

### Evo Embedding Backend

Current source: `embedding_pipeline/`, `models/`, `models/evo_enhanced/`

Responsibilities:

- Load Evo model/tokenizer.
- Extract layer-selected genome embeddings.
- Support pooled and per-window outputs.
- Write HDF5/NumPy artifacts with manifest metadata.

### Layer Diagnostic Backend

Current source: `diagnostics/`, `models/evo_1_131k_base/modified_model.py`

Responsibilities:

- Sweep Evo layers under native bfloat16 inference.
- Track activation scale, isotropy, effective rank, token concentration, and cross-seed stability.
- Encode the thesis decision that Layer 10 is the deepest stable extraction point before the Layer 11 boundary.

### MiniRocket Backend

Current source: `minirocket/minirocket_pipeline/`

Responsibilities:

- Convert ordered embeddings into local-pattern features.
- Provide PCA and SRP comparison tracks.
- Train downstream classifiers.
- Export metrics and neighbor-audit artifacts.

### Classical Baseline Backend

Current source: `benchmarking/`

Responsibilities:

- Execute Kover, ResFinder, PhenotypeSeeker, Seq2Geno, Aytan-Aktug, and majority baselines.
- Preserve benchmark compatibility with published AMR baseline protocols.
- Export normalized result tables into the common evaluation schema.
- Keep the registered backend list inspectable even before heavyweight wrappers are executable.

### Legacy Training Backend

Current source: `prediction/`

Responsibilities:

- Preserve prior ESM and PyTorch Lightning experiments.
- Provide reusable split, metric, and SLURM-launch logic.
- Serve as a source for code migration into `src/evo_amr/`.
- Promote path, split, metric, and launcher concepts first; model internals should remain wrapped until the public experiment schema is stable.

## Design Rules

- Absolute server paths belong only in ignored path profile files.
- Experiment definitions should be YAML, not edited Python globals.
- Shell scripts should become thin launchers or backend adapters.
- Backends may be server-dependent; the public interface should still support dry-run validation.
- Generated artifacts do not belong in git.
- Legacy code is deprecated by documentation and wrappers before removal.
- Public CLI stages should support dry-runs before execution.

## Reconstruction Order

1. Define the clean target package and configuration style.
2. Classify old code as migrate, wrap, vendor, legacy, or ignore.
3. Stabilize one tiny end-to-end workflow.
4. Wrap one real backend at a time.
5. Promote reusable legacy code only after tests exist.

## Pipeline Rebuild Layer

The project now has a separate pipeline layer for rebuilding the old systems:

- `evo_amr.pipelines.catalog` records the target research pipelines and their
  source systems.
- `evo-amr list-pipelines` prints the rebuilt project map for reviewers.
- `docs/pipeline_rebuild_plan.md` explains how the old folders become a single
  structured framework.

This is deliberate. It lets the GitHub repository show the breadth of the work
without turning the clean package into a dump of nested repositories, private
data, binaries, checkpoints, and server-specific scripts.
