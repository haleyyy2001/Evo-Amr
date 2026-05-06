# Migration Map

This document classifies the existing repository against the target Evo-AMR architecture. It is intentionally a plan, not a file-move log.

## Action Labels

- **MIGRATE**: move stable logic into `src/evo_amr/`.
- **WRAP**: keep code in place for now and call through an adapter.
- **VENDOR**: preserve external/reference software under `external/`.
- **LEGACY**: keep for historical inspection, but do not support as part of the clean interface.
- **IGNORE**: data, logs, results, notebooks, virtualenvs, caches, generated artifacts.

## Target Modules

```text
src/evo_amr/
├── config/       # config loading, path profiles, validation
├── data/         # manifests, phenotype normalization, dataset schemas
├── splits/       # random/species-holdout split generation and leakage checks
├── embeddings/   # Evo/ESM representation extraction adapters
├── features/     # pooling, PCA, SRP, MiniRocket transforms
├── baselines/    # adapters for Kover, ResFinder, PhenotypeSeeker, Seq2Geno, etc.
├── models/       # trainable AMR prediction models
├── evaluation/   # metrics and result schemas
├── reporting/    # markdown/table reports
├── visualization/# figures and neighbor analysis
├── workflows/    # orchestration and SLURM/local launch helpers
└── cli.py
```

## Current To Target Mapping

| Old path | Current role | New home | Action | Notes |
| --- | --- | --- | --- | --- |
| `config/config_manager.py` | Root path/env config | `src/evo_amr/config/` | MIGRATE | Keep current behavior but replace hard-coded defaults with profiles. |
| `config/*.yaml` | Current path/env settings | `configs/paths/` | MIGRATE | Split public examples from ignored local profiles. |
| `data_preprocessing/*.py` | Species/drug subset scripts | `src/evo_amr/data/` and `src/evo_amr/splits/` | MIGRATE | First extract reusable manifest/split logic; leave script wrappers. |
| `embedding_pipeline/embedding_generator.py` | Evo genome embedding extraction | `src/evo_amr/embeddings/` | WRAP then MIGRATE | Start with adapter; later split config, model loading, writing. |
| `models/evo_1_131k_base/` | Evo architecture code | `src/evo_amr/embeddings/evo/` or `src/evo_amr/models/evo/` | WRAP | Treat as model backend; avoid unnecessary refactor. |
| `models/evo_enhanced/` | Enhanced Evo embedding utilities | `src/evo_amr/embeddings/evo/` | MIGRATE | Useful for layer diagnostics and extraction. |
| `diagnostics/*.sbatch` | Layer/numerical diagnostics | `src/evo_amr/workflows/` and `configs/slurm/` | WRAP | Convert server paths to profile variables. |
| `diagnostics/*.sbatch` + thesis diagnostic logic | Evo layer stability sweep | `src/evo_amr/diagnostics/` | MIGRATE | Layer 10/11 decision is now encoded in `diagnostics/layers.py`; full metric computation still needs migration. |
| `minirocket/minirocket_pipeline/core/*.py` | PCA/SRP/MiniRocket transforms | `src/evo_amr/features/` | WRAP then MIGRATE | Preserve algorithm code; create stable config adapter first. |
| `minirocket/minirocket_pipeline/amr_classification_pipeline.py` | Classifier comparison | `src/evo_amr/models/` and `src/evo_amr/evaluation/` | WRAP | Too large to rewrite first; expose via CLI adapter later. |
| `minirocket/minirocket_pipeline/visualization/` | Method and neighbor plots | `src/evo_amr/visualization/` | MIGRATE | Keep plotting code but separate data loading from rendering. |
| `minirocket/minirocket_pipeline/trash/` | Old experiment attempts | `legacy/minirocket_experiments/` | LEGACY | Preserve only if useful; not part of supported CLI. |
| `benchmarking/src/data_preprocess/` | PATRIC benchmark preprocessing | `src/evo_amr/data/filters.py` | MIGRATE started | QC and species-drug eligibility are now represented as config primitives. Full PATRIC parsing remains wrapped. |
| `benchmarking/src/cv_folds/` | CV and phylogeny folds | `src/evo_amr/splits/designs.py` | MIGRATE started | Split design vocabulary now covers random, clustered, KMA/phylogeny, and species-holdout. Actual KMA parser still needs a wrapper. |
| `benchmarking/scripts/model/*.sh` | Baseline shell runners | `src/evo_amr/baselines/registry.py` and adapters | WRAP started | Registered backends now expose action labels, capabilities, and script locations for dry-runs. |
| `benchmarking/AMR_software/` | Adapted third-party tools | `external/benchmarking/AMR_software/` | VENDOR | Preserve as reference backend, not core package code. |
| `benchmarking/data/` | PATRIC metadata/data artifacts | external data storage | IGNORE | Do not commit wholesale; document required sources. |
| `benchmarking/main.sh` | Monolithic original workflow | `docs/legacy_workflows.md` reference | LEGACY | Useful for provenance, not a supported entry point. |
| `prediction/data_prep/` | BV-BRC download, metadata, splits | `src/evo_amr/data/` and `src/evo_amr/splits/` | MIGRATE started | Clustered/random split defaults are now represented in `splits/designs.py`; download and metadata code remain legacy. |
| `prediction/lib/data_modules/` | Lightning data modules | `src/evo_amr/models/` | WRAP | Keep until trainable models are a priority. |
| `prediction/lib/metrics/` | AMR metrics | `src/evo_amr/evaluation/grouped.py` | MIGRATE started | Grouped species/drug metric summaries and inverse-frequency weighting now exist. TorchMetrics code remains wrapped. |
| `prediction/lib/model/` | Single/multi-drug models | `src/evo_amr/models/families.py` | MIGRATE started | Model families are registered as config-level descriptors; PyTorch implementations remain legacy until training wrappers are stable. |
| `prediction/train.py`, `prediction/evaluate.py` | Lightning train/eval entry points | `src/evo_amr/workflows/` adapters | WRAP | Keep behavior, replace pickle configs later. |
| `prediction/utils/path_utils.py` | Environment-driven path roots | `src/evo_amr/config/paths.py` | MIGRATE started | Converted to named path profiles and template resolution. |
| `prediction/workflows/` | SLURM launchers | `src/evo_amr/workflows/slurm.py` | MIGRATE started | Dry-run SLURM rendering exists; submission remains intentionally disabled. |
| `prediction/configs/` | Python/pickle experiment configs | `configs/experiments/` | LEGACY then MIGRATE | Use as templates, but target YAML. |
| `prediction/testtt/` | Exploratory Kover/Evo/domain-transfer work | `legacy/exploratory/` | LEGACY | Mine useful scripts after stable architecture exists. |
| `prediction/notebooks/` | EDA and analysis notebooks | `docs/figures` or ignored archive | IGNORE/LEGACY | Do not rely on notebooks as pipeline steps. |
| `prediction/data/`, `backup_data*`, logs | Data/artifacts | external storage | IGNORE | Keep out of git. |

## First Workflow To Stabilize

The first real migration should be:

```text
manifest
→ species-holdout split validation
→ Evo Layer 10 embedding manifest
→ MiniRocket/PCA feature adapter
→ classifier comparison
→ metrics table
→ markdown report
```

The tiny fixture in `examples/` demonstrates this shape without requiring private data or server tools.
