# Pipeline Rebuild Plan

The goal is not to label old work as disposable. The goal is to rebuild all
historical research pipelines into one structured Evo-AMR project.

The public architecture is:

```text
Dataset
  -> Manifest
  -> Split
  -> Representation
  -> Feature Transformation
  -> Model / Baseline
  -> Evaluation
  -> Report
  -> HPC Orchestration
```

Each old system contributes real project work:

- `benchmarking/` contributes classical AMR preprocessing, folds,
  baselines, and benchmark reporting.
- `prediction/` contributes BV-BRC/ESM data preparation, trainable AMR models,
  metrics, configs, and SLURM workflows.
- root Evo-AMR contributes Evo embedding extraction, layer diagnostics,
  MiniRocket/PCA/SRP experiments, and thesis results.

## Rebuilt Pipeline Map

| Pipeline | Sources | Clean home | Public entrypoint | Status |
| --- | --- | --- | --- | --- |
| Dataset manifest | `benchmarking/src/data_preprocess`, `prediction/data_prep` | `evo_amr.data` | `evo-amr prepare-data` | scaffold ready |
| Split construction | `benchmarking/src/cv_folds`, `prediction/data_prep/data_splits` | `evo_amr.splits` | `evo-amr create-splits` | scaffold ready |
| Genome representations | `embedding_pipeline`, `prediction/data_prep/embeddings`, `models/evo_1_131k_base` | `evo_amr.embeddings` | `evo-amr embed` | adapter ready |
| Feature transforms | `minirocket/minirocket_pipeline` | `evo_amr.features` | `evo-amr train` | scaffold ready |
| Classical benchmarking | `benchmarking/scripts/model`, `benchmarking/AMR_software` | `evo_amr.baselines` | `evo-amr run-baseline` | adapter ready |
| Trainable AMR models | `prediction/lib/model`, `prediction/lib/lightning_modules` | `evo_amr.models` | `evo-amr train` | model families registered |
| Evaluation/audits | `prediction/lib/metrics`, `benchmarking/src/analysis_utility` | `evo_amr.evaluation` | `evo-amr evaluate` | scaffold ready |
| Reporting/visualization | `benchmarking/src/benchmark_utility`, MiniRocket visualization | `evo_amr.reporting`, `evo_amr.visualization` | `evo-amr report` | scaffold ready |
| HPC orchestration | `prediction/workflows`, `diagnostics/*.sbatch` | `evo_amr.workflows` | `evo-amr <stage> --profile server` | dry-run ready |

The same map is available from the CLI:

```bash
evo-amr list-pipelines
evo-amr list-baselines
evo-amr list-models
```

## Why The Old Trees Are Not Copied Wholesale Yet

The old folders contain valuable pipelines, but also nested git repos, generated
files, private/server paths, checkpoints, environment dumps, and third-party
bundles. A showcase repository is stronger if it presents those pipelines as a
designed system instead of committing every artifact.

The rebuild strategy is:

1. Keep all source systems documented.
2. Build clean modules and CLI surfaces first.
3. Migrate small stable logic into `src/evo_amr`.
4. Wrap large tools with adapters.
5. Vendor or move historical code only after data/artifact cleanup.

This preserves the scope of the work while making the project readable as a
single research framework.
