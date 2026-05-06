# Legacy Code Mining Notes

This repository now treats `benchmarking/` and `prediction/` as source material
for a cleaner Evo-AMR framework, not as directories to commit wholesale.

## AMR_benchmarking Contributions

`AMR_benchmarking` is strongest as a classical AMR benchmarking suite. It
contains:

- PATRIC/BV-BRC-style metadata preprocessing, phenotype quality filtering, and
  species-antibiotic task selection.
- KMA, random, and phylogeny-aware fold generation.
- Classical baselines: Kover, ResFinder/PointFinder, PhenotypeSeeker,
  Seq2Geno2Pheno, majority baseline, and Aytan-Aktug neural baselines.
- Large result-analysis and visualization scripts for per-species,
  per-antibiotic, and multi-model comparisons.

The clean framework maps those into:

- `evo_amr.data.filters` for QC and species-antibiotic eligibility policy.
- `evo_amr.splits.designs` for random, clustered, KMA, phylogeny, and
  species-holdout split descriptions.
- `evo_amr.baselines.registry` for wrapper metadata around the legacy baselines.
- `evo_amr.visualization` and `evo_amr.reporting` for future migrated report
  generation.

## amr_pred Contributions

`amr_pred` is strongest as the embedding/modeling training system. It contains:

- Random and taxonomy/cluster-aware split construction.
- ESM/proteome embedding extraction and cached representation loading.
- Single-drug and multi-drug datasets, PyTorch modules, Lightning data modules,
  optimizers, checkpoint utilities, and evaluation scripts.
- Taxonomy-aware metrics and weighted cluster metrics.
- Server/HPC path utilities and dynamic SLURM launcher scripts.

The clean framework maps those into:

- `evo_amr.config.paths` for named local/server path profiles.
- `evo_amr.splits.designs.default_amr_pred_clustered_design` for migrated split
  defaults.
- `evo_amr.embeddings` for representation adapters.
- `evo_amr.models.families` for single-drug probes, bag-of-proteins models,
  attention encoders, and multi-drug shared-trunk models.
- `evo_amr.workflows.slurm` for non-executing SLURM dry-run rendering.
- `evo_amr.evaluation` for metrics, grouped species/drug audits, weighting, and
  mechanism-aware analysis.

## Migration Policy

Use these labels when deciding what enters the showcase package:

- `MIGRATE`: small, generally useful logic that should become native
  `src/evo_amr` code.
- `WRAP`: useful backend script that should remain external but be called
  through typed adapters.
- `VENDOR`: third-party or large reference code that should be preserved under
  `external/` only after a clear provenance decision.
- `LEGACY`: historical research code to preserve for context, but not support as
  the public API.
- `IGNORE`: datasets, logs, virtual environments, checkpoints, generated
  artifacts, `.DS_Store`, caches, and server-only outputs.

The current reconstruction follows that policy: build clean interfaces first,
then migrate or wrap one workflow at a time.
