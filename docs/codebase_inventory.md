# Codebase Inventory

## Root Evo-AMR

- `embedding_pipeline/`: Evo embedding extraction and HDF5 writing.
- `models/`: Evo model code and enhanced wrappers.
- `minirocket/`: MiniRocket, PCA, SRP, classifier comparison, and visualization code.
- `data_preprocessing/`: species and antibiotic subset scripts.
- `diagnostics/`: Evo layer stability and numerical diagnostic SLURM scripts.
- `config/`: current path and environment configuration.
- `scripts/`: standalone analysis scripts.

## AMR_benchmarking

`benchmarking/` is an imported classical AMR benchmark suite. It includes:

- `AMR_software/`: adapted third-party software backends.
- `scripts/model/`: shell runners for Kover, ResFinder, PhenotypeSeeker, Seq2Geno, Aytan-Aktug, and majority baseline.
- `src/data_preprocess/`: PATRIC metadata and quality filtering utilities.
- `src/cv_folds/`: cross-validation fold construction.
- `src/benchmark_utility/`: benchmark result tables and plots.
- `Config.yaml` and `main.sh`: original monolithic benchmark orchestration.

This should be treated as an external backend, not as the primary framework code.

## amr_pred

`prediction/` is a larger legacy/internal AMR prediction framework. It includes:

- `data_prep/`: BVBRC/PATRIC metadata, sequence download, and split generation.
- `lib/`: PyTorch Lightning data modules, datasets, models, losses, and metrics.
- `train.py` and `evaluate.py`: train/eval entry points.
- `workflows/`: SLURM job generation for training and sweeps.
- `configs/`: Python and pickle configs for model experiments.
- `testtt/`: exploratory Evo, Kover, domain-transfer, and plotting experiments.
- `notebooks/`: EDA and output analysis.

This should become `legacy/amr_pred` during migration, with stable pieces promoted into `src/evo_amr/`.

## Key Risks

- Many scripts hard-code server paths.
- Multiple nested git repositories exist.
- Generated artifacts are mixed with source code.
- Several workflows rely on conda environment names and SLURM assumptions.
- Similar logic appears in multiple places for metrics, splits, Kover preparation, and plotting.

## Mined Into `src/evo_amr`

- `benchmarking/src/data_preprocess/preprocess.py` inspired `DatasetFilterConfig`, phenotype normalization, and species-antibiotic task counting.
- `benchmarking/src/cv_folds/` and `prediction/data_prep/data_splits/` inspired split design dataclasses for random, clustered, and species-holdout evaluations.
- `benchmarking/scripts/model/` inspired the baseline backend registry.
- `prediction/utils/path_utils.py` inspired config-driven path profiles.
- `prediction/utils/workflow_utils.py` inspired non-executing SLURM rendering.
- `prediction/lib/model/` inspired the model-family registry for single-drug
  probes, bag-of-proteins models, attention encoders, and multi-drug heads.
- `prediction/lib/metrics/` inspired grouped evaluation and inverse-frequency
  weighting utilities.
