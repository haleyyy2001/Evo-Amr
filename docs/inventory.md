# Evo-AMR Codebase Inventory

**Phase 1 deliverable.** Every major pipeline and script across the three source systems,
mapped to the target pipeline spine and classified for migration strategy.

**Target pipeline spine:**
```
Dataset → Manifest → Split → Representation → FeatureTransform
→ Model/Baseline → Evaluation → Report → HPCOrchestration
```

---

## Source System 1: Repo Root

Evo embedding extraction, MiniRocket/PCA/SRP experiments, layer diagnostics,
data preprocessing, dataset statistics.

### embedding_pipeline/embedding_generator.py
**What it does:** Production GPU embedding generator. Reads genome sequences
(CSV manifest, FASTA files, or directory of `.fna.gz`), tokenises with the
ByteTokenizer, extracts hidden states from `models/evo_1_131k_base/modified_model.py`
via a sliding window (window=8192, stride=4096), writes per-token and pooled
embeddings to HDF5. Multi-GPU via `ThreadPoolExecutor`, CUDA graph optimisation,
atomic write with crash recovery. Heavy (700+ lines).

**Pipeline stage:** Representation  
**Strategy:** Adapt — keep as the authoritative backend script, wrap in
`src/evo_amr/embeddings/` adapter that generates the CLI call and validates
manifest/output paths. Do not reimplement GPU logic.

---

### minirocket/minirocket_pipeline/amr_classification_pipeline.py
**What it does:** The core ML experiment driver. Reads HDF5 embeddings → applies
MiniRocket (2 000 random convolutional kernels → PPV features), global mean pooling,
PCA (41-dim), and SRP as alternative feature transforms → fits 10+ classifiers
(LR, SVM, RF, ExtraTrees, HistGBT, KNN, NearestCentroid, MLP, CatBoost) with
GridSearchCV → evaluates on val_overlapped / val_outside / test_overlapped /
test_outside partitions → writes metrics CSV and diagnostic plots. Also implements
Multiple Instance Learning (MIL) via bag-of-windows aggregation. Very large (~2000 lines).

**Pipeline stage:** FeatureTransform → Model → Evaluation  
**Strategy:** Migrate in parts. The MiniRocket transform, pooling logic, and
per-partition evaluation loop are stable and should be migrated into
`src/evo_amr/features/` and `src/evo_amr/evaluation/`. The classifier grid and
MIL aggregation belong in `src/evo_amr/models/`. Visualisation moves to
`src/evo_amr/reporting/`.

---

### diagnostics/run_diagnostics.sbatch
**What it does:** SLURM array job that runs the 32-layer Evo diagnostic sweep
(mean_norm, std_norm, effective_rank, angular_div per layer) on Insomnia/Manitou.
Reads config via `config/config_manager.py` for portable paths.

**Pipeline stage:** Representation (layer selection)  
**Strategy:** Adapter — wrap in `src/evo_amr/embeddings/` SLURM adapter that
generates and submits this sbatch. The diagnostic logic itself is now also
packaged in `evo-hidden/examples/layer_sweep.py`.

---

### diagnostics/diagnostic_analysis.sbatch
**What it does:** Second SLURM job. Runs post-hoc analysis of layer sweep results
(reads output files, generates stability boundary report).

**Pipeline stage:** Report  
**Strategy:** Adapter (SLURM shell-out).

---

### data_preprocessing/extract_balanced_species_subsets.py
**What it does:** Balanced genome subset extraction with a deduplication registry.
Reads BV-BRC metadata and a Kover-style label CSV, samples up to N genomes per
species/antibiotic/phenotype cell, tracks already-used genome IDs across extraction
rounds to avoid overlap.

**Pipeline stage:** Dataset → Manifest  
**Strategy:** Migrate — stable Pandas logic, no heavy dependencies.

---

### data_preprocessing/extract_proper_species_subsets.py
**What it does:** Earlier version of balanced extraction (natural-proportion sampling
without the registry).

**Pipeline stage:** Dataset  
**Strategy:** Migrate alongside the balanced version; share the loader.

---

### data_preprocessing/create_balanced_subsets.py
**What it does:** Variant that creates fixed-size subsets per antibiotic with
explicit R/S balancing.

**Pipeline stage:** Dataset  
**Strategy:** Migrate.

---

### data_preprocessing/compare_distributions.py
**What it does:** Compares R/S distributions across partition labels and across
extraction rounds. Diagnostic / data-quality check.

**Pipeline stage:** Dataset → Report  
**Strategy:** Migrate as a report helper.

---

### data_preprocessing/extract_species_subsets_gentamicin.py
**What it does:** Gentamicin-specific subset extraction (same pattern as general
version but hardcoded for that drug).

**Pipeline stage:** Dataset  
**Strategy:** Migrate — generalise the drug-specific parameter.

---

### scripts/dataset_analyzer.py
**What it does:** Partition-wise dataset statistics: species counts per partition,
R/S ratios per antibiotic per partition, cross-partition overlap checks. Writes
a log file and console report.

**Pipeline stage:** Dataset → Report  
**Strategy:** Migrate as `src/evo_amr/data/` analyser.

---

### scripts/Z_stats_pca_analysis.py
**What it does:** Loads `*_pca_stats_dim41.npz` files and a Kover metadata CSV,
computes per-partition data distribution statistics for PCA-transformed embeddings.

**Pipeline stage:** Evaluation → Report  
**Strategy:** Migrate as a reporting helper (feature-transform diagnostic).

---

### config/config_manager.py
**What it does:** Reads `config/paths.yaml` and `config/environment.yaml` to
expose portable server path configuration (model dirs, data dirs, env names).
Used by SLURM scripts and the embedding generator.

**Pipeline stage:** Cross-cutting (HPCOrchestration)  
**Strategy:** Migrate into `src/evo_amr/config/` (already partially done).

---

### embedding_pipeline/config.yaml
**What it does:** Embedding job config: model path, layer index, window/stride,
output format, dtype.

**Pipeline stage:** Representation (config)  
**Strategy:** Migrate schema into `configs/experiments/` YAML format.

---

## Source System 2: benchmarking/

External baseline suite: Kover, ResFinder, PhenotypeSeeker, Seq2Geno, Aytan-Aktug.
Fold generation. Result aggregation.

### AMR_software/Kover/main_kover.py
**What it does:** Prepares input files for Kover 2.0 (k-mer set cover machine
learning). Reads species/antibiotic metadata, extracts CV fold assignments,
writes TSV manifests listing sequence paths and phenotype labels, then invokes
the `kover` CLI for training and prediction.

**Pipeline stage:** Baseline (k-mer rule learner)  
**Strategy:** Adapter — shell out to the Kover binary. Do not reimplement.
Wrap in `src/evo_amr/baselines/kover.py` adapter that takes a manifest and
fold index, generates the TSV inputs, and invokes `kover dataset create`,
`kover learn`, `kover predict`.

---

### AMR_software/resfinder/main_resfinder_folds.py
**What it does:** Runs ResFinder (gene/point-mutation database lookup via BLAST/KMA)
per genome, extracts the `pheno_table.txt` output, converts to binary resistance
calls, computes classification metrics per fold. Handles both zip and unzip output.

**Pipeline stage:** Baseline (resistance gene database)  
**Strategy:** Adapter — shell out to the `resfinder` CLI. Wrap phenotype table
parsing in `src/evo_amr/baselines/resfinder.py`.

---

### AMR_software/PhenotypeSeeker/main_pts.py
**What it does:** Runs PhenotypeSeeker (k-mer frequency + elastic net). Calls
`kmer.sh` to compute k-mer matrices, `map.sh` to run PhenotypeSeeker, parses
output. Per-species, per-antibiotic, per-fold.

**Pipeline stage:** Baseline (k-mer ML)  
**Strategy:** Adapter — shell out to PhenotypeSeeker CLI.

---

### AMR_software/seq2geno/main_s2p.py
**What it does:** Seq2Geno → Seq2Pheno pipeline. Assembly (SPAdes/SKESA),
variant calling, expression profiles, gene presence/absence → feature matrix →
ML classification. Requires Snakemake + many external tools.

**Pipeline stage:** Baseline (whole-genome assembly pipeline)  
**Strategy:** Adapter (heavy; shell out to the Snakemake workflow).

---

### AMR_software/AytanAktug/main_MSMA_concat.py (+ SSMA, SSSA, MSMA_discrete)
**What it does:** Aytan-Aktug neural network models. MSMA = Multi-Species
Multi-Antibiotic; SSMA = Single-Species Multi-Antibiotic; SSSA = Single-Species
Single-Antibiotic. Reads ResFinder feature vectors (gene presence/absence encoded
from BLAST output), trains small MLPs, evaluates with nested CV.

**Pipeline stage:** Baseline (resistance gene feature NN)  
**Strategy:** Adapt — the feature encoding (ResFinder BLAST → binary vector) is
stable logic worth migrating. The NN training loop can be wrapped or replaced
with a scikit-learn compatible interface.

---

### AMR_software/majority/main_majority.py
**What it does:** Majority-vote (most frequent phenotype in training set) baseline.
Trivially simple.

**Pipeline stage:** Baseline  
**Strategy:** Migrate directly (< 50 lines of logic).

---

### src/cv_folds/prepare_folds.py
**What it does:** Generates CV fold assignments for a species/antibiotic pair.
Supports three strategies: random, KMA cluster-based (genomic sequence clusters
to avoid leakage), phylogeny-based (Seq2Geno tree). Writes fold JSON files.

**Pipeline stage:** Split  
**Strategy:** Migrate — this is the central split logic. Abstracts over all
three CV designs. Maps to `src/evo_amr/splits/`.

---

### src/cv_folds/cluster2folds.py
**What it does:** Assigns cluster membership → fold index. Reads KMA cluster
files, groups genomes, balances fold sizes.

**Pipeline stage:** Split  
**Strategy:** Migrate alongside `prepare_folds.py`.

---

### src/cv_folds/generate_random_folds.py
**What it does:** Purely random fold generation (no clustering).

**Pipeline stage:** Split  
**Strategy:** Migrate.

---

### src/amr_utility/load_data.py
**What it does:** Canonical data loader for the benchmarking system. Reads the
main species/antibiotic metadata TSV, extracts genome IDs, phenotype labels,
and antibiotics list per species. Also handles balance checking and downsampling.

**Pipeline stage:** Dataset → Manifest  
**Strategy:** Migrate core loading logic into `src/evo_amr/data/manifest.py`.

---

### src/amr_utility/name_utility.py
**What it does:** Path naming conventions — `GETname_*` functions return paths
to metadata files, model output dirs, fold files, score files. Encodes the
benchmarking system's directory layout.

**Pipeline stage:** Cross-cutting  
**Strategy:** Migrate naming logic into the config/path system in `src/evo_amr/config/`.

---

### src/amr_utility/file_utility.py + math_utility.py
**What it does:** File I/O helpers (make_dir, safe CSV write) and math utilities
(AUC, MCC wrappers).

**Pipeline stage:** Cross-cutting  
**Strategy:** Migrate into `src/evo_amr/` utilities.

---

### src/data_preprocess/preprocess.py
**What it does:** Reads raw metadata, filters by species/antibiotic availability,
normalises phenotype labels, exports clean TSVs.

**Pipeline stage:** Dataset  
**Strategy:** Migrate.

---

### src/analysis_utility/result_analysis.py
**What it does:** Aggregates nested-CV JSON score files into summary tables
(mean ± std over folds). Produces per-antibiotic, per-species, per-classifier
tables with F1, AUC, MCC, etc. Both clinical and standard thresholds.

**Pipeline stage:** Report  
**Strategy:** Migrate — this is the primary reporting logic for the baseline suite.

---

### src/benchmark_utility/benchmark.py
**What it does:** Orchestrates the full benchmark loop: for each species × antibiotic
× classifier, set up data, run model, collect scores.

**Pipeline stage:** Evaluation  
**Strategy:** Migrate as the evaluation coordinator.

---

### scripts/model/*.sh (kover.sh, resfinder.sh, phenotypeseeker.sh, seq2geno.sh, etc.)
**What it does:** Shell launchers for each baseline. Set environment variables,
call the appropriate Python main script with the right flags.

**Pipeline stage:** HPCOrchestration  
**Strategy:** Adapter — these become the backend scripts that
`src/evo_amr/baselines/<tool>.py` wrappers invoke.

---

### main.sh
**What it does:** Top-level orchestration for the full benchmarking run. Sequences
preprocessing → fold generation → baseline training → result analysis.

**Pipeline stage:** HPCOrchestration  
**Strategy:** Adapter — serves as reference for the unified CLI orchestration.

---

### Config.yaml
**What it does:** Benchmarking system config: species list, antibiotics, fold
count, CV strategy, tool paths.

**Pipeline stage:** Cross-cutting (config)  
**Strategy:** Migrate schema into `configs/baselines/` YAML format.

---

## Source System 3: prediction/

BV-BRC / ESM embedding data pipeline, PyTorch Lightning training and evaluation,
SLURM workflow orchestration.

### train.py
**What it does:** PyTorch Lightning training entry point. Parses a `.pkl` config,
initialises `SingleDrugBOPLitModule` or multi-drug `BOPLitModule`, configures
W&B logger, checkpoint callbacks, runs `Trainer.fit()`.

**Pipeline stage:** Model  
**Strategy:** Migrate — this is the authoritative training loop for the DL models.

---

### evaluate.py
**What it does:** PyTorch Lightning evaluation entry point. Loads a checkpoint,
runs `Trainer.predict()` on a specified split, writes prediction CSVs.

**Pipeline stage:** Evaluation  
**Strategy:** Migrate.

---

### lib/lightning_modules.py
**What it does:** `BOPLitModule` and `SingleDrugBOPLitModule` — PyTorch Lightning
modules. Forward pass calls the MLP trunk, computes loss, logs per-step and
per-epoch metrics. Handles both single-drug (binary) and multi-drug (multi-task)
settings. Supports species-stratified metrics via `ClusterWeightScheme`.

**Pipeline stage:** Model  
**Strategy:** Migrate as the DL model backbone in `src/evo_amr/models/`.

---

### lib/modules.py
**What it does:** `SimpleMLPTrunk`, `ResidualBlock`, attention head variants.
Configurable MLP architectures built from a layer spec list. Used by Lightning
modules as the prediction head.

**Pipeline stage:** Model  
**Strategy:** Migrate.

---

### lib/loss.py
**What it does:** `MultiDrugBCEWithLogitsLoss` — multi-task binary cross-entropy
with NaN masking (handles missing antibiotic labels), per-task reduction, and
cluster weighting schemes (uniform, sqrt, log2).

**Pipeline stage:** Model  
**Strategy:** Migrate.

---

### lib/data.py
**What it does:** Data pipeline for amr_pred. Loads ESM embeddings from `.pt` files
(per-protein), aggregates to genome-level representations (sum/mean over proteins
in the proteome), handles multi-drug multi-label targets with NaN for missing.
`BasicBatchConverter` and `MultiDrugBatchConverter` for DataLoader collation.

**Pipeline stage:** Dataset → FeatureTransform  
**Strategy:** Migrate the collation and aggregation logic. The ESM loading path
is ESM-specific; generalise to also support Evo embeddings loaded from HDF5.

---

### lib/cv.py
**What it does:** CV split generation. `define_cv_kfold` (KFold, StratifiedKFold,
GroupKFold by species/genus, StratifiedGroupKFold) and `define_cv_logo`
(Leave-One-Species/Group-Out with minimum threshold). Returns `{fold_name: (train_idx, test_idx)}` dicts.

**Pipeline stage:** Split  
**Strategy:** Migrate into `src/evo_amr/splits/`. This is the most complete CV
library across all three systems — use it as the canonical implementation.

---

### lib/enums.py
**What it does:** Enums for `Representations`, `Embedding_Models`, `DataSplit`,
`CV_Design`, `ClusterWeightScheme`, `ModelStage`, `MultiDrugEvalMetrics`,
`SingleDrugEvalMetrics`, and more. Central vocabulary for the system.

**Pipeline stage:** Cross-cutting  
**Strategy:** Migrate into `src/evo_amr/` as the canonical enum module; extend
with Evo-specific embedding models and partition types.

---

### analysis/metrics.py
**What it does:** `compute_classification_metrics` — computes AUROC, AUPRC,
accuracy, sensitivity (recall positive), specificity (recall negative), F1
from raw predictions. Handles edge case of all-one-class labels.

**Pipeline stage:** Evaluation  
**Strategy:** Migrate as the primary metric computation function in
`src/evo_amr/evaluation/metrics.py` (supplements the existing stub there).

---

### analysis/plotting.py
**What it does:** `plot_cat_param`, `plot_bool_param` — seaborn boxplot + swarmplot
for hyperparameter sweep results; `spearmanr`/`mannwhitneyu` annotations.

**Pipeline stage:** Report  
**Strategy:** Migrate into `src/evo_amr/reporting/`.

---

### simple_model/simple_model.py
**What it does:** Logistic regression baseline on ESM embeddings. `compute_or_load_X`
loads or computes genome-level ESM representations, `simple_model` fits LR,
`reconcile_data` aligns metadata with embedding files.

**Pipeline stage:** Baseline → Model  
**Strategy:** Migrate — generalise the embedding loading to support Evo HDF5 as well.

---

### simple_model/cv_opt.py
**What it does:** 10-fold CV hyperparameter search over LR penalty/C for the
simple model. Hardcoded dataset/run params at top of file.

**Pipeline stage:** Model  
**Strategy:** Migrate into a configurable experiment runner.

---

### data_prep/utils.py
**What it does:** GTDB-Tk taxonomy parsing (`create_and_save_gtdbtk_summary_df`,
`standardize_gtdbtk_warnings`), data split config reading, metadata loading
helpers. Manages GTDB rank taxonomy (domain → species).

**Pipeline stage:** Dataset  
**Strategy:** Migrate taxonomy parsing into `src/evo_amr/data/`.

---

### utils/data_utils.py
**What it does:** Load metadata CSVs/TSVs, load pre-cached embedding representations,
load data split configs. `edit_species` normalises noisy species labels. Core
data access layer for amr_pred.

**Pipeline stage:** Dataset → Manifest  
**Strategy:** Migrate — this and `benchmarking/src/amr_utility/load_data.py`
are the two candidate authoritative loaders; merge into one manifest reader.

---

### utils/path_utils.py
**What it does:** All path construction for amr_pred: data splits, metadata,
embeddings, cache, model checkpoints, eval outputs, staging dirs. Depends on
`*.env` environment variables.

**Pipeline stage:** Cross-cutting (HPCOrchestration)  
**Strategy:** Migrate path logic; replace env-var dependency with the config
system already started in `src/evo_amr/config/`.

---

### utils/metric_utils.py
**What it does:** Initialises `torchmetrics.MetricCollection` objects for training
and eval. Maps enum keys to metric class + init kwargs for single-drug and
multi-drug settings.

**Pipeline stage:** Evaluation  
**Strategy:** Migrate.

---

### utils/checkpoint_utils.py
**What it does:** Find best checkpoint by metric, extract epoch number from filename,
multi-drug checkpoint callback configuration.

**Pipeline stage:** Model → HPCOrchestration  
**Strategy:** Migrate.

---

### utils/workflow_utils.py
**What it does:** `submit_dynamic_sbatch` — dynamically generates and submits a
SLURM job from a template, waits for job ID, returns it. Used by all `workflows/`
launchers.

**Pipeline stage:** HPCOrchestration  
**Strategy:** Adapter (shell out to `sbatch`).

---

### workflows/launch_training.py
**What it does:** Configures a single-drug or multi-drug training run: loads
base config, merges run-specific overrides, calls `submit_dynamic_sbatch` to
submit the SLURM job. Hardcoded environment constants at the top (server paths,
env names, SLURM args).

**Pipeline stage:** HPCOrchestration  
**Strategy:** Adapter — the config merging logic is migratable; the SLURM
submission shells out.

---

### workflows/launch_baseline_training.py
**What it does:** Same pattern as `launch_training.py` but for simple model
baselines.

**Pipeline stage:** HPCOrchestration  
**Strategy:** Adapter.

---

### workflows/launch_sweep.py
**What it does:** Configures and launches a W&B hyperparameter sweep, submits
SLURM array jobs via `submit_dynamic_sbatch`.

**Pipeline stage:** HPCOrchestration  
**Strategy:** Adapter.

---

### sweep.py
**What it does:** W&B sweep config (Bayes optimisation over LR, hidden dim,
dropout, weight decay for multi-drug model).

**Pipeline stage:** Model (hyperparameter search config)  
**Strategy:** Migrate config format into `configs/experiments/` YAML.

---

### compute_drug_split_stats.py
**What it does:** Per-drug summary statistics across data splits: genome counts,
R/S ratios, species breakdown. Writes a report CSV.

**Pipeline stage:** Split → Report  
**Strategy:** Migrate.

---

### lib/enums.py → Embedding_Models
**Note:** amr_pred currently tracks ESM1b, ESM2-650M, ESM2-3B, ESM2-15B as
`Embedding_Models`. Evo models need to be added here during migration.

---

### configs/single_drug/attention_v0/ + configs/multi_drug/precomp/trunk_2_head_2.py
**What it does:** Pickle-serialised model configs for attention-based and
precomputed-embedding MLP runs. Specifies model architecture, optimizer, loss,
metrics, data split.

**Pipeline stage:** Model (config)  
**Strategy:** Migrate to YAML format in `configs/experiments/`.

---

### envs/*.env (harry_manitou.env, noga_manitou.env, huilin_manitou.env, etc.)
**What it does:** Per-user, per-cluster environment variable files with private
server paths (`/burg/pmg/`, `/pmglocal/`, `/manitou/`, W&B credentials).

**Pipeline stage:** HPCOrchestration  
**Strategy:** **Gitignore.** These contain private server paths and credentials.
Provide `envs/template.env` instead.

---

### testtt/ (auto_stat_drug.py, cut_both_approaches.py, rank.py, etc.)
**What it does:** Exploratory/scratch scripts for data inspection, distribution
checks, ranking experiments. Not part of the main pipeline.

**Strategy:** Keep in source tree for reference; do not migrate. Mark in
`src/evo_amr/` registry as `status: reference`.

---

### debug/ (debug_eval.py, debug_train.py)
**What it does:** Debug harnesses that run training/eval with small data samples.

**Strategy:** Keep as reference; do not migrate.

---

## Gitignore Recommendations

Items to add/confirm in `.gitignore`:

```
# Private server environment files (contain cluster paths and credentials)
prediction/envs/*.env
!prediction/envs/template.env

# Nested git repos (tracked separately or not at all)
prediction/.git
benchmarking/.git
evo-hidden/.git

# Raw genome data
benchmarking/data/
prediction/data/_raw_md/
prediction/data/*/

# Old backups
prediction/backup_data*/

# Conda/pip environment artifacts inside the repo
prediction/envs/*.yml   # keep — these document dependencies
prediction/create_env.log
prediction/install_log.txt

# Model weights (already in root .gitignore but confirm)
models/evo-1-8k-base/
models/evo_1_131k_base/*.pt
models/evo_1_131k_base/*.safetensors
models/evo_enhanced/

# Scratch / temp
temp_trash/
prediction/debug.txt
prediction/testtt/    # (optional: keep for reference, or gitignore)

# SLURM logs already in root .gitignore
*.out
*.err
```

Currently `.gitignore` already excludes `benchmarking/`, `prediction/`, and
`evo-hidden/` at the bottom (treating them as untracked imported systems).
That blanket exclusion should be **removed** once migration begins, replaced
with targeted exclusions above so source files are tracked.

---

## Migration Priority Order

| Priority | Source | Files | Reason |
|---|---|---|---|
| 1 | Root | `embedding_pipeline/embedding_generator.py` | Core to all experiments |
| 2 | Root | `minirocket/…/amr_classification_pipeline.py` | MiniRocket is primary method |
| 3 | Root | `data_preprocessing/*.py` | Needed to reproduce dataset |
| 4 | `amr_pred` | `lib/cv.py` | Most complete CV implementation |
| 5 | `amr_pred` | `analysis/metrics.py` | Metric standard across systems |
| 6 | `amr_pred` | `lib/lightning_modules.py`, `lib/modules.py`, `lib/loss.py` | DL backbone |
| 7 | `AMR_benchmarking` | `src/cv_folds/`, `src/amr_utility/` | Split + data load |
| 8 | `AMR_benchmarking` | `src/analysis_utility/result_analysis.py` | Baseline reporting |
| 9 | `AMR_benchmarking` | `AMR_software/*/main_*.py` | Baseline adapters |
| 10 | `amr_pred` | `workflows/`, `utils/workflow_utils.py` | SLURM orchestration |
