# Thesis Structure To Framework Map

This document maps the thesis to the reconstructed Evo-AMR framework. The goal
is for the repository to make the research structure visible even when private
data and HPC resources are unavailable.

## Thesis Core Claim

Cross-species AMR prediction is an out-of-distribution generalization problem.
Representations must preserve transferable resistance mechanisms without
collapsing into phylogenetic shortcuts.

The project has three experimental pillars:

1. Classical baseline stress test with Kover under species holdout.
2. Diagnostic-driven Evo layer selection.
3. Mechanism-aware aggregation comparison: Global Pooling vs MiniRocket.

## Chapter Map

| Thesis section | Research role | Framework home | Example config / artifact |
| --- | --- | --- | --- |
| Chapter 2: Data | Dataset construction, filtering, five-way partitions | `src/evo_amr/data`, `src/evo_amr/splits` | `configs/datasets/bvbrc_multi_drug.example.yaml` |
| 2.3 Retention criteria | Antibiotic/species filtering and power constraints | `src/evo_amr/data` | future `DatasetFilterConfig` |
| 2.5 Split design | train / val_overlapped / val_outside / test_overlapped / test_outside | `src/evo_amr/splits` | `examples/tiny_manifest.csv` |
| 2.6 Leakage checks | Guard against species overlap in outside partitions | `src/evo_amr/splits/leakage.py` | tested in `tests/test_framework_primitives.py` |
| Chapter 3: Kover baseline | Interpretable k-mer baseline and OOD degradation | `src/evo_amr/baselines` | `configs/experiments/kover_species_holdout_six_drugs.example.yaml` |
| 3.3 Experimental setup | Kover SCM, k=31, tune on val_outside, three replicates | baseline adapter + config | `configs/baselines/kover.example.yaml` |
| Chapter 4: Layer diagnostics | Activation scale, isotropy, effective rank, cross-seed stability | `src/evo_amr/diagnostics` | `configs/experiments/evo_layer_diagnostics.example.yaml` |
| 4.4 L11 stability boundary | L10 is deepest stable extraction layer | `src/evo_amr/diagnostics/layers.py` | `validate_layer_choice(10)` |
| Chapter 5: Local pattern preservation | Compare Global Pooling vs MiniRocket on Evo L10 embeddings | `src/evo_amr/features`, `src/evo_amr/evaluation` | `configs/experiments/evo_minirocket_ampicillin.example.yaml` |
| 5.3 Experimental design | 3,388 genomes, 126 species, ampicillin, common classifier panel | `src/evo_amr/config` | `ExperimentConfig` |
| 5.4 Aggregate metrics | MCC, F1, AUROC, AUPRC across partitions | `src/evo_amr/evaluation/results.py` | `ResultRecord` |
| 5.5 Neighbor analysis | k=20 neighbor audit, species-level mechanism evidence | `src/evo_amr/visualization`, `src/evo_amr/evaluation/mechanisms.py` | `configs/experiments/mechanism_neighbor_audit.example.yaml` |
| Appendix A | Replicate robustness across v1-2 and v1-3 | `src/evo_amr/evaluation` | future replicate report config |

## Thesis-Specific Artifacts

The clean framework should eventually emit these artifact families:

- `manifest.csv`: normalized genome/species/phenotype/split table.
- `split_audit.json`: species leakage and partition statistics.
- `layer_diagnostics.csv`: per-layer stability metrics.
- `embedding_manifest.csv`: genome to embedding artifact map.
- `feature_manifest.csv`: aggregation method to feature artifact map.
- `predictions.csv`: genome-level predictions and probabilities.
- `metrics_by_partition.csv`: AUROC/AUPRC/F1/MCC by split.
- `metrics_by_species.csv`: species-level performance.
- `neighbor_audit.csv`: nearest-neighbor species/phenotype composition.
- `report.md`: run summary with config snapshot and thesis interpretation.

## Mechanism-Mix Hypothesis In Code

The thesis conclusion is not “MiniRocket always wins.” It is:

| Mechanism regime | Expected best aggregation | Code concept |
| --- | --- | --- |
| Cassette-mediated / horizontally transferred | MiniRocket | `CASSETTE_MEDIATED` |
| Chromosomal / diffuse / lineage-coupled | Global Pooling | `CHROMOSOMAL_DIFFUSE` |
| Mixed or unknown | Compare or ensemble both | `MIXED_OR_UNKNOWN` |

This is represented in `src/evo_amr/evaluation/mechanisms.py`.

## What Still Needs Migration

The old code contains more implementation than the clean framework currently
wraps. The next migrations should be:

1. Promote species/antibiotic filtering into `src/evo_amr/data`.
2. Wrap the Kover species-holdout pipeline as a dry-run adapter.
3. Wrap Evo Layer 10 extraction using the existing `embedding_pipeline`.
4. Wrap MiniRocket and Global Pooling feature transforms.
5. Normalize real result tables into `ResultRecord` rows.
