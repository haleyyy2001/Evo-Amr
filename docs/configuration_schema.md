# Configuration Schema

Evo-AMR uses YAML files to describe experiments before any backend code is
launched. The schema is intentionally lightweight and inspectable.

## Top-Level Blocks

```yaml
experiment:
  name: evo_l10_minirocket_ampicillin_species_holdout
  task: binary_amr_prediction
  antibiotic: ampicillin
  evaluation: species_holdout

dataset:
  config: configs/datasets/bvbrc_multi_drug.example.yaml
  manifest: ${artifact_root}/manifests/bvbrc_multi_drug_aug2024.csv
  split_design: clustered_3_v1

representation:
  model: evo-1-8k-base
  extraction_layer: 10
  dtype: bfloat16
  output: ${embedding_root}/evo_l10/ampicillin

aggregation:
  methods: [mean_pooling, pca, sparse_random_projection, minirocket]

models:
  classifiers: [logistic_regression, linear_svm, random_forest, lightgbm, knn_cosine]

metrics: [auroc, auprc, f1, mcc, balanced_accuracy]

outputs:
  run_dir: ${artifact_root}/experiments/evo_l10_minirocket_ampicillin
  report: ${artifact_root}/reports/evo_l10_minirocket_ampicillin.md
```

Specialized thesis configs include:

- `configs/experiments/evo_layer_diagnostics.example.yaml`
- `configs/experiments/kover_species_holdout_six_drugs.example.yaml`
- `configs/experiments/mechanism_neighbor_audit.example.yaml`

## Manifest Contract

Every backend should converge on a manifest with these required columns:

| Column | Meaning |
| --- | --- |
| `genome_id` | Stable genome identifier used across metadata, sequences, embeddings, and predictions. |
| `species` | Species or GTDB species label used for leakage checks. |
| `antibiotic` | Antibiotic or drug class being predicted. |
| `phenotype` | Binary label: `1` resistant, `0` sensitive. |
| `partition_label` | Split assignment such as `train`, `val_outside`, or `test_outside`. |

Optional columns:

| Column | Meaning |
| --- | --- |
| `sequence_path` | FASTA/FNA path for embedding or baseline tools. |
| `embedding_path` | Representation artifact path. |
| `source_dataset` | Dataset provenance. |
| `split_design` | Split generator or protocol identifier. |

## Dry-Run Philosophy

The CLI should be useful even without private data:

- validate config shape
- print planned inputs and outputs
- print backend commands without launching them
- explain when a dataset depends on restricted/local paths
- show which old backend will eventually be wrapped

Execution is added only after a backend has an adapter and tests.
