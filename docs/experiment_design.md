# Experiment Design

Evo-AMR experiments should be defined by configuration, not by editing scripts.

## Experiment Unit

An experiment is the tuple:

```text
dataset + split design + representation + aggregation + model + metrics
```

For example:

```text
BV-BRC multi-drug Aug 2024
+ clustered species holdout split
+ Evo-1-8k-base Layer 10 embeddings
+ MiniRocket aggregation
+ k-NN / tree / linear classifiers
+ AUROC, AUPRC, F1, MCC
```

## Species-Holdout Evaluation

The key research setting is out-of-distribution AMR prediction. Splits should make species membership explicit:

- `train`: species used for model fitting
- `val_overlapped`: validation genomes from species represented in training
- `val_outside`: validation genomes from held-out species
- `test_overlapped`: test genomes from training-overlapped species
- `test_outside`: test genomes from held-out species

Every split artifact should be auditable for species leakage.

## Manifest Schema

All backends should converge on a common manifest table:

```text
genome_id
species
antibiotic
phenotype
partition_label
sequence_path
embedding_path
source_dataset
split_design
```

Backend-specific fields are allowed, but these columns should be stable.

## Representation Families

### Classical Baselines

- Kover k-mer set covering machines
- PhenotypeSeeker k-mer models
- ResFinder determinant calls
- Seq2Geno feature pipelines
- Aytan-Aktug neural baselines
- Majority baseline

### Foundation Model Representations

- Evo genome embeddings
- ESM protein/genome-derived embeddings
- Layer-selected hidden states
- Pooled genome vectors
- Ordered per-window embedding streams

## Aggregation Choices

Aggregation is a scientific variable, not just implementation detail.

- Global pooling captures broad genomic background.
- PCA and SRP provide compact baselines.
- MiniRocket preserves local activation patterns, useful for cassette-scale resistance mechanisms.

## Result Schema

Each run should emit:

- config snapshot
- manifest snapshot or hash
- per-partition metrics
- per-species metrics
- predictions table
- optional neighbor-audit table
- environment/profile metadata

This makes old experiments easier to compare even when full data reproduction is restricted.
