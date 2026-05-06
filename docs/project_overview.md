# Project Overview

## The Problem: Antimicrobial Resistance Prediction Under Phylogenetic Shift

Antimicrobial resistance (AMR) kills an estimated 1.27 million people per year and is projected to become the leading cause of infectious-disease death by 2050. Routine susceptibility testing — growing the pathogen in culture, exposing it to antibiotics, observing growth — takes 48 to 72 hours. During that window clinicians must choose between waiting (risky) and prescribing broad-spectrum antibiotics empirically (accelerates resistance). Modern short-read sequencing can decode a bacterial genome in hours, but translating that sequence into a reliable resistance prediction is an unsolved problem.

The core difficulty is **biological heterogeneity across species**. Resistance arises from different mechanisms in different organisms:

- **Cassette-mediated resistance** — β-lactamase genes, efflux pump operons, or integron-carried determinants spread horizontally via plasmids, transposons, and genomic islands. These elements are physically compact, functionally conserved, and can transfer across distant taxa. A model that recognizes the blaZ or blaTEM sequence in *Staphylococcus aureus* should in principle recognize it in *Klebsiella pneumoniae* if it has learned the relevant features.

- **Chromosomal / diffuse resistance** — altered promoters, regulatory mutations, loss-of-function in membrane channel genes, or progressive accumulation of point mutations in drug-target genes. These signals are embedded in the broader lineage-specific genomic background and do not transfer cleanly across species.

No single representation captures both. A model trained on one mechanism profile and evaluated on another may fail silently. This is the central challenge the project addresses.

---

## The Evaluation Gap: Why Random Splits Overestimate Performance

Standard train-test splits sample genomes randomly within a dataset. When train and test sets share species, they share background genomic signals — GC content, codon usage, chromosomal architecture, and lineage-specific mutation patterns. A model can learn these species-level fingerprints and predict resistance labels without actually learning transferable resistance biology.

**Species-holdout evaluation** removes this confound by ensuring that all genomes from a given species appear exclusively in training or exclusively in evaluation, never both. This is a strict out-of-distribution (OOD) generalization setting. Performance on held-out species reflects what the model has actually learned about resistance mechanisms rather than about taxonomic identity.

The project demonstrates that this distinction matters quantitatively. For the same model, same data, and same training procedure:

- **Same-species evaluation** (val_overlapped, test_overlapped): model appears to generalize well
- **Cross-species evaluation** (val_outside, test_outside): performance can collapse unless the representation is mechanism-appropriate

This is not a methodological curiosity. In clinical deployment, a model predicting AMR for a newly sequenced pathogen will almost always encounter species the model has not seen before.

---

## Why Genomic Foundation Models

Classical AMR baselines — Kover k-mer set covering machines, ResFinder determinant calls, PhenotypeSeeker — work from explicit sequence features or known resistance determinant databases. They are interpretable and fast but rely on pre-specified patterns.

Genomic foundation models offer a different approach: instead of specifying what to look for, train a deep model on massive unlabeled genomic corpora to learn general-purpose sequence representations, then fine-tune or probe those representations for AMR labels.

This project uses **Evo-1-8k-base**, a Hyena SSM-based (State Space Model) foundation model trained on prokaryotic DNA. Unlike protein-centric models (ESM, ProtTrans), Evo operates on raw DNA sequences without translating to amino acids, which means it can capture non-coding regulatory elements, intergenic resistance determinants, and whole-genome chromosomal context.

Key properties:
- Tokenizes raw DNA at single-nucleotide resolution
- 8k-token context window (≈ 8 kb, covers typical AMR gene clusters)
- 32 transformer-like (Hyena) layers
- Trained on diverse prokaryotic genomes
- Extracted as per-token hidden states at any layer

---

## Evo Embedding Strategy: Layer Diagnostics and Layer 10 Selection

Not all layers of a deep model produce equally useful representations. Early layers capture low-level sequence statistics; later layers integrate longer-range context but may also encode information specific to pre-training objectives that does not transfer to downstream tasks.

This project implements a **systematic diagnostic suite** across all 32 Evo layers under native bfloat16 inference. For each layer and each of several random seeds, the following metrics are computed:

| Metric | What it measures |
| --- | --- |
| Activation magnitude | average L2 norm of hidden states |
| Isotropic angular diversity | uniformity of direction distribution |
| Effective rank | rank of the activation covariance matrix |
| Singular spectrum | shape of the singular value distribution |
| Token-norm concentration | whether activations collapse to a subspace |
| Cross-seed stability | variance of the above metrics across random seeds |

The diagnostics reveal a **sharp stability boundary at Layer 11**: metrics become highly variable across seeds and across sequence inputs, indicating that the model's internal representations undergo a qualitative transition at that point. **Layer 10 is the deepest jointly stable extraction layer** — it captures rich contextual information without the instability that makes Layer 11 and beyond unreliable for downstream use.

All downstream AMR experiments use Layer 10 embeddings extracted under bfloat16 inference. The diagnostic code lives in `diagnostics/` and is parameterized by SLURM job scripts; the extraction pipeline lives in `embedding_pipeline/` and is being wrapped into `src/evo_amr/embeddings/`.

### Implementation: `modified_model.py`

The original `StripedHyena` model class was extended to expose hidden states without breaking the original checkpoint or inference path. The modified class at `models/evo_1_131k_base/modified_model.py` adds:

- `output_hidden_states: bool` and `return_dict: bool` flags on `forward()` — backward-compatible with the original signature
- Hidden states collected as `x.clone()` **after each block**, building a list of 32 tensors of shape `(batch, seq_len, hidden_dim)`
- Two convenience extraction methods:

```python
# Single-layer extraction (used for AMR experiments)
outputs = model.extract_layer_hidden_states(input_ids, target_layer=10)
hidden = outputs['hidden_states']   # shape: (1, n_tokens, hidden_dim)

# Multi-layer extraction (used for layer diagnostic suite)
layer_states = model.extract_multiple_layers(input_ids, layer_list=list(range(32)))
```

`hidden_states[10]` corresponds to the output of block 10 in the 0-indexed 32-block model. The `last_hidden_state` returned by `return_dict=True` is the post-RMSNorm representation used for next-token prediction and is **not** used for AMR tasks.

Memory and precision setup for GPU inference:

```python
model.to_bfloat16_except_poles_residues()   # poles/residues stay in float32 for stability
model.precompute_filters(L=8192, device=device)   # log-exp trick, stored in bfloat16
```

---

## Feature Aggregation: From Per-Token Embeddings to Genome Vectors

An Evo-1-8k-base inference pass over a genome produces a matrix of shape `(n_tokens, hidden_dim)` at the selected layer. For a full bacterial genome (≈ 3–6 Mb), this is processed in overlapping 8k-token windows, yielding an ordered sequence of per-window embedding matrices.

**Aggregation converts this sequence into a fixed-size feature vector for classification.** This project treats aggregation as a scientific variable and compares several strategies:

### Global Mean Pooling

Average all token-position hidden states across all windows. Produces a single vector of shape `(hidden_dim,)`. Captures the broad genomic background signal. Works well for diffuse or chromosomal resistance mechanisms.

### PCA

Apply principal component analysis to retain 90% of variance. Reduces dimensionality for downstream classifiers while preserving major geometric structure.

### Sparse Random Projection (SRP)

Project embeddings into a lower-dimensional space via a sparse random matrix (Johnson-Lindenstrauss guarantee). Parameter-free, reproducible, and a useful dimensionality-reduction baseline.

### MiniRocket

Convert the ordered sequence of per-window embeddings into local pattern features using random convolutional kernels:

1. Treat the ordered embedding sequence as a multivariate time series.
2. Apply 2,000 random convolutional kernels (varying lengths, dilations, biases).
3. Compute PPV (proportion-of-positive-values) pooling for each kernel — the fraction of positions where the convolution output is positive.
4. Concatenate PPV features across all kernels.

This produces a feature vector that preserves **local order information** — whether resistance-associated patterns appear at particular genomic positions or in particular proximity. This matters for cassette-mediated resistance, where gene cluster geometry is biologically meaningful.

Key hyperparameters: window size 2048 tokens, stride 1024 tokens, 2,000 kernels.

---

## Experimental Design at a Glance

| Component | Details |
| --- | --- |
| Dataset | BV-BRC / PATRIC genomic phenotype records |
| Antibiotics studied | ampicillin, gentamicin, ciprofloxacin, tetracycline, and others |
| Ampicillin dataset | 3,388 genomes from 126 species |
| Split design | strict species-holdout (train / val_overlapped / val_outside / test_overlapped / test_outside) |
| Split replicates | 3 independent versions (v1-1, v1-2, v1-3) |
| Representations | Evo L10, ESM, Kover k-mer, ResFinder, PhenotypeSeeker, majority |
| Aggregations | mean pooling, PCA, SRP, MiniRocket |
| Classifiers | k-NN (cosine), logistic regression, linear SVM, random forest, LightGBM |
| Metrics | AUROC, AUPRC, F1, MCC, balanced accuracy |
| Evaluation granularity | per-partition and per-species |

---

## Representative Results (Ampicillin)

| Partition | Model | Aggregation | AUROC | MCC |
| --- | --- | --- | --- | --- |
| val_outside | k-NN cosine | MiniRocket | 0.926 | 0.753 |
| val_outside | k-NN cosine | Global pooling | 0.515 | 0.148 |
| test_outside | LightGBM | Global pooling | — | 0.932 |
| test_outside | k-NN cosine | MiniRocket | — | 0.798 |

The split-level divergence reflects mechanism dependency. The val_outside species are predominantly affected by cassette-mediated resistance (MiniRocket excels). The test_outside species include more chromosomal/diffuse mechanisms (global pooling recovers).

**Core design principle:** match aggregation to expected biology. Use MiniRocket when cassette-scale local patterns are plausible; use global pooling when broad genomic context is more informative. Compare both when the mechanism profile of held-out species is unknown.

---

## How to Navigate This Repository

| Directory | Role |
| --- | --- |
| `src/evo_amr/` | Clean framework interface, CLI, config, adapters |
| `embedding_pipeline/` | Evo embedding extraction — being wrapped |
| `diagnostics/` | Layer diagnostic SLURM jobs |
| `minirocket/minirocket_pipeline/` | MiniRocket/PCA/SRP/classifier pipeline — being wrapped |
| `models/` | Evo model architecture (Hyena SSM) |
| `benchmarking/` | Classical baseline suite (Kover, ResFinder, etc.) |
| `prediction/` | Legacy PyTorch Lightning training framework |
| `data_preprocessing/` | Dataset filtering and species-balancing scripts |
| `configs/` | Example YAML configs for experiments, baselines, paths, SLURM |
| `examples/` | Tiny fixture manifest and fake outputs for dry-run testing |
| `docs/` | This document, thesis PDF, research summary, architecture plan |
| `tests/` | Lightweight configuration and interface sanity checks |

The thesis PDF at [`docs/cross_species_amr_prediction_thesis.pdf`](cross_species_amr_prediction_thesis.pdf) contains the full scientific writeup. The [`docs/research_summary.md`](research_summary.md) provides a concise findings summary.

---

## What Requires Private Data or HPC

Full reproduction of thesis experiments requires:

- The BV-BRC genomic dataset (not redistributable without registration)
- Generated Evo Layer 10 HDF5 embeddings (terabytes; not tracked in git)
- Installed Kover, ResFinder, PhenotypeSeeker binaries
- Columbia University HPC cluster (SLURM paths in `configs/paths/server.example.yaml`)

The repository is designed to be **inspectable without this infrastructure**:

- `examples/` provides toy fixtures with correct schemas
- `configs/` provides documented parameter shapes
- The CLI runs in dry-run mode without launching compute
- Tests validate configuration and interface correctness without data or GPU

## Full Pipeline Rebuild

The clean package is not meant to erase the earlier work. It is the stable
interface for rebuilding the full project as a structured set of research
pipelines:

- `benchmarking/` contributes classical baseline systems, PATRIC
  preprocessing, KMA/phylogeny folds, and benchmark reporting.
- `prediction/` contributes BV-BRC/ESM data preparation, PyTorch/Lightning
  single-drug and multi-drug models, taxonomy-aware metrics, experiment configs,
  and SLURM workflows.

The rebuild map is in [`docs/pipeline_rebuild_plan.md`](pipeline_rebuild_plan.md).
The same information is exposed from the CLI:

```bash
evo-amr list-pipelines
evo-amr list-baselines
evo-amr list-models
```

This makes the scope of the old pipelines visible while rebuilding them into one
coherent project and keeping private data, generated artifacts, nested
repositories, and server-only files out of the public API.
