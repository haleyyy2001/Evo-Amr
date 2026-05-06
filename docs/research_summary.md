# Research Summary

This repository supports the thesis:

**Cross-Species Antimicrobial Resistance Prediction from Genomic Foundation Models**

Huilin Tai, Columbia University, 2025

The full thesis PDF is included at [cross_species_amr_prediction_thesis.pdf](cross_species_amr_prediction_thesis.pdf).

## Research Question

Can genomic foundation model embeddings improve antimicrobial resistance prediction when training and evaluation species are phylogenetically distinct?

This is an out-of-distribution generalization problem. Standard random splits can overestimate performance because train and test genomes often share species-specific background signals. This project instead uses strict species holdout splits to evaluate whether learned representations transfer across bacterial taxa.

## Main Contributions

1. Species holdout evaluation protocol that exposes cross-species degradation hidden by same-species evaluation.
2. Diagnostic framework for selecting stable Evo-1-8k-base embedding layers under native bfloat16 inference.
3. Layer 10 extraction strategy, selected as the deepest jointly stable layer before the Layer 11 stability boundary.
4. MiniRocket aggregation over ordered per-window embeddings to preserve local resistance patterns.
5. Species-level neighbor analysis showing when local pattern preservation helps and when global pooling is more appropriate.

## Key Findings

- Kover provides a strong interpretable k-mer baseline, but performance drops under true cross-species evaluation.
- Evo activation diagnostics show a sharp stability boundary at Layer 11; Layer 10 is the preferred extraction layer for downstream AMR experiments.
- MiniRocket can reorganize Evo embedding space toward phenotype-aligned neighborhoods, especially for species where cassette-mediated resistance is plausible.
- Global pooling remains competitive for chromosomal or diffuse resistance mechanisms.
- Aggregation strategy is mechanism-dependent; no single representation dominates every held-out species set.

## Representative Ampicillin Results

The thesis reports experiments across 3,388 genomes from 126 species for ampicillin resistance.

- On `val_outside`, MiniRocket k-NN achieved MCC 0.753, compared with Global Pooling k-NN MCC 0.148.
- On `test_outside`, Global Pooling with LightGBM achieved MCC 0.997, while MiniRocket achieved MCC 0.956.
- For k-NN on `val_outside`, MiniRocket increased AUROC from 0.515 to 0.926.

These split-level differences motivated species-level analysis. MiniRocket performed best when local, horizontally transferable resistance elements were likely to dominate. Global Pooling performed better when broad chromosomal background was more informative.

## Practical Design Principle

Match embedding aggregation to the expected biology:

- Use local pattern aggregation, such as MiniRocket, when resistance may be driven by plasmids, transposons, genomic islands, or other cassette-scale mechanisms.
- Use global pooling when resistance is primarily chromosomal, diffuse, or lineage-coupled.
- Compare both approaches when the mechanism profile is unknown or mixed.

## Code Map

- `embedding_pipeline/`: Evo embedding extraction and pooling options
- `diagnostics/`: model-layer diagnostic runs and SLURM entry points
- `minirocket/minirocket_pipeline/core/`: MiniRocket, PCA, and SRP downstream pipelines
- `minirocket/minirocket_pipeline/visualization/`: neighbor, phylogenetic, and method comparison analysis
- `data_preprocessing/`: dataset filtering, balancing, and species subset generation

For the reconstructed framework view, see [thesis_structure.md](thesis_structure.md).
