# Reproducibility Notes

This repository is structured as a research showcase. Some experiments depend on restricted/local datasets, large generated embeddings, installed external AMR tools, and HPC paths. Full reproduction requires access to the original data and compute environment.

The repository is therefore designed to be inspectable:

- example configs document intended parameters
- dry-run CLI commands validate experiment definitions
- tiny example manifests show the expected schemas
- architecture docs explain how the earlier research systems connect
- tests focus on configuration and interface sanity checks

## What Is Not Tracked

The following should stay out of git:

- raw genome FASTA/FNA files
- generated HDF5/NumPy embeddings
- model checkpoints and weights
- Kover datasets and result dumps
- notebook outputs and large figures
- local conda or virtual environments
- SLURM logs and caches

## Reproducible Artifact Strategy

For future papers, each experiment should save:

1. YAML config snapshot
2. manifest snapshot or checksum
3. code commit hash
4. environment/profile name
5. per-partition metrics
6. per-species metrics
7. prediction table
8. report markdown

This is enough to audit the experiment even when raw data cannot be redistributed.
