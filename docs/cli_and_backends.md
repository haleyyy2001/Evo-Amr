# CLI And Backends

The `evo-amr` command is the stable public interface for the reconstructed project.
It is designed around dry-runs first because most real experiments depend on
restricted data, external AMR tools, and HPC environments.

## Commands

```bash
evo-amr prepare-data --config configs/datasets/bvbrc_multi_drug.example.yaml
evo-amr create-splits --config configs/experiments/evo_minirocket_ampicillin.example.yaml
evo-amr embed --config configs/experiments/evo_minirocket_ampicillin.example.yaml
evo-amr run-baseline --config configs/baselines/kover.example.yaml
evo-amr train --config configs/experiments/evo_minirocket_ampicillin.example.yaml
evo-amr evaluate --config configs/experiments/evo_minirocket_ampicillin.example.yaml
evo-amr report --config examples/example_experiment.yaml --execute
```

Without `--execute`, commands print a plan and do not launch heavy jobs.
Plans include expected inputs, outputs, notes about restricted data, and backend
commands where an adapter exists.

## Backend Strategy

Backends should be added in this order:

1. Dry-run adapter that prints the exact command or Python call.
2. Input validator for manifests, split columns, and expected artifacts.
3. Execution adapter for local or SLURM launch.
4. Result normalizer that emits the shared Evo-AMR result schema.

## Current Backend Status

| Backend | Status | Notes |
| --- | --- | --- |
| Tiny fixture workflow | implemented | Runs without private data or heavy dependencies. |
| Kover | dry-run adapter planned | Config exists; old backend remains untracked. |
| ResFinder | dry-run adapter planned | Config exists; old backend remains untracked. |
| Evo embeddings | scaffolded | Existing root code should be wrapped next. |
| MiniRocket/PCA/SRP | scaffolded | Existing pipeline should be wrapped after manifest schema stabilizes. |
| PyTorch/Lightning models | legacy | Mine from `amr_pred` after config conventions stabilize. |

## Why Dry-Run First

Dry-runs make the repo useful as a showcase even when the full compute environment
is unavailable. They demonstrate that experiments are config-driven, that stages
are explicit, and that the earlier systems are being rebuilt behind a coherent API.
