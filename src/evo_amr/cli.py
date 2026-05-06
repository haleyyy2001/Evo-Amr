"""Command line interface for the Evo-AMR showcase framework.

The CLI is intentionally conservative: by default it validates configuration and
prints the planned workflow. Heavy research jobs still run through the legacy
scripts and server-specific launchers until each backend is migrated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evo_amr.baselines import (
    ShellBaselineAdapter,
    get_baseline_backend,
    list_baseline_backends,
)
from evo_amr.config import ExperimentConfig, load_yaml
from evo_amr.embeddings import EvoEmbeddingConfig, EvoEmbeddingAdapter
from evo_amr.models import get_model_family, list_model_families
from evo_amr.pipelines import list_pipelines
from evo_amr.workflows.tiny import run_tiny_workflow
from evo_amr.workflows.plans import build_stage_plan


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML")
    parser.add_argument(
        "--profile",
        default="local",
        help="Path profile name, e.g. local, burg, insomnia",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the backend command. Omit for dry-run planning.",
    )


def _print_plan(stage: str, args: argparse.Namespace) -> int:
    config_data = load_yaml(args.config)
    experiment_config = ExperimentConfig.from_dict(config_data)
    mode = "execute" if args.execute else "dry-run"
    experiment = experiment_config.name

    print(f"[evo-amr] stage={stage}")
    print(f"[evo-amr] mode={mode}")
    print(f"[evo-amr] profile={args.profile}")
    print(f"[evo-amr] experiment={experiment}")
    print(f"[evo-amr] config={args.config}")
    print(build_stage_plan(stage, experiment_config).render())

    if stage == "run-baseline":
        baseline = config_data.get("baseline", {})
        execution = config_data.get("execution", {})
        script = execution.get("script")
        if script:
            adapter = ShellBaselineAdapter(
                name=baseline.get("name", args.config.stem),
                script=Path(script),
                backend_root=Path(baseline["backend"]) if baseline.get("backend") else None,
            )
            print(f"[evo-amr] backend command: {adapter.dry_run_command(args.config).render()}")

    if stage == "embed" and experiment_config.dataset.manifest:
        adapter = EvoEmbeddingAdapter(EvoEmbeddingConfig.from_experiment_yaml(config_data))
        print("[evo-amr] embedding adapter:")
        print(adapter.describe().rstrip())
        print(
            "[evo-amr] backend command: "
            f"{adapter.dry_run_command(experiment_config.dataset.manifest)}"
        )

    if not args.execute:
        print("[evo-amr] dry-run only; backend command was not launched")
        return 0

    if stage == "report":
        print(run_tiny_workflow(args.config))
        return 0

    raise NotImplementedError(
        "Execution adapters are intentionally staged. Use the documented legacy "
        "backend scripts or add a backend wrapper before enabling --execute."
    )


def _list_baselines(args: argparse.Namespace) -> int:
    """Print registered legacy baseline backends."""
    print("[evo-amr] registered baseline backends")
    for name in list_baseline_backends():
        backend = get_baseline_backend(name)
        if args.family and backend.family != args.family:
            continue
        capabilities = []
        if backend.supports_multi_species:
            capabilities.append("multi_species")
        if backend.supports_multi_drug:
            capabilities.append("multi_drug")
        caps = ",".join(capabilities) if capabilities else "single_task"
        print(
            f"- {backend.name}: family={backend.family}; action={backend.action}; "
            f"capabilities={caps}; script={backend.script}"
        )
    return 0


def _list_models(args: argparse.Namespace) -> int:
    """Print registered trainable model families."""
    print("[evo-amr] registered model families")
    for name in list_model_families(task=args.task):
        family = get_model_family(name)
        print(
            f"- {family.name}: task={family.task}; representation={family.representation}; "
            f"output={family.output_mode}; source={family.legacy_source}"
        )
    return 0


def _list_pipelines(args: argparse.Namespace) -> int:
    """Print the rebuilt research pipeline catalog."""
    print("[evo-amr] rebuilt research pipelines")
    for pipeline in list_pipelines(status=args.status):
        capabilities = "; ".join(pipeline.capabilities)
        sources = ", ".join(pipeline.source_systems)
        modules = ", ".join(pipeline.clean_modules)
        print(
            f"- {pipeline.name}: status={pipeline.status}; stage={pipeline.stage}; "
            f"entrypoint={pipeline.entrypoint}; modules={modules}; sources={sources}; "
            f"capabilities={capabilities}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evo-amr",
        description="Config-driven AMR benchmarking research interface",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in [
        "prepare-data",
        "create-splits",
        "embed",
        "run-baseline",
        "train",
        "evaluate",
        "report",
    ]:
        child = subparsers.add_parser(command)
        _add_common_args(child)
        child.set_defaults(func=lambda args, stage=command: _print_plan(stage, args))

    baselines = subparsers.add_parser("list-baselines")
    baselines.add_argument("--family", help="Filter by backend family")
    baselines.set_defaults(func=_list_baselines)

    models = subparsers.add_parser("list-models")
    models.add_argument("--task", help="Filter by task, e.g. single_drug or multi_drug")
    models.set_defaults(func=_list_models)

    pipelines = subparsers.add_parser("list-pipelines")
    pipelines.add_argument("--status", help="Filter by rebuild status")
    pipelines.set_defaults(func=_list_pipelines)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
