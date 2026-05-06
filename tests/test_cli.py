from pathlib import Path

from evo_amr.cli import build_parser
from evo_amr.config import ExperimentConfig
from evo_amr.config import load_yaml
from evo_amr.workflows.plans import build_stage_plan
from evo_amr.workflows import run_tiny_workflow


def test_example_experiment_config_loads():
    config = load_yaml(Path("examples/example_experiment.yaml"))

    assert config["experiment"]["name"] == "tiny_demo_evo_minirocket"
    assert config["dataset"]["manifest"] == "examples/tiny_manifest.csv"


def test_cli_parses_dry_run_command():
    parser = build_parser()
    args = parser.parse_args(
        ["train", "--config", "examples/example_experiment.yaml", "--profile", "local"]
    )

    assert args.command == "train"
    assert args.config == Path("examples/example_experiment.yaml")
    assert args.profile == "local"
    assert args.execute is False


def test_cli_parses_baseline_inventory_command():
    parser = build_parser()
    args = parser.parse_args(["list-baselines", "--family", "curated_gene_baseline"])

    assert args.command == "list-baselines"
    assert args.family == "curated_gene_baseline"


def test_cli_parses_model_inventory_command():
    parser = build_parser()
    args = parser.parse_args(["list-models", "--task", "multi_drug"])

    assert args.command == "list-models"
    assert args.task == "multi_drug"


def test_cli_parses_pipeline_inventory_command():
    parser = build_parser()
    args = parser.parse_args(["list-pipelines", "--status", "SCAFFOLD_READY"])

    assert args.command == "list-pipelines"
    assert args.status == "SCAFFOLD_READY"


def test_tiny_workflow_renders_report():
    report = run_tiny_workflow("examples/example_experiment.yaml")

    assert "# tiny_demo_evo_minirocket" in report
    assert "| train | 2 | 1 |" in report
    assert "accuracy:" in report
    assert "mcc:" in report


def test_stage_plan_includes_inputs_outputs_and_notes():
    config = ExperimentConfig.from_yaml(
        "configs/experiments/evo_minirocket_ampicillin.example.yaml"
    )
    rendered = build_stage_plan("train", config).render()

    assert "[evo-amr] inputs:" in rendered
    assert "manifest" in rendered
    assert "feature_methods=mean_pooling, pca, sparse_random_projection, minirocket" in rendered
