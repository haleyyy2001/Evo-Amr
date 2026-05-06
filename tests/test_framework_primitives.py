from pathlib import Path

import pytest

from evo_amr.baselines import ShellBaselineAdapter
from evo_amr.config import ExperimentConfig
from evo_amr.data import load_manifest, manifest_schema_status, validate_manifest_rows
from evo_amr.evaluation import binary_classification_summary
from evo_amr.features import mean_pool
from evo_amr.models import MajorityClassifier
from evo_amr.splits import outside_species_leakage, summarize_partitions


def test_experiment_config_supports_experiment_yaml():
    config = ExperimentConfig.from_yaml("examples/example_experiment.yaml")

    assert config.name == "tiny_demo_evo_minirocket"
    assert config.task == "binary_amr_prediction"
    assert config.dataset.manifest == "examples/tiny_manifest.csv"
    assert config.features.methods == ("mean_pooling", "minirocket")
    assert config.model.classifiers == ("logistic_regression", "knn_cosine")


def test_experiment_config_supports_baseline_yaml():
    config = ExperimentConfig.from_yaml("configs/baselines/kover.example.yaml")

    assert config.name == "kover"
    assert config.task == "baseline"
    assert config.evaluation == "species_holdout"


def test_manifest_split_summary_and_leakage_check():
    rows = load_manifest("examples/tiny_manifest.csv")

    validate_manifest_rows(rows)
    assert summarize_partitions(rows)["train"]["n_genomes"] == 2
    assert outside_species_leakage(rows) == {}


def test_manifest_schema_status_reports_missing_optional_columns():
    status = manifest_schema_status({"genome_id", "species", "phenotype"})

    assert "antibiotic" in status["required_missing"]
    assert "sequence_path" in status["optional_missing"]


def test_validate_manifest_rows_rejects_non_binary_phenotype():
    with pytest.raises(ValueError, match="non-binary phenotype"):
        validate_manifest_rows(
            [
                {
                    "genome_id": "G001",
                    "species": "Escherichia_coli",
                    "phenotype": "maybe",
                    "partition_label": "train",
                    "antibiotic": "ampicillin",
                }
            ]
        )


def test_leakage_check_detects_train_species_in_outside_partition():
    rows = [
        {"partition_label": "train", "species": "Escherichia_coli"},
        {"partition_label": "val_outside", "species": "Escherichia_coli"},
    ]

    assert outside_species_leakage(rows) == {"val_outside": {"Escherichia_coli"}}


def test_majority_classifier_and_binary_metrics():
    y_true = [1, 1, 0, 1]
    y_pred = MajorityClassifier().fit(y_true).predict(len(y_true))
    metrics = binary_classification_summary(y_true, y_pred)

    assert y_pred == [1, 1, 1, 1]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(1.0)


def test_mean_pool_validates_shapes():
    assert mean_pool([[1.0, 3.0], [3.0, 5.0]]) == [2.0, 4.0]

    with pytest.raises(ValueError):
        mean_pool([[1.0], [1.0, 2.0]])


def test_shell_baseline_adapter_renders_command():
    adapter = ShellBaselineAdapter(
        name="kover",
        script=Path("scripts/model/kover.sh"),
        backend_root=Path("external/AMR_benchmarking"),
    )

    command = adapter.dry_run_command("configs/baselines/kover.example.yaml")
    assert "kover.sh" in command.render()
    assert "external/AMR_benchmarking" in command.render()
