import pytest

from evo_amr.config import ExperimentConfig
from evo_amr.diagnostics import (
    RECOMMENDED_EVO_LAYER,
    STABILITY_BOUNDARY_LAYER,
    LayerDiagnosticConfig,
    validate_layer_choice,
)
from evo_amr.evaluation import (
    CASSETTE_MEDIATED,
    CHROMOSOMAL_DIFFUSE,
    MIXED_OR_UNKNOWN,
    ResultRecord,
    recommend_aggregation,
)


def test_layer_diagnostic_constants_match_thesis():
    assert RECOMMENDED_EVO_LAYER == 10
    assert STABILITY_BOUNDARY_LAYER == 11

    ok, message = validate_layer_choice(10)
    assert ok is True
    assert "deepest stable" in message

    ok, message = validate_layer_choice(11)
    assert ok is False
    assert "stability boundary" in message


def test_layer_diagnostic_config_lists_thesis_metrics():
    config = LayerDiagnosticConfig()

    assert "isotropy" in config.metrics
    assert "effective_rank" in config.metrics
    assert "cross_seed_stability" in config.metrics


@pytest.mark.parametrize(
    ("mechanism", "expected"),
    [
        (CASSETTE_MEDIATED, "minirocket"),
        (CHROMOSOMAL_DIFFUSE, "global_pooling"),
        (MIXED_OR_UNKNOWN, "ensemble_or_compare"),
    ],
)
def test_mechanism_regime_recommends_expected_aggregation(mechanism, expected):
    recommendation = recommend_aggregation(mechanism)

    assert recommendation.preferred_aggregation == expected
    assert recommendation.rationale


def test_result_record_serializes_thesis_metric_row():
    record = ResultRecord(
        experiment="evo_l10_minirocket_ampicillin_species_holdout",
        antibiotic="ampicillin",
        partition="val_outside",
        representation="evo_l10",
        aggregation="minirocket",
        model="knn_cosine",
        metric="mcc",
        value=0.753,
        split_replicate="v1-1",
    )

    row = record.as_dict()
    assert row["metric"] == "mcc"
    assert row["value"] == 0.753
    assert row["partition"] == "val_outside"


def test_thesis_specialized_configs_parse():
    diagnostic = ExperimentConfig.from_yaml(
        "configs/experiments/evo_layer_diagnostics.example.yaml"
    )
    kover = ExperimentConfig.from_yaml(
        "configs/experiments/kover_species_holdout_six_drugs.example.yaml"
    )
    neighbor = ExperimentConfig.from_yaml(
        "configs/experiments/mechanism_neighbor_audit.example.yaml"
    )

    assert diagnostic.task == "representation_diagnostics"
    assert kover.evaluation == "species_holdout"
    assert neighbor.task == "mechanism_analysis"
