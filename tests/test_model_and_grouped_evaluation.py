import pytest

from evo_amr.evaluation import grouped_binary_classification_summary, inverse_frequency_weights
from evo_amr.models import get_model_family, list_model_families


def test_model_family_registry_documents_amr_pred_architectures():
    names = list_model_families(task="single_drug")
    family = get_model_family("multi_drug_mlp")

    assert "bag_of_proteins" in names
    assert family.task == "multi_drug"
    assert family.output_mode == "per_drug_binary_logits"
    assert "prediction/lib/model/multi_drug_model.py" in family.legacy_source


def test_model_family_registry_rejects_unknown_model():
    with pytest.raises(KeyError, match="unknown model family"):
        get_model_family("not_a_model")


def test_grouped_binary_summary_computes_species_metrics():
    rows = [
        {"species": "A", "phenotype": 1, "prediction": 1},
        {"species": "A", "phenotype": 0, "prediction": 1},
        {"species": "B", "phenotype": 0, "prediction": 0},
    ]

    metrics = grouped_binary_classification_summary(rows, group_key="species")

    assert metrics["A"]["accuracy"] == pytest.approx(0.5)
    assert metrics["B"]["accuracy"] == pytest.approx(1.0)


def test_inverse_frequency_weights_give_equal_group_mass():
    weights = inverse_frequency_weights(["A", "A", "B"])

    assert weights == pytest.approx([0.25, 0.25, 0.5])
