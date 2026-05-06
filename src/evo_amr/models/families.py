"""Model family descriptors for migrated AMR prediction architectures.

The old amr_pred package contains concrete PyTorch/Lightning implementations.
This module records the supported architecture families as configuration-level
objects first, so experiments can be documented before heavyweight model code is
migrated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelFamily:
    """Inspectable description of a trainable AMR model family."""

    name: str
    task: str
    representation: str
    output_mode: str
    legacy_source: str
    notes: str = ""


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "single_drug_lr": ModelFamily(
        name="single_drug_lr",
        task="single_drug",
        representation="precomputed_embedding",
        output_mode="binary_logit",
        legacy_source="prediction/lib/model/single_drug_model.py::PrecompSingleDrugLR",
        notes="Linear probe over precomputed genome/proteome representation.",
    ),
    "single_drug_mlp": ModelFamily(
        name="single_drug_mlp",
        task="single_drug",
        representation="precomputed_embedding",
        output_mode="binary_logit",
        legacy_source="prediction/lib/model/single_drug_model.py::PrecompSingleDrugMLP",
        notes="Configurable MLP probe.",
    ),
    "bag_of_proteins": ModelFamily(
        name="bag_of_proteins",
        task="single_drug",
        representation="protein_embedding_set",
        output_mode="binary_logit",
        legacy_source="prediction/lib/model/single_drug_model.py::SimpleSingleDrugBOPModel",
        notes="Mean/sum reduction over protein embeddings before classification.",
    ),
    "attention_bag_of_proteins": ModelFamily(
        name="attention_bag_of_proteins",
        task="single_drug",
        representation="protein_embedding_set",
        output_mode="binary_logit",
        legacy_source="prediction/lib/model/single_drug_model.py::MHAModel",
        notes="Transformer-style encoder over protein embeddings.",
    ),
    "multi_drug_mlp": ModelFamily(
        name="multi_drug_mlp",
        task="multi_drug",
        representation="precomputed_embedding",
        output_mode="per_drug_binary_logits",
        legacy_source="prediction/lib/model/multi_drug_model.py::PrecompMultiDrugMLP",
        notes="Shared trunk with antibiotic-specific prediction heads.",
    ),
}


def get_model_family(name: str) -> ModelFamily:
    """Return a registered model family by name."""
    try:
        return MODEL_FAMILIES[name]
    except KeyError as exc:
        raise KeyError(f"unknown model family: {name}") from exc


def list_model_families(task: str | None = None) -> tuple[str, ...]:
    """Return registered model family names, optionally filtered by task."""
    names = [
        name
        for name, family in MODEL_FAMILIES.items()
        if task is None or family.task == task
    ]
    return tuple(sorted(names))
