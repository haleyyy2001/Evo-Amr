"""Tests for the embedding adapter and visualization spec helpers.

These tests require no GPU, no model weights, and no private data.
They validate the dry-run interface only.
"""

from pathlib import Path

from evo_amr.embeddings import EvoEmbeddingAdapter, EvoEmbeddingConfig
from evo_amr.visualization import (
    PlotSpec,
    method_comparison_spec,
    neighbour_audit_spec,
    phylogenetic_curve_spec,
    species_heatmap_spec,
)


# ---------------------------------------------------------------------------
# EvoEmbeddingConfig
# ---------------------------------------------------------------------------


def test_evo_embedding_config_defaults():
    config = EvoEmbeddingConfig()

    assert config.extraction_layer == 10
    assert config.dtype == "bfloat16"
    assert config.window_size == 8192
    assert config.model == "evo-1-8k-base"


def test_evo_embedding_config_from_experiment_yaml():
    raw = {
        "representation": {
            "model": "evo-1-8k-base",
            "extraction_layer": 10,
            "dtype": "bfloat16",
            "window_size": 8192,
            "output": "embeddings/evo_l10/ampicillin",
        }
    }
    config = EvoEmbeddingConfig.from_experiment_yaml(raw)

    assert config.extraction_layer == 10
    assert config.output_dir == Path("embeddings/evo_l10/ampicillin")


def test_evo_embedding_config_from_example_yaml():
    from evo_amr.config import load_yaml

    raw = load_yaml("configs/experiments/evo_minirocket_ampicillin.example.yaml")
    config = EvoEmbeddingConfig.from_experiment_yaml(raw)

    assert config.model == "evo-1-8k-base"
    assert config.extraction_layer == 10
    assert config.dtype == "bfloat16"


# ---------------------------------------------------------------------------
# EvoEmbeddingAdapter
# ---------------------------------------------------------------------------


def test_evo_adapter_dry_run_command_contains_layer():
    config = EvoEmbeddingConfig(extraction_layer=10)
    adapter = EvoEmbeddingAdapter(config)

    cmd = adapter.dry_run_command("examples/tiny_manifest.csv")

    assert "--layer 10" in cmd
    assert "embedding_pipeline/embedding_generator.py" in cmd
    assert "examples/tiny_manifest.csv" in cmd


def test_evo_adapter_custom_backend_script():
    config = EvoEmbeddingConfig()
    adapter = EvoEmbeddingAdapter(config, backend_script=Path("scripts/embed.py"))

    cmd = adapter.dry_run_command("manifest.csv")

    assert "scripts/embed.py" in cmd


def test_evo_adapter_describe_mentions_extract_method():
    config = EvoEmbeddingConfig()
    adapter = EvoEmbeddingAdapter(config)

    desc = adapter.describe()

    assert "extract_layer_hidden_states" in desc
    assert "target_layer=10" in desc
    assert "bfloat16" in desc


# ---------------------------------------------------------------------------
# PlotSpec
# ---------------------------------------------------------------------------


def test_plot_spec_describe_contains_kind_and_output():
    spec = PlotSpec(
        kind="method_comparison",
        title="Test figure",
        x_label="classifier",
        y_label="aggregation",
        output_path="results/fig.png",
        note="test note",
    )
    desc = spec.describe()

    assert "[plot] kind=method_comparison" in desc
    assert "[plot] output=results/fig.png" in desc
    assert "[plot] note=test note" in desc


def test_neighbour_audit_spec():
    spec = neighbour_audit_spec("G001", "minirocket", "results/neighbours.png", k=5)

    assert spec.kind == "neighbour_audit"
    assert "G001" in spec.title
    assert "k=5" in spec.note


def test_phylogenetic_curve_spec():
    spec = phylogenetic_curve_spec("ampicillin", "auroc", "results/phylo.png")

    assert spec.kind == "phylogenetic_curve"
    assert "AUROC" in spec.y_label
    assert "ampicillin" in spec.title


def test_method_comparison_spec():
    spec = method_comparison_spec("ampicillin", "val_outside", "mcc", "results/heatmap.png")

    assert spec.kind == "method_comparison"
    assert "MCC" in spec.note


def test_species_heatmap_spec():
    spec = species_heatmap_spec("ampicillin", "test_outside", "mcc", "results/species.png")

    assert spec.kind == "species_heatmap"
    assert "MiniRocket" in spec.note
