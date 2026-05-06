import pytest

from evo_amr.baselines import get_baseline_backend, list_baseline_backends
from evo_amr.config import PathProfile
from evo_amr.data import (
    DatasetFilterConfig,
    eligible_species_antibiotic_tasks,
    normalize_binary_phenotype,
)
from evo_amr.splits import (
    ClusteredSplitConfig,
    SpeciesHoldoutConfig,
    SplitDesign,
    default_amr_pred_clustered_design,
)
from evo_amr.workflows import SlurmConfig, render_sbatch_dry_run


def test_dataset_filter_config_counts_species_antibiotic_tasks():
    rows = [
        {"species": "Escherichia coli", "antibiotic": "ampicillin"},
        {"species": "Escherichia coli", "antibiotic": "ampicillin"},
        {"species": "Escherichia coli", "antibiotic": "ciprofloxacin"},
    ]

    eligible = eligible_species_antibiotic_tasks(
        rows, DatasetFilterConfig(min_genomes_per_species_antibiotic=2)
    )

    assert eligible == {("Escherichia coli", "ampicillin"): 2}


def test_normalize_binary_phenotype_handles_common_amr_labels():
    assert normalize_binary_phenotype("Resistant") == 1
    assert normalize_binary_phenotype("S") == 0
    assert normalize_binary_phenotype("intermediate") is None


def test_clustered_split_defaults_match_amr_pred_proportions():
    design = default_amr_pred_clustered_design("taxa", seed=7)

    design.validate()
    assert design.clustered is not None
    assert design.clustered.val_fraction == pytest.approx(0.17)
    assert design.clustered.test_fraction == pytest.approx(0.13)
    assert "prediction" in design.describe()


def test_split_design_validates_species_holdout_overlap():
    design = SplitDesign(
        name="bad_holdout",
        strategy="species_holdout",
        species_holdout=SpeciesHoldoutConfig(
            holdout_species=("Klebsiella pneumoniae",),
            validation_species=("Klebsiella pneumoniae",),
        ),
    )

    with pytest.raises(ValueError, match="both validation and test"):
        design.validate()


def test_clustered_split_rejects_unknown_level():
    with pytest.raises(ValueError, match="unsupported cluster level"):
        ClusteredSplitConfig(level="unknown").validate()


def test_path_profile_resolves_template_without_filesystem_dependency():
    profile = PathProfile(
        name="server",
        roots={"data_root": "/restricted/data", "run_root": "/scratch/runs"},
    )

    assert profile.resolve("$data_root/manifests/train.csv") == "/restricted/data/manifests/train.csv"
    assert "run_root=/scratch/runs" in profile.describe()


def test_baseline_registry_exposes_amr_benchmarking_backends():
    names = list_baseline_backends()
    backend = get_baseline_backend("resfinder")

    assert "kover_ms" in names
    assert "aytanaktug_msma_concat" in names
    assert backend.family == "curated_gene_baseline"
    assert backend.action == "WRAP"


def test_slurm_dry_run_renders_sbatch_script():
    command = render_sbatch_dry_run(
        SlurmConfig(job_name="evo-amr-train", partition="gpu", gres="gpu:1"),
        "evo-amr train --config configs/experiments/demo.yaml",
    )

    script = command.script()
    assert "#SBATCH --job-name=evo-amr-train" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "set -euo pipefail" in script
    assert "evo-amr train" in script
