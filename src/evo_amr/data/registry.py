"""Registry of Dataset → Manifest source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataSourceScript:
    """A legacy data loading or preprocessing script."""

    name: str
    source: Path
    action: str
    output: str
    notes: str = ""


DATA_SOURCES: dict[str, DataSourceScript] = {
    "bvbrc_metadata": DataSourceScript(
        name="bvbrc_metadata",
        source=Path("prediction/data_prep/utils.py"),
        action="MIGRATE",
        output="manifest rows with GTDB taxonomy",
        notes="Authoritative GTDB-Tk taxonomy parsing; edit_species normalisation.",
    ),
    "amr_benchmarking_loader": DataSourceScript(
        name="amr_benchmarking_loader",
        source=Path("benchmarking/src/amr_utility/load_data.py"),
        action="MIGRATE",
        output="genome ID / phenotype / antibiotic lists per species",
        notes="Primary loader for AMR_benchmarking system; includes balance checks.",
    ),
    "amr_benchmarking_preprocess": DataSourceScript(
        name="amr_benchmarking_preprocess",
        source=Path("benchmarking/src/data_preprocess/preprocess.py"),
        action="MIGRATE",
        output="cleaned metadata TSV",
        notes="Filters by species/antibiotic availability; normalises phenotype labels.",
    ),
    "extract_balanced_subsets": DataSourceScript(
        name="extract_balanced_subsets",
        source=Path("data_preprocessing/extract_balanced_species_subsets.py"),
        action="MIGRATE",
        output="balanced subset CSV with dedup registry",
        notes="Prioritises sensitive samples in val_outside; tracks used genomes across rounds.",
    ),
    "extract_proper_subsets": DataSourceScript(
        name="extract_proper_subsets",
        source=Path("data_preprocessing/extract_proper_species_subsets.py"),
        action="MIGRATE",
        output="natural-proportion subset CSV",
        notes="Earlier version of balanced extraction; no registry.",
    ),
    "create_balanced_subsets": DataSourceScript(
        name="create_balanced_subsets",
        source=Path("data_preprocessing/create_balanced_subsets.py"),
        action="MIGRATE",
        output="fixed-size R/S balanced subset CSV",
    ),
    "extract_gentamicin_subsets": DataSourceScript(
        name="extract_gentamicin_subsets",
        source=Path("data_preprocessing/extract_species_subsets_gentamicin.py"),
        action="MIGRATE",
        output="gentamicin-specific subset CSV",
        notes="Generalise the drug-specific parameter during migration.",
    ),
    "compare_distributions": DataSourceScript(
        name="compare_distributions",
        source=Path("data_preprocessing/compare_distributions.py"),
        action="MIGRATE",
        output="R/S distribution comparison report",
        notes="Data-quality diagnostic; migrate as a report helper.",
    ),
    "dataset_analyzer": DataSourceScript(
        name="dataset_analyzer",
        source=Path("scripts/dataset_analyzer.py"),
        action="MIGRATE",
        output="partition-wise species/R-S statistics log",
        notes="Professional dataset statistics tool; stable pandas logic.",
    ),
}


def get_data_source(name: str) -> DataSourceScript:
    try:
        return DATA_SOURCES[name]
    except KeyError as exc:
        raise KeyError(f"unknown data source: {name}") from exc


def list_data_sources(action: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, src in sorted(DATA_SOURCES.items())
        if action is None or src.action == action
    )
