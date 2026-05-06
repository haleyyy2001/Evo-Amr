"""Adapter stubs for Dataset → Manifest source scripts.

Each class wraps one legacy source. The `load` method raises NotImplementedError
until the migration for that source is complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BVBRCMetadataAdapter:
    """Adapter for prediction/data_prep/utils.py — BV-BRC GTDB metadata."""

    metadata_dir: Path
    dataset: str

    def load(self) -> list[dict[str, str]]:
        raise NotImplementedError(
            "Migrate prediction/data_prep/utils.py::create_and_save_gtdbtk_summary_df "
            "and load_metadata into src/evo_amr/data/."
        )

    def describe(self) -> str:
        return f"BVBRCMetadataAdapter(dataset={self.dataset}, dir={self.metadata_dir})"


@dataclass(frozen=True)
class AMRBenchmarkingLoaderAdapter:
    """Adapter for benchmarking/src/amr_utility/load_data.py."""

    metadata_path: Path
    species: str
    antibiotic: str

    def load(self) -> list[dict[str, str]]:
        raise NotImplementedError(
            "Migrate benchmarking/src/amr_utility/load_data.py::extract_info "
            "into src/evo_amr/data/."
        )

    def describe(self) -> str:
        return (
            f"AMRBenchmarkingLoaderAdapter("
            f"species={self.species}, antibiotic={self.antibiotic})"
        )


@dataclass(frozen=True)
class BalancedSubsetAdapter:
    """Adapter for data_preprocessing/extract_balanced_species_subsets.py."""

    metadata_path: Path
    kover_csv_path: Path
    output_dir: Path
    antibiotic: str | None = None

    def load(self) -> list[dict[str, str]]:
        raise NotImplementedError(
            "Migrate data_preprocessing/extract_balanced_species_subsets.py "
            "into src/evo_amr/data/."
        )

    def describe(self) -> str:
        return (
            f"BalancedSubsetAdapter("
            f"antibiotic={self.antibiotic or 'all'}, output={self.output_dir})"
        )
