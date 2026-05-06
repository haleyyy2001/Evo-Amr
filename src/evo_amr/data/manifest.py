"""Manifest loading and validation."""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "genome_id",
    "species",
    "antibiotic",
    "phenotype",
    "partition_label",
}

OPTIONAL_COLUMNS = {
    "sequence_path",
    "embedding_path",
    "source_dataset",
    "split_design",
}


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV manifest and validate the shared Evo-AMR columns."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        return list(reader)


def manifest_schema_status(columns: set[str]) -> dict[str, list[str]]:
    """Summarize required and optional manifest columns."""
    return {
        "required_present": sorted(REQUIRED_COLUMNS & columns),
        "required_missing": sorted(REQUIRED_COLUMNS - columns),
        "optional_present": sorted(OPTIONAL_COLUMNS & columns),
        "optional_missing": sorted(OPTIONAL_COLUMNS - columns),
    }


def validate_manifest_rows(rows: list[dict[str, str]]) -> None:
    """Validate basic manifest row values."""
    for idx, row in enumerate(rows, start=1):
        phenotype = row.get("phenotype")
        if phenotype not in {"0", "1", 0, 1}:
            raise ValueError(f"Row {idx} has non-binary phenotype: {phenotype!r}")
        if not row.get("genome_id"):
            raise ValueError(f"Row {idx} is missing genome_id")
        if not row.get("species"):
            raise ValueError(f"Row {idx} is missing species")
