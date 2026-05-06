"""Registry of legacy AMR baseline backends.

AMR_benchmarking includes several mature external methods. The showcase
framework treats them as backend capabilities that can be wrapped one by one
instead of importing their full source tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineBackend:
    """Metadata for a legacy baseline backend."""

    name: str
    family: str
    script: Path
    supports_multi_species: bool = False
    supports_multi_drug: bool = False
    action: str = "WRAP"
    notes: str = ""


DEFAULT_BASELINE_BACKENDS: dict[str, BaselineBackend] = {
    "majority": BaselineBackend(
        name="majority",
        family="sanity_check",
        script=Path("benchmarking/scripts/model/majority.sh"),
        supports_multi_species=True,
        supports_multi_drug=True,
        action="MIGRATE",
        notes="Useful as an in-package baseline; already mirrored by MajorityClassifier.",
    ),
    "kover": BaselineBackend(
        name="kover",
        family="kmer_rule_learning",
        script=Path("benchmarking/scripts/model/kover.sh"),
        notes="External rule-learning backend; wrap command and parse reports.",
    ),
    "kover_ms": BaselineBackend(
        name="kover_ms",
        family="kmer_rule_learning",
        script=Path("benchmarking/scripts/model/kover_MS.sh"),
        supports_multi_species=True,
        notes="Multi-species Kover workflow from AMR_benchmarking.",
    ),
    "phenotypeseeker": BaselineBackend(
        name="phenotypeseeker",
        family="kmer_statistical_baseline",
        script=Path("benchmarking/scripts/model/phenotypeseeker.sh"),
        notes="External k-mer association baseline.",
    ),
    "phenotypeseeker_ms": BaselineBackend(
        name="phenotypeseeker_ms",
        family="kmer_statistical_baseline",
        script=Path("benchmarking/scripts/model/phenotypeseeker_MS.sh"),
        supports_multi_species=True,
        notes="Multi-species PhenotypeSeeker workflow.",
    ),
    "resfinder": BaselineBackend(
        name="resfinder",
        family="curated_gene_baseline",
        script=Path("benchmarking/scripts/model/resfinder.sh"),
        notes="Curated gene/point-mutation baseline; preserve as external backend.",
    ),
    "seq2geno": BaselineBackend(
        name="seq2geno",
        family="genotype_to_phenotype_pipeline",
        script=Path("benchmarking/scripts/model/seq2geno.sh"),
        action="VENDOR",
        notes="Large third-party style pipeline; keep as external reference.",
    ),
    "aytanaktug_sssa": BaselineBackend(
        name="aytanaktug_sssa",
        family="neural_baseline",
        script=Path("benchmarking/scripts/model/AytanAktug_SSSA.sh"),
        notes="Single species single antibiotic neural baseline.",
    ),
    "aytanaktug_ssma": BaselineBackend(
        name="aytanaktug_ssma",
        family="neural_baseline",
        script=Path("benchmarking/scripts/model/AytanAktug_SSMA.sh"),
        supports_multi_drug=True,
        notes="Single species multi-antibiotic neural baseline.",
    ),
    "aytanaktug_msma_discrete": BaselineBackend(
        name="aytanaktug_msma_discrete",
        family="neural_baseline",
        script=Path("benchmarking/scripts/model/AytanAktug_MSMA_discrete.sh"),
        supports_multi_species=True,
        supports_multi_drug=True,
        notes="Multi-species multi-antibiotic discrete-output neural baseline.",
    ),
    "aytanaktug_msma_concat": BaselineBackend(
        name="aytanaktug_msma_concat",
        family="neural_baseline",
        script=Path("benchmarking/scripts/model/AytanAktug_MSMA_concat.sh"),
        supports_multi_species=True,
        supports_multi_drug=True,
        notes="Multi-species multi-antibiotic concatenated-input neural baseline.",
    ),
}


def get_baseline_backend(name: str) -> BaselineBackend:
    """Return a registered backend by name."""
    try:
        return DEFAULT_BASELINE_BACKENDS[name]
    except KeyError as exc:
        raise KeyError(f"unknown baseline backend: {name}") from exc


def list_baseline_backends() -> tuple[str, ...]:
    """Return registered backend names in stable order."""
    return tuple(sorted(DEFAULT_BASELINE_BACKENDS))
