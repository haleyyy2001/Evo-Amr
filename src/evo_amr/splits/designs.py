"""Split design descriptors for random, clustered, and species-holdout studies.

These objects are clean counterparts to AMR_benchmarking's KMA/phylogeny folds
and amr_pred's random and clustered split scripts. They intentionally describe
the design without launching legacy code or mutating data.
"""

from __future__ import annotations

from dataclasses import dataclass


CLUSTER_LEVELS = ("genus", "species", "taxa", "genome_id", "kma_cluster", "phylogeny")


@dataclass(frozen=True)
class RandomSplitConfig:
    """Random train/validation/test split proportions."""

    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0

    def validate(self) -> None:
        """Validate fractions before use."""
        if self.val_fraction < 0 or self.test_fraction < 0:
            raise ValueError("split fractions must be non-negative")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("validation + test fractions must be less than 1")


@dataclass(frozen=True)
class ClusteredSplitConfig:
    """Cluster-aware split design used to reduce taxonomic leakage."""

    level: str = "genus"
    val_fraction: float = 0.2
    test_fraction: float = 0.15
    seed: int = 0
    min_cluster_size: int = 2
    max_cluster_size: int | None = None

    def validate(self) -> None:
        """Validate cluster split settings."""
        if self.level not in CLUSTER_LEVELS:
            raise ValueError(f"unsupported cluster level: {self.level}")
        RandomSplitConfig(self.val_fraction, self.test_fraction, self.seed).validate()
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be positive")
        if self.max_cluster_size is not None and self.max_cluster_size < self.min_cluster_size:
            raise ValueError("max_cluster_size must be >= min_cluster_size")


@dataclass(frozen=True)
class SpeciesHoldoutConfig:
    """Evaluation design that holds out entire species from training."""

    holdout_species: tuple[str, ...]
    validation_species: tuple[str, ...] = ()

    def validate(self) -> None:
        """Validate species-holdout setup."""
        if not self.holdout_species:
            raise ValueError("holdout_species must not be empty")
        overlap = set(self.holdout_species) & set(self.validation_species)
        if overlap:
            raise ValueError(f"species cannot be in both validation and test holdout: {sorted(overlap)}")


@dataclass(frozen=True)
class SplitDesign:
    """Named split design for config files and dry-run plans."""

    name: str
    strategy: str
    random: RandomSplitConfig | None = None
    clustered: ClusteredSplitConfig | None = None
    species_holdout: SpeciesHoldoutConfig | None = None
    legacy_source: str | None = None

    def validate(self) -> None:
        """Validate the active strategy config."""
        strategies = {
            "random": self.random,
            "clustered": self.clustered,
            "species_holdout": self.species_holdout,
        }
        if self.strategy not in strategies:
            raise ValueError(f"unsupported split strategy: {self.strategy}")
        active = strategies[self.strategy]
        if active is None:
            raise ValueError(f"missing config for split strategy: {self.strategy}")
        active.validate()

    def describe(self) -> str:
        """Render a compact description for reports."""
        source = f", source={self.legacy_source}" if self.legacy_source else ""
        return f"SplitDesign(name={self.name}, strategy={self.strategy}{source})"


def default_amr_pred_clustered_design(level: str = "genus", seed: int = 0) -> SplitDesign:
    """Return defaults matching the amr_pred clustered split scripts."""
    proportions = {
        "genus": (0.2, 0.15),
        "taxa": (0.17, 0.13),
        "genome_id": (0.1, 0.1),
    }
    val_fraction, test_fraction = proportions.get(level, (0.2, 0.15))
    return SplitDesign(
        name=f"clustered_{level}",
        strategy="clustered",
        clustered=ClusteredSplitConfig(
            level=level,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        ),
        legacy_source="prediction/data_prep/data_splits/clustered_splits.py",
    )
