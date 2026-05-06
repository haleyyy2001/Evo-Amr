"""Registry of FeatureTransform → Model source scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelScript:
    """A legacy model training or inference script."""

    name: str
    source: Path
    framework: str
    task: str
    action: str
    notes: str = ""


MODEL_SCRIPTS: dict[str, ModelScript] = {
    "amr_pred_train": ModelScript(
        name="amr_pred_train",
        source=Path("prediction/train.py"),
        framework="pytorch_lightning",
        task="single_drug / multi_drug",
        action="MIGRATE",
        notes=(
            "Authoritative PyTorch Lightning training entry point. "
            "Configures W&B, trainer, callbacks, initialises BOPLitModule. "
            "Migrate into src/evo_amr/models/."
        ),
    ),
    "amr_pred_lightning_modules": ModelScript(
        name="amr_pred_lightning_modules",
        source=Path("prediction/lib/lightning_modules.py"),
        framework="pytorch_lightning",
        task="single_drug / multi_drug",
        action="MIGRATE",
        notes="BOPLitModule / SingleDrugBOPLitModule; forward, loss, metrics, logging.",
    ),
    "amr_pred_modules": ModelScript(
        name="amr_pred_modules",
        source=Path("prediction/lib/modules.py"),
        framework="pytorch",
        task="single_drug / multi_drug",
        action="MIGRATE",
        notes="SimpleMLPTrunk, ResidualBlock; configurable from layer spec list.",
    ),
    "amr_pred_loss": ModelScript(
        name="amr_pred_loss",
        source=Path("prediction/lib/loss.py"),
        framework="pytorch",
        task="multi_drug",
        action="MIGRATE",
        notes="MultiDrugBCEWithLogitsLoss with NaN masking and cluster weighting.",
    ),
    "amr_pred_simple_model": ModelScript(
        name="amr_pred_simple_model",
        source=Path("prediction/simple_model/simple_model.py"),
        framework="sklearn",
        task="single_drug",
        action="MIGRATE",
        notes="Logistic regression baseline over ESM embeddings; generalise to Evo.",
    ),
    "amr_pred_cv_opt": ModelScript(
        name="amr_pred_cv_opt",
        source=Path("prediction/simple_model/cv_opt.py"),
        framework="sklearn",
        task="single_drug",
        action="MIGRATE",
        notes="10-fold CV hyperparameter search; make configurable during migration.",
    ),
    "amr_pred_sweep": ModelScript(
        name="amr_pred_sweep",
        source=Path("prediction/sweep.py"),
        framework="wandb",
        task="multi_drug",
        action="ADAPT",
        notes="W&B Bayes sweep; migrate config to YAML in configs/experiments/.",
    ),
    "minirocket_classifier_grid": ModelScript(
        name="minirocket_classifier_grid",
        source=Path("minirocket/minirocket_pipeline/amr_classification_pipeline.py"),
        framework="sklearn",
        task="single_drug",
        action="MIGRATE",
        notes=(
            "GridSearchCV over 10+ classifiers (LR, SVM, RF, MLP, KNN, etc.). "
            "Migrate classifier grid into models/; leave MiniRocket in features/."
        ),
    ),
}


def get_model_script(name: str) -> ModelScript:
    try:
        return MODEL_SCRIPTS[name]
    except KeyError as exc:
        raise KeyError(f"unknown model script: {name}") from exc


def list_model_scripts(framework: str | None = None, task: str | None = None) -> tuple[str, ...]:
    return tuple(
        name
        for name, s in sorted(MODEL_SCRIPTS.items())
        if (framework is None or s.framework == framework)
        and (task is None or task in s.task)
    )
