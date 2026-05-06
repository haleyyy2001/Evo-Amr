"""Trainable AMR prediction model interfaces."""

from .families import ModelFamily, get_model_family, list_model_families
from .majority import MajorityClassifier

__all__ = [
    "MajorityClassifier",
    "ModelFamily",
    "get_model_family",
    "list_model_families",
]
