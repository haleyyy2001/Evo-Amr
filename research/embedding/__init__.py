"""
Embedding Pipeline Module

This module provides functionality for generating embeddings from genomic sequences
using evolutionary models. It includes utilities for sequence tokenization,
model inference, and embedding extraction.

Main components:
- embedding_generator.py: Main script for generating embeddings
- Configuration management for different model types
- Batch processing utilities for large datasets
"""

from .embedding_generator import *

__all__ = ["embedding_generator"]