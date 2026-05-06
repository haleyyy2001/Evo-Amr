"""
Models Module

This module contains evolutionary model implementations and utilities.

Components:
- evo_1_131k_base: Classic Evo model architecture and utilities
- evo-1-8k-base: Pre-trained model weights
- evo_8: Modified Evo model with enhanced diagnostic capabilities

The models support:
- Genomic sequence processing
- Hidden state extraction
- Attention and Hyena-based architectures
- Configurable precision (fp16, bf16, fp32)
"""

__all__ = ["evo_1_131k_base", "evo_8"]