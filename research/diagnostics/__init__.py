"""
Diagnostics Module

This module provides comprehensive diagnostic tools for analyzing evolutionary model
behavior, including numerical stability analysis, activation pattern detection,
and statistical robustness testing.

Main components:
- run_diagnostics.sbatch: Portable SLURM script for multi-seed analysis
- diagnostic_analysis.sbatch: Original diagnostic script
- Statistical analysis utilities
- Visualization tools for diagnostic results
"""

__all__ = ["run_diagnostics", "diagnostic_analysis"]