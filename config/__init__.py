"""
Configuration Module

Centralized configuration management for the Evo-AMR pipeline.
Handles path resolution, environment setup, and parameter management.

Main components:
- config_manager.py: Main configuration manager class
- paths.yaml: Path configuration file
- environment.yaml: Environment settings
"""

from .config_manager import ConfigManager, get_config, setup_paths_from_config

__all__ = ["ConfigManager", "get_config", "setup_paths_from_config"]