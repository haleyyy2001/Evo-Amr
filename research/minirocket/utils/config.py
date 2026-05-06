#!/usr/bin/env python3
"""
Configuration management for MiniRocket AMR Pipeline
====================================================

This module provides centralized configuration management for all pipeline components,
eliminating hardcoded paths and making the pipeline portable across different environments.

Usage:
    from config import get_config
    
    config = get_config()
    print(config.data_dir)
    print(config.output_dir)

Environment Variables:
    AMR_DATA_DIR: Directory containing embedding files
    AMR_OUTPUT_DIR: Base output directory for results
    AMR_LABELS_DIR: Directory containing label CSV files
    AMR_CONFIG_FILE: Path to custom config file (optional)

Authors: Evo-AMR Team
License: MIT
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration class for MiniRocket AMR Pipeline."""
    
    # Core directories
    data_dir: str = "data/embeddings"
    output_dir: str = "results"
    labels_dir: str = "data/labels"
    models_dir: str = "models"
    
    # Processing parameters
    projection_dim: int = 41
    window_size: int = 2048
    stride: int = 1024
    num_kernels: int = 2000
    max_dilations_per_kernel: int = 8
    
    # Training parameters
    max_files: int = 60
    max_tokens: int = 20000
    batch_size: int = 131072
    window_batch: int = 128
    
    # Pipeline options
    run_tracks: list = field(default_factory=lambda: ["baseline", "minirocket"])
    skip_mil: bool = False
    tune_on_val_outside: bool = True
    use_zscore: bool = True
    shuffle: bool = True
    cover_tail: bool = True
    
    # File patterns
    embedding_patterns: list = field(default_factory=lambda: ["*.h5", "*.hdf5"])
    label_pattern: str = "*_kover.csv"
    
    # Logging and output
    log_level: str = "INFO"
    save_pca_stats: bool = True
    
    def __post_init__(self):
        """Convert relative paths to absolute paths."""
        base_dir = Path(__file__).parent
        
        # Convert to absolute paths
        self.data_dir = str((base_dir / self.data_dir).resolve())
        self.output_dir = str((base_dir / self.output_dir).resolve())
        self.labels_dir = str((base_dir / self.labels_dir).resolve())
        self.models_dir = str((base_dir / self.models_dir).resolve())
        
        # Create directories if they don't exist
        for dir_path in [self.output_dir, self.models_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_embedding_files(self, limit: Optional[int] = None) -> list:
        """Get list of embedding files."""
        files = []
        data_path = Path(self.data_dir)
        
        if data_path.is_file():
            return [str(data_path)]
        elif data_path.is_dir():
            for pattern in self.embedding_patterns:
                files.extend(data_path.glob(pattern))
            files = sorted([str(f) for f in files])
            return files[:limit] if limit else files
        else:
            return []
    
    def get_label_files(self) -> list:
        """Get list of label CSV files."""
        labels_path = Path(self.labels_dir)
        if labels_path.is_dir():
            return sorted([str(f) for f in labels_path.glob(self.label_pattern)])
        return []
    
    def get_output_path(self, *parts: str) -> str:
        """Get output path for results."""
        return str(Path(self.output_dir) / Path(*parts))
    
    def get_models_path(self, *parts: str) -> str:
        """Get path for model files."""
        return str(Path(self.models_dir) / Path(*parts))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PipelineConfig':
        """Load config from file (JSON or YAML)."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    def save_to_file(self, config_path: str) -> None:
        """Save config to file (JSON or YAML)."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            else:
                json.dump(self.to_dict(), f, indent=2)


def get_config(config_file: Optional[str] = None) -> PipelineConfig:
    """
    Get pipeline configuration.
    
    Priority order:
    1. Custom config file (if provided)
    2. Environment variable AMR_CONFIG_FILE
    3. config.yaml in pipeline directory
    4. config.json in pipeline directory
    5. Default configuration with environment variable overrides
    
    Args:
        config_file: Path to custom config file
        
    Returns:
        PipelineConfig instance
    """
    # Try to find config file
    if config_file is None:
        config_file = os.getenv('AMR_CONFIG_FILE')
    
    if config_file is None:
        # Look for default config files
        base_dir = Path(__file__).parent
        for filename in ['config.yaml', 'config.yml', 'config.json']:
            candidate = base_dir / filename
            if candidate.exists():
                config_file = str(candidate)
                break
    
    # Load from file if found
    if config_file and Path(config_file).exists():
        config = PipelineConfig.from_file(config_file)
    else:
        config = PipelineConfig()
    
    # Override with environment variables
    env_overrides = {
        'data_dir': os.getenv('AMR_DATA_DIR'),
        'output_dir': os.getenv('AMR_OUTPUT_DIR'),
        'labels_dir': os.getenv('AMR_LABELS_DIR'),
        'projection_dim': os.getenv('AMR_PROJECTION_DIM'),
        'window_size': os.getenv('AMR_WINDOW_SIZE'),
        'stride': os.getenv('AMR_STRIDE'),
        'num_kernels': os.getenv('AMR_NUM_KERNELS'),
        'max_files': os.getenv('AMR_MAX_FILES'),
        'max_tokens': os.getenv('AMR_MAX_TOKENS'),
        'log_level': os.getenv('AMR_LOG_LEVEL'),
    }
    
    # Apply environment overrides
    for key, value in env_overrides.items():
        if value is not None:
            if hasattr(config, key):
                field_type = type(getattr(config, key))
                if field_type == int:
                    setattr(config, key, int(value))
                elif field_type == bool:
                    setattr(config, key, value.lower() in ['true', '1', 'yes'])
                else:
                    setattr(config, key, value)
    
    # Handle special environment variables
    run_tracks_env = os.getenv('AMR_RUN_TRACKS')
    if run_tracks_env:
        config.run_tracks = [t.strip() for t in run_tracks_env.split(',') if t.strip()]
    
    if os.getenv('AMR_SKIP_MIL', '0') == '1':
        config.skip_mil = True
    
    if os.getenv('AMR_TUNE_ON_VAL_OUTSIDE', '1') == '0':
        config.tune_on_val_outside = False
    
    return config


def create_example_config() -> None:
    """Create example configuration files."""
    base_dir = Path(__file__).parent
    
    config = PipelineConfig()
    
    # Save YAML example
    config.save_to_file(str(base_dir / 'config.example.yaml'))
    
    # Save JSON example
    config.save_to_file(str(base_dir / 'config.example.json'))
    
    print("Example configuration files created:")
    print(f"  - {base_dir / 'config.example.yaml'}")
    print(f"  - {base_dir / 'config.example.json'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration management for MiniRocket AMR Pipeline')
    parser.add_argument('--create-examples', action='store_true', 
                       help='Create example configuration files')
    parser.add_argument('--show-config', action='store_true',
                       help='Show current configuration')
    
    args = parser.parse_args()
    
    if args.create_examples:
        create_example_config()
    elif args.show_config:
        config = get_config()
        print("Current configuration:")
        print(json.dumps(config.to_dict(), indent=2))
    else:
        parser.print_help()