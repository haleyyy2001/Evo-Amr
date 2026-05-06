#!/usr/bin/env python3
"""
Configuration Manager for Evo-AMR Pipeline
Handles path resolution and environment setup
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import string

class ConfigManager:
    """Manages configuration for Evo-AMR pipeline"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            base_dir: Base directory for the pipeline. If None, uses current file's parent directory
        """
        if base_dir is None:
            # Auto-detect base directory from this file's location
            self.base_dir = Path(__file__).parent.parent.absolute()
        else:
            self.base_dir = Path(base_dir).absolute()
            
        self.config_dir = self.base_dir / "config"
        
        # Load configurations
        self.paths = self._load_yaml("paths.yaml")
        self.env = self._load_yaml("environment.yaml")
        
        # Resolve paths
        self._resolve_paths()
        
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _resolve_paths(self):
        """Resolve all paths using string interpolation"""
        # Set base variables for interpolation
        variables = {
            'base_dir': str(self.base_dir),
            'data_dir': self.paths.get('data_dir', str(self.base_dir / 'data'))
        }
        
        # Recursively resolve paths
        self.paths = self._interpolate_dict(self.paths, variables)
        
    def _interpolate_dict(self, data: Any, variables: Dict[str, str]) -> Any:
        """Recursively interpolate variables in dictionary values"""
        if isinstance(data, dict):
            return {k: self._interpolate_dict(v, variables) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._interpolate_dict(item, variables) for item in data]
        elif isinstance(data, str):
            try:
                # Use string.Template for safe interpolation
                template = string.Template(data)
                return template.safe_substitute(variables)
            except (KeyError, ValueError):
                return data
        else:
            return data
    
    def get_path(self, *keys: str) -> Path:
        """Get a path from the configuration
        
        Args:
            *keys: Nested keys to access the path
            
        Returns:
            Resolved Path object
        """
        current = self.paths
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise KeyError(f"Path not found: {'.'.join(keys)}")
        
        if isinstance(current, str):
            return Path(current)
        else:
            raise ValueError(f"Path value is not a string: {'.'.join(keys)}")
    
    def get_env(self, *keys: str) -> Any:
        """Get an environment setting
        
        Args:
            *keys: Nested keys to access the setting
            
        Returns:
            Environment setting value
        """
        current = self.env
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                raise KeyError(f"Environment setting not found: {'.'.join(keys)}")
        
        return current
    
    def ensure_dirs(self, *path_keys: str):
        """Ensure directories exist for given path keys"""
        for path_key in path_keys:
            keys = path_key.split('.')
            path = self.get_path(*keys)
            path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to a specific model"""
        return self.get_path("models", model_name)
    
    def get_cache_path(self, cache_type: str) -> Path:
        """Get path to a specific cache directory"""
        return self.get_path("cache", cache_type)
    
    def get_output_path(self, output_type: str) -> Path:
        """Get path to a specific output directory"""
        return self.get_path("output", output_type)
    
    def setup_environment(self) -> Dict[str, str]:
        """Setup environment variables for the pipeline
        
        Returns:
            Dictionary of environment variables to set
        """
        env_vars = {}
        
        # Cache paths
        cache_root = self.get_cache_path("root")
        env_vars.update({
            'EVOCACHE_ROOT': str(cache_root),
            'XDG_CACHE_HOME': str(cache_root / "xdg"),
            'HF_HOME': str(self.get_cache_path("hf")),
            'TRANSFORMERS_CACHE': str(self.get_cache_path("hf")),
            'TORCH_HOME': str(self.get_cache_path("torch")),
            'TORCHINDUCTOR_CACHE_DIR': str(cache_root / "inductor"),
            'TRITON_CACHE_DIR': str(self.get_cache_path("triton")),
            'FLASH_ATTENTION_CACHE_DIR': str(self.get_cache_path("flashattn")),
            'CUDA_CACHE_PATH': str(cache_root / "cuda"),
        })
        
        # Performance settings
        env_vars.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'TRANSFORMERS_VERBOSITY': 'error',
            'TOKENIZERS_PARALLELISM': 'false',
            'PYTHONUNBUFFERED': '1',
        })
        
        # CUDA settings
        if cuda_devices := self.get_env("cuda", "visible_devices"):
            if cuda_devices != "auto":
                env_vars['CUDA_VISIBLE_DEVICES'] = cuda_devices
        
        return env_vars
    
    def get_slurm_config(self) -> Dict[str, Any]:
        """Get SLURM configuration"""
        slurm_config = {
            'partition': self.get_env("slurm", "partition"),
            'account': self.get_env("slurm", "account"),
            'log_dir': self.get_path("slurm", "log_dir"),
        }
        
        # Ensure log directory exists
        slurm_config['log_dir'].mkdir(parents=True, exist_ok=True)
        
        return slurm_config

# Global instance for easy access
_config_manager = None

def get_config(base_dir: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(base_dir)
    return _config_manager

def setup_paths_from_config(base_dir: Optional[str] = None) -> Dict[str, str]:
    """Setup environment variables from configuration
    
    Returns:
        Dictionary of environment variables to set
    """
    config = get_config(base_dir)
    return config.setup_environment()

if __name__ == "__main__":
    # Test the configuration manager
    config = ConfigManager()
    print(f"Base directory: {config.base_dir}")
    print(f"Model path: {config.get_model_path('evo_1_131k_base')}")
    print(f"Cache path: {config.get_cache_path('hf')}")
    print("Environment variables:")
    for k, v in config.setup_environment().items():
        print(f"  {k}={v}")