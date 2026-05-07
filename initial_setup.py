#!/usr/bin/env python3
"""
Setup script for Evo-AMR pipeline
Handles initial configuration and dependency checking
"""

import os
import sys
import subprocess
from pathlib import Path
import yaml

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def check_conda():
    """Check if conda is available"""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Conda available: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        print("✗ Conda not found in PATH")
        return False

def check_gpu():
    """Check if CUDA GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA available: {gpu_count} GPU(s), {gpu_name}")
            return True
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("? PyTorch not installed - cannot check GPU")
        return None

def setup_config():
    """Setup initial configuration files"""
    base_dir = Path(__file__).parent.absolute()
    config_dir = base_dir / "config"

    paths_file = config_dir / "paths.yaml"
    if paths_file.exists():
        with open(paths_file, 'r') as f:
            paths_config = yaml.safe_load(f)
    else:
        print("ERROR: config/paths.yaml not found")
        return False

    paths_config['base_dir'] = str(base_dir)

    current_data_dir = paths_config.get('data_dir', str(base_dir / 'data'))
    print(f"\nCurrent data directory: {current_data_dir}")
    new_data_dir = input("Enter data directory path (or press Enter to keep current): ").strip()

    if new_data_dir:
        paths_config['data_dir'] = new_data_dir
        print(f"✓ Updated data directory to: {new_data_dir}")

    with open(paths_file, 'w') as f:
        yaml.dump(paths_config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Configuration updated in: {paths_file}")
    return True

def create_directories():
    """Create necessary directories"""
    base_dir = Path(__file__).parent.absolute()

    dirs_to_create = ["results", "logs", "cache", "data", "embeddings"]

    for dirname in dirs_to_create:
        dir_path = base_dir / dirname
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def test_config_manager():
    """Test the configuration manager"""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "config"))
        from config_manager import ConfigManager

        config = ConfigManager()
        print(f"✓ Configuration manager loaded successfully")
        print(f"  Base directory: {config.base_dir}")
        print(f"  Model path: {config.get_model_path('evo_1_131k_base')}")
        return True
    except Exception as e:
        print(f"✗ Configuration manager test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Evo-AMR Setup")
    print("=" * 50)

    print("\n1. Checking system requirements...")
    check_python_version()
    conda_available = check_conda()
    gpu_available = check_gpu()

    print("\n2. Setting up configuration...")
    if not setup_config():
        print("✗ Configuration setup failed")
        return 1

    print("\n3. Creating directories...")
    create_directories()

    print("\n4. Testing configuration...")
    if not test_config_manager():
        print("✗ Configuration test failed")
        return 1

    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update config/paths.yaml with your data paths")
    print("2. Update config/environment.yaml with your SLURM settings")
    print("3. Install dependencies: conda env create -f environment.yml")
    print("4. Run diagnostics: sbatch diagnostics/run_diagnostics.sbatch")

    if not conda_available:
        print("\nNote: Install conda/mamba for dependency management")

    if gpu_available is False:
        print("\nWarning: No CUDA GPU detected - some features may not work")

    return 0

if __name__ == "__main__":
    sys.exit(main())
