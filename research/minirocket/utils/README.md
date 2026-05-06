# Utility Modules

This directory contains utility modules, configuration management, and supporting tools for the MiniRocket AMR pipeline.

## Files

### Configuration Management
- **`config.py`** - Centralized configuration system with YAML/JSON support
  - Environment variable overrides
  - Default configuration with portable paths
  - Example configuration generation

### Visualization Tools
- **`embedding_visualization.py`** - Embedding analysis and visualization utilities
  - PCA/t-SNE dimensionality reduction plots
  - Feature distribution analysis
  - AMR resistance pattern visualization

### Environment Setup
- **`setup_environment.sh`** - Environment setup script for dependencies and paths

## Configuration System

The configuration system provides flexible, portable configuration management:

### Basic Usage
```python
from utils.config import get_config

config = get_config()
print(config.data_dir)
print(config.output_dir)
```

### Environment Variables
Key environment variables for configuration override:
- `AMR_DATA_DIR`: Directory containing embedding files
- `AMR_OUTPUT_DIR`: Base output directory for results
- `AMR_LABELS_DIR`: Directory containing label CSV files
- `AMR_CONFIG_FILE`: Path to custom configuration file

### Configuration Files
Supports both YAML and JSON formats:
```bash
# Create example configurations
python utils/config.py --create-examples

# Use custom configuration
python pipeline.py --config path/to/config.yaml
```

## Visualization Tools

### Embedding Analysis
```python
from utils.embedding_visualization import visualize_embeddings

# Generate PCA and t-SNE plots
visualize_embeddings(
    embedding_dir="data/embeddings",
    labels_csv="data/labels/drug_labels.csv",
    output_dir="results/visualization"
)
```

### Features
- Principal Component Analysis (PCA) plots
- t-SNE embedding visualization
- Feature importance analysis
- Resistance pattern clustering
- Interactive plots with resistance annotations

## Environment Setup

### Quick Setup
```bash
# Run setup script
./utils/setup_environment.sh

# Or manual setup
pip install -r requirements.txt
export AMR_DATA_DIR="/path/to/data"
export AMR_OUTPUT_DIR="/path/to/results"
```

### Dependencies
Core dependencies managed by the configuration system:
- numpy, pandas, h5py
- scikit-learn, joblib
- sktime (MiniRocket)
- numba (performance)
- matplotlib, seaborn (visualization)
- PyYAML (configuration)

Optional dependencies:
- PyTorch (MIL models)
- plotly (interactive plots)

## Integration

All utility modules are designed to integrate seamlessly with the core pipeline:
- Configuration system used by all pipeline components
- Visualization tools generate publication-ready figures
- Setup scripts ensure consistent environments across systems