# MiniRocket AMR Pipeline

A comprehensive pipeline for antimicrobial resistance (AMR) prediction using MiniRocket time-series classification and baseline methods with evolutionary protein embeddings.

## Directory Structure

```
minirocket_pipeline/
├── core/                     # Core pipeline implementations
│   ├── minirocket_combined_pipeline.py    # Optimized combined pipeline
│   ├── combined_pipeline.py               # Professional pipeline
│   ├── minirocket_classifier.py           # Core MiniRocket implementation
│   ├── pca_analyzer.py                    # PCA analysis tools
│   ├── minirocket_mil_pipeline.py         # Multiple Instance Learning
│   ├── mini_pca_pipeline.py               # PCA feature extraction
│   ├── mini_srp_pipeline.py               # Sparse Random Projection
│   └── make_fit_manifest.py               # Training utilities
├── experiments/              # Testing and experimental scripts
│   ├── amr_unified_test.py                # Unified comparison tests
│   └── amr_general_test.py                # Flexible testing pipeline
├── utils/                    # Utilities and configuration
│   ├── config.py                          # Configuration management
│   ├── embedding_visualization.py         # Visualization tools
│   └── setup_environment.sh               # Environment setup
├── examples/                 # Examples and templates
│   └── config.example.yaml                # Example configuration
├── visualization/            # Analysis and visualization results
├── trash/                    # Deprecated/unused files
├── amr_classification_pipeline.py         # Main comprehensive pipeline
└── README.md                 # This file
```

## Features

- **Multiple Approaches**: MiniRocket time-series classification and PCA baseline methods
- **Configurable Pipeline**: Environment-based configuration with YAML/JSON support
- **Professional Architecture**: Modular design with proper separation of concerns
- **Cross-validation**: Proper train/validation/test splits with hyperparameter tuning
- **Multiple Instance Learning (MIL)**: Sequence-level predictions for genomic data
- **Comprehensive Evaluation**: Detailed metrics and visualization tools

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment (optional)
./setup_environment.sh
```

### 2. Configure Pipeline

```bash
# Create configuration file
python config.py --create-examples

# Copy and edit configuration
cp config.example.yaml config.yaml
# Edit config.yaml with your paths
```

### 3. Basic Usage

```bash
# Using configuration file
python amr_classification_pipeline.py --config config.yaml

# Using environment variables
export AMR_DATA_DIR="/path/to/embeddings"
export AMR_OUTPUT_DIR="/path/to/results"
export AMR_LABELS_DIR="/path/to/labels"
python amr_classification_pipeline.py

# Run specific tracks only
export AMR_RUN_TRACKS="minirocket"
python amr_classification_pipeline.py
```

## Core Components

### Main Pipeline Scripts

- **`amr_classification_pipeline.py`**: Main comprehensive pipeline with multiple algorithms
- **`amr_unified_test.py`**: Unified testing pipeline for MiniRocket vs baseline comparison
- **`amr_general_test.py`**: General testing pipeline with command-line flexibility
- **`minirocket_combined_pipeline.py`**: Optimized combined PCA + MiniRocket pipeline
- **`combined_pipeline.py`**: Professional combined pipeline with proper logging

### Specialized Components

- **`minirocket_classifier.py`**: Core MiniRocket windowed pipeline
- **`minirocket_mil_pipeline.py`**: Multiple Instance Learning implementation
- **`pca_analyzer.py`**: PCA analysis and dimensionality reduction
- **`embedding_visualization.py`**: Embedding visualization tools

### Pipeline Modules

- **`pipelines/mini_pca_pipeline.py`**: PCA-based feature extraction pipeline
- **`pipelines/mini_srp_pipeline.py`**: Sparse Random Projection pipeline
- **`pipelines/make_fit_manifest.py`**: Training manifest generation

### Visualization Tools

- **`visualization/comprehensive_analysis.py`**: Comprehensive result analysis
- **`visualization/knn/analyze_split_results.py`**: K-NN analysis for data splits
- **`visualization/knn_ultra_analysis_all_partitions.py`**: Cross-partition analysis

## Configuration

### Configuration File (config.yaml)

```yaml
# Core directories
data_dir: "data/embeddings"
output_dir: "results"
labels_dir: "data/labels"
models_dir: "models"

# Processing parameters
projection_dim: 41
window_size: 2048
stride: 1024
num_kernels: 2000

# Pipeline options
run_tracks:
  - "baseline"
  - "minirocket"
skip_mil: false
tune_on_val_outside: true
```

### Environment Variables

Key environment variables for configuration:

- `AMR_DATA_DIR`: Directory containing embedding files
- `AMR_OUTPUT_DIR`: Base output directory for results
- `AMR_LABELS_DIR`: Directory containing label CSV files
- `AMR_RUN_TRACKS`: Comma-separated list of tracks to run
- `AMR_SKIP_MIL`: Set to '1' to skip Multiple Instance Learning
- `AMR_CONFIG_FILE`: Path to custom configuration file

## Data Requirements

### Input Data Structure

```
data/
├── embeddings/           # H5/HDF5 embedding files
│   ├── genome1.h5
│   ├── genome2.h5
│   └── ...
└── labels/               # CSV label files
    ├── drug1_kover.csv
    ├── drug2_kover.csv
    └── ...
```

### Embedding File Format

H5/HDF5 files with:
- `/embeddings`: `[T, D]` float32/float64 array (e.g., D=4096)
- `/valid_mask`: `[T]` optional boolean mask

### Label File Format

CSV files with columns:
- `genome_id`: Unique genome identifier
- `partition_label`: train/val_outside/val_overlapped/test_outside/test_overlapped
- `label`: Binary resistance label (0/1)

## Pipeline Workflows

### 1. Training Workflow

```bash
# Fit phase: Train PCA projector and MiniRocket
python minirocket_combined_pipeline.py fit /path/to/embeddings \
  --outdir models/ \
  --labels-csv data/labels/drug_kover.csv \
  --proj-dim 41 \
  --num-kernels 2000
```

### 2. Extraction Workflow

```bash
# Combined extraction: Apply PCA and MiniRocket in single pass
python minirocket_combined_pipeline.py extract_combined /path/to/embeddings \
  --projector models/projector.npz \
  --minirocket models/minirocket.joblib \
  --outdir results/ \
  --pca-stats-dir results/pca_stats
```

### 3. Classification Workflow

```bash
# Run comprehensive classification pipeline
python amr_classification_pipeline.py \
  --config config.yaml \
  --drug ampicillin \
  --split clustered_3_v1-2
```

## Output Structure

```
results/
├── baseline/             # PCA baseline results
│   ├── models/
│   ├── features/
│   └── evaluation/
├── minirocket/           # MiniRocket results
│   ├── models/
│   ├── features/
│   └── evaluation/
└── comparison/           # Cross-method comparison
    ├── roc_curves.png
    ├── metrics_table.csv
    └── feature_analysis.png
```

## Performance Tips

1. **Memory Management**: Use `--batch-size` to control memory usage
2. **Parallelization**: Set `NUMBA_NUM_THREADS` for optimal performance
3. **Storage**: Use `--cover-tail` for complete sequence coverage
4. **Validation**: Use `--tune-on-val-outside` for proper hyperparameter tuning

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Path Issues**: Use absolute paths or proper relative paths from pipeline root
3. **Memory Issues**: Reduce batch sizes or use streaming processing
4. **Configuration**: Verify config.yaml syntax and paths

### Debug Mode

```bash
# Enable debug logging
export AMR_LOG_LEVEL=DEBUG
python pipeline.py --config config.yaml
```

## Contributing

1. Follow the existing code structure
2. Use the configuration system for all paths
3. Add proper logging and error handling
4. Update documentation for new features
5. Test with multiple data splits and drugs

## License

MIT License - see LICENSE file for details.

## Authors

Evo-AMR Team