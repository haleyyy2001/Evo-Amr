# Examples and Configuration Templates

This directory contains example configurations, usage examples, and templates for the MiniRocket AMR pipeline.

## Files

### Configuration Examples
- **`config.example.yaml`** - Complete example configuration file with all options documented
  - Core directory settings
  - Processing parameters
  - Pipeline options and flags
  - Detailed comments explaining each setting

## Configuration Examples

### Basic Configuration (config.example.yaml)
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

## Usage Examples

### 1. Basic Pipeline Execution
```bash
# Copy example configuration
cp examples/config.example.yaml config.yaml

# Edit paths for your data
vim config.yaml

# Run main classification pipeline
python amr_classification_pipeline.py --config config.yaml
```

### 2. Environment-Based Configuration
```bash
# Set environment variables
export AMR_DATA_DIR="/path/to/your/embeddings"
export AMR_OUTPUT_DIR="/path/to/your/results"
export AMR_LABELS_DIR="/path/to/your/labels"

# Run without config file
python amr_classification_pipeline.py
```

### 3. Experimental Comparison
```bash
# Run unified test for method comparison
export AMR_RUN_TRACKS="baseline,minirocket"
python experiments/amr_unified_test.py

# Run only MiniRocket for faster testing
export AMR_RUN_TRACKS="minirocket"
python experiments/amr_unified_test.py
```

### 4. Custom Drug-Split Analysis
```bash
# Use general test with command-line arguments
python experiments/amr_general_test.py \
  --split clustered_3_v1-2 \
  --drug ampicillin \
  --kover_csv data/labels/ampicillin_v1-2_kover.csv \
  --embedding_dir data/embeddings/v1-2
```

### 5. Core Pipeline Components
```bash
# Run optimized combined pipeline
python core/minirocket_combined_pipeline.py fit data/embeddings \
  --outdir models/ \
  --labels-csv data/labels/drug_labels.csv

# Extract features
python core/minirocket_combined_pipeline.py extract_combined data/embeddings \
  --projector models/projector.npz \
  --minirocket models/minirocket.joblib \
  --outdir results/
```

## Data Structure Examples

### Expected Directory Structure
```
project/
├── data/
│   ├── embeddings/           # H5 embedding files
│   │   ├── genome1.h5
│   │   ├── genome2.h5
│   │   └── ...
│   └── labels/               # CSV label files
│       ├── ampicillin_kover.csv
│       ├── gentamicin_kover.csv
│       └── ...
├── config.yaml              # Main configuration
├── results/                  # Output directory
└── models/                   # Trained models
```

### Label File Format (CSV)
```csv
genome_id,partition_label,label
1001.1001,train,0
1002.1002,train,1
1003.1003,val_outside,0
1004.1004,test_outside,1
```

### Embedding File Format (H5)
```
genome1.h5
├── /embeddings     # [T, D] float array (e.g., [2000, 4096])
└── /valid_mask     # [T] boolean mask (optional)
```

## Performance Tuning Examples

### Memory-Optimized Configuration
```yaml
# Reduce memory usage
batch_size: 65536
window_batch: 64
max_files: 30
max_tokens: 10000
skip_mil: true
```

### Speed-Optimized Configuration
```yaml
# Faster processing
run_tracks: ["minirocket"]  # Skip baseline
num_kernels: 1000          # Fewer kernels
projection_dim: 32         # Lower dimensions
```

### High-Accuracy Configuration
```yaml
# Maximum accuracy
num_kernels: 4000
max_dilations_per_kernel: 16
tune_on_val_outside: true
use_zscore: true
```

## Troubleshooting Examples

### Common Issues and Solutions
```bash
# 1. Import errors - check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/minirocket_pipeline"

# 2. Memory issues - reduce batch sizes
export AMR_BATCH_SIZE=32768

# 3. Performance issues - set thread counts
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8

# 4. Debug mode
export AMR_LOG_LEVEL=DEBUG
python pipeline.py --config config.yaml
```