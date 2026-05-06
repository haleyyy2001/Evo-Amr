# Experimental and Testing Modules

This directory contains experimental scripts and comprehensive testing pipelines for comparing different AMR prediction approaches.

## Files

### Comprehensive Test Pipelines
- **`amr_unified_test.py`** - Unified testing pipeline for MiniRocket vs baseline comparison
  - Compares MiniRocket time-series classification with PCA baseline methods
  - Supports both tabular and Multiple Instance Learning (MIL) approaches
  - Environment-driven configuration with flexible run tracks

- **`amr_general_test.py`** - General testing pipeline with command-line flexibility
  - Command-line arguments for flexible split-drug pairs
  - Configurable for different experimental setups
  - Supports custom data splits and drug configurations

## Usage

### Quick Comparison Test
```bash
# Run unified test with default configuration
python experiments/amr_unified_test.py

# Run only MiniRocket track
AMR_RUN_TRACKS=minirocket python experiments/amr_unified_test.py

# Skip MIL to save memory
AMR_SKIP_MIL=1 python experiments/amr_unified_test.py
```

### Custom Experimental Setup
```bash
# Run general test with specific parameters
python experiments/amr_general_test.py \
  --split clustered_3_v1-2 \
  --drug gentamicin \
  --kover_csv /path/to/labels.csv \
  --embedding_dir /path/to/embeddings
```

## Environment Variables

### Common Configuration
- `AMR_RUN_TRACKS`: Comma-separated list of tracks ("baseline", "minirocket")
- `AMR_SKIP_MIL`: Set to '1' to skip Multiple Instance Learning
- `AMR_TUNE_ON_VAL_OUTSIDE`: Use validation set for hyperparameter tuning

### Data Paths
- `AMR_OUTRUN`: Output directory for MiniRocket results
- `AMR_PCA_DIR`: Output directory for PCA baseline results
- `AMR_LABELS`: Path to labels CSV file
- `AMR_ROOT_OUT`: Root output directory for all results

## Output Structure

Results are organized by method and evaluation type:
```
results/
├── baseline/
│   ├── tabular_results.json
│   └── feature_analysis/
├── minirocket/
│   ├── tabular_results.json
│   ├── mil_results.json (if enabled)
│   └── feature_analysis/
└── comparison/
    ├── method_comparison.png
    └── performance_summary.csv
```

## Experimental Design

Both scripts implement proper experimental methodology:
- Train/validation/test splits with no data leakage
- Hyperparameter tuning on validation set
- Multiple evaluation metrics (AUROC, AUPRC, accuracy, etc.)
- Cross-validation for robust estimates
- Statistical significance testing

## Performance Tips

1. **Memory Management**: Use `AMR_SKIP_MIL=1` for large datasets
2. **Speed**: Start with `AMR_RUN_TRACKS=minirocket` for faster testing
3. **Validation**: Use `AMR_TUNE_ON_VAL_OUTSIDE=1` for proper evaluation