# Core Pipeline Modules

This directory contains the essential pipeline implementations for the MiniRocket AMR project.

## Files (Cleaned and Streamlined)

### Main Pipeline
- **`minirocket_combined_pipeline.py`** - ⭐ **PRIMARY PIPELINE** - Optimized combined PCA + MiniRocket with single-pass extraction
  - Use this for most applications
  - ~40-50% faster than separate PCA+MiniRocket
  - Commands: `fit`, `extract_combined`

### Specialized Pipelines
- **`mini_pca_pipeline.py`** - PCA-only feature extraction pipeline
  - For PCA baseline comparisons
  - Commands: `fit`, `run`, `pool`

- **`mini_srp_pipeline.py`** - Sparse Random Projection alternative to PCA
  - For cases where PCA bias is undesirable
  - Johnson-Lindenstrauss random projections
  - Commands: `fit`, `run`, `pool`

### Advanced Methods
- **`minirocket_mil_pipeline.py`** - Multiple Instance Learning implementation
  - For sequence-level predictions from window-level features
  - Noisy-OR aggregation with PyTorch

## Recommended Usage

### 🚀 **Start Here (Most Users)**
```bash
# Use the main combined pipeline for fastest results
python core/minirocket_combined_pipeline.py fit /path/to/embeddings --outdir models/
python core/minirocket_combined_pipeline.py extract_combined /path/to/embeddings \
  --projector models/projector.npz --minirocket models/minirocket.joblib --outdir results/
```

### 📊 **For Baseline Comparisons**
```bash
# Use PCA-only pipeline for baseline
python core/mini_pca_pipeline.py fit /path/to/embeddings --outdir models/
python core/mini_pca_pipeline.py run /path/to/embeddings --projector models/projector.npz --outdir results/
```

### 🎯 **For Alternative Methods**
```bash
# Use SRP pipeline for unbiased dimensionality reduction
python core/mini_srp_pipeline.py fit /path/to/embeddings --outdir models/
python core/mini_srp_pipeline.py run /path/to/embeddings --projector models/projector.npz --outdir results/
```

## Dependencies

Core requirements:
- numpy, pandas, h5py, joblib
- scikit-learn, sktime (MiniRocket)
- numba (performance optimization)

Optional:
- PyTorch (for MIL models)
- matplotlib/seaborn (visualization)