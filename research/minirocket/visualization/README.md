# Visualization and Analysis Tools

This directory contains visualization and analysis tools for understanding AMR prediction methods and exploring the processed embedding space through K-Nearest Neighbor (KNN) analysis.

## Directory Structure

### 🔍 **KNN Neighbor Analysis** (`knn_neighbor_analysis/`)
**Purpose: Understanding neighbors in processed embedding space**

The core focus of this directory is analyzing how genomes cluster and relate to each other in the processed feature space after PCA/MiniRocket transformation. KNN analysis helps us understand:

- **Species clustering patterns** - Do genomes from the same species cluster together?
- **Resistance patterns** - Do resistant/susceptible genomes form distinct clusters?
- **Method effectiveness** - Which feature extraction method (PCA vs MiniRocket) produces better clustering?
- **Cross-partition consistency** - Are clustering patterns consistent across data splits?

#### Key Files:
- **`knn_cosine_analysis.py`** - Core KNN analysis using cosine similarity
- **`knn_ultra_analysis_all_partitions.py`** - Cross-partition KNN analysis
- **`analyze_knn_species_summary_v2.py`** - Species-level clustering analysis
- **`analyze_split_results.py`** - Split-specific result analysis
- **`run_ultra_knn.sh`** - Batch script for running KNN analysis

#### Results Structure:
```
knn_neighbor_analysis/
├── v1-1/                    # Data split v1-1 results
├── v1-2/                    # Data split v1-2 results  
├── v1-3/                    # Data split v1-3 results
├── species_comprehensive_summary_*.txt  # Species clustering summaries
└── knn_*_comparative_summary_*.txt      # Method comparison summaries
```

### 📊 **Method Comparison** (Root Level)
- **`method_comparison_analysis.py`** - Compare MiniRocket vs PCA baseline methods
- **`method_comparison_results/`** - Comparative analysis results and plots
- **`run_method_comparison.sh`** - Script to run method comparisons

### 🗑️ **Deprecated Files** (`trash/`)
- Obsolete visualization scripts and redundant analysis tools

## Key Analysis Questions Answered

### 1. **Neighbor Quality in Embedding Space**
- Are the nearest neighbors of a genome from the same species?
- Do resistant genomes cluster with other resistant genomes?
- How does the choice of feature extraction (PCA vs MiniRocket) affect clustering?

### 2. **Species-Level Clustering**
```bash
# Analyze species clustering patterns
python knn_neighbor_analysis/analyze_knn_species_summary_v2.py

# Results show:
# - Percentage of genomes whose nearest neighbors are same-species
# - Resistance prediction accuracy based on neighbor labels
# - Species diversity in neighborhood analysis
```

### 3. **Cross-Method Comparison**
```bash
# Compare PCA baseline vs MiniRocket neighbor quality
python knn_neighbor_analysis/knn_ultra_analysis_all_partitions.py

# Results show:
# - Which method produces better species clustering
# - Which method better separates resistant/susceptible genomes
# - Consistency across different data splits
```

### 4. **Resistance Pattern Analysis**
The KNN analysis reveals whether:
- Resistant genomes form distinct clusters in feature space
- Nearest neighbor voting can predict resistance effectively
- Feature space preserves biologically meaningful relationships

## Usage Examples

### Basic KNN Analysis
```bash
# Run comprehensive KNN analysis across all partitions
cd visualization/knn_neighbor_analysis/
./run_ultra_knn.sh

# Analyze results for specific species patterns
python analyze_knn_species_summary_v2.py
```

### Method Comparison
```bash
# Compare different feature extraction methods
cd visualization/
python method_comparison_analysis.py

# Generate comparison plots
./run_method_comparison.sh
```

## Interpretation Guidelines

### **Good Neighbor Quality Indicators:**
- **High same-species percentage** (>70%) - Feature space preserves taxonomic relationships
- **High resistance prediction accuracy** - Neighbors share similar resistance patterns
- **Consistent cross-split results** - Method is robust across data partitions

### **Feature Space Quality Metrics:**
- **Species homogeneity** - Nearest neighbors from same species
- **Resistance clustering** - Resistant/susceptible genomes form distinct groups
- **Biological consistency** - Related organisms cluster together

## Files Overview

| File/Directory | Purpose | Key Output |
|---------------|---------|------------|
| `knn_neighbor_analysis/` | Core KNN analysis | Neighbor quality metrics |
| `method_comparison_analysis.py` | Method comparison | Comparative performance plots |
| `method_comparison_results/` | Stored results | Publication-ready figures |
| `trash/` | Deprecated files | Legacy analysis scripts |

## Dependencies

- **scipy** - Cosine similarity calculations
- **scikit-learn** - KNN implementation
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn** - Visualization
- **networkx** (optional) - Network analysis of neighbor relationships