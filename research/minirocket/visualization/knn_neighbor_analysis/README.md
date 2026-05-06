# KNN Neighbor Analysis for Processed Embedding Space

This directory contains a tool for analyzing K-Nearest Neighbors (KNN) in the processed embedding space to understand how well different feature extraction methods preserve biological and resistance-related relationships.

## 🎯 **Core Purpose**

**Understanding neighbors in processed embedding space after PCA/MiniRocket transformation**

We use KNN analysis to evaluate:
1. **Species clustering quality** - Do genomes from the same species end up as neighbors?
2. **Resistance pattern preservation** - Do resistant/susceptible genomes cluster appropriately?
3. **Method comparison** - Which feature extraction method (PCA vs MiniRocket) produces better biological clustering?
4. **Cross-partition robustness** - Are patterns consistent across different data splits?

## 📁 **Single Analysis Script**

### **`knn_species_analysis.py`** - Complete KNN + Species Analysis
This script does **everything** you need:
- ✅ **KNN computation** using cosine similarity
- ✅ **Species clustering analysis** (same-species neighbor percentages)
- ✅ **AMR prediction accuracy** via neighbor voting
- ✅ **Cross-partition analysis** (val_outside, test_outside, etc.)
- ✅ **Method comparison** (PCA baseline vs MiniRocket)
- ✅ **Rich output summaries** with biological interpretation

## 🚀 **Usage**

### **Basic Usage**
```bash
# Analyze your processed features
python knn_species_analysis.py

# The script will look for your data in standard locations:
# - Processed features: /path/to/processed_embedding/
# - Labels: /path/to/kover_input/labels.csv
```

### **Custom Paths**
```bash
# Specify custom data locations
python knn_species_analysis.py \
  --data-dir /path/to/your/processed/features \
  --labels-csv /path/to/your/labels.csv \
  --output-dir results/
```

### **Compare Methods**
```bash
# Analyze MiniRocket features
python knn_species_analysis.py --feature-type minirocket

# Analyze PCA baseline features  
python knn_species_analysis.py --feature-type baseline
```

## 📊 **Expected Data Structure**

### **Input: Labels CSV Format**
```csv
genome_id,partition_label,label
1001.1001,train,0
1002.1002,train,1
1003.1003,val_outside,0
1004.1004,test_outside,1
1005.1005,val_overlapped,1
1006.1006,test_overlapped,0
```

### **Input: Processed Features Directory**
```
processed_features/
├── 1001.1001_mr_win2048_s1024.npz     # MiniRocket features
├── 1001.1001_pca_stats_dim41.npz      # PCA baseline features
├── 1002.1002_mr_win2048_s1024.npz
├── 1002.1002_pca_stats_dim41.npz
└── ...
```

### **Output: Rich Analysis Results**
```
Species Clustering Analysis Results:
================================
Total genomes analyzed: 1,234
Feature type: minirocket

KNN Performance Metrics:
- Neighbor voting accuracy: 82.3%
- Precision: 84.5%
- Recall: 79.1%

Species Clustering Quality:
- Same-species neighbors: 78.3%
- Average species per 10-NN: 2.1
- Top species clusters:
  1. Escherichia coli: 89.2% same-species
  2. Klebsiella pneumoniae: 85.7% same-species
  3. Staphylococcus aureus: 91.4% same-species

Cross-Partition Consistency:
- val_outside: 82.1% accuracy
- test_outside: 83.4% accuracy
- val_overlapped: 84.2% accuracy
- test_overlapped: 81.9% accuracy
```

## 📈 **Interpreting Results**

### **✅ Good Neighbor Quality Indicators**
- **Same-species percentage > 70%** - Feature space preserves taxonomic relationships
- **KNN accuracy > 75%** - Neighbors share similar resistance patterns
- **Consistent cross-partition results** - Method is robust across data splits
- **Balanced species clustering** - Not too perfect (avoids species bias)

### **⚠️ Potential Issues to Watch**
- **Same-species percentage < 50%** - Feature space may be too noisy
- **Perfect clustering (>95%)** - Possible overfitting to species
- **Large cross-partition variation** - Unstable feature extraction
- **Very low accuracy (<60%)** - Features don't capture resistance signals

### **🎯 Target Performance Metrics**
```
Ideal KNN Analysis Results:
├── Same-species clustering: 70-85%
├── KNN accuracy: 75-90%
├── Cross-partition consistency: <0.1 variation
└── Species diversity: 1.5-3.0 species per 10-NN
```

## 🔬 **What This Analysis Tells You**

### **Biological Validation**
- **High same-species clustering** = Features preserve important biological relationships
- **Moderate clustering** = Ideal balance - preserves biology without species bias  
- **Random clustering** = Features may be too abstract or noisy

### **Method Effectiveness**
- **High KNN accuracy** = Feature space captures resistance-related patterns
- **Cross-partition consistency** = Method is robust and generalizable
- **Method comparison** = Shows whether MiniRocket or PCA baseline works better

### **Quality Control**
- **Species distribution analysis** = Identifies potential species bias
- **Resistance pattern analysis** = Validates biological relevance
- **Neighbor diversity metrics** = Ensures clustering isn't too narrow

## 🛠️ **Dependencies**

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 📋 **Quick Start**

1. **Prepare your data** (processed features + labels CSV)
2. **Run analysis**: `python knn_species_analysis.py`  
3. **Check results** for same-species clustering % and KNN accuracy
4. **Compare methods** by running with `--feature-type baseline` and `--feature-type minirocket`

This single script provides comprehensive analysis of neighbor quality in your processed embedding space, combining both machine learning performance metrics and biological relationship validation.