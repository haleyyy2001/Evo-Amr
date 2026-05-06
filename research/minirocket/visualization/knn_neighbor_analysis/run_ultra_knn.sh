#!/usr/bin/env bash
#SBATCH --job-name=knn_analysis
#SBATCH --partition=cpu
#SBATCH --account=pmg
#SBATCH --cpus-per-task=6
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=knn_analysis.log
#SBATCH --error=knn_analysis.err

# Ultra k-NN Analysis Runner Script
# =================================
# Runs the comprehensive k-NN analysis for all partitions using config-based paths.
#
# USAGE:
# 1. Set environment variables to configure data paths (optional):
#    export AMR_DATA_DIR=/path/to/embeddings
#    export AMR_OUTPUT_DIR=/path/to/results
#    export AMR_LABELS_DIR=/path/to/labels
#
# 2. Run with SLURM:
#    sbatch run_ultra_knn.sh
#
# 3. Or run directly:
#    bash run_ultra_knn.sh
#
# This script will use the config system to determine paths automatically.
# The analysis will create .out files that can be analyzed with knn_species_analysis.py

echo "🚀 Starting Ultra k-NN Analysis for All Partitions..."

# Use config-based paths - no hardcoded paths
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PYTHON_ENV="${AMR_PYTHON_ENV:-python}"

echo "🧬 Analyzing MiniRocket features..."

# Run MiniRocket analysis - uses config for paths
$PYTHON_ENV $SCRIPT_DIR/knn_ultra_analysis_all_partitions.py \
    --feature-type minirocket \
    --k 20

echo "🎯 Analyzing Baseline features..."

# Run Baseline analysis - uses config for paths
$PYTHON_ENV $SCRIPT_DIR/knn_ultra_analysis_all_partitions.py \
    --feature-type baseline \
    --k 20

echo "✅ Ultra k-NN Analysis Complete!"
echo "📊 Check results in the configured output directory"