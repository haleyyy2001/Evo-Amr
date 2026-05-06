#!/usr/bin/env bash
#SBATCH --job-name=comprehensive_viz
#SBATCH --partition=cpu
#SBATCH --account=pmg
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/insomnia001/depts/pmg/users/ht2666/log/comprehensive_viz_%j.log
#SBATCH --error=/insomnia001/depts/pmg/users/ht2666/log/comprehensive_viz_%j.err

set -euo pipefail

echo "[INFO] $(date) Starting comprehensive analysis"
echo "[INFO] Analyzing MiniRocket (6-stat pooled) vs Baseline (PCA)"
echo "[INFO] Detailed visualizations: PCA, t-SNE, UMAP, clustering, species analysis"

export MPLBACKEND=Agg
PY=/insomnia001/depts/pmg/users/ht2666/mambaforge/envs/evo-amr/bin/python

echo "[INFO] Installing UMAP if not available..."
$PY -c "
try:
    import umap
    print('UMAP already available')
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'umap-learn'], check=True)
    print('UMAP installed successfully')
"

echo "[INFO] Running comprehensive analysis..."
$PY /insomnia001/depts/pmg/users/ht2666/minirocket_pipeline/visualization/comprehensive_analysis.py

echo "[DONE] $(date) Comprehensive analysis completed"
echo "[INFO] Check results in /insomnia001/depts/pmg/users/ht2666/minirocket_pipeline/visualization/comprehensive_results/"