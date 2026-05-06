#!/bin/bash

# Setup conda environment for genome embedding visualization
cd /insomnia001/depts/pmg/users/ht2666/minirocket_pipeline

# Set conda path
export PATH="/insomnia001/depts/pmg/users/ht2666/miniconda3/bin:$PATH"

echo "Creating conda environment 'genome_viz'..."

# Create conda environment
conda create -n genome_viz python=3.9 -y

echo "Activating environment and installing packages..."

# Activate environment and install packages
conda activate genome_viz

# Install core packages
conda install -c conda-forge numpy scipy matplotlib seaborn pandas h5py scikit-learn tqdm -y

# Install additional packages with pip
pip install umap-learn sktime

echo "Environment setup complete!"
echo "To activate: conda activate genome_viz"