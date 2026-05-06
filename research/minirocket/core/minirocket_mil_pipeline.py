#!/usr/bin/env python3
"""
Corrected pipeline for genome embedding processing:
1. Dimension reduction on 4096-dim (PCA: 4096 → 128)
2. MiniROCKET on genome length dimension (treating 128 dims as channels)
3. Output fixed-length features per genome
"""

import os
import h5py
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate
import warnings
warnings.filterwarnings('ignore')

class GenomeMiniROCKETProcessor:
    def __init__(self, reduced_dim=128, rocket_kernels=10000):
        self.reduced_dim = reduced_dim
        self.rocket_kernels = rocket_kernels
        
        # Dimension reduction components
        self.dimension_reducer = None
        self.scaler = None
        
        # MiniROCKET component
        self.minirocket = None
        
    def fit_dimension_reduction(self, all_files, max_samples_per_file=50000):
        """Fit PCA for dimension reduction on 4096 → reduced_dim."""
        print(f"Training dimension reduction: 4096 → {self.reduced_dim}")
        
        self.dimension_reducer = IncrementalPCA(
            n_components=self.reduced_dim, 
            batch_size=min(5000, max_samples_per_file // 10)
        )
        self.scaler = StandardScaler()
        
        # Collect data from all genomes for PCA training
        total_samples_processed = 0
        
        for file_path in tqdm(all_files, desc="Fitting dimension reducer"):
            try:
                with h5py.File(file_path, 'r') as f:
                    embeddings = f['embeddings'][:]  # Shape: (genome_length, 4096)
                
                # Sample from this genome to avoid memory issues
                if len(embeddings) > max_samples_per_file:
                    # Uniform sampling across genome length
                    indices = np.linspace(0, len(embeddings)-1, max_samples_per_file, dtype=int)
                    sampled_embeddings = embeddings[indices]
                else:
                    sampled_embeddings = embeddings
                
                # Incremental fitting
                self.scaler.partial_fit(sampled_embeddings)
                
                # Scale and fit PCA
                scaled_embeddings = self.scaler.transform(sampled_embeddings)
                self.dimension_reducer.partial_fit(scaled_embeddings)
                
                total_samples_processed += len(sampled_embeddings)
                
            except Exception as e:
                print(f"Error processing {file_path} for dimension reduction: {e}")
                continue
        
        print(f"Dimension reduction fitted on {total_samples_processed} total samples")
        print(f"Explained variance ratio (first 10): {self.dimension_reducer.explained_variance_ratio_[:10]}")
        
    def fit_minirocket(self, all_files, max_genomes_for_rocket=20):
        """Fit MiniROCKET on reduced genome sequences."""
        print(f"Training MiniROCKET on genome length dimension...")
        
        self.minirocket = MiniRocketMultivariate(
            num_kernels=self.rocket_kernels,
            random_state=42
        )
        
        # Collect sample genomes for MiniROCKET fitting
        sample_genomes = []
        processed_count = 0
        
        for file_path in tqdm(all_files[:max_genomes_for_rocket], desc="Collecting genomes for MiniROCKET"):
            try:
                with h5py.File(file_path, 'r') as f:
                    embeddings = f['embeddings'][:]  # Shape: (genome_length, 4096)
                
                # Apply dimension reduction
                scaled_embeddings = self.scaler.transform(embeddings)
                reduced_genome = self.dimension_reducer.transform(scaled_embeddings)
                # Shape: (genome_length, reduced_dim)
                
                # Prepare for MiniROCKET: (n_instances, n_channels, n_timepoints)
                # We want: (1, reduced_dim, genome_length)
                rocket_input = reduced_genome.T[np.newaxis, :, :]  # (1, reduced_dim, genome_length)
                sample_genomes.append(rocket_input)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {file_path} for MiniROCKET: {e}")
                continue
        
        if sample_genomes:
            # Stack all sample genomes
            X_train = np.vstack(sample_genomes)  # (n_genomes, reduced_dim, max_genome_length)
            print(f"MiniROCKET training data shape: {X_train.shape}")
            
            # Fit MiniROCKET
            self.minirocket.fit(X_train)
            print(f"MiniROCKET fitted on {len(sample_genomes)} genomes")
        else:
            raise ValueError("No valid genomes found for MiniROCKET training")
    
    def fit(self, all_files, max_samples_per_file=50000, max_genomes_for_rocket=10):
        """Fit the entire pipeline."""
        print("=== Training Complete Pipeline ===")
        
        # Step 1: Fit dimension reduction
        self.fit_dimension_reduction(all_files, max_samples_per_file)
        
        # Step 2: Fit MiniROCKET  
        self.fit_minirocket(all_files, max_genomes_for_rocket)
        
        print("=== Pipeline Training Complete ===")
    
    def transform_genome(self, file_path):
        """Transform a single genome file."""
        try:
            with h5py.File(file_path, 'r') as f:
                embeddings = f['embeddings'][:]  # Shape: (genome_length, 4096)
            
            print(f"Processing genome of length: {len(embeddings)}")
            
            # Step 1: Apply dimension reduction
            scaled_embeddings = self.scaler.transform(embeddings)
            reduced_genome = self.dimension_reducer.transform(scaled_embeddings)
            # Shape: (genome_length, reduced_dim)
            
            # Step 2: Prepare for MiniROCKET
            rocket_input = reduced_genome.T[np.newaxis, :, :]  # (1, reduced_dim, genome_length)
            
            # Step 3: Apply MiniROCKET
            genome_features = self.minirocket.transform(rocket_input)  # (1, n_features)
            
            return genome_features[0]  # Return 1D feature vector
            
        except Exception as e:
            print(f"Error transforming {file_path}: {e}")
            return None
    
    def save_model(self, save_path):
        """Save the fitted pipeline."""
        model_data = {
            'dimension_reducer': self.dimension_reducer,
            'scaler': self.scaler, 
            'minirocket': self.minirocket,
            'reduced_dim': self.reduced_dim,
            'rocket_kernels': self.rocket_kernels
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Pipeline saved to {save_path}")
    
    def load_model(self, load_path):
        """Load a fitted pipeline."""
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.dimension_reducer = model_data['dimension_reducer']
        self.scaler = model_data['scaler']
        self.minirocket = model_data['minirocket'] 
        self.reduced_dim = model_data['reduced_dim']
        self.rocket_kernels = model_data['rocket_kernels']
        print(f"Pipeline loaded from {load_path}")

def main():
    # Paths
    input_dir = "/insomnia001/depts/pmg/users/ht2666/ultra_evo_embeddings_fixed"
    output_dir = "/insomnia001/depts/pmg/users/ht2666/ultra_lowdim_evo_embedding"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get all h5 files
    all_files = [f for f in Path(input_dir).glob("*.h5") if not f.name.endswith('.tmp')]
    print(f"Found {len(all_files)} embedding files")
    
    if len(all_files) == 0:
        print("No valid .h5 files found!")
        return
    
    # Initialize processor
    processor = GenomeMiniROCKETProcessor(
        reduced_dim=128,      # 4096 → 128 dimension reduction
        rocket_kernels=10000  # MiniROCKET feature count
    )
    
    # Fit the pipeline
    processor.fit(
        all_files=all_files,
        max_samples_per_file=50000,  # Sample per genome for PCA training
        max_genomes_for_rocket=min(10, len(all_files))  # Genomes for MiniROCKET training
    )
    
    # Process all genomes
    print("\n=== Processing All Genomes ===")
    results = {}
    failed_files = []
    
    for file_path in tqdm(all_files, desc="Extracting genome features"):
        file_id = file_path.stem
        genome_features = processor.transform_genome(file_path)
        
        if genome_features is not None:
            results[file_id] = genome_features
            print(f"✓ {file_id}: {len(genome_features)} features extracted")
        else:
            failed_files.append(file_path)
            print(f"✗ {file_id}: Failed")
    
    # Save results
    if results:
        output_file = Path(output_dir) / "genome_minirocket_features.h5"
        
        print(f"\nSaving results to {output_file}")
        with h5py.File(output_file, 'w') as f:
            file_ids = list(results.keys())
            features = np.array(list(results.values()))
            
            f.create_dataset('file_ids', data=[fid.encode('utf-8') for fid in file_ids])
            f.create_dataset('features', data=features)
            f.attrs['n_genomes'] = len(file_ids)
            f.attrs['feature_dim'] = features.shape[1]
            f.attrs['reduced_dim'] = processor.reduced_dim
            f.attrs['rocket_kernels'] = processor.rocket_kernels
        
        # Save model
        model_file = Path(output_dir) / "genome_minirocket_pipeline.pkl"
        processor.save_model(model_file)
        
        # Print summary
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {len(results)} genomes")
        print(f"Failed: {len(failed_files)} genomes")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Results: {output_file}")
        print(f"Model: {model_file}")
        
        # Show feature statistics
        print(f"\nFeature Statistics:")
        print(f"Mean: {np.mean(features):.4f}")
        print(f"Std: {np.std(features):.4f}")
        print(f"Min: {np.min(features):.4f}")
        print(f"Max: {np.max(features):.4f}")
        
    else:
        print("No genomes were successfully processed!")
    
    if failed_files:
        print(f"\nFailed files ({len(failed_files)}):")
        for f in failed_files[:10]:  # Show first 10
            print(f"  {f.name}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files)-10} more")

if __name__ == "__main__":
    main()