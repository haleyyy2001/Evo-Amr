#!/usr/bin/env python3
"""
Extract properly species-balanced subsets using actual GTDB metadata

Uses the real metadata file to map genome_id -> gtdb_species
Then creates subsets with proper species balance:
- Training + Overlap: same species
- Outside: different species  
- All balanced for R/S labels
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
from glob import glob

# Fixed seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_used_genomes_registry(registry_path):
    """Load registry of already used genomes"""
    if os.path.exists(registry_path):
        print(f"Loading existing registry: {registry_path}")
        registry_df = pd.read_csv(registry_path)
        used_genomes = set(registry_df['genome_id'].unique())
        print(f"Found {len(used_genomes)} previously used genomes")
        return used_genomes, registry_df
    else:
        print(f"No existing registry found at {registry_path}")
        return set(), pd.DataFrame(columns=['genome_id', 'subset_name', 'extraction_round'])

def save_used_genomes_registry(registry_path, registry_df):
    """Save updated registry of used genomes"""
    registry_df.to_csv(registry_path, index=False)
    print(f"Updated registry saved: {registry_path}")

def get_next_subset_number(output_dir):
    """Find the next subset number based on existing files"""
    pattern = os.path.join(output_dir, 'natural_proportion_sampling/ampicillin/subsets_100_*.csv')
    existing_files = glob(pattern)
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        basename = os.path.basename(f)
        # Extract number from subsets_100_X.csv
        if basename.startswith('subsets_100_') and basename.endswith('.csv'):
            try:
                num_part = basename[12:-4]  # Remove 'subsets_100_' and '.csv'
                if num_part.isdigit():
                    numbers.append(int(num_part))
            except:
                continue
    
    return max(numbers) + 1 if numbers else 1

def get_next_subset_number_gentamicin(output_dir):
    """Find the next subset number for gentamicin based on existing files"""
    pattern = os.path.join(output_dir, 'natural_proportion_sampling/gentamicin/subsets_*.csv')
    existing_files = glob(pattern)
    if not existing_files:
        return 1
    
    numbers = []
    for f in existing_files:
        basename = os.path.basename(f)
        # Extract number from subsets_X.csv
        if basename.startswith('subsets_') and basename.endswith('.csv') and '_report' not in basename:
            try:
                num_part = basename[8:-4]  # Remove 'subsets_' and '.csv'
                if num_part.isdigit():
                    numbers.append(int(num_part))
            except:
                continue
    
    return max(numbers) + 1 if numbers else 1

def load_kover_data(kover_path, metadata_path=None):
    """Load kover data and optionally merge with metadata"""
    print("=== Loading Kover Data ===")
    
    # Load kover data (has genome_id, partition_label, antibiotic_label)
    print(f"Loading kover data from: {kover_path}")
    kover_df = pd.read_csv(kover_path)
    print(f"Loaded kover data: {len(kover_df)} genomes")
    print(f"Columns: {list(kover_df.columns)}")
    
    # Check data structure
    partition_counts = kover_df['partition_label'].value_counts()
    print(f"\nPartition distribution:")
    for partition, count in partition_counts.items():
        r_count = len(kover_df[(kover_df['partition_label'] == partition) & (kover_df['antibiotic_label'] == 1)])
        s_count = len(kover_df[(kover_df['partition_label'] == partition) & (kover_df['antibiotic_label'] == 0)])
        print(f"  {partition}: {count} genomes (R:{r_count}, S:{s_count})")
    
    # If metadata provided, try to merge for species information
    if metadata_path and os.path.exists(metadata_path):
        print(f"\nAttempting to merge with metadata: {metadata_path}")
        try:
            metadata_df = pd.read_csv(metadata_path, sep='\t')
            print(f"Loaded metadata: {len(metadata_df)} genomes")
            
            # Try to merge on genome_id
            if 'genome_id' in metadata_df.columns and 'gtdb_species' in metadata_df.columns:
                merged_df = kover_df.merge(metadata_df[['genome_id', 'gtdb_species']], 
                                          on='genome_id', how='left')
                n_with_species = merged_df['gtdb_species'].notna().sum()
                print(f"After merge: {n_with_species}/{len(merged_df)} genomes have species information")
                return merged_df
            else:
                print("Metadata doesn't have expected columns (genome_id, gtdb_species)")
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    print("Proceeding without species information")
    return kover_df

def analyze_species_distribution(df):
    """Analyze species distribution across partitions"""
    print("\n=== Species Distribution Analysis ===")
    
    # Overall species counts
    species_counts = df['gtdb_species'].value_counts()
    print(f"\nTotal unique species: {len(species_counts)}")
    print("\nTop 10 species by genome count:")
    for species, count in species_counts.head(10).items():
        print(f"  {species}: {count} genomes")
    
    # Species by partition
    print(f"\nSpecies distribution by partition:")
    partition_species = df.groupby(['partition_label', 'gtdb_species']).size().unstack(fill_value=0)
    
    # Return ALL species available in training partition (no top-k restriction)
    if 'train' in partition_species.index:
        train_species = partition_species.loc['train']
        train_species_with_data = train_species[train_species > 0].sort_values(ascending=False)
        print(f"\nAll training species ({len(train_species_with_data)}):")
        for species, count in train_species_with_data.head(10).items():  # Show top 10 for display
            print(f"  {species}: {count} training genomes")
        if len(train_species_with_data) > 10:
            print(f"  ... and {len(train_species_with_data)-10} more species")
        
        return train_species_with_data.index.tolist()  # Return ALL species with training data
    
    return species_counts.index.tolist()

def extract_kover_subsets(df, used_genomes):
    """Extract subsets from Kover data with natural proportions"""
    print(f"\n=== Extracting Kover Subsets ===")
    
    # Target sample sizes
    target_sizes = {
        'train': 1200,
        'test_overlapped': 200, 
        'val_overlapped': 200,
        'test_outside': -1,  # Take all available
        'val_outside': -1    # Take all available
    }
    
    subsets = {}
    
    for partition_name, target_size in target_sizes.items():
        print(f"\n--- Extracting {partition_name} ---")
        
        # Get available data for this partition
        partition_data = df[
            (df['partition_label'] == partition_name) &
            (~df['genome_id'].isin(used_genomes))  # Exclude already used genomes
        ].copy()
        
        if len(partition_data) == 0:
            print(f"  Warning: No unused genomes for {partition_name}")
            subsets[partition_name] = pd.DataFrame()
            continue
        
        # Calculate current proportions
        r_count = len(partition_data[partition_data['antibiotic_label'] == 1])
        s_count = len(partition_data[partition_data['antibiotic_label'] == 0])
        total_available = len(partition_data)
        r_prop = r_count / total_available if total_available > 0 else 0
        s_prop = s_count / total_available if total_available > 0 else 0
        
        print(f"  Available: {total_available} genomes (R:{r_count} [{r_prop:.1%}], S:{s_count} [{s_prop:.1%}])")
        
        # Sample strategy
        if target_size == -1:
            # Take all available (for outside sets)
            subset_df = partition_data.copy()
            print(f"  Taking all {len(subset_df)} genomes")
        else:
            # Sample maintaining natural proportions
            if total_available <= target_size:
                # Take all if not enough available
                subset_df = partition_data.copy()
                print(f"  Taking all {len(subset_df)} genomes (less than target {target_size})")
            else:
                # Stratified sampling to maintain natural proportions
                r_data = partition_data[partition_data['antibiotic_label'] == 1]
                s_data = partition_data[partition_data['antibiotic_label'] == 0]
                
                # Calculate how many of each class to sample
                target_r = int(target_size * r_prop)
                target_s = target_size - target_r
                
                # Ensure we don't exceed available data
                actual_r = min(target_r, len(r_data))
                actual_s = min(target_s, len(s_data))
                
                # If we can't get enough of one class, adjust the other
                if actual_r < target_r and len(s_data) > actual_s:
                    additional_s = min(target_r - actual_r, len(s_data) - actual_s)
                    actual_s += additional_s
                elif actual_s < target_s and len(r_data) > actual_r:
                    additional_r = min(target_s - actual_s, len(r_data) - actual_r)
                    actual_r += additional_r
                
                sampled_parts = []
                if actual_r > 0:
                    sampled_r = r_data.sample(n=actual_r, random_state=RANDOM_SEED)
                    sampled_parts.append(sampled_r)
                if actual_s > 0:
                    sampled_s = s_data.sample(n=actual_s, random_state=RANDOM_SEED)
                    sampled_parts.append(sampled_s)
                
                if sampled_parts:
                    subset_df = pd.concat(sampled_parts, ignore_index=True)
                else:
                    subset_df = pd.DataFrame()
                
                final_r = len(subset_df[subset_df['antibiotic_label'] == 1])
                final_s = len(subset_df[subset_df['antibiotic_label'] == 0])
                final_r_prop = final_r / len(subset_df) if len(subset_df) > 0 else 0
                final_s_prop = final_s / len(subset_df) if len(subset_df) > 0 else 0
                
                print(f"  Sampled: {len(subset_df)} genomes (R:{final_r} [{final_r_prop:.1%}], S:{final_s} [{final_s_prop:.1%}])")
        
        subsets[partition_name] = subset_df
        
        # Final statistics
        if len(subset_df) > 0:
            final_r = len(subset_df[subset_df['antibiotic_label'] == 1])
            final_s = len(subset_df[subset_df['antibiotic_label'] == 0])
            if 'gtdb_species' in subset_df.columns:
                n_species = subset_df['gtdb_species'].nunique()
                print(f"  Final: {len(subset_df)} genomes (R:{final_r}, S:{final_s}) from {n_species} species")
            else:
                print(f"  Final: {len(subset_df)} genomes (R:{final_r}, S:{final_s})")
    
    return subsets

def main():
    parser = argparse.ArgumentParser(description='Extract properly species-balanced subsets using GTDB metadata with genome tracking')
    parser.add_argument('--metadata', 
                       default='/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/data/multi-drug_aug2024/data_splits/clustered_3_v1/metadata_collapsed_filt.tsv',
                       help='Metadata TSV file with gtdb_species')
    parser.add_argument('--kover_csv', 
                       default='/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-2_gentamicin_kover.csv',
                       help='Input kover CSV file')
    parser.add_argument('--output_dir',
                       default='/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data',
                       help='Output directory for subset CSVs')
    
    args = parser.parse_args()
    
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Metadata file: {args.metadata}")
    print(f"Kover CSV: {args.kover_csv}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup output directory structure for gentamicin
    gentamicin_dir = os.path.join(args.output_dir, 'natural_proportion_sampling/gentamicin')
    os.makedirs(gentamicin_dir, exist_ok=True)
    
    # Registry for tracking used genomes
    registry_path = os.path.join(gentamicin_dir, '._registry_used_genomes.csv')
    used_genomes, registry_df = load_used_genomes_registry(registry_path)
    
    # Get next subset number for gentamicin
    subset_number = get_next_subset_number_gentamicin(args.output_dir)
    print(f"Creating subset number: {subset_number}")
    
    # Load data
    merged_df = load_kover_data(args.kover_csv, args.metadata)
    
    # Analyze species distribution if available
    training_species = []
    if 'gtdb_species' in merged_df.columns:
        training_species = analyze_species_distribution(merged_df)
    else:
        print("\n=== No Species Information Available ===")
        print("Proceeding with partition-based extraction only")
    
    # Extract subsets using new strategy
    subsets = extract_kover_subsets(merged_df, used_genomes)
    
    # Save subsets in the same format as existing files
    print(f"\n=== Saving Subsets ===")
    
    all_subsets = []
    new_genomes_for_registry = []
    
    for name, subset_df in subsets.items():
        if len(subset_df) > 0:
            # Save with available columns
            base_cols = ['genome_id', 'partition_label', 'antibiotic_label']
            if 'gtdb_species' in subset_df.columns:
                output_df = subset_df[base_cols + ['gtdb_species']].copy()
            else:
                output_df = subset_df[base_cols].copy()
            all_subsets.append(output_df)
            
            # Add to registry tracking
            for _, row in output_df.iterrows():
                new_genomes_for_registry.append({
                    'genome_id': row['genome_id'],
                    'subset_name': f'subsets_{subset_number}',
                    'extraction_round': subset_number
                })
        else:
            print(f"  {name}: EMPTY - no genomes extracted")
    
    if all_subsets:
        # Combine all subsets into single file
        combined_df = pd.concat(all_subsets, ignore_index=True)
        
        # Save main subset file
        main_output_path = os.path.join(gentamicin_dir, f'subsets_{subset_number}.csv')
        combined_df.to_csv(main_output_path, index=False)
        
        total_r = len(combined_df[combined_df['antibiotic_label'] == 1])
        total_s = len(combined_df[combined_df['antibiotic_label'] == 0])
        print(f"\nMain file: {len(combined_df)} genomes (R:{total_r}, S:{total_s}) -> {main_output_path}")
        
        # Create report
        report = []
        for partition in combined_df['partition_label'].unique():
            partition_data = combined_df[combined_df['partition_label'] == partition]
            r_count = len(partition_data[partition_data['antibiotic_label'] == 1])
            s_count = len(partition_data[partition_data['antibiotic_label'] == 0])
            
            report_entry = {
                'partition_label': partition,
                'n_genomes': len(partition_data),
                'r_count': r_count,
                's_count': s_count,
                'r_proportion': r_count / len(partition_data) if len(partition_data) > 0 else 0
            }
            
            # Add species information if available
            if 'gtdb_species' in partition_data.columns:
                species_counts = partition_data['gtdb_species'].value_counts()
                species_list = '; '.join(species_counts.index.tolist())
                report_entry.update({
                    'n_species': len(species_counts),
                    'species_list': species_list
                })
            else:
                report_entry.update({
                    'n_species': 'N/A',
                    'species_list': 'No species information'
                })
            
            report.append(report_entry)
        
        # Save report
        report_df = pd.DataFrame(report)
        report_path = os.path.join(gentamicin_dir, f'subsets_{subset_number}_report.csv')
        report_df.to_csv(report_path, index=False)
        print(f"Report: {report_path}")
        
        # Update registry
        if new_genomes_for_registry:
            new_registry_entries = pd.DataFrame(new_genomes_for_registry)
            updated_registry = pd.concat([registry_df, new_registry_entries], ignore_index=True)
            save_used_genomes_registry(registry_path, updated_registry)
            print(f"Added {len(new_genomes_for_registry)} genomes to registry")
    
    else:
        print("\n⚠️  No genomes extracted - all may have been used already")
    
    print(f"\n✅ Gentamicin subset extraction completed for subset {subset_number}!")

if __name__ == "__main__":
    main()