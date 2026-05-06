#!/usr/bin/env python3
"""
Create balanced subset 7 with focus on:
1. More randomly extracted train samples
2. ALL sensitive samples from val_outside partition
3. Maintain other partitions as needed
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

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

def get_all_training_species():
    """Get all species that were used in training across all previous extractions"""
    all_train_species = set()
    
    # Check all existing subset files
    for i in [1, 2, 3, 4]:
        subset_path = f"natural_proportion_sampling/ampicillin/subsets_100_{i}.csv"
        if os.path.exists(subset_path):
            df = pd.read_csv(subset_path)
            if 'gtdb_species' in df.columns:
                train_data = df[df['partition_label'] == 'train']
                species = set(train_data['gtdb_species'].unique())
                all_train_species.update(species)
                print(f"  Subset {i}: {len(species)} training species")
    
    print(f"Total unique training species from all extractions: {len(all_train_species)}")
    return list(all_train_species)

def load_metadata_and_kover(metadata_path, kover_path):
    """Load metadata and merge with kover data"""
    print("=== Loading Metadata and Kover Data ===")
    
    # Load metadata (has genome_id and gtdb_species)
    print(f"Loading metadata from: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    print(f"Loaded metadata: {len(metadata_df)} genomes")
    
    # Load kover data (has genome_id, partition_label, antibiotic_label)
    print(f"Loading kover data from: {kover_path}")
    kover_df = pd.read_csv(kover_path)
    print(f"Loaded kover data: {len(kover_df)} genomes")
    
    # Merge on genome_id
    print(f"Merging data on genome_id...")
    merged_df = kover_df.merge(metadata_df[['genome_id', 'gtdb_species', 'ampicillin']], 
                              on='genome_id', how='inner')
    
    print(f"After merge: {len(merged_df)} genomes with species information")
    return merged_df

def create_subset_7(df, training_species, used_genomes):
    """Create subset 7 with MUCH more training data and balanced sensitive samples for other partitions"""
    print(f"\n=== Creating Subset 7 (Balanced & More Training) ===")
    
    # Get remaining species for "outside" subsets
    all_species = set(df['gtdb_species'].unique())
    outside_species = list(all_species - set(training_species))
    print(f"Training species: {len(training_species)}")
    print(f"Outside species: {len(outside_species)}")
    
    subsets = {}
    
    # 1. Training set: MUCH MORE TRAINING DATA - we have 14,122 unused!
    print(f"\n--- Extracting MUCH MORE Training Data ---")
    train_data = df[
        (df['partition_label'] == 'train') &
        (~df['genome_id'].isin(used_genomes))
    ].copy()
    
    if len(train_data) > 0:
        # Sample a lot more training data - we have tons available
        target_train = min(1000, len(train_data))  # Get up to 1000 training samples!
        train_selection = train_data.sample(n=target_train, random_state=RANDOM_SEED)
        subsets['train'] = train_selection
        
        r_count = len(train_selection[train_selection['antibiotic_label'] == 1])
        s_count = len(train_selection[train_selection['antibiotic_label'] == 0])
        species_count = train_selection['gtdb_species'].nunique()
        print(f"  LARGE train sampling: {len(train_selection)} genomes (R:{r_count}, S:{s_count}) from {species_count} species")
    else:
        subsets['train'] = pd.DataFrame()
        print(f"  No training data available")
    
    # 2-5. OTHER PARTITIONS: Prioritize SENSITIVE samples to balance the dataset
    print(f"\n--- Extracting Other Partitions (PRIORITIZE SENSITIVE for Balance) ---")
    
    # Define partition strategies
    partition_strategies = [
        ('val_outside', 100, 'Get what we can (no sensitive left)'),
        ('test_outside', 36, 'Get all remaining'),  # We know there are 36 left
        ('val_overlapped', 200, 'Prioritize sensitive samples'),
        ('test_overlapped', 200, 'Prioritize sensitive samples')
    ]
    
    for partition_name, target_size, strategy in partition_strategies:
        print(f"\n  --- {partition_name} ({strategy}) ---")
        
        # Get all available data for this partition
        partition_data = df[
            (df['partition_label'] == partition_name) &
            (~df['genome_id'].isin(used_genomes))
        ].copy()
        
        if len(partition_data) == 0:
            subsets[partition_name] = pd.DataFrame()
            print(f"    {partition_name}: 0 genomes available")
            continue
        
        # Separate sensitive and resistant
        sensitive_data = partition_data[partition_data['antibiotic_label'] == 0]
        resistant_data = partition_data[partition_data['antibiotic_label'] == 1]
        
        print(f"    Available: {len(partition_data)} total (S:{len(sensitive_data)}, R:{len(resistant_data)})")
        
        selected_data = []
        
        # Strategy: Target 65:35 (resistant:sensitive) ratio for better balance
        if len(sensitive_data) > 0:
            # Target 35% sensitive (65% resistant)
            target_sensitive = int(target_size * 0.35)
            max_sensitive = min(len(sensitive_data), target_sensitive)
            if max_sensitive > 0:
                selected_sensitive = sensitive_data.sample(n=max_sensitive, random_state=RANDOM_SEED)
                selected_data.append(selected_sensitive)
                print(f"    Selected {len(selected_sensitive)} SENSITIVE samples (targeting 35%)")
        
        # Fill remaining with resistant (should be ~65%)
        remaining_target = target_size - (len(selected_data[0]) if selected_data else 0)
        if remaining_target > 0 and len(resistant_data) > 0:
            max_resistant = min(len(resistant_data), remaining_target)
            selected_resistant = resistant_data.sample(n=max_resistant, random_state=RANDOM_SEED)
            selected_data.append(selected_resistant)
            print(f"    Selected {len(selected_resistant)} resistant samples (targeting 65%)")
        
        # Combine selections
        if selected_data:
            subset_df = pd.concat(selected_data, ignore_index=True)
            subsets[partition_name] = subset_df
            
            r_count = len(subset_df[subset_df['antibiotic_label'] == 1])
            s_count = len(subset_df[subset_df['antibiotic_label'] == 0])
            species_count = subset_df['gtdb_species'].nunique()
            balance_pct = s_count / len(subset_df) * 100 if len(subset_df) > 0 else 0
            print(f"    FINAL {partition_name}: {len(subset_df)} genomes (S:{s_count}, R:{r_count}) - {balance_pct:.1f}% sensitive, {species_count} species")
        else:
            subsets[partition_name] = pd.DataFrame()
            print(f"    {partition_name}: 0 genomes selected")
    
    return subsets

def main():
    print(f"Random seed: {RANDOM_SEED}")
    
    # Setup paths
    ampicillin_dir = "natural_proportion_sampling/ampicillin"
    os.makedirs(ampicillin_dir, exist_ok=True)
    
    # Registry for tracking used genomes
    registry_path = os.path.join(ampicillin_dir, '._registry_used_genomes.csv')
    used_genomes, registry_df = load_used_genomes_registry(registry_path)
    
    # Get all training species from previous extractions
    print(f"\n=== Analyzing Previous Training Species ===")
    training_species = get_all_training_species()
    
    # Load and merge data
    metadata_path = '/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/data/multi-drug_aug2024/data_splits/clustered_3_v1/metadata_collapsed_filt.tsv'
    kover_path = '/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-1_ampicillin_kover.csv'
    
    merged_df = load_metadata_and_kover(metadata_path, kover_path)
    
    # Extract subset 7
    subsets = create_subset_7(merged_df, training_species, used_genomes)
    
    # Save subsets
    print(f"\n=== Saving Subset 7 ===")
    
    all_subsets = []
    new_genomes_for_registry = []
    
    for name, subset_df in subsets.items():
        if len(subset_df) > 0:
            all_subsets.append(subset_df)
            
            # Add to registry tracking
            for _, row in subset_df.iterrows():
                new_genomes_for_registry.append({
                    'genome_id': row['genome_id'],
                    'subset_name': 'subsets_100_7',
                    'extraction_round': 7
                })
    
    if all_subsets:
        # Combine all subsets into single file
        combined_df = pd.concat(all_subsets, ignore_index=True)
        
        # Save main subset file
        main_output_path = os.path.join(ampicillin_dir, 'subsets_100_7.csv')
        combined_df.to_csv(main_output_path, index=False)
        
        total_r = len(combined_df[combined_df['antibiotic_label'] == 1])
        total_s = len(combined_df[combined_df['antibiotic_label'] == 0])
        print(f"Main file: {len(combined_df)} genomes (R:{total_r}, S:{total_s}) -> {main_output_path}")
        
        # Create species report
        species_report = []
        for partition in combined_df['partition_label'].unique():
            partition_data = combined_df[combined_df['partition_label'] == partition]
            species_counts = partition_data['gtdb_species'].value_counts()
            species_list = '; '.join(species_counts.index.tolist())
            r_count = len(partition_data[partition_data['antibiotic_label'] == 1])
            s_count = len(partition_data[partition_data['antibiotic_label'] == 0])
            
            species_report.append({
                'partition_label': partition,
                'n_genomes': len(partition_data),
                'n_species': len(species_counts),
                'species_list': species_list,
                'r_count': r_count,
                's_count': s_count
            })
        
        # Save species report
        species_report_df = pd.DataFrame(species_report)
        species_report_path = os.path.join(ampicillin_dir, 'subsets_100_7_species_report.csv')
        species_report_df.to_csv(species_report_path, index=False)
        print(f"Species report: {species_report_path}")
        
        # Update registry
        if new_genomes_for_registry:
            new_registry_entries = pd.DataFrame(new_genomes_for_registry)
            updated_registry = pd.concat([registry_df, new_registry_entries], ignore_index=True)
            save_used_genomes_registry(registry_path, updated_registry)
            print(f"Added {len(new_genomes_for_registry)} genomes to registry")
        
        print(f"\n✅ Subset 7 extraction completed!")
        print(f"Focus: More random train samples + ALL sensitive val_outside")
    else:
        print(f"\n⚠️  No genomes extracted")

if __name__ == "__main__":
    main()