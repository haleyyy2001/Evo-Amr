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

def load_metadata_and_kover(metadata_path, kover_path):
    """Load metadata and merge with kover data"""
    print("=== Loading Metadata and Kover Data ===")
    
    # Load metadata (has genome_id and gtdb_species)
    print(f"Loading metadata from: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    print(f"Loaded metadata: {len(metadata_df)} genomes")
    print(f"Metadata columns include: genome_id, gtdb_species, ampicillin")
    
    # Load kover data (has genome_id, partition_label, antibiotic_label)
    print(f"\nLoading kover data from: {kover_path}")
    kover_df = pd.read_csv(kover_path)
    print(f"Loaded kover data: {len(kover_df)} genomes")
    
    # Merge on genome_id
    print(f"\nMerging data on genome_id...")
    merged_df = kover_df.merge(metadata_df[['genome_id', 'gtdb_species', 'ampicillin']], 
                              on='genome_id', how='inner')
    
    print(f"After merge: {len(merged_df)} genomes with species information")
    print(f"Lost {len(kover_df) - len(merged_df)} genomes in merge")
    
    # Check ampicillin consistency
    # kover antibiotic_label should match metadata ampicillin column
    if 'ampicillin' in merged_df.columns:
        # Convert boolean to int if needed
        if merged_df['ampicillin'].dtype == 'bool':
            merged_df['ampicillin_meta'] = merged_df['ampicillin'].astype(int)
        else:
            merged_df['ampicillin_meta'] = merged_df['ampicillin']
        
        # Check consistency
        consistent = merged_df['antibiotic_label'] == merged_df['ampicillin_meta']
        print(f"Ampicillin label consistency: {consistent.sum()}/{len(merged_df)} ({consistent.mean()*100:.1f}%)")
    
    return merged_df

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

def extract_species_balanced_subsets(df, training_species, used_genomes):
    """Extract subsets with proper species strategy"""
    print(f"\n=== Extracting Species-Balanced Subsets ===")
    
    print(f"Available training species ({len(training_species)}):")
    for i, species in enumerate(training_species[:10], 1):  # Show first 10 for display
        train_count = len(df[(df['gtdb_species'] == species) & (df['partition_label'] == 'train')])
        print(f"  {i}. {species}: {train_count} training genomes")
    if len(training_species) > 10:
        print(f"  ... and {len(training_species)-10} more species")
    
    # Get remaining species for "outside" subsets
    all_species = set(df['gtdb_species'].unique())
    outside_species = list(all_species - set(training_species))
    print(f"\nRemaining species for 'outside' subsets: {len(outside_species)}")
    
    subsets = {}
    
    # 1. Training set: balanced across top species, excluding used genomes
    print(f"\n--- Extracting Training Set ---")
    train_genomes = []
    target_total_train = 100  # Target 100 training genomes total
    target_per_species = max(1, target_total_train // min(len(training_species), 20))  # Distribute across species
    
    for species in training_species:
        species_train = df[
            (df['gtdb_species'] == species) & 
            (df['partition_label'] == 'train') &
            (~df['genome_id'].isin(used_genomes))  # Exclude already used genomes
        ].copy()
        
        if len(species_train) == 0:
            print(f"  Warning: No unused training genomes for {species}")
            continue
        
        # Random sampling without considering R/S balance
        n_to_sample = min(target_per_species, len(species_train))
        species_selection = species_train.sample(n=n_to_sample, random_state=RANDOM_SEED)
        if len(species_selection) > 0:
            train_genomes.append(species_selection)
            r_count = len(species_selection[species_selection['antibiotic_label'] == 1])
            s_count = len(species_selection[species_selection['antibiotic_label'] == 0])
            print(f"  {species}: {len(species_selection)} genomes (R:{r_count}, S:{s_count})")
    
    if train_genomes:
        subsets['train'] = pd.concat(train_genomes, ignore_index=True)
        print(f"  Total training: {len(subsets['train'])} genomes")
    else:
        subsets['train'] = pd.DataFrame()
        print("  Warning: No training genomes extracted!")
    
    # Get species that were actually selected for training
    actual_training_species = set()
    if len(subsets['train']) > 0:
        actual_training_species = set(subsets['train']['gtdb_species'].unique())
        print(f"  Actually selected training species: {len(actual_training_species)} species")
    
    # 2. Overlap subsets: from same species as actually selected training genomes
    print(f"\n--- Extracting Overlap Subsets ---")
    
    for subset_name, partition in [('test_overlapped', 'test_overlapped'), ('val_overlapped', 'val_overlapped')]:
        overlap_genomes = []
        
        for species in actual_training_species:
            species_data = df[
                (df['gtdb_species'] == species) & 
                (df['partition_label'] == partition) &
                (~df['genome_id'].isin(used_genomes))  # Exclude already used genomes
            ].copy()
            
            if len(species_data) > 0:
                # Sample without label constraints as requested
                n_to_sample = min(2, len(species_data))  # Get up to 2 per species
                species_selection = species_data.sample(n=n_to_sample, random_state=RANDOM_SEED)
                if len(species_selection) > 0:
                    overlap_genomes.append(species_selection)
        
        if overlap_genomes:
            subset_df = pd.concat(overlap_genomes, ignore_index=True)
            
            # Increase to ~75 genomes per overlap subset (150 total for both)
            target_overlap = 75
            if len(subset_df) > target_overlap:
                subset_df = subset_df.sample(n=target_overlap, random_state=RANDOM_SEED)
            
            subsets[subset_name] = subset_df
        else:
            subsets[subset_name] = pd.DataFrame()
        
        print(f"  {subset_name}: {len(subsets[subset_name])} genomes")
    
    # 3. Outside subsets: from different species
    print(f"\n--- Extracting Outside Subsets ---")
    
    for subset_name, partition in [('test_outside', 'test_outside'), ('val_outside', 'val_outside')]:
        outside_data = df[
            (df['gtdb_species'].isin(outside_species)) & 
            (df['partition_label'] == partition) &
            (~df['genome_id'].isin(used_genomes))  # Exclude already used genomes
        ].copy()
        
        if len(outside_data) > 0:
            # Random sampling from outside species without label constraints
            # Increase to ~75 genomes per outside subset (150 total for both)
            target_outside = 75
            n_to_sample = min(target_outside, len(outside_data))
            subset_df = outside_data.sample(n=n_to_sample, random_state=RANDOM_SEED)
            subsets[subset_name] = subset_df
            
            # Show species diversity in outside sets
            if len(subset_df) > 0:
                outside_species_used = subset_df['gtdb_species'].value_counts()
                r_count = len(subset_df[subset_df['antibiotic_label'] == 1])
                s_count = len(subset_df[subset_df['antibiotic_label'] == 0])
                print(f"  {subset_name}: {len(subset_df)} genomes (R:{r_count}, S:{s_count}) from {len(outside_species_used)} species")
                for species, count in outside_species_used.head(3).items():
                    print(f"    {species}: {count} genomes")
        else:
            subsets[subset_name] = pd.DataFrame()
            print(f"  {subset_name}: 0 genomes (no data in partition)")
    
    return subsets

def main():
    parser = argparse.ArgumentParser(description='Extract properly species-balanced subsets using GTDB metadata with genome tracking')
    parser.add_argument('--metadata', 
                       default='/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/data/multi-drug_aug2024/data_splits/clustered_3_v1/metadata_collapsed_filt.tsv',
                       help='Metadata TSV file with gtdb_species')
    parser.add_argument('--kover_csv', 
                       default='/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-1_ampicillin_kover.csv',
                       help='Input kover CSV file')
    parser.add_argument('--output_dir',
                       default='/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data',
                       help='Output directory for subset CSVs')
    
    args = parser.parse_args()
    
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Metadata file: {args.metadata}")
    print(f"Kover CSV: {args.kover_csv}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup output directory structure
    ampicillin_dir = os.path.join(args.output_dir, 'natural_proportion_sampling/ampicillin')
    os.makedirs(ampicillin_dir, exist_ok=True)
    
    # Registry for tracking used genomes
    registry_path = os.path.join(ampicillin_dir, '._registry_used_genomes.csv')
    used_genomes, registry_df = load_used_genomes_registry(registry_path)
    
    # Get next subset number
    subset_number = get_next_subset_number(args.output_dir)
    print(f"Creating subset number: {subset_number}")
    
    # Load and merge data
    merged_df = load_metadata_and_kover(args.metadata, args.kover_csv)
    
    # Analyze species distribution
    training_species = analyze_species_distribution(merged_df)
    
    # Extract subsets
    subsets = extract_species_balanced_subsets(merged_df, training_species, used_genomes)
    
    # Save subsets in the same format as existing files
    print(f"\n=== Saving Subsets ===")
    
    all_subsets = []
    new_genomes_for_registry = []
    
    for name, subset_df in subsets.items():
        if len(subset_df) > 0:
            # Save with gtdb_species column to match existing format
            output_df = subset_df[['genome_id', 'partition_label', 'antibiotic_label', 'gtdb_species']].copy()
            all_subsets.append(output_df)
            
            # Add to registry tracking
            for _, row in output_df.iterrows():
                new_genomes_for_registry.append({
                    'genome_id': row['genome_id'],
                    'subset_name': f'subsets_100_{subset_number}',
                    'extraction_round': subset_number
                })
        else:
            print(f"  {name}: EMPTY - no genomes extracted")
    
    if all_subsets:
        # Combine all subsets into single file
        combined_df = pd.concat(all_subsets, ignore_index=True)
        
        # Save main subset file
        main_output_path = os.path.join(ampicillin_dir, f'subsets_100_{subset_number}.csv')
        combined_df.to_csv(main_output_path, index=False)
        
        total_r = len(combined_df[combined_df['antibiotic_label'] == 1])
        total_s = len(combined_df[combined_df['antibiotic_label'] == 0])
        print(f"\nMain file: {len(combined_df)} genomes (R:{total_r}, S:{total_s}) -> {main_output_path}")
        
        # Create species report matching existing format
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
        species_report_path = os.path.join(ampicillin_dir, f'subsets_100_{subset_number}_species_report.csv')
        species_report_df.to_csv(species_report_path, index=False)
        print(f"Species report: {species_report_path}")
        
        # Update registry
        if new_genomes_for_registry:
            new_registry_entries = pd.DataFrame(new_genomes_for_registry)
            updated_registry = pd.concat([registry_df, new_registry_entries], ignore_index=True)
            save_used_genomes_registry(registry_path, updated_registry)
            print(f"Added {len(new_genomes_for_registry)} genomes to registry")
    
    else:
        print("\n⚠️  No genomes extracted - all may have been used already")
    
    print(f"\n✅ Natural proportion sampling extraction completed for subset {subset_number}!")

if __name__ == "__main__":
    main()