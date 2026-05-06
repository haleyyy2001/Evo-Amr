#!/usr/bin/env python3
"""
Extract balanced subsets with focus on underrepresented categories
Modified from extract_proper_species_subsets.py to:
1. Prioritize sensitive samples in val_outside 
2. Maintain better R/S balance across all partitions
3. Extract ~1000 additional genomes total
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
        if basename.startswith('subsets_100_') and basename.endswith('.csv'):
            try:
                num_part = basename[12:-4]
                if num_part.isdigit():
                    numbers.append(int(num_part))
            except:
                continue
    
    return max(numbers) + 1 if numbers else 1

def load_metadata_and_kover(metadata_path, kover_path):
    """Load metadata and merge with kover data"""
    print("=== Loading Metadata and Kover Data ===")
    
    print(f"Loading metadata from: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    print(f"Loaded metadata: {len(metadata_df)} genomes")
    
    print(f"\nLoading kover data from: {kover_path}")
    kover_df = pd.read_csv(kover_path)
    print(f"Loaded kover data: {len(kover_df)} genomes")
    
    print(f"\nMerging data on genome_id...")
    merged_df = kover_df.merge(metadata_df[['genome_id', 'gtdb_species', 'ampicillin']], 
                              on='genome_id', how='inner')
    
    print(f"After merge: {len(merged_df)} genomes with species information")
    print(f"Lost {len(kover_df) - len(merged_df)} genomes in merge")
    
    return merged_df

def analyze_current_imbalance(merged_df, used_genomes):
    """Analyze current imbalance in the dataset"""
    print("\n=== Current Dataset Imbalance Analysis ===")
    
    # Get unused genomes
    available_df = merged_df[~merged_df['genome_id'].isin(used_genomes)].copy()
    
    print(f"Total available (unused) genomes: {len(available_df)}")
    
    # Analyze by partition and label
    imbalance_stats = []
    for partition in ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']:
        partition_data = available_df[available_df['partition_label'] == partition]
        if len(partition_data) > 0:
            resistant = (partition_data['antibiotic_label'] == 1).sum()
            sensitive = (partition_data['antibiotic_label'] == 0).sum()
            total = len(partition_data)
            rate = (resistant / total * 100) if total > 0 else 0
            
            imbalance_stats.append({
                'Partition': partition,
                'Available': total,
                'Resistant': resistant,
                'Sensitive': sensitive,
                'R_Rate': f"{rate:.1f}%"
            })
    
    imbalance_df = pd.DataFrame(imbalance_stats)
    print("\nAvailable genomes by partition:")
    print(imbalance_df.to_string(index=False))
    
    return available_df, imbalance_df

def calculate_balanced_targets(imbalance_df):
    """Calculate target samples to achieve balance"""
    print("\n=== Calculating Balanced Extraction Targets ===")
    
    # Define target extractions with focus on balance
    # Prioritize val_outside sensitive samples
    targets = {
        'train': {
            'total_target': 400,
            'sensitive_ratio': 0.35,  # Aim for 35% sensitive
            'priority': 2
        },
        'val_outside': {
            'total_target': 250,  # More samples for val_outside
            'sensitive_ratio': 0.40,  # Higher sensitive ratio
            'priority': 1  # Highest priority
        },
        'val_overlapped': {
            'total_target': 150,
            'sensitive_ratio': 0.35,
            'priority': 3
        },
        'test_outside': {
            'total_target': 150,
            'sensitive_ratio': 0.30,
            'priority': 4
        },
        'test_overlapped': {
            'total_target': 100,
            'sensitive_ratio': 0.35,
            'priority': 5
        }
    }
    
    extraction_plan = {}
    
    for _, row in imbalance_df.iterrows():
        partition = row['Partition']
        if partition in targets:
            target = targets[partition]
            available_sensitive = row['Sensitive']
            available_resistant = row['Resistant']
            
            # Calculate desired sensitive samples
            desired_sensitive = int(target['total_target'] * target['sensitive_ratio'])
            desired_resistant = target['total_target'] - desired_sensitive
            
            # Adjust based on availability
            actual_sensitive = min(desired_sensitive, available_sensitive)
            actual_resistant = min(desired_resistant, available_resistant)
            
            # Special handling for val_outside - prioritize getting more sensitive
            if partition == 'val_outside':
                # Try to get as many sensitive as possible
                actual_sensitive = min(available_sensitive, 100)  # Get up to 100 sensitive
                # Fill remainder with resistant
                remaining_budget = target['total_target'] - actual_sensitive
                actual_resistant = min(remaining_budget, available_resistant)
            
            extraction_plan[partition] = {
                'sensitive': actual_sensitive,
                'resistant': actual_resistant,
                'total': actual_sensitive + actual_resistant,
                'priority': target['priority']
            }
            
            print(f"\n{partition}:")
            print(f"  Available: R={available_resistant}, S={available_sensitive}")
            print(f"  Target: R={actual_resistant}, S={actual_sensitive}")
            print(f"  New ratio: {(actual_resistant/(actual_sensitive+actual_resistant)*100):.1f}% resistant")
    
    return extraction_plan

def extract_balanced_subsets_with_species(available_df, extraction_plan):
    """Extract balanced subsets considering species diversity"""
    print("\n=== Extracting Balanced Subsets ===")
    
    subsets = {}
    
    # Sort by priority
    sorted_partitions = sorted(extraction_plan.items(), key=lambda x: x[1]['priority'])
    
    for partition, plan in sorted_partitions:
        print(f"\n--- Extracting {partition} ---")
        
        partition_data = available_df[available_df['partition_label'] == partition].copy()
        
        if len(partition_data) == 0:
            print(f"  No available data for {partition}")
            subsets[partition] = pd.DataFrame()
            continue
        
        extracted_genomes = []
        
        # Extract sensitive samples
        if plan['sensitive'] > 0:
            sensitive_data = partition_data[partition_data['antibiotic_label'] == 0]
            
            if len(sensitive_data) > 0:
                # For val_outside, try to maximize species diversity
                if partition == 'val_outside':
                    # Group by species and sample proportionally
                    species_groups = sensitive_data.groupby('gtdb_species')
                    sensitive_samples = []
                    
                    # First, get at least 1 from each species if possible
                    for species, group in species_groups:
                        if len(sensitive_samples) < plan['sensitive']:
                            sample = group.sample(n=min(1, len(group)), random_state=RANDOM_SEED)
                            sensitive_samples.append(sample)
                    
                    # If we need more, sample randomly from remaining
                    if len(pd.concat(sensitive_samples)) < plan['sensitive']:
                        already_sampled = pd.concat(sensitive_samples)['genome_id'].tolist()
                        remaining = sensitive_data[~sensitive_data['genome_id'].isin(already_sampled)]
                        if len(remaining) > 0:
                            additional_needed = plan['sensitive'] - len(already_sampled)
                            additional = remaining.sample(n=min(additional_needed, len(remaining)), 
                                                        random_state=RANDOM_SEED)
                            sensitive_samples.append(additional)
                    
                    sensitive_selection = pd.concat(sensitive_samples, ignore_index=True)
                    sensitive_selection = sensitive_selection.head(plan['sensitive'])
                else:
                    # Regular sampling for other partitions
                    n_sensitive = min(plan['sensitive'], len(sensitive_data))
                    sensitive_selection = sensitive_data.sample(n=n_sensitive, random_state=RANDOM_SEED)
                
                extracted_genomes.append(sensitive_selection)
                print(f"  Extracted {len(sensitive_selection)} sensitive samples")
                
                # Show species diversity for val_outside
                if partition == 'val_outside':
                    species_count = sensitive_selection['gtdb_species'].nunique()
                    print(f"    From {species_count} different species")
        
        # Extract resistant samples
        if plan['resistant'] > 0:
            resistant_data = partition_data[partition_data['antibiotic_label'] == 1]
            
            if len(resistant_data) > 0:
                n_resistant = min(plan['resistant'], len(resistant_data))
                resistant_selection = resistant_data.sample(n=n_resistant, random_state=RANDOM_SEED)
                extracted_genomes.append(resistant_selection)
                print(f"  Extracted {len(resistant_selection)} resistant samples")
        
        # Combine
        if extracted_genomes:
            subset_df = pd.concat(extracted_genomes, ignore_index=True)
            subsets[partition] = subset_df
            
            # Summary
            total_r = (subset_df['antibiotic_label'] == 1).sum()
            total_s = (subset_df['antibiotic_label'] == 0).sum()
            species_count = subset_df['gtdb_species'].nunique()
            print(f"  Total: {len(subset_df)} genomes (R:{total_r}, S:{total_s}) from {species_count} species")
        else:
            subsets[partition] = pd.DataFrame()
            print(f"  No genomes extracted")
    
    return subsets

def main():
    parser = argparse.ArgumentParser(description='Extract balanced subsets with focus on underrepresented categories')
    parser.add_argument('--metadata', 
                       default='/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/data/multi-drug_aug2024/data_splits/clustered_3_v1/metadata_collapsed_filt.tsv',
                       help='Metadata TSV file with gtdb_species')
    parser.add_argument('--kover_csv', 
                       default='/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-1_ampicillin_kover.csv',
                       help='Input kover CSV file')
    parser.add_argument('--output_dir',
                       default='/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data',
                       help='Output directory for subset CSVs')
    parser.add_argument('--focus_balance', action='store_true', default=True,
                       help='Focus on balancing R/S labels especially for val_outside')
    
    args = parser.parse_args()
    
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Metadata file: {args.metadata}")
    print(f"Kover CSV: {args.kover_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Focus on balance: {args.focus_balance}")
    
    # Setup output directory
    ampicillin_dir = os.path.join(args.output_dir, 'natural_proportion_sampling/ampicillin')
    os.makedirs(ampicillin_dir, exist_ok=True)
    
    # Load registry
    registry_path = os.path.join(ampicillin_dir, '._registry_used_genomes.csv')
    used_genomes, registry_df = load_used_genomes_registry(registry_path)
    
    # Get next subset number
    subset_number = get_next_subset_number(args.output_dir)
    print(f"Creating subset number: {subset_number}")
    
    # Load and merge data
    merged_df = load_metadata_and_kover(args.metadata, args.kover_csv)
    
    # Analyze current imbalance
    available_df, imbalance_df = analyze_current_imbalance(merged_df, used_genomes)
    
    # Calculate balanced targets
    extraction_plan = calculate_balanced_targets(imbalance_df)
    
    # Extract balanced subsets
    subsets = extract_balanced_subsets_with_species(available_df, extraction_plan)
    
    # Save subsets
    print(f"\n=== Saving Balanced Subsets ===")
    
    all_subsets = []
    new_genomes_for_registry = []
    
    for name, subset_df in subsets.items():
        if len(subset_df) > 0:
            output_df = subset_df[['genome_id', 'partition_label', 'antibiotic_label', 'gtdb_species']].copy()
            all_subsets.append(output_df)
            
            for _, row in output_df.iterrows():
                new_genomes_for_registry.append({
                    'genome_id': row['genome_id'],
                    'subset_name': f'balanced_subsets_{subset_number}',
                    'extraction_round': subset_number
                })
    
    if all_subsets:
        # Combine all subsets
        combined_df = pd.concat(all_subsets, ignore_index=True)
        
        # Save main subset file  
        main_output_path = os.path.join(ampicillin_dir, f'subsets_100_{subset_number}.csv')
        combined_df.to_csv(main_output_path, index=False)
        
        print(f"\n=== Final Statistics ===")
        
        # Show final distribution
        for partition in combined_df['partition_label'].unique():
            partition_data = combined_df[combined_df['partition_label'] == partition]
            total = len(partition_data)
            resistant = (partition_data['antibiotic_label'] == 1).sum()
            sensitive = (partition_data['antibiotic_label'] == 0).sum()
            rate = (resistant / total * 100) if total > 0 else 0
            species_count = partition_data['gtdb_species'].nunique()
            
            print(f"{partition:20s}: {total:4d} total (R:{resistant:3d}, S:{sensitive:3d}) = {rate:5.1f}% resistant, {species_count} species")
        
        total_genomes = len(combined_df)
        total_r = (combined_df['antibiotic_label'] == 1).sum()
        total_s = (combined_df['antibiotic_label'] == 0).sum()
        overall_rate = (total_r / total_genomes * 100) if total_genomes > 0 else 0
        
        print(f"\nOverall: {total_genomes} genomes (R:{total_r}, S:{total_s}) = {overall_rate:.1f}% resistant")
        print(f"Saved to: {main_output_path}")
        
        # Create species report
        species_report = []
        for partition in combined_df['partition_label'].unique():
            partition_data = combined_df[combined_df['partition_label'] == partition]
            species_counts = partition_data['gtdb_species'].value_counts()
            species_list = '; '.join(species_counts.head(5).index.tolist())
            if len(species_counts) > 5:
                species_list += f'; ... and {len(species_counts)-5} more'
            
            r_count = (partition_data['antibiotic_label'] == 1).sum()
            s_count = (partition_data['antibiotic_label'] == 0).sum()
            
            species_report.append({
                'partition_label': partition,
                'n_genomes': len(partition_data),
                'n_species': len(species_counts),
                'r_count': r_count,
                's_count': s_count,
                'resistance_rate': f"{(r_count/(r_count+s_count)*100):.1f}%",
                'top_species': species_list
            })
        
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
        
        print(f"\n✅ Balanced extraction completed! Subset {subset_number} created with focus on underrepresented categories.")
    else:
        print("\n⚠️  No genomes extracted - check availability")

if __name__ == "__main__":
    main()