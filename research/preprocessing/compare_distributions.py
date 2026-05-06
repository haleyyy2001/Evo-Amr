#!/usr/bin/env python3
"""
Analyze statistics of CSV files
Analyzes: basic stats, distributions, data quality
"""
import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import chi2_contingency, ks_2samp
import warnings
warnings.filterwarnings('ignore')

def load_all_current_samples():
    """Load all current sample files"""
    current_files = [
        '/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data/natural_proportion_sampling/ampicillin/subsets_100_1.csv',
        '/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data/natural_proportion_sampling/ampicillin/subsets_100_2.csv',
        '/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data/natural_proportion_sampling/ampicillin/subsets_100_3.csv',
        '/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data/natural_proportion_sampling/ampicillin/subsets_100_4.csv',
        '/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data/natural_proportion_sampling/ampicillin/subsets_100_5.csv',
        '/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data/natural_proportion_sampling/ampicillin/subsets_100_6.csv',
    ]
    
    dfs = []
    for file in current_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except:
            continue
    
    # Combine and deduplicate
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['genome_id'])
    return combined

def load_original_with_species():
    """Load original Kover data with species information"""
    # Load metadata for species
    metadata = pd.read_csv(
        '/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/data/multi-drug_aug2024/data_splits/clustered_3_v1/metadata_collapsed_filt.tsv',
        sep='\t'
    )
    
    # Load Kover data
    kover = pd.read_csv(
        '/insomnia001/depts/pmg/pmg_burg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-1_ampicillin_kover.csv'
    )
    
    # Merge to get species
    merged = kover.merge(metadata[['genome_id', 'gtdb_species']], on='genome_id', how='left')
    return merged

def compare_resistance_rates(current_df, original_df):
    """Compare resistance rates between datasets"""
    print("="*70)
    print("RESISTANCE RATE COMPARISON")
    print("="*70)
    
    results = []
    
    # Overall comparison
    curr_overall_r = (current_df['antibiotic_label'] == 1).mean() * 100
    orig_overall_r = (original_df['antibiotic_label'] == 1).mean() * 100
    
    print(f"\n{'Overall Resistance Rate':30s}")
    print(f"  Original: {orig_overall_r:6.2f}%")
    print(f"  Current:  {curr_overall_r:6.2f}%")
    print(f"  Difference: {curr_overall_r - orig_overall_r:+.2f}%")
    
    # By partition
    print(f"\n{'Partition':<20s} {'Original':>12s} {'Current':>12s} {'Difference':>12s}")
    print("-"*60)
    
    for partition in ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']:
        orig_part = original_df[original_df['partition_label'] == partition]
        curr_part = current_df[current_df['partition_label'] == partition]
        
        if len(orig_part) > 0:
            orig_rate = (orig_part['antibiotic_label'] == 1).mean() * 100
        else:
            orig_rate = 0
            
        if len(curr_part) > 0:
            curr_rate = (curr_part['antibiotic_label'] == 1).mean() * 100
        else:
            curr_rate = 0
        
        diff = curr_rate - orig_rate
        
        print(f"{partition:<20s} {orig_rate:11.2f}% {curr_rate:11.2f}% {diff:+11.2f}%")
        
        results.append({
            'partition': partition,
            'original_rate': orig_rate,
            'current_rate': curr_rate,
            'difference': diff
        })
    
    # Chi-square test for independence (simplified)
    print("\n" + "="*70)
    print("STATISTICAL TEST")
    
    # Create simple contingency table
    orig_r = (original_df['antibiotic_label'] == 1).sum()
    orig_s = (original_df['antibiotic_label'] == 0).sum()
    curr_r = (current_df['antibiotic_label'] == 1).sum()
    curr_s = (current_df['antibiotic_label'] == 0).sum()
    
    contingency = np.array([[orig_r, orig_s], [curr_r, curr_s]])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"Contingency table:")
    print(f"              Resistant  Sensitive")
    print(f"Original:     {orig_r:9d}  {orig_s:9d}")
    print(f"Current:      {curr_r:9d}  {curr_s:9d}")
    print(f"\nChi-square statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Distributions are SIGNIFICANTLY different (p < 0.05)")
    else:
        print("Result: Distributions are NOT significantly different (p >= 0.05)")
    
    return results

def compare_partition_proportions(current_df, original_df):
    """Compare partition size proportions"""
    print("\n" + "="*70)
    print("PARTITION SIZE PROPORTION COMPARISON")
    print("="*70)
    
    orig_props = original_df['partition_label'].value_counts(normalize=True) * 100
    curr_props = current_df['partition_label'].value_counts(normalize=True) * 100
    
    print(f"\n{'Partition':<20s} {'Original':>12s} {'Current':>12s} {'Difference':>12s}")
    print("-"*60)
    
    for partition in ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']:
        orig_prop = orig_props.get(partition, 0)
        curr_prop = curr_props.get(partition, 0)
        diff = curr_prop - orig_prop
        
        print(f"{partition:<20s} {orig_prop:11.2f}% {curr_prop:11.2f}% {diff:+11.2f}%")

def compare_species_diversity(current_df, original_df):
    """Compare species diversity between datasets"""
    print("\n" + "="*70)
    print("SPECIES DIVERSITY COMPARISON")
    print("="*70)
    
    # Filter to only rows with species information
    orig_with_species = original_df[original_df['gtdb_species'].notna()]
    curr_with_species = current_df[current_df['gtdb_species'].notna()]
    
    # Overall species count
    orig_species = set(orig_with_species['gtdb_species'].unique())
    curr_species = set(curr_with_species['gtdb_species'].unique())
    
    print(f"\nTotal unique species:")
    print(f"  Original: {len(orig_species)}")
    print(f"  Current:  {len(curr_species)}")
    print(f"  Coverage: {len(curr_species)/len(orig_species)*100:.1f}% of original species")
    
    # Top species comparison
    orig_top = orig_with_species['gtdb_species'].value_counts().head(10)
    curr_top = curr_with_species['gtdb_species'].value_counts().head(10)
    
    print(f"\nTop 10 species representation:")
    print(f"{'Species':<50s} {'Original %':>12s} {'Current %':>12s}")
    print("-"*75)
    
    all_top_species = set(orig_top.index) | set(curr_top.index)
    
    for species in sorted(all_top_species):
        orig_pct = orig_with_species[orig_with_species['gtdb_species'] == species].shape[0] / len(orig_with_species) * 100
        curr_pct = curr_with_species[curr_with_species['gtdb_species'] == species].shape[0] / len(curr_with_species) * 100
        
        # Truncate species name if too long
        species_display = species[:47] + "..." if len(species) > 50 else species
        print(f"{species_display:<50s} {orig_pct:11.2f}% {curr_pct:11.2f}%")
    
    # Species diversity by partition
    print(f"\nSpecies diversity by partition:")
    print(f"{'Partition':<20s} {'Orig Species':>15s} {'Curr Species':>15s} {'Coverage':>12s}")
    print("-"*65)
    
    for partition in ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']:
        orig_part_species = orig_with_species[orig_with_species['partition_label'] == partition]['gtdb_species'].nunique()
        curr_part_species = curr_with_species[curr_with_species['partition_label'] == partition]['gtdb_species'].nunique()
        
        if orig_part_species > 0:
            coverage = curr_part_species / orig_part_species * 100
        else:
            coverage = 0
        
        print(f"{partition:<20s} {orig_part_species:>15d} {curr_part_species:>15d} {coverage:11.1f}%")

def calculate_representation_score(current_df, original_df):
    """Calculate overall representation score"""
    print("\n" + "="*70)
    print("OVERALL REPRESENTATION SCORE")
    print("="*70)
    
    scores = []
    
    # 1. Resistance rate similarity (by partition)
    for partition in ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']:
        orig_part = original_df[original_df['partition_label'] == partition]
        curr_part = current_df[current_df['partition_label'] == partition]
        
        if len(orig_part) > 0 and len(curr_part) > 0:
            orig_rate = (orig_part['antibiotic_label'] == 1).mean()
            curr_rate = (curr_part['antibiotic_label'] == 1).mean()
            
            # Score based on difference (max 20 points per partition, 100 total)
            diff = abs(orig_rate - curr_rate)
            score = max(0, 20 * (1 - diff))  # 20 points if perfect match, 0 if 100% different
            scores.append(('resistance_' + partition, score))
    
    # 2. Partition proportion similarity (20 points)
    orig_props = original_df['partition_label'].value_counts(normalize=True)
    curr_props = current_df['partition_label'].value_counts(normalize=True)
    
    prop_diffs = []
    for partition in orig_props.index:
        if partition in curr_props.index:
            prop_diffs.append(abs(orig_props[partition] - curr_props[partition]))
    
    avg_prop_diff = np.mean(prop_diffs) if prop_diffs else 1
    prop_score = 20 * (1 - avg_prop_diff)
    scores.append(('partition_proportions', prop_score))
    
    # 3. Species coverage (20 points)
    orig_species = set(original_df[original_df['gtdb_species'].notna()]['gtdb_species'].unique())
    curr_species = set(current_df[current_df['gtdb_species'].notna()]['gtdb_species'].unique())
    
    if len(orig_species) > 0:
        species_coverage = len(curr_species) / len(orig_species)
        species_score = 20 * min(1, species_coverage)  # Cap at 100%
    else:
        species_score = 0
    scores.append(('species_coverage', species_score))
    
    # Calculate total
    total_score = sum(score for _, score in scores)
    max_score = 140  # 5 partitions * 20 + 20 + 20
    
    print("\nScoring breakdown:")
    for name, score in scores:
        print(f"  {name:<30s}: {score:6.2f} / 20.0")
    
    print(f"\nTotal Score: {total_score:.2f} / {max_score:.0f} ({total_score/max_score*100:.1f}%)")
    
    # Interpretation
    percentage = total_score / max_score * 100
    if percentage >= 90:
        interpretation = "EXCELLENT - Very representative of original dataset"
    elif percentage >= 75:
        interpretation = "GOOD - Reasonably representative with minor differences"
    elif percentage >= 60:
        interpretation = "MODERATE - Some significant differences from original"
    else:
        interpretation = "POOR - Major differences from original dataset"
    
    print(f"Interpretation: {interpretation}")

def analyze_partition_proportions(registry_file, kover_file):
    """Analyze partition label proportions for genomes in registry file"""
    print("="*70)
    print(f"PARTITION LABEL PROPORTION ANALYSIS")
    print(f"Registry file: {registry_file}")
    print(f"Kover file: {kover_file}")
    print("="*70)
    
    try:
        # Load the registry file
        print("Loading registry file...")
        registry_df = pd.read_csv(registry_file)
        print(f"Registry genomes: {len(registry_df):,}")
        
        # Load the Kover file with partition labels
        print("Loading Kover file with partition labels...")
        kover_df = pd.read_csv(kover_file)
        print(f"Kover genomes: {len(kover_df):,}")
        
        # Convert genome_id to string for consistent merging
        registry_df['genome_id'] = registry_df['genome_id'].astype(str)
        kover_df['genome_id'] = kover_df['genome_id'].astype(str)
        
        # Merge to get partition labels for registry genomes
        print("Merging datasets...")
        merged_df = registry_df.merge(kover_df[['genome_id', 'partition_label', 'antibiotic_label']], 
                                     on='genome_id', how='left')
        
        print(f"Successfully merged: {(merged_df['partition_label'].notna()).sum():,} genomes")
        print(f"Missing partition labels: {(merged_df['partition_label'].isna()).sum():,} genomes")
        
        # Filter to only genomes with partition labels
        labeled_df = merged_df[merged_df['partition_label'].notna()]
        
        if len(labeled_df) == 0:
            print("ERROR: No genomes found with partition labels!")
            return
        
        # Basic registry statistics
        print(f"\nREGISTRY FILE BASIC STATS:")
        print(f"Total rows: {len(registry_df):,}")
        print(f"Unique genome_ids: {registry_df['genome_id'].nunique():,}")
        print(f"Duplicates: {registry_df.duplicated().sum():,}")
        
        if 'subset_name' in registry_df.columns:
            print(f"\nSUBSET DISTRIBUTION:")
            subset_dist = registry_df['subset_name'].value_counts()
            for subset, count in subset_dist.items():
                pct = count / len(registry_df) * 100
                print(f"  {str(subset):<30}: {count:>8,} ({pct:>5.1f}%)")
        
        # Partition label analysis
        print(f"\nPARTITION LABEL DISTRIBUTION (for {len(labeled_df):,} labeled genomes):")
        print(f"{'Partition':<20} {'Count':>10} {'Percentage':>12}")
        print("-" * 45)
        
        partition_dist = labeled_df['partition_label'].value_counts()
        total_labeled = len(labeled_df)
        
        # Ensure we show all 5 partition types
        all_partitions = ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']
        
        for partition in all_partitions:
            count = partition_dist.get(partition, 0)
            pct = count / total_labeled * 100
            print(f"{partition:<20} {count:>10,} {pct:>11.2f}%")
        
        # Detailed resistance analysis by partition
        print(f"\nDETAILED RESISTANCE BREAKDOWN BY PARTITION:")
        print(f"{'Partition':<20} {'Total':>8} {'Sensitive (0)':>13} {'Resistant (1)':>14} {'Rate':>8}")
        print("-" * 68)
        
        for partition in all_partitions:
            part_data = labeled_df[labeled_df['partition_label'] == partition]
            if len(part_data) > 0:
                sensitive = (part_data['antibiotic_label'] == 0).sum()
                resistant = (part_data['antibiotic_label'] == 1).sum()
                rate = resistant / len(part_data) * 100
                print(f"{partition:<20} {len(part_data):>8,} {sensitive:>13,} {resistant:>14,} {rate:>7.1f}%")
            else:
                print(f"{partition:<20} {0:>8,} {0:>13,} {0:>14,} {0:>7.1f}%")
        
        # Overall resistance rate
        overall_sensitive = (labeled_df['antibiotic_label'] == 0).sum()
        overall_resistant = (labeled_df['antibiotic_label'] == 1).sum()
        overall_rate = overall_resistant / len(labeled_df) * 100
        
        print(f"\nOVERALL SUMMARY:")
        print(f"Total genomes: {len(labeled_df):,}")
        print(f"Sensitive (0): {overall_sensitive:,} ({100-overall_rate:.2f}%)")
        print(f"Resistant (1): {overall_resistant:,} ({overall_rate:.2f}%)")
        
        # Compare with full Kover dataset
        print(f"\nCOMPARISON WITH FULL KOVER DATASET:")
        kover_partition_dist = kover_df['partition_label'].value_counts()
        kover_total = len(kover_df)
        
        print(f"{'Partition':<20} {'Kover %':>10} {'Registry %':>12} {'Difference':>12}")
        print("-" * 58)
        
        for partition in all_partitions:
            kover_count = kover_partition_dist.get(partition, 0)
            kover_pct = kover_count / kover_total * 100
            
            registry_count = partition_dist.get(partition, 0)
            registry_pct = registry_count / total_labeled * 100
            
            diff = registry_pct - kover_pct
            print(f"{partition:<20} {kover_pct:>9.2f}% {registry_pct:>11.2f}% {diff:>+11.2f}%")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"Error analyzing files: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_all_ampicillin_subsets():
    """Analyze all subset files in the ampicillin directory and save CSV summary"""
    print("="*70)
    print("AMPICILLIN SUBSETS ANALYSIS")
    print("="*70)
    
    ampicillin_dir = "/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo_8/data/natural_proportion_sampling/ampicillin"
    registry_file = os.path.join(ampicillin_dir, "._registry_used_genomes.csv")
    kover_file = "/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-1_ampicillin_kover.csv"
    
    try:
        # Load kover data for partition labels
        print("Loading Kover data...")
        kover_df = pd.read_csv(kover_file)
        kover_df['genome_id'] = kover_df['genome_id'].astype(str)
        
        # Find all subset files
        import glob
        subset_files = glob.glob(os.path.join(ampicillin_dir, "subsets_*.csv"))
        subset_files.extend(glob.glob(os.path.join(ampicillin_dir, "balanced_*.csv")))
        
        print(f"Found {len(subset_files)} subset files:")
        for f in subset_files:
            print(f"  {os.path.basename(f)}")
        
        summary_data = []
        
        # Analyze each subset file
        for subset_file in sorted(subset_files):
            subset_name = os.path.basename(subset_file).replace('.csv', '')
            print(f"\n--- Analyzing {subset_name} ---")
            
            try:
                # Load subset data
                subset_df = pd.read_csv(subset_file)
                subset_df['genome_id'] = subset_df['genome_id'].astype(str)
                
                # Merge with kover to get partition labels if not present
                if 'partition_label' not in subset_df.columns:
                    subset_df = subset_df.merge(kover_df[['genome_id', 'partition_label', 'antibiotic_label']], 
                                              on='genome_id', how='left')
                
                # Analyze by partition
                all_partitions = ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']
                
                for partition in all_partitions:
                    partition_data = subset_df[subset_df['partition_label'] == partition]
                    
                    if len(partition_data) > 0:
                        sensitive_count = len(partition_data[partition_data['antibiotic_label'] == 0])
                        resistant_count = len(partition_data[partition_data['antibiotic_label'] == 1])
                        total_count = len(partition_data)
                        sensitive_pct = sensitive_count / total_count * 100 if total_count > 0 else 0
                        resistant_pct = resistant_count / total_count * 100 if total_count > 0 else 0
                        
                        summary_data.append({
                            'subset_name': subset_name,
                            'partition_label': partition,
                            'total_count': total_count,
                            'sensitive_0_count': sensitive_count,
                            'resistant_1_count': resistant_count,
                            'sensitive_0_percentage': round(sensitive_pct, 2),
                            'resistant_1_percentage': round(resistant_pct, 2)
                        })
                        
                        print(f"  {partition}: {total_count} total (S:{sensitive_count}, R:{resistant_count}) - {resistant_pct:.1f}% resistant")
                    else:
                        # Add empty entry for missing partitions
                        summary_data.append({
                            'subset_name': subset_name,
                            'partition_label': partition,
                            'total_count': 0,
                            'sensitive_0_count': 0,
                            'resistant_1_count': 0,
                            'sensitive_0_percentage': 0.0,
                            'resistant_1_percentage': 0.0
                        })
                
            except Exception as e:
                print(f"  Error analyzing {subset_name}: {str(e)}")
        
        # Save individual summary to CSV
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            output_file = os.path.join(ampicillin_dir, "partition_analysis_summary.csv")
            summary_df.to_csv(output_file, index=False)
            print(f"\n✅ Individual summary saved to: {output_file}")
            
            # AGGREGATE ALL SUBSETS TOGETHER
            print(f"\n" + "="*70)
            print("COMBINED TOTALS ACROSS ALL SUBSETS")
            print("="*70)
            
            # Aggregate by partition across all subsets
            combined_data = []
            all_partitions = ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']
            
            for partition in all_partitions:
                partition_records = summary_df[summary_df['partition_label'] == partition]
                
                total_count = partition_records['total_count'].sum()
                total_sensitive = partition_records['sensitive_0_count'].sum()
                total_resistant = partition_records['resistant_1_count'].sum()
                
                if total_count > 0:
                    sensitive_pct = total_sensitive / total_count * 100
                    resistant_pct = total_resistant / total_count * 100
                else:
                    sensitive_pct = 0
                    resistant_pct = 0
                
                combined_data.append({
                    'partition_label': partition,
                    'total_count': total_count,
                    'sensitive_0_count': total_sensitive,
                    'resistant_1_count': total_resistant,
                    'sensitive_0_percentage': round(sensitive_pct, 2),
                    'resistant_1_percentage': round(resistant_pct, 2)
                })
                
                print(f"{partition:<20}: {total_count:>6} total (S:{total_sensitive:>4}, R:{total_resistant:>4}) - {resistant_pct:>5.1f}% resistant")
            
            # Save combined totals to CSV
            combined_df = pd.DataFrame(combined_data)
            combined_output_file = os.path.join(ampicillin_dir, "combined_totals_analysis.csv")
            combined_df.to_csv(combined_output_file, index=False)
            print(f"\n✅ Combined totals saved to: {combined_output_file}")
            
            # Overall totals across ALL partitions
            grand_total_count = combined_df['total_count'].sum()
            grand_total_sensitive = combined_df['sensitive_0_count'].sum()
            grand_total_resistant = combined_df['resistant_1_count'].sum()
            
            if grand_total_count > 0:
                grand_sensitive_pct = grand_total_sensitive / grand_total_count * 100
                grand_resistant_pct = grand_total_resistant / grand_total_count * 100
            else:
                grand_sensitive_pct = 0
                grand_resistant_pct = 0
            
            print(f"\nGRAND TOTALS (All Partitions Combined):")
            print(f"Total genomes extracted: {grand_total_count:,}")
            print(f"Sensitive (0): {grand_total_sensitive:,} ({grand_sensitive_pct:.2f}%)")
            print(f"Resistant (1): {grand_total_resistant:,} ({grand_resistant_pct:.2f}%)")
            
            # Show individual subset summary
            print(f"\nSUMMARY BY INDIVIDUAL SUBSET:")
            for subset in summary_df['subset_name'].unique():
                subset_data = summary_df[summary_df['subset_name'] == subset]
                total_genomes = subset_data['total_count'].sum()
                total_resistant = subset_data['resistant_1_count'].sum()
                total_sensitive = subset_data['sensitive_0_count'].sum()
                if total_genomes > 0:
                    resistance_rate = total_resistant / total_genomes * 100
                    print(f"  {subset}: {total_genomes} genomes (R:{total_resistant}, S:{total_sensitive}) - {resistance_rate:.1f}% resistant")
        else:
            print("\n⚠️  No data to save")
            
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Auto-analyze all ampicillin subsets
    analyze_all_ampicillin_subsets()

if __name__ == "__main__":
    main()