#!/usr/bin/env python3
"""
k-NN Ultra Analysis for ALL Partitions: val_outside, test_outside, val_overlapped, test_overlapped
===============================================================================================
Ultra comprehensive k-NN cosine similarity analysis across all data partitions.
Shows which training genomes are used to classify samples in each partition using k=20 cosine similarity.
Extends the original val_outside analysis to cover all four split types.

USAGE:
-----
1. Basic usage with default config:
   python knn_ultra_analysis_all_partitions.py --feature-type minirocket

2. With custom paths via environment variables:
   export AMR_DATA_DIR=/path/to/embeddings
   export AMR_OUTPUT_DIR=/path/to/results
   export AMR_LABELS_DIR=/path/to/labels
   python knn_ultra_analysis_all_partitions.py --feature-type minirocket

3. Analyze both feature types:
   python knn_ultra_analysis_all_partitions.py --feature-type both

4. Custom k value:
   python knn_ultra_analysis_all_partitions.py --feature-type minirocket --k 10

OUTPUT:
------
Generates .out files that can be analyzed by knn_species_analysis.py
Results include detailed neighbor analysis and accuracy metrics per partition.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import argparse
from datetime import datetime
import warnings
import sys
import os

# Add parent directories to path to import config
sys.path.append(str(Path(__file__).parent.parent.parent / 'utils'))
from config import get_config

warnings.filterwarnings('ignore')

def compute_6_stats(data, axis=0):
    """Compute 6 statistics: mean, std, max, min, q25, q75"""
    stats = []
    stats.append(np.mean(data, axis=axis))
    stats.append(np.std(data, axis=axis))
    stats.append(np.max(data, axis=axis))
    stats.append(np.min(data, axis=axis))
    stats.append(np.percentile(data, 25, axis=axis))
    stats.append(np.percentile(data, 75, axis=axis))
    return np.concatenate(stats)

def load_all_features(data_dir, sample_ids, feature_type="minirocket"):
    """Load all available features for given sample IDs"""
    features = []
    valid_ids = []
    failed_ids = []
    
    data_path = Path(data_dir)
    print(f"Loading {feature_type} features from: {data_path}")
    
    for i, sample_id in enumerate(sample_ids):
        if feature_type == "minirocket":
            data_file = data_path / f"{sample_id}_mr_win2048_s1024.npz"
            key = 'X'
        else:  # baseline/pca
            data_file = data_path / f"{sample_id}_pca_stats_dim41.npz"
            key = 'V'
        
        if not data_file.exists():
            failed_ids.append(sample_id)
            continue
            
        try:
            data = np.load(data_file)
            if key not in data:
                failed_ids.append(sample_id)
                continue
            
            if feature_type == "minirocket":
                # Apply 6-stat pooling for MiniRocket
                window_features = data[key]
                pooled_features = compute_6_stats(window_features, axis=0)
            else:
                # Use pre-pooled features for baseline
                pooled_features = data[key].flatten()
            
            features.append(pooled_features)
            valid_ids.append(sample_id)
            
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1} samples, loaded {len(features)} features")
                
        except Exception as e:
            failed_ids.append(sample_id)
            if len(failed_ids) <= 5:  # Only show first few errors
                print(f"  Error loading {sample_id}: {e}")
    
    print(f"Successfully loaded {len(features)} {feature_type} features")
    print(f"Failed to load {len(failed_ids)} samples")
    
    return np.array(features), valid_ids

def knn_ultra_analysis(features, labels, target_partition, k=20, verbose=True):
    """
    Perform k-NN cosine similarity analysis for any target partition vs train
    
    Args:
        features: Feature matrix
        labels: Labels DataFrame
        target_partition: The partition to analyze (e.g., 'val_outside', 'test_outside', etc.)
        k: Number of neighbors
        verbose: If True, print detailed neighbor information for each sample
    
    Returns:
        List of result dictionaries with neighbor details
    """
    
    # Separate target partition samples from train
    target_mask = labels['partition_label'] == target_partition
    train_mask = labels['partition_label'] == 'train'
    
    target_features = features[target_mask]
    train_features = features[train_mask]
    
    target_labels = labels[target_mask].copy()
    train_labels = labels[train_mask].copy()
    
    print(f"\nk-NN Analysis Setup for {target_partition}:")
    print(f"  {target_partition} samples: {len(target_features)}")
    print(f"  train samples: {len(train_features)}")
    print(f"  k neighbors: {k}")
    
    if len(target_features) == 0:
        print(f"No {target_partition} samples found!")
        return []
    
    if len(train_features) < k:
        print(f"Not enough train samples ({len(train_features)}) for k={k}")
        k = len(train_features)
        print(f"Reduced k to {k}")
    
    # Fit k-NN model on train data using cosine similarity
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(train_features)
    
    # Find neighbors for each target sample
    distances, neighbor_indices = knn.kneighbors(target_features)
    
    print(f"\n{'='*100}")
    print(f"k-NN COSINE (k={k}) ANALYSIS: Training Genomes Used for {target_partition}")
    print(f"{'='*100}")
    
    results = []
    
    for i, (target_genome, target_row) in enumerate(target_labels.iterrows()):
        target_genome_id = target_row['genome_id']
        target_species = target_row['species']
        target_amr = target_row['antibiotic_label']
        target_amr_label = 'R' if target_amr == 1 else 'S'
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"[{i+1}/{len(target_labels)}] {target_partition}: {target_genome_id}")
            print(f"Species: {target_species}, True AMR: {target_amr_label}")
            print(f"\n{k} Training Genomes Used:")
            print(f"{'#':<3} {'Training Genome':<25} {'Species':<12} {'AMR':<4} {'Sim':<7}")
            print(f"{'-'*55}")
        
        neighbor_data = []
        species_count = {}
        amr_count = {'S': 0, 'R': 0}
        
        for j, neighbor_idx in enumerate(neighbor_indices[i]):
            neighbor_row = train_labels.iloc[neighbor_idx]
            neighbor_id = neighbor_row['genome_id']
            neighbor_species = neighbor_row['species']
            neighbor_amr = neighbor_row['antibiotic_label']
            neighbor_amr_label = 'R' if neighbor_amr == 1 else 'S'
            
            # Convert cosine distance to similarity
            cosine_sim = 1 - distances[i][j]
            
            neighbor_data.append({
                'rank': j + 1,
                'genome_id': neighbor_id,
                'species': neighbor_species,
                'amr_label': neighbor_amr_label,
                'cosine_similarity': cosine_sim,
                'species_match': neighbor_species == target_species,
                'amr_match': neighbor_amr_label == target_amr_label
            })
            
            # Count species and AMR labels
            species_count[neighbor_species] = species_count.get(neighbor_species, 0) + 1
            amr_count[neighbor_amr_label] += 1
            
            if verbose:
                print(f"{j+1:<3} {neighbor_id:<25} {neighbor_species:<12} {neighbor_amr_label:<4} {cosine_sim:.4f}")
        
        # k-NN prediction (majority vote)
        predicted_amr = 'R' if amr_count['R'] > amr_count['S'] else 'S'
        is_correct = predicted_amr == target_amr_label
        
        if verbose:
            print(f"\nVotes: R={amr_count['R']}, S={amr_count['S']} → Prediction: {predicted_amr} {'✓ CORRECT' if is_correct else '✗ WRONG'}")
        
        # Store results
        results.append({
            'target_genome_id': target_genome_id,
            'target_species': target_species,
            'true_amr': target_amr_label,
            'predicted_amr': predicted_amr,
            'correct': is_correct,
            'neighbor_species': dict(species_count),
            'neighbor_amr': dict(amr_count),
            'neighbors': neighbor_data,
            'partition': target_partition
        })
    
    # Summary analysis
    print(f"\n{'='*100}")
    print(f"SUMMARY - {target_partition}")
    print(f"{'='*100}")
    
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    print(f"\nAccuracy: {correct_predictions}/{total_samples} ({accuracy:.1%})")
    
    # Most used training genomes
    genome_usage = {}
    for result in results:
        for neighbor in result['neighbors']:
            neighbor_id = neighbor['genome_id']
            if neighbor_id not in genome_usage:
                genome_usage[neighbor_id] = 0
            genome_usage[neighbor_id] += 1
    
    sorted_usage = sorted(genome_usage.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 30 Most Frequently Used Training Genomes:")
    print(f"{'Rank':<5} {'Training Genome':<30} {'Times Used':<12} {'Percentage'}")
    print(f"{'-'*60}")
    
    total_uses = len(results) * k
    for i, (genome_id, count) in enumerate(sorted_usage[:30]):
        percentage = count / total_uses * 100
        print(f"{i+1:<5} {genome_id:<30} {count:<12} {percentage:.1f}%")
    
    print(f"\nUsage Statistics:")
    print(f"  Total unique training genomes used: {len(genome_usage)}/{len(train_labels)}")
    print(f"  Average usage per genome: {np.mean(list(genome_usage.values())):.1f}")
    print(f"  Max usage: {max(genome_usage.values()) if genome_usage else 0}")
    print(f"  Min usage: {min(genome_usage.values()) if genome_usage else 0}")
    
    return results

def analyze_all_partitions(features, labels, feature_name, k=20, output_dir=None, verbose=False):
    """
    Analyze all partitions sequentially
    """
    target_partitions = ['val_outside', 'test_outside', 'val_overlapped', 'test_overlapped']
    all_results = {}
    
    print(f"\n{'🔬' * 50}")
    print(f"ULTRA k-NN ANALYSIS FOR ALL PARTITIONS - {feature_name}")
    print(f"{'🔬' * 50}")
    
    for partition in target_partitions:
        partition_results = knn_ultra_analysis(features, labels, partition, k, verbose)
        
        if partition_results:
            all_results[partition] = partition_results
            
            # Save individual partition results
            if output_dir:
                save_partition_results(partition_results, partition, feature_name, output_dir)
        else:
            print(f"⚠️  No results for {partition}")
    
    # Generate comparative summary
    if all_results:
        generate_comparative_summary(all_results, feature_name, output_dir)
    
    return all_results

def save_partition_results(results, partition, feature_name, output_dir):
    """Save results for a single partition"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed text output (similar to original format)
    text_file = output_dir / f"knn_{feature_name.lower()}_{partition}_{timestamp}.out"
    
    with open(text_file, 'w') as f:
        f.write(f"k-NN COSINE (k=20) ANALYSIS: Training Genomes Used for {partition} ({feature_name})\n")
        f.write("=" * 100 + "\n")
        
        # Write detailed results for each sample
        for i, result in enumerate(results):
            f.write(f"\n{'='*80}\n")
            f.write(f"[{i+1}/{len(results)}] {partition}: {result['target_genome_id']}\n")
            f.write(f"Species: {result['target_species']}, True AMR: {result['true_amr']}\n")
            f.write(f"\n20 Training Genomes Used:\n")
            f.write(f"{'#':<3} {'Training Genome':<25} {'Species':<12} {'AMR':<4} {'Sim':<7}\n")
            f.write(f"{'-'*55}\n")
            
            for neighbor in result['neighbors']:
                f.write(f"{neighbor['rank']:<3} {neighbor['genome_id']:<25} "
                       f"{neighbor['species']:<12} {neighbor['amr_label']:<4} "
                       f"{neighbor['cosine_similarity']:.4f}\n")
            
            votes_r = result['neighbor_amr']['R']
            votes_s = result['neighbor_amr']['S']
            status = "✓ CORRECT" if result['correct'] else "✗ WRONG"
            f.write(f"\nVotes: R={votes_r}, S={votes_s} → Prediction: {result['predicted_amr']} {status}\n")
        
        # Write summary
        total_samples = len(results)
        correct_predictions = sum(1 for r in results if r['correct'])
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        f.write(f"\n{'='*100}\n")
        f.write(f"SUMMARY - {partition}\n")
        f.write(f"{'='*100}\n")
        f.write(f"\nAccuracy: {correct_predictions}/{total_samples} ({accuracy:.1%})\n")
    
    print(f"📁 Saved {partition} results: {text_file}")
    
    # Save CSV summary
    csv_data = []
    for result in results:
        csv_data.append({
            'genome_id': result['target_genome_id'],
            'species': result['target_species'],
            'true_amr': result['true_amr'],
            'predicted_amr': result['predicted_amr'],
            'correct': result['correct'],
            'partition': partition,
            'feature_type': feature_name
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_file = output_dir / f"knn_{feature_name.lower()}_{partition}_{timestamp}.csv"
    csv_df.to_csv(csv_file, index=False)
    print(f"📁 Saved {partition} CSV: {csv_file}")

def generate_comparative_summary(all_results, feature_name, output_dir):
    """Generate comparative summary across all partitions"""
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Text summary
    summary_file = output_dir / f"knn_{feature_name.lower()}_comparative_summary_{timestamp}.txt"
    
    with open(summary_file, 'w') as f:
        f.write(f"🎯 COMPARATIVE SUMMARY - {feature_name} k-NN Analysis\n")
        f.write("=" * 80 + "\n")
        
        partition_stats = {}
        
        for partition, results in all_results.items():
            total = len(results)
            correct = sum(1 for r in results if r['correct'])
            accuracy = correct / total if total > 0 else 0
            
            partition_stats[partition] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            }
            
            f.write(f"\n{partition.upper()}:\n")
            f.write(f"  Samples: {total}\n")
            f.write(f"  Correct: {correct}\n")
            f.write(f"  Accuracy: {accuracy:.1%}\n")
        
        f.write(f"\nRANKING BY ACCURACY:\n")
        sorted_partitions = sorted(partition_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for i, (partition, stats) in enumerate(sorted_partitions):
            f.write(f"  {i+1}. {partition}: {stats['accuracy']:.1%} ({stats['correct']}/{stats['total']})\n")
    
    print(f"📊 Saved comparative summary: {summary_file}")
    
    # Create visualization
    create_summary_visualization(all_results, feature_name, output_dir, timestamp)

def create_summary_visualization(all_results, feature_name, output_dir, timestamp):
    """Create summary visualization comparing all partitions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{feature_name} k-NN Analysis: All Partitions Comparison', fontsize=16, fontweight='bold')
    
    partitions = list(all_results.keys())
    
    # 1. Accuracy comparison
    accuracies = []
    sample_counts = []
    
    for partition in partitions:
        results = all_results[partition]
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total if total > 0 else 0
        
        accuracies.append(accuracy * 100)
        sample_counts.append(total)
    
    bars = ax1.bar(partitions, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax1.set_title('Accuracy by Partition')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xlabel('Partition')
    ax1.set_ylim(0, 100)
    
    # Add accuracy labels on bars
    for bar, acc, count in zip(bars, accuracies, sample_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    # 2. Sample size comparison
    ax2.bar(partitions, sample_counts, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax2.set_title('Sample Sizes by Partition')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xlabel('Partition')
    
    # Add sample count labels
    for i, count in enumerate(sample_counts):
        ax2.text(i, count + max(sample_counts) * 0.01, str(count), ha='center', va='bottom')
    
    # 3. Species distribution
    all_species = set()
    for results in all_results.values():
        for result in results:
            all_species.add(result['target_species'])
    
    species_partition_matrix = np.zeros((len(all_species), len(partitions)))
    species_list = sorted(list(all_species))
    
    for j, partition in enumerate(partitions):
        results = all_results[partition]
        species_counts = {}
        for result in results:
            species = result['target_species']
            species_counts[species] = species_counts.get(species, 0) + 1
        
        for i, species in enumerate(species_list):
            species_partition_matrix[i, j] = species_counts.get(species, 0)
    
    im = ax3.imshow(species_partition_matrix, cmap='Blues', aspect='auto')
    ax3.set_title('Species Distribution Across Partitions')
    ax3.set_xlabel('Partition')
    ax3.set_ylabel('Species')
    ax3.set_xticks(range(len(partitions)))
    ax3.set_xticklabels(partitions, rotation=45)
    ax3.set_yticks(range(len(species_list)))
    ax3.set_yticklabels([s[:15] for s in species_list])  # Truncate long species names
    plt.colorbar(im, ax=ax3)
    
    # 4. AMR distribution
    amr_data = {partition: {'R': 0, 'S': 0} for partition in partitions}
    
    for partition, results in all_results.items():
        for result in results:
            amr_data[partition][result['true_amr']] += 1
    
    r_counts = [amr_data[p]['R'] for p in partitions]
    s_counts = [amr_data[p]['S'] for p in partitions]
    
    x = range(len(partitions))
    width = 0.35
    
    ax4.bar([i - width/2 for i in x], r_counts, width, label='Resistant', color='red', alpha=0.7)
    ax4.bar([i + width/2 for i in x], s_counts, width, label='Susceptible', color='blue', alpha=0.7)
    
    ax4.set_title('AMR Distribution by Partition')
    ax4.set_ylabel('Number of Samples')
    ax4.set_xlabel('Partition')
    ax4.set_xticks(x)
    ax4.set_xticklabels(partitions)
    ax4.legend()
    
    plt.tight_layout()
    
    plot_file = output_dir / f"knn_{feature_name.lower()}_summary_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved summary visualization: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description='Ultra k-NN Analysis for All Partitions')
    parser.add_argument('--labels', type=str, 
                       help='Path to labels CSV file (uses config default if not specified)')
    parser.add_argument('--mr-dir', type=str,
                       help='Directory containing MiniRocket features (uses config default if not specified)')
    parser.add_argument('--baseline-dir', type=str,
                       help='Directory containing baseline features (uses config default if not specified)')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of neighbors for k-NN (default: 20)')
    parser.add_argument('--output-dir', type=str, 
                       help='Directory for output files (uses config default if not specified)')
    parser.add_argument('--feature-type', type=str, choices=['minirocket', 'baseline', 'both'], 
                       default='both', help='Which features to analyze')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed neighbor information for each sample')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config()
    
    # Use config defaults if arguments not provided
    labels_path = args.labels if args.labels else config.get_label_files()[0] if config.get_label_files() else None
    mr_dir = args.mr_dir if args.mr_dir else config.data_dir
    baseline_dir = args.baseline_dir if args.baseline_dir else config.data_dir
    output_dir = args.output_dir if args.output_dir else config.get_output_path('knn_analysis')
    
    if not labels_path:
        print("❌ No labels file found. Please specify --labels or configure AMR_LABELS_DIR")
        sys.exit(1)
    
    print("🚀" * 50)
    print("k-NN ULTRA ANALYSIS FOR ALL PARTITIONS")
    print("🚀" * 50)
    print(f"Labels CSV: {labels_path}")
    print(f"MiniRocket dir: {mr_dir}")
    print(f"Baseline dir: {baseline_dir}")
    print(f"k neighbors: {args.k}")
    print(f"Output directory: {output_dir}")
    print(f"Feature type: {args.feature_type}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load labels
    print("\n📊 Loading labels...")
    labels_df = pd.read_csv(labels_path)
    labels_df['species'] = labels_df['genome_id'].apply(
        lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)[:10]
    )
    
    # Check partition distribution
    partition_dist = labels_df['partition_label'].value_counts()
    print(f"\nPartition distribution:")
    for partition, count in partition_dist.items():
        print(f"  {partition}: {count}")
    
    sample_ids = labels_df['genome_id'].astype(str).tolist()
    
    # Analyze MiniRocket features
    if args.feature_type in ['minirocket', 'both']:
        print(f"\n{'🔬' * 30}")
        print("MINIROCKET ULTRA ANALYSIS")
        print(f"{'🔬' * 30}")
        
        mr_features, mr_valid_ids = load_all_features(mr_dir, sample_ids, "minirocket")
        
        if len(mr_features) > 0:
            # Create matching labels DataFrame
            mr_labels_list = []
            for valid_id in mr_valid_ids:
                matching = labels_df[labels_df['genome_id'].astype(str) == str(valid_id)]
                if not matching.empty:
                    mr_labels_list.append(matching.iloc[0])
            mr_labels = pd.DataFrame(mr_labels_list)
            
            print(f"\nMiniRocket dataset loaded: {mr_features.shape}")
            analyze_all_partitions(mr_features, mr_labels, "MiniRocket", args.k, output_dir, args.verbose)
        else:
            print("❌ No MiniRocket features loaded!")
    
    # Analyze Baseline features
    if args.feature_type in ['baseline', 'both']:
        print(f"\n{'🎯' * 30}")
        print("BASELINE ULTRA ANALYSIS")
        print(f"{'🎯' * 30}")
        
        bl_features, bl_valid_ids = load_all_features(baseline_dir, sample_ids, "baseline")
        
        if len(bl_features) > 0:
            # Create matching labels DataFrame
            bl_labels_list = []
            for valid_id in bl_valid_ids:
                matching = labels_df[labels_df['genome_id'].astype(str) == str(valid_id)]
                if not matching.empty:
                    bl_labels_list.append(matching.iloc[0])
            bl_labels = pd.DataFrame(bl_labels_list)
            
            print(f"\nBaseline dataset loaded: {bl_features.shape}")
            analyze_all_partitions(bl_features, bl_labels, "Baseline", args.k, output_dir, args.verbose)
        else:
            print("❌ No Baseline features loaded!")
    
    print(f"\n{'🎉' * 50}")
    print("ULTRA k-NN ANALYSIS COMPLETE")
    print(f"{'🎉' * 50}")
    print(f"\n📁 All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()