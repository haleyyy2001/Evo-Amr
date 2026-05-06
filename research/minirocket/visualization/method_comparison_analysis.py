#!/insomnia001/depts/pmg/users/ht2666/mambaforge/envs/evo-amr/bin/python
"""
Comprehensive Detailed Analysis of MiniRocket vs Baseline Features
================================================================
Detailed visualization comparing MiniRocket (6-stat pooled) vs Baseline (PCA) features
with clustering analysis, species patterns, and statistical comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, fallback if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

plt.style.use('default')
sns.set_palette("husl")

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

def load_minirocket_features(mr_dir, sample_ids, max_samples=200):
    """Load MiniRocket features and apply 6-stat pooling"""
    features = []
    valid_ids = []
    
    mr_path = Path(mr_dir)
    print(f"Loading MiniRocket features from: {mr_path}")
    
    for i, sample_id in enumerate(sample_ids[:max_samples]):
        mr_file = mr_path / f"{sample_id}_mr_win2048_s1024.npz"
        if not mr_file.exists():
            continue
            
        try:
            data = np.load(mr_file)
            if 'X' not in data:
                continue
            
            window_features = data['X']
            pooled_features = compute_6_stats(window_features, axis=0)
            
            features.append(pooled_features)
            valid_ids.append(sample_id)
            
            if len(features) % 25 == 0:
                print(f"  Loaded {len(features)} MiniRocket samples...")
                
        except Exception as e:
            if len(features) < 5:  # Only show first few errors
                print(f"  Error loading {sample_id}: {e}")
            continue
    
    return np.array(features), valid_ids

def load_baseline_features(pca_dir, sample_ids, max_samples=200):
    """Load baseline (PCA) features"""
    features = []
    valid_ids = []
    
    pca_path = Path(pca_dir)
    print(f"Loading baseline (PCA) features from: {pca_path}")
    
    for i, sample_id in enumerate(sample_ids[:max_samples]):
        pca_file = pca_path / f"{sample_id}_pca_stats_dim41.npz"
        if not pca_file.exists():
            continue
            
        try:
            data = np.load(pca_file)
            if 'V' not in data:
                continue
            
            pooled_features = data['V'].flatten()
            features.append(pooled_features)
            valid_ids.append(sample_id)
            
            if len(features) % 25 == 0:
                print(f"  Loaded {len(features)} baseline samples...")
                
        except Exception as e:
            if len(features) < 5:
                print(f"  Error loading {sample_id}: {e}")
            continue
    
    return np.array(features), valid_ids

def create_dimensionality_reduction_plots(features, labels, title, output_path):
    """Create comprehensive dimensionality reduction plots"""
    
    # Create subplots
    if HAS_UMAP:
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    fig.suptitle(f'{title} - Dimensionality Reduction Analysis', fontsize=16)
    
    # PCA Analysis
    if features.shape[0] > 2 and features.shape[1] > 2:
        pca = PCA(n_components=min(3, features.shape[0], features.shape[1]))
        features_pca = pca.fit_transform(features)
        
        # PCA - AMR coloring
        ax = axes[0, 0]
        if 'antibiotic_label' in labels.columns:
            scatter = ax.scatter(features_pca[:, 0], features_pca[:, 1], 
                               c=labels['antibiotic_label'], cmap='RdYlBu', 
                               alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA - AMR Labels')
        ax.grid(True, alpha=0.3)
        
        # PCA - Species coloring
        ax = axes[0, 1]
        if 'species' in labels.columns:
            unique_species = labels['species'].value_counts().head(10).index
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_species)))
            for i, species in enumerate(unique_species):
                mask = labels['species'] == species
                if mask.sum() > 0:
                    ax.scatter(features_pca[mask, 0], features_pca[mask, 1], 
                             color=colors[i], label=species, alpha=0.7, s=50,
                             edgecolors='black', linewidth=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('PCA - Species')
        ax.grid(True, alpha=0.3)
        
        # t-SNE Analysis
        if features.shape[0] > 5:
            ax = axes[1, 0]
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
                features_tsne = tsne.fit_transform(features)
                
                if 'antibiotic_label' in labels.columns:
                    scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], 
                                       c=labels['antibiotic_label'], cmap='RdYlBu', 
                                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
                    plt.colorbar(scatter, ax=ax)
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_title('t-SNE - AMR Labels')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f't-SNE failed: {str(e)[:30]}...', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('t-SNE - Failed')
        
        # UMAP Analysis (if available)
        if HAS_UMAP and features.shape[0] > 5:
            ax = axes[1, 1]
            try:
                umap_reducer = UMAP(n_components=2, random_state=42, 
                                  n_neighbors=min(15, features.shape[0]-1))
                features_umap = umap_reducer.fit_transform(features)
                
                if 'antibiotic_label' in labels.columns:
                    scatter = ax.scatter(features_umap[:, 0], features_umap[:, 1], 
                                       c=labels['antibiotic_label'], cmap='RdYlBu', 
                                       alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
                    plt.colorbar(scatter, ax=ax)
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.set_title('UMAP - AMR Labels')
                ax.grid(True, alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f'UMAP failed: {str(e)[:30]}...', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('UMAP - Failed')
        elif not HAS_UMAP:
            # Additional PCA variance plot
            ax = axes[1, 1] if not HAS_UMAP else axes[1, 2]
            n_components = min(20, features.shape[1], features.shape[0])
            pca_full = PCA(n_components=n_components)
            pca_full.fit(features)
            
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            ax.plot(range(1, len(cumvar)+1), cumvar, 'bo-', markersize=4)
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% variance')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.set_title('PCA Variance Explained')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Clustering analysis
        if HAS_UMAP:
            ax = axes[1, 2]
        else:
            ax = axes[0, 2] if features.shape[1] > 2 else axes[1, 0]
            
        if features.shape[0] > 3:
            # K-means clustering
            silhouette_scores = []
            k_range = range(2, min(8, features.shape[0]))
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(features)
                    score = silhouette_score(features, cluster_labels)
                    silhouette_scores.append(score)
                except:
                    silhouette_scores.append(0)
            
            if silhouette_scores:
                ax.plot(k_range, silhouette_scores, 'bo-', markersize=6)
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel('Silhouette Score')
                ax.set_title('Clustering Analysis')
                ax.grid(True, alpha=0.3)
                
                # Highlight best k
                best_k = k_range[np.argmax(silhouette_scores)]
                ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, 
                          label=f'Best k={best_k}')
                ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_feature_analysis_plots(features, labels, title, output_path):
    """Create detailed feature analysis plots"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'{title} - Feature Analysis', fontsize=16)
    
    # 1. Feature distributions by AMR
    ax = axes[0, 0]
    if 'antibiotic_label' in labels.columns:
        for amr_val in [0, 1]:
            mask = labels['antibiotic_label'] == amr_val
            if mask.sum() > 0:
                label = 'Resistant' if amr_val == 1 else 'Susceptible'
                feature_means = features[mask].mean(axis=0)
                ax.hist(feature_means, bins=30, alpha=0.6, label=label, density=True)
        ax.set_xlabel('Mean Feature Value')
        ax.set_ylabel('Density')
        ax.set_title('Feature Value Distributions by AMR')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Sample norms by AMR
    ax = axes[0, 1]
    sample_norms = np.linalg.norm(features, axis=1)
    if 'antibiotic_label' in labels.columns:
        for amr_val in [0, 1]:
            mask = labels['antibiotic_label'] == amr_val
            if mask.sum() > 0:
                label = 'Resistant' if amr_val == 1 else 'Susceptible'
                ax.hist(sample_norms[mask], bins=20, alpha=0.6, label=label, density=True)
        ax.set_xlabel('Sample L2 Norm')
        ax.set_ylabel('Density')
        ax.set_title('Sample Norms by AMR')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Feature variance distribution
    ax = axes[0, 2]
    feature_vars = features.var(axis=0)
    ax.hist(feature_vars, bins=40, alpha=0.7, edgecolor='black')
    ax.axvline(np.median(feature_vars), color='red', linestyle='--', 
              label=f'Median: {np.median(feature_vars):.4f}')
    ax.axvline(np.mean(feature_vars), color='orange', linestyle='--', 
              label=f'Mean: {np.mean(feature_vars):.4f}')
    ax.set_xlabel('Feature Variance')
    ax.set_ylabel('Count')
    ax.set_title('Feature Variance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Top varying features
    ax = axes[1, 0]
    top_var_indices = np.argsort(feature_vars)[-20:]
    ax.barh(range(len(top_var_indices)), feature_vars[top_var_indices])
    ax.set_xlabel('Variance')
    ax.set_ylabel('Feature Index')
    ax.set_title('Top 20 Most Variable Features')
    ax.grid(True, alpha=0.3)
    
    # 5. Feature correlation heatmap (sample)
    ax = axes[1, 1]
    if features.shape[1] > 1:
        n_features = min(50, features.shape[1])
        sample_features = features[:, :n_features]
        feature_corr = np.corrcoef(sample_features.T)
        
        im = ax.imshow(feature_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Feature Correlations (first {n_features})')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
    
    # 6. Species distribution
    ax = axes[1, 2]
    if 'species' in labels.columns:
        species_counts = labels['species'].value_counts().head(15)
        bars = ax.barh(range(len(species_counts)), species_counts.values)
        ax.set_xlabel('Count')
        ax.set_ylabel('Species')
        ax.set_title('Top 15 Species Distribution')
        ax.set_yticks(range(len(species_counts)))
        ax.set_yticklabels(species_counts.index, fontsize=8)
        
        # Add count labels
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                   str(species_counts.values[i]), ha='left', va='center', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 7. AMR distribution by partition
    ax = axes[2, 0]
    if 'antibiotic_label' in labels.columns and 'partition_label' in labels.columns:
        try:
            crosstab = pd.crosstab(labels['partition_label'], labels['antibiotic_label'])
            if not crosstab.empty:
                crosstab.plot(kind='bar', ax=ax, color=['lightblue', 'salmon'])
                ax.set_xlabel('Partition')
                ax.set_ylabel('Count')
                ax.set_title('AMR Distribution by Partition')
                ax.legend(['Susceptible', 'Resistant'])
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        except Exception:
            ax.text(0.5, 0.5, 'Crosstab unavailable', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('AMR Distribution by Partition')
    
    # 8. Feature statistics boxplot
    ax = axes[2, 1]
    stats_data = {
        'Mean': features.mean(axis=0),
        'Std': features.std(axis=0),
        'Min': features.min(axis=0),
        'Max': features.max(axis=0)
    }
    
    box_data = [stats_data[key] for key in stats_data.keys()]
    bp = ax.boxplot(box_data, labels=stats_data.keys(), patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Statistics Distribution')
    ax.grid(True, alpha=0.3)
    
    # 9. Sample distance matrix (subset)
    ax = axes[2, 2]
    if features.shape[0] > 1:
        n_samples = min(30, features.shape[0])
        sample_subset = features[:n_samples]
        distances = np.array([[np.linalg.norm(sample_subset[i] - sample_subset[j]) 
                             for j in range(n_samples)] for i in range(n_samples)])
        
        im = ax.imshow(distances, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_title(f'Sample Distance Matrix (first {n_samples})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sample Index')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_plots(mr_features, mr_labels, bl_features, bl_labels, output_path):
    """Create comparison plots between MiniRocket and Baseline"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MiniRocket vs Baseline Comparison', fontsize=16)
    
    # Find common samples
    mr_ids = set(mr_labels['genome_id'].astype(str))
    bl_ids = set(bl_labels['genome_id'].astype(str))
    common_ids = mr_ids & bl_ids
    
    print(f"Common samples between MiniRocket and Baseline: {len(common_ids)}")
    
    # 1. Feature dimensionality comparison
    ax = axes[0, 0]
    categories = ['MiniRocket', 'Baseline']
    dimensions = [mr_features.shape[1], bl_features.shape[1]]
    bars = ax.bar(categories, dimensions, color=['skyblue', 'lightcoral'])
    ax.set_ylabel('Number of Features')
    ax.set_title('Feature Dimensionality')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, dim in zip(bars, dimensions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
               str(dim), ha='center', va='bottom', fontweight='bold')
    
    # 2. Sample count comparison
    ax = axes[0, 1]
    sample_counts = [mr_features.shape[0], bl_features.shape[0], len(common_ids)]
    labels = ['MiniRocket', 'Baseline', 'Common']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax.bar(labels, sample_counts, color=colors)
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Count Comparison')
    ax.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, sample_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               str(count), ha='center', va='bottom', fontweight='bold')
    
    # 3. AMR distribution comparison
    ax = axes[0, 2]
    if 'antibiotic_label' in mr_labels.columns and 'antibiotic_label' in bl_labels.columns:
        mr_amr = mr_labels['antibiotic_label'].value_counts()
        bl_amr = bl_labels['antibiotic_label'].value_counts()
        
        x = np.arange(2)
        width = 0.35
        
        mr_counts = [mr_amr.get(0, 0), mr_amr.get(1, 0)]
        bl_counts = [bl_amr.get(0, 0), bl_amr.get(1, 0)]
        
        ax.bar(x - width/2, mr_counts, width, label='MiniRocket', color='skyblue')
        ax.bar(x + width/2, bl_counts, width, label='Baseline', color='lightcoral')
        
        ax.set_ylabel('Count')
        ax.set_title('AMR Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(['Susceptible', 'Resistant'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Feature variance comparison
    ax = axes[1, 0]
    mr_var = mr_features.var(axis=0)
    bl_var = bl_features.var(axis=0)
    
    ax.hist(mr_var, bins=30, alpha=0.6, label='MiniRocket', density=True, color='skyblue')
    ax.hist(bl_var, bins=30, alpha=0.6, label='Baseline', density=True, color='lightcoral')
    ax.set_xlabel('Feature Variance')
    ax.set_ylabel('Density')
    ax.set_title('Feature Variance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Sample norm comparison
    ax = axes[1, 1]
    mr_norms = np.linalg.norm(mr_features, axis=1)
    bl_norms = np.linalg.norm(bl_features, axis=1)
    
    ax.hist(mr_norms, bins=20, alpha=0.6, label='MiniRocket', density=True, color='skyblue')
    ax.hist(bl_norms, bins=20, alpha=0.6, label='Baseline', density=True, color='lightcoral')
    ax.set_xlabel('Sample L2 Norm')
    ax.set_ylabel('Density')
    ax.set_title('Sample Norm Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    stats_data = [
        ['Metric', 'MiniRocket', 'Baseline'],
        ['Samples', f"{mr_features.shape[0]}", f"{bl_features.shape[0]}"],
        ['Features', f"{mr_features.shape[1]}", f"{bl_features.shape[1]}"],
        ['Mean Variance', f"{mr_var.mean():.4f}", f"{bl_var.mean():.4f}"],
        ['Mean Norm', f"{mr_norms.mean():.2f}", f"{bl_norms.mean():.2f}"],
        ['Value Range', f"[{mr_features.min():.3f}, {mr_features.max():.3f}]", 
         f"[{bl_features.min():.3f}, {bl_features.max():.3f}]"]
    ]
    
    table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0], 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data)):
        for j in range(3):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 1:  # MiniRocket column
                    table[(i, j)].set_facecolor('#e1f5fe')
                elif j == 2:  # Baseline column
                    table[(i, j)].set_facecolor('#fce4ec')
    
    ax.set_title('Summary Statistics Comparison', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_species_clustering(features, labels, name):
    """Detailed species clustering analysis"""
    print(f"\n=== {name} Species Clustering Analysis ===")
    
    if 'species' in labels.columns and 'antibiotic_label' in labels.columns:
        # Species-AMR cross-tabulation
        crosstab = pd.crosstab(labels['species'], labels['antibiotic_label'], margins=True)
        print(f"\nSpecies vs AMR Cross-tabulation (top 15 species):")
        print(crosstab.head(15))
        
        # Calculate species purity and coverage
        species_stats = []
        for species in labels['species'].unique():
            species_mask = labels['species'] == species
            species_amr = labels[species_mask]['antibiotic_label']
            
            if len(species_amr) > 1:
                # Purity: most common AMR label ratio
                purity = max(species_amr.value_counts()) / len(species_amr)
                
                # AMR distribution
                resistant_count = (species_amr == 1).sum()
                susceptible_count = (species_amr == 0).sum()
                
                species_stats.append({
                    'species': species,
                    'total_samples': len(species_amr),
                    'resistant': resistant_count,
                    'susceptible': susceptible_count,
                    'purity': purity,
                    'dominant_label': 'R' if resistant_count > susceptible_count else 'S'
                })
        
        # Sort by sample count
        species_stats.sort(key=lambda x: x['total_samples'], reverse=True)
        
        print(f"\nTop Species Analysis (by sample count):")
        print(f"{'Species':<15} {'Total':<6} {'Resist':<6} {'Susc':<6} {'Purity':<7} {'Dom':<4}")
        print("-" * 55)
        for stat in species_stats[:15]:
            print(f"{stat['species']:<15} {stat['total_samples']:<6} {stat['resistant']:<6} "
                  f"{stat['susceptible']:<6} {stat['purity']:<7.3f} {stat['dominant_label']:<4}")
        
        # Clustering analysis if enough samples
        if features.shape[0] > 5 and features.shape[1] > 1:
            try:
                # Try k-means with different k values
                silhouette_scores = []
                ari_scores = []
                
                # Create species labels for comparison
                species_labels = labels['species'].astype('category').cat.codes
                
                for k in range(2, min(8, len(labels['species'].unique()), features.shape[0])):
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    cluster_labels = kmeans.fit_predict(features)
                    
                    sil_score = silhouette_score(features, cluster_labels)
                    ari_score = adjusted_rand_score(species_labels, cluster_labels)
                    
                    silhouette_scores.append(sil_score)
                    ari_scores.append(ari_score)
                
                if silhouette_scores:
                    best_k = np.argmax(silhouette_scores) + 2
                    best_sil = max(silhouette_scores)
                    best_ari = ari_scores[np.argmax(silhouette_scores)]
                    
                    print(f"\nClustering Analysis:")
                    print(f"  Best k: {best_k}")
                    print(f"  Best silhouette score: {best_sil:.3f}")
                    print(f"  ARI with species labels: {best_ari:.3f}")
                    
            except Exception as e:
                print(f"\nClustering analysis failed: {e}")

def print_detailed_summary(features, labels, name):
    """Print comprehensive summary statistics"""
    print(f"\n{'='*20} {name} DETAILED SUMMARY {'='*20}")
    print(f"Dataset Shape: {features.shape}")
    print(f"Memory Usage: {features.nbytes / 1024**2:.2f} MB")
    print(f"Data Type: {features.dtype}")
    
    print(f"\nFeature Statistics:")
    print(f"  Mean range:     [{features.mean(axis=0).min():.6f}, {features.mean(axis=0).max():.6f}]")
    print(f"  Std range:      [{features.std(axis=0).min():.6f}, {features.std(axis=0).max():.6f}]")
    print(f"  Min value:      {features.min():.6f}")
    print(f"  Max value:      {features.max():.6f}")
    print(f"  Median:         {np.median(features):.6f}")
    print(f"  Mean sparsity:  {(features == 0).mean():.3f}")
    
    if 'antibiotic_label' in labels.columns:
        amr_dist = labels['antibiotic_label'].value_counts()
        total = amr_dist.sum()
        print(f"\nAMR Distribution:")
        print(f"  Susceptible (0): {amr_dist.get(0, 0)} ({amr_dist.get(0, 0)/total:.1%})")
        print(f"  Resistant (1):   {amr_dist.get(1, 0)} ({amr_dist.get(1, 0)/total:.1%})")
    
    if 'partition_label' in labels.columns:
        part_dist = labels['partition_label'].value_counts()
        print(f"\nPartition Distribution:")
        for partition, count in part_dist.items():
            print(f"  {partition}: {count} ({count/part_dist.sum():.1%})")
    
    if 'species' in labels.columns:
        species_count = labels['species'].nunique()
        total_samples = len(labels)
        print(f"\nSpecies Diversity:")
        print(f"  Unique species: {species_count}")
        print(f"  Samples per species (avg): {total_samples/species_count:.1f}")
        
        top_species = labels['species'].value_counts().head(8)
        print(f"  Top species:")
        for species, count in top_species.items():
            print(f"    {species}: {count} ({count/total_samples:.1%})")

def main():
    # Configuration
    LABELS_CSV = "/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-1_ampicillin_kover.csv"
    MR_DIR = "/insomnia001/depts/pmg/users/ht2666/mr2048_s1024_41"
    BASELINE_DIR = "/insomnia001/depts/pmg/users/ht2666/mr_pca"  # This is actually PCA features
    OUTPUT_DIR = "/insomnia001/depts/pmg/users/ht2666/minirocket_pipeline/visualization/comprehensive_results"
    MAX_SAMPLES = 300  # Increased for more comprehensive analysis
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE ANALYSIS: MINIROCKET vs BASELINE")
    print("="*80)
    print(f"Labels CSV: {LABELS_CSV}")
    print(f"MiniRocket dir: {MR_DIR}")
    print(f"Baseline dir: {BASELINE_DIR}")
    print(f"Max samples: {MAX_SAMPLES}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"UMAP available: {HAS_UMAP}")
    
    # Load labels and add species info
    print("\nLoading labels...")
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df['species'] = labels_df['genome_id'].apply(
        lambda x: str(x).split('.')[0] if '.' in str(x) else str(x)[:10]
    )
    sample_ids = labels_df['genome_id'].astype(str).tolist()
    print(f"Total samples available: {len(sample_ids)}")
    
    # Analyze MiniRocket features
    print(f"\n{'='*60}")
    print("MINIROCKET ANALYSIS")
    print(f"{'='*60}")
    
    mr_features, mr_valid_ids = load_minirocket_features(MR_DIR, sample_ids, MAX_SAMPLES)
    
    if len(mr_features) > 0:
        # Create matching labels
        mr_labels_list = []
        for valid_id in mr_valid_ids:
            matching = labels_df[labels_df['genome_id'].astype(str) == str(valid_id)]
            if not matching.empty:
                mr_labels_list.append(matching.iloc[0])
        mr_labels = pd.DataFrame(mr_labels_list)
        
        print_detailed_summary(mr_features, mr_labels, "MiniRocket")
        analyze_species_clustering(mr_features, mr_labels, "MiniRocket")
        
        # Generate visualizations
        mr_dimred_output = f"{OUTPUT_DIR}/minirocket_dimensionality_reduction.png"
        mr_features_output = f"{OUTPUT_DIR}/minirocket_feature_analysis.png"
        
        create_dimensionality_reduction_plots(mr_features, mr_labels, "MiniRocket", mr_dimred_output)
        create_feature_analysis_plots(mr_features, mr_labels, "MiniRocket", mr_features_output)
        
        print(f"\nMiniRocket visualizations saved:")
        print(f"  - {mr_dimred_output}")
        print(f"  - {mr_features_output}")
    else:
        print("No MiniRocket features loaded!")
        mr_features = None
        mr_labels = None
    
    # Analyze Baseline features
    print(f"\n{'='*60}")
    print("BASELINE (PCA) ANALYSIS")
    print(f"{'='*60}")
    
    bl_features, bl_valid_ids = load_baseline_features(BASELINE_DIR, sample_ids, MAX_SAMPLES)
    
    if len(bl_features) > 0:
        # Create matching labels
        bl_labels_list = []
        for valid_id in bl_valid_ids:
            matching = labels_df[labels_df['genome_id'].astype(str) == str(valid_id)]
            if not matching.empty:
                bl_labels_list.append(matching.iloc[0])
        bl_labels = pd.DataFrame(bl_labels_list)
        
        print_detailed_summary(bl_features, bl_labels, "Baseline")
        analyze_species_clustering(bl_features, bl_labels, "Baseline")
        
        # Generate visualizations
        bl_dimred_output = f"{OUTPUT_DIR}/baseline_dimensionality_reduction.png"
        bl_features_output = f"{OUTPUT_DIR}/baseline_feature_analysis.png"
        
        create_dimensionality_reduction_plots(bl_features, bl_labels, "Baseline", bl_dimred_output)
        create_feature_analysis_plots(bl_features, bl_labels, "Baseline", bl_features_output)
        
        print(f"\nBaseline visualizations saved:")
        print(f"  - {bl_dimred_output}")
        print(f"  - {bl_features_output}")
    else:
        print("No Baseline features loaded!")
        bl_features = None
        bl_labels = None
    
    # Create comparison analysis
    if mr_features is not None and bl_features is not None:
        print(f"\n{'='*60}")
        print("COMPARATIVE ANALYSIS")
        print(f"{'='*60}")
        
        comparison_output = f"{OUTPUT_DIR}/minirocket_vs_baseline_comparison.png"
        create_comparison_plots(mr_features, mr_labels, bl_features, bl_labels, comparison_output)
        
        print(f"\nComparison visualization saved:")
        print(f"  - {comparison_output}")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved in: {OUTPUT_DIR}")
    
    # List all generated files
    result_files = list(Path(OUTPUT_DIR).glob("*.png"))
    if result_files:
        print(f"\nGenerated {len(result_files)} visualization files:")
        for file in sorted(result_files):
            print(f"  - {file.name}")

if __name__ == "__main__":
    main()