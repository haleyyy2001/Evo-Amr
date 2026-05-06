#!/usr/bin/env python3
"""
AMR Classification Pipeline
==========================

Comprehensive pipeline for antimicrobial resistance (AMR) prediction comparing
multiple approaches: MiniRocket time-series classification and baseline methods
using PCA/pooled embeddings.

Features:
- Multiple classification algorithms with hyperparameter tuning
- Cross-validation with proper train/validation/test splits
- Multiple Instance Learning (MIL) for sequence-level predictions
- Comprehensive evaluation metrics and visualization
- Configurable via environment variables
- Support for both tabular and sequence-based approaches

Author: Evo-AMR Team
License: MIT
"""

import os
import re
import json
import glob
import random
import warnings
import sys
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

# Configure output for immediate feedback
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Visualization
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Core ML libraries
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, PredefinedSplit
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.exceptions import FitFailedWarning

# Metrics
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, matthews_corrcoef, balanced_accuracy_score, f1_score, auc,
    precision_score, recall_score
)

warnings.filterwarnings("ignore", category=FitFailedWarning)

# Optional ML libraries
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

# PyTorch for MIL
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, MIL model will be skipped")


class Configuration:
    """Pipeline configuration management."""
    
    def __init__(self):
        # Parse run tracks from environment
        run_tracks_env = os.getenv("AMR_RUN_TRACKS", None)
        if run_tracks_env:
            self.run_tracks = [t.strip() for t in run_tracks_env.split(",") if t.strip()]
        else:
            self.run_tracks = ["baseline", "minirocket"]
        
        # Pipeline options
        self.skip_mil = os.getenv("AMR_SKIP_MIL", "0") == "1"
        self.tune_on_val_outside = os.getenv("AMR_TUNE_ON_VAL_OUTSIDE", "1") == "1"
        
        # Paths
        self.outrun = os.getenv("AMR_OUTRUN", "/tmp/mr2048_s1024_41")
        self.pca_dir = os.getenv("AMR_PCA_DIR", "/tmp/mr_pca")
        self.labels_path = os.getenv("AMR_LABELS", "labels.csv")
        self.root_output = os.getenv("AMR_ROOT_OUT", "/tmp/evo_ampicillin_results")
        self.window_tag = os.getenv("AMR_WIN_TAG", "_mr_win2048_s1024")
        
        # Output directories
        self.minirocket_dir = os.path.join(self.root_output, "minirocket")
        self.baseline_dir = os.path.join(self.root_output, "baseline")
        
        # Constants
        self.n_top_features = 50
        self.target_partitions = ['val_overlapped', 'val_outside', 'test_overlapped', 'test_outside']
        self.partitions_canvas = ['train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped']
        self.save_individual_figs = False
        
        # Partition display names and colors
        self.partition_names = {
            'train': 'Train', 'val_outside': 'Val-Out', 'val_overlapped': 'Val-Ovr',
            'test_outside': 'Test-Out', 'test_overlapped': 'Test-Ovr',
        }
        self.partition_colors = {
            'train': '#1f77b4', 'val_outside': '#ff7f0e', 'val_overlapped': '#2ca02c',
            'test_outside': '#d62728', 'test_overlapped': '#9467bd'
        }
        
        # Model configuration
        self.auroc_scorer = 'roc_auc'
        self.calibrate_on_train_cv = True
        self.calibration_method = 'isotonic'
        
        # Cross-validation strategy
        if self.tune_on_val_outside:
            self.cv_robust = 3
        else:
            self.cv_robust = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)


class DataLoader:
    """Handles data loading and preprocessing."""
    
    @staticmethod
    def ensure_directories(output_root: str):
        """Create necessary output directories."""
        Path(output_root).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def extract_genome_id_minirocket(file_path: str) -> str:
        """Extract genome ID from MiniRocket file path."""
        stem = Path(file_path).stem
        return stem.replace("_mr_win2048_s1024", "")
    
    @staticmethod
    def extract_genome_id_baseline(file_path: str) -> str:
        """Extract genome ID from baseline file path."""
        stem = Path(file_path).stem
        # Remove common suffixes
        for suffix in ["_pca_stats_dim41", "_pooled_embeddings"]:
            if stem.endswith(suffix):
                return stem[:-len(suffix)]
        return stem
    
    @staticmethod
    def load_minirocket_windows(file_path: str) -> np.ndarray:
        """Load MiniRocket window data."""
        data = np.load(file_path)
        return data['windows']  # [n_windows, n_features]
    
    @staticmethod
    def enhanced_pooled_vector(embeddings: np.ndarray) -> np.ndarray:
        """
        Create enhanced pooled vector with multiple statistics.
        
        Args:
            embeddings: Input embeddings [seq_len, dim]
            
        Returns:
            Pooled feature vector [6 * dim]
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D")
        
        # Compute statistics along sequence dimension
        mean_vec = np.mean(embeddings, axis=0)
        std_vec = np.std(embeddings, axis=0)
        max_vec = np.max(embeddings, axis=0)
        min_vec = np.min(embeddings, axis=0)
        q25_vec = np.percentile(embeddings, 25, axis=0)
        q75_vec = np.percentile(embeddings, 75, axis=0)
        
        # Concatenate all statistics
        return np.concatenate([mean_vec, std_vec, max_vec, min_vec, q25_vec, q75_vec])
    
    @staticmethod
    def load_baseline_vector(file_path: str) -> np.ndarray:
        """Load baseline feature vector."""
        if file_path.endswith('_pca_stats_dim41.npz'):
            # PCA statistical features
            data = np.load(file_path)
            return data['V'].flatten()  # Flatten to 1D
        else:
            # Pooled embeddings
            data = np.load(file_path)
            embeddings = data['embeddings']  # [seq_len, dim]
            return DataLoader.enhanced_pooled_vector(embeddings)
    
    @staticmethod
    def normalize_numeric_id(identifier: str) -> str:
        """Normalize numeric identifiers to avoid float precision issues."""
        # Convert "23.20" -> "23.2" for consistent matching
        if re.match(r'^\d+\.\d+$', identifier):
            return str(float(identifier))
        return identifier


class IDManager:
    """Handles genome ID management and validation."""
    
    @staticmethod
    def warn_ambiguous_ids(set_a: set, set_b: set, title_a: str = "A", title_b: str = "B"):
        """Warn about potentially ambiguous ID matches."""
        norm_a = {IDManager._normalize_for_comparison(id_) for id_ in set_a}
        norm_b = {IDManager._normalize_for_comparison(id_) for id_ in set_b}
        
        # Check for normalized matches that weren't exact matches
        exact_overlap = set_a & set_b
        norm_overlap = norm_a & norm_b
        
        if len(norm_overlap) > len(exact_overlap):
            ambiguous = norm_overlap - {IDManager._normalize_for_comparison(id_) for id_ in exact_overlap}
            if ambiguous:
                print(f"WARNING: Found {len(ambiguous)} potentially ambiguous IDs between {title_a} and {title_b}")
                for amb_id in sorted(list(ambiguous)[:5]):  # Show first 5
                    matches_a = [id_ for id_ in set_a if IDManager._normalize_for_comparison(id_) == amb_id]
                    matches_b = [id_ for id_ in set_b if IDManager._normalize_for_comparison(id_) == amb_id]
                    print(f"  '{amb_id}' matches: {title_a}={matches_a}, {title_b}={matches_b}")
    
    @staticmethod
    def _normalize_for_comparison(identifier: str) -> str:
        """Normalize ID for comparison purposes."""
        return DataLoader.normalize_numeric_id(identifier).lower().strip()
    
    @staticmethod
    def warn_internal_ambiguity(id_list: List[str], scope: str = "set"):
        """Warn about internal ID ambiguities within a single collection."""
        normalized = [IDManager._normalize_for_comparison(id_) for id_ in id_list]
        unique_normalized = set(normalized)
        
        if len(unique_normalized) < len(id_list):
            duplicates = len(id_list) - len(unique_normalized)
            print(f"WARNING: {scope} has {duplicates} potential internal ID duplicates")
            
            # Find specific duplicates
            from collections import Counter
            counts = Counter(normalized)
            for norm_id, count in counts.items():
                if count > 1:
                    originals = [id_list[i] for i, norm in enumerate(normalized) if norm == norm_id]
                    print(f"  Normalized '{norm_id}' appears {count} times: {originals}")


class LabelManager:
    """Handles label loading and processing."""
    
    def __init__(self, labels_path: str):
        self.labels_path = labels_path
        self._labels_df = None
        self._id_index = None
    
    def load_labels(self) -> pd.DataFrame:
        """Load and validate labels CSV."""
        if self._labels_df is not None:
            return self._labels_df
        
        try:
            df = pd.read_csv(self.labels_path)
        except Exception as e:
            raise FileNotFoundError(f"Cannot load labels from {self.labels_path}: {e}")
        
        # Validate required columns
        required_cols = ['genome_id', 'label', 'partition_label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in labels CSV: {missing_cols}")
        
        # Convert genome_id to string and normalize
        df['genome_id'] = df['genome_id'].astype(str)
        df['genome_id_norm'] = df['genome_id'].apply(DataLoader.normalize_numeric_id)
        
        # Validate partitions
        valid_partitions = {'train', 'val_outside', 'val_overlapped', 'test_outside', 'test_overlapped'}
        invalid_partitions = set(df['partition_label'].unique()) - valid_partitions
        if invalid_partitions:
            print(f"WARNING: Found unexpected partition labels: {invalid_partitions}")
        
        # Check for internal ID ambiguities
        IDManager.warn_internal_ambiguity(df['genome_id'].tolist(), "labels CSV")
        
        print(f"[LABELS] Loaded {len(df)} samples from {self.labels_path}")
        print(f"[LABELS] Partitions: {dict(df['partition_label'].value_counts())}")
        print(f"[LABELS] Labels: {dict(df['label'].value_counts())}")
        
        self._labels_df = df
        return df
    
    def build_id_index(self) -> Dict[str, int]:
        """Build normalized genome ID to index mapping."""
        if self._id_index is not None:
            return self._id_index
        
        df = self.load_labels()
        self._id_index = {row['genome_id_norm']: idx for idx, row in df.iterrows()}
        return self._id_index
    
    def get_label_for_genome(self, genome_id: str) -> Tuple[Optional[int], Optional[str]]:
        """Get label and partition for genome ID."""
        df = self.load_labels()
        id_index = self.build_id_index()
        
        # Try exact match first
        if genome_id in df['genome_id'].values:
            row = df[df['genome_id'] == genome_id].iloc[0]
            return int(row['label']), row['partition_label']
        
        # Try normalized match
        norm_id = DataLoader.normalize_numeric_id(genome_id)
        if norm_id in id_index:
            idx = id_index[norm_id]
            row = df.iloc[idx]
            return int(row['label']), row['partition_label']
        
        return None, None


class ModelBuilder:
    """Builds and configures ML models."""
    
    def __init__(self, config: Configuration):
        self.config = config
    
    def build_model_suite(self) -> Dict[str, Any]:
        """Build comprehensive suite of ML models."""
        models = {}
        
        # Logistic Regression variants
        models['LogReg_L2'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ])
        
        models['LogReg_L1'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty='l1', solver='liblinear', random_state=42, 
                                     max_iter=1000, class_weight='balanced'))
        ])
        
        # Ridge Classifier
        models['Ridge'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RidgeClassifier(random_state=42, class_weight='balanced'))
        ])
        
        # SVM variants
        models['LinearSVM'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', CalibratedClassifierCV(
                LinearSVC(random_state=42, max_iter=10000, class_weight='balanced'),
                method=self.config.calibration_method, cv=3
            ))
        ])
        
        models['RBF_SVM'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
        ])
        
        # Tree-based models
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1
        )
        
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1
        )
        
        models['HistGBM'] = HistGradientBoostingClassifier(
            random_state=42, class_weight='balanced'
        )
        
        # k-NN
        models['kNN'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ])
        
        # Nearest Centroid
        models['NearestCentroid'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', NearestCentroid())
        ])
        
        # Neural Network
        models['MLP'] = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, 
                                 max_iter=500, early_stopping=True))
        ])
        
        # Gradient Boosting (if available)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                random_state=42, eval_metric='logloss', use_label_encoder=False
            )
        
        if LGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(
                random_state=42, verbosity=-1, class_weight='balanced'
            )
        
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(
                random_state=42, verbose=False, class_weight='balanced'
            )
        
        return models
    
    def get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for model tuning."""
        return {
            'LogReg_L2': {'clf__C': [0.01, 0.1, 1.0, 10.0]},
            'LogReg_L1': {'clf__C': [0.01, 0.1, 1.0, 10.0]},
            'Ridge': {'clf__alpha': [0.1, 1.0, 10.0, 100.0]},
            'RBF_SVM': {'clf__C': [0.1, 1.0, 10.0], 'clf__gamma': ['scale', 'auto']},
            'RandomForest': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
            'ExtraTrees': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]},
            'HistGBM': {'max_depth': [6, 10, 15], 'learning_rate': [0.01, 0.1, 0.2]},
            'kNN': {'clf__n_neighbors': [3, 5, 7, 10], 'clf__weights': ['uniform', 'distance']},
            'MLP': {'clf__hidden_layer_sizes': [(50,), (100,), (100, 50)], 
                   'clf__alpha': [0.001, 0.01, 0.1]},
        }


class MILModel(nn.Module):
    """Multiple Instance Learning model for sequence-level prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: [batch_size, n_instances, features]
        batch_size, n_instances, _ = x.shape
        
        # Extract features for each instance
        h = self.feature_extractor(x.view(-1, x.size(-1)))  # [batch_size * n_instances, hidden]
        h = h.view(batch_size, n_instances, -1)  # [batch_size, n_instances, hidden]
        
        # Attention pooling
        a = self.attention(h)  # [batch_size, n_instances, 1]
        a = torch.softmax(a, dim=1)
        
        # Weighted average
        z = torch.sum(a * h, dim=1)  # [batch_size, hidden]
        
        # Final classification
        y = self.classifier(z)  # [batch_size, 1]
        return torch.sigmoid(y)


class Evaluator:
    """Handles model evaluation and metrics computation."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = (y_true == y_pred).mean()
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # ROC metrics
        if len(np.unique(y_true)) > 1:  # Need both classes for ROC
            metrics['auroc'] = roc_auc_score(y_true, y_scores)
            metrics['auprc'] = average_precision_score(y_true, y_scores)
        else:
            metrics['auroc'] = np.nan
            metrics['auprc'] = np.nan
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        return metrics
    
    @staticmethod
    def optimal_threshold_by_mcc(y_true: np.ndarray, y_scores: np.ndarray, 
                                n_thresholds: int = 100) -> Tuple[float, float]:
        """Find optimal threshold by maximizing MCC."""
        thresholds = np.linspace(0, 1, n_thresholds)
        best_mcc = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            mcc = matthews_corrcoef(y_true, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
        
        return best_threshold, best_mcc


class Visualizer:
    """Handles visualization and plotting."""
    
    def __init__(self, config: Configuration):
        self.config = config
    
    def create_confusion_matrix_grid(self, results: Dict, output_dir: str):
        """Create confusion matrix visualization grid."""
        models = list(results.keys())
        partitions = self.config.target_partitions
        
        n_models = len(models)
        n_partitions = len(partitions)
        
        fig, axes = plt.subplots(n_models, n_partitions, figsize=(4*n_partitions, 3*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        if n_partitions == 1:
            axes = axes.reshape(-1, 1)
        
        for i, model_name in enumerate(models):
            for j, partition in enumerate(partitions):
                ax = axes[i, j]
                
                if partition in results[model_name]:
                    metrics = results[model_name][partition]
                    cm = np.array([[metrics['true_negatives'], metrics['false_positives']],
                                  [metrics['false_negatives'], metrics['true_positives']]])
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                              xticklabels=['Resistant', 'Susceptible'],
                              yticklabels=['Resistant', 'Susceptible'])
                    ax.set_title(f'{model_name} - {self.config.partition_names[partition]}')
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{model_name} - {self.config.partition_names[partition]}')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_curves(self, results: Dict, output_dir: str):
        """Create ROC and PR curves overlay."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC curves
        for model_name, model_results in results.items():
            for partition in self.config.target_partitions:
                if partition in model_results and 'y_true' in model_results[partition]:
                    y_true = model_results[partition]['y_true']
                    y_scores = model_results[partition]['y_scores']
                    
                    if len(np.unique(y_true)) > 1:
                        fpr, tpr, _ = roc_curve(y_true, y_scores)
                        auroc = roc_auc_score(y_true, y_scores)
                        
                        label = f'{model_name} {self.config.partition_names[partition]} (AUC={auroc:.3f})'
                        ax1.plot(fpr, tpr, label=label, color=self.config.partition_colors[partition])
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # PR curves
        for model_name, model_results in results.items():
            for partition in self.config.target_partitions:
                if partition in model_results and 'y_true' in model_results[partition]:
                    y_true = model_results[partition]['y_true']
                    y_scores = model_results[partition]['y_scores']
                    
                    if len(np.unique(y_true)) > 1:
                        precision, recall, _ = precision_recall_curve(y_true, y_scores)
                        auprc = average_precision_score(y_true, y_scores)
                        
                        label = f'{model_name} {self.config.partition_names[partition]} (AP={auprc:.3f})'
                        ax2.plot(recall, precision, label=label, color=self.config.partition_colors[partition])
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / 'performance_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


class AMRClassificationPipeline:
    """Main pipeline class orchestrating the entire AMR classification workflow."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self.label_manager = LabelManager(config.labels_path)
        self.model_builder = ModelBuilder(config)
        self.evaluator = Evaluator()
        self.visualizer = Visualizer(config)
    
    def run_pipeline(self):
        """Run the complete AMR classification pipeline."""
        print(f"[PIPELINE] Starting AMR classification pipeline")
        print(f"[PIPELINE] Run tracks: {self.config.run_tracks}")
        
        # Load labels
        labels_df = self.label_manager.load_labels()
        
        # Ensure output directories
        DataLoader.ensure_directories(self.config.root_output)
        
        # Run each track
        all_results = {}
        for track in self.config.run_tracks:
            print(f"\n[PIPELINE] Running track: {track}")
            track_results = self.run_track(track, labels_df)
            all_results[track] = track_results
        
        # Create comparison visualizations if multiple tracks
        if len(self.config.run_tracks) > 1:
            self.create_track_comparison(all_results)
        
        print(f"\n[PIPELINE] Pipeline completed successfully")
        print(f"[PIPELINE] Results saved to: {self.config.root_output}")
    
    def run_track(self, track: str, labels_df: pd.DataFrame) -> Dict:
        """Run a specific track (baseline or minirocket)."""
        if track == "baseline":
            return self.run_baseline_track(labels_df)
        elif track == "minirocket":
            return self.run_minirocket_track(labels_df)
        else:
            raise ValueError(f"Unknown track: {track}")
    
    def run_baseline_track(self, labels_df: pd.DataFrame) -> Dict:
        """Run baseline PCA/pooled features track."""
        print(f"[BASELINE] Loading baseline features from {self.config.pca_dir}")
        
        # Find baseline files
        baseline_files = list(Path(self.config.pca_dir).glob("*_pca_stats_dim*.npz"))
        baseline_files.extend(Path(self.config.pca_dir).glob("*_pooled_embeddings.npz"))
        
        if not baseline_files:
            raise FileNotFoundError(f"No baseline files found in {self.config.pca_dir}")
        
        print(f"[BASELINE] Found {len(baseline_files)} baseline files")
        
        # Load data
        data_dict = {}
        for file_path in baseline_files:
            genome_id = DataLoader.extract_genome_id_baseline(str(file_path))
            label, partition = self.label_manager.get_label_for_genome(genome_id)
            
            if label is not None:
                try:
                    features = DataLoader.load_baseline_vector(str(file_path))
                    data_dict[genome_id] = {
                        'features': features,
                        'label': label,
                        'partition': partition
                    }
                except Exception as e:
                    print(f"[BASELINE] Error loading {file_path}: {e}")
        
        print(f"[BASELINE] Loaded {len(data_dict)} samples with labels")
        
        # Run classification
        return self.run_classification(data_dict, self.config.baseline_dir, "baseline")
    
    def run_minirocket_track(self, labels_df: pd.DataFrame) -> Dict:
        """Run MiniRocket time-series features track."""
        print(f"[MINIROCKET] Loading MiniRocket features from {self.config.outrun}")
        
        # Find MiniRocket files
        pattern = f"*{self.config.window_tag}.npz"
        minirocket_files = list(Path(self.config.outrun).glob(pattern))
        
        if not minirocket_files:
            raise FileNotFoundError(f"No MiniRocket files found matching {pattern} in {self.config.outrun}")
        
        print(f"[MINIROCKET] Found {len(minirocket_files)} MiniRocket files")
        
        # Load data (for MIL, we keep windows; for tabular, we pool)
        data_dict = {}
        for file_path in minirocket_files:
            genome_id = DataLoader.extract_genome_id_minirocket(str(file_path))
            label, partition = self.label_manager.get_label_for_genome(genome_id)
            
            if label is not None:
                try:
                    windows = DataLoader.load_minirocket_windows(str(file_path))
                    # For tabular models, pool the windows
                    pooled_features = DataLoader.enhanced_pooled_vector(windows)
                    
                    data_dict[genome_id] = {
                        'features': pooled_features,  # Pooled for tabular models
                        'windows': windows,           # Raw windows for MIL
                        'label': label,
                        'partition': partition
                    }
                except Exception as e:
                    print(f"[MINIROCKET] Error loading {file_path}: {e}")
        
        print(f"[MINIROCKET] Loaded {len(data_dict)} samples with labels")
        
        # Run classification
        results = self.run_classification(data_dict, self.config.minirocket_dir, "minirocket")
        
        # Add MIL results if available and not skipped
        if TORCH_AVAILABLE and not self.config.skip_mil:
            mil_results = self.run_mil_classification(data_dict, self.config.minirocket_dir)
            results['MIL_Model'] = mil_results
        
        return results
    
    def run_classification(self, data_dict: Dict, output_dir: str, track_name: str) -> Dict:
        """Run tabular classification pipeline."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        X, y, partitions, genome_ids = self.prepare_tabular_data(data_dict)
        
        # Build models
        models = self.model_builder.build_model_suite()
        param_grids = self.model_builder.get_hyperparameter_grids()
        
        # Results storage
        all_results = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"[{track_name.upper()}] Training {model_name}...")
            
            try:
                model_results = self.train_and_evaluate_model(
                    model, model_name, X, y, partitions, genome_ids, param_grids.get(model_name, {})
                )
                all_results[model_name] = model_results
                
            except Exception as e:
                print(f"[{track_name.upper()}] Error with {model_name}: {e}")
                continue
        
        # Save results
        results_path = Path(output_dir) / 'classification_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self.convert_results_for_json(all_results)
            json.dump(json_results, f, indent=2)
        
        # Create visualizations
        self.visualizer.create_confusion_matrix_grid(all_results, output_dir)
        self.visualizer.create_performance_curves(all_results, output_dir)
        
        print(f"[{track_name.upper()}] Results saved to {output_dir}")
        return all_results
    
    def prepare_tabular_data(self, data_dict: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare data for tabular classification."""
        genome_ids = list(data_dict.keys())
        X = np.array([data_dict[gid]['features'] for gid in genome_ids])
        y = np.array([data_dict[gid]['label'] for gid in genome_ids])
        partitions = np.array([data_dict[gid]['partition'] for gid in genome_ids])
        
        print(f"[DATA] Prepared {len(X)} samples with {X.shape[1]} features")
        print(f"[DATA] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        return X, y, partitions, genome_ids
    
    def train_and_evaluate_model(self, model, model_name: str, X: np.ndarray, y: np.ndarray,
                                partitions: np.ndarray, genome_ids: List[str], param_grid: Dict) -> Dict:
        """Train and evaluate a single model."""
        results = {}
        
        # Split data
        train_mask = partitions == 'train'
        X_train, y_train = X[train_mask], y[train_mask]
        
        # Hyperparameter tuning
        if param_grid and self.config.tune_on_val_outside:
            # Use validation set for tuning
            val_mask = partitions == 'val_outside'
            if val_mask.sum() > 0:
                X_val, y_val = X[val_mask], y[val_mask]
                # Create PredefinedSplit for validation
                test_fold = np.ones(len(X_train) + len(X_val)) * -1
                test_fold[len(X_train):] = 0
                ps = PredefinedSplit(test_fold)
                
                X_tune = np.vstack([X_train, X_val])
                y_tune = np.hstack([y_train, y_val])
                
                grid_search = GridSearchCV(model, param_grid, cv=ps, scoring=self.config.auroc_scorer, n_jobs=-1)
                grid_search.fit(X_tune, y_tune)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
        else:
            # Use cross-validation on training set
            if param_grid:
                grid_search = GridSearchCV(model, param_grid, cv=self.config.cv_robust, 
                                         scoring=self.config.auroc_scorer, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
            else:
                best_model = model
        
        # Train final model
        best_model.fit(X_train, y_train)
        
        # Evaluate on all partitions
        for partition in self.config.partitions_canvas:
            partition_mask = partitions == partition
            if partition_mask.sum() == 0:
                continue
            
            X_part, y_part = X[partition_mask], y[partition_mask]
            
            # Predictions
            if hasattr(best_model, 'predict_proba'):
                y_scores = best_model.predict_proba(X_part)[:, 1]
            else:
                # For models without probability output
                decision_scores = best_model.decision_function(X_part)
                y_scores = 1 / (1 + np.exp(-decision_scores))  # Sigmoid transformation
            
            # Find optimal threshold
            if partition == 'train':
                optimal_threshold, _ = self.evaluator.optimal_threshold_by_mcc(y_part, y_scores)
            else:
                optimal_threshold = 0.5  # Use default for other partitions
            
            y_pred = (y_scores >= optimal_threshold).astype(int)
            
            # Compute metrics
            metrics = self.evaluator.compute_metrics(y_part, y_pred, y_scores)
            metrics['optimal_threshold'] = optimal_threshold
            metrics['y_true'] = y_part
            metrics['y_pred'] = y_pred
            metrics['y_scores'] = y_scores
            
            results[partition] = metrics
        
        return results
    
    def run_mil_classification(self, data_dict: Dict, output_dir: str) -> Dict:
        """Run Multiple Instance Learning classification."""
        print("[MIL] Running Multiple Instance Learning...")
        
        # Prepare MIL data
        mil_data = []
        mil_labels = []
        mil_partitions = []
        mil_genome_ids = []
        
        for genome_id, sample in data_dict.items():
            mil_data.append(sample['windows'])
            mil_labels.append(sample['label'])
            mil_partitions.append(sample['partition'])
            mil_genome_ids.append(genome_id)
        
        # Convert to arrays
        mil_labels = np.array(mil_labels)
        mil_partitions = np.array(mil_partitions)
        
        # Get dimensions
        n_features = mil_data[0].shape[1]
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MILModel(n_features).to(device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Train/val split
        train_mask = mil_partitions == 'train'
        train_data = [mil_data[i] for i in range(len(mil_data)) if train_mask[i]]
        train_labels = mil_labels[train_mask]
        
        # Training loop
        model.train()
        for epoch in range(100):  # Simplified training
            total_loss = 0
            for i, (windows, label) in enumerate(zip(train_data, train_labels)):
                optimizer.zero_grad()
                
                # Convert to tensor
                x = torch.FloatTensor(windows).unsqueeze(0).to(device)  # [1, n_windows, n_features]
                y = torch.FloatTensor([label]).to(device)
                
                # Forward pass
                output = model(x).squeeze()
                loss = criterion(output, y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"[MIL] Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
        
        # Evaluation
        model.eval()
        results = {}
        
        with torch.no_grad():
            for partition in self.config.partitions_canvas:
                partition_mask = mil_partitions == partition
                if partition_mask.sum() == 0:
                    continue
                
                partition_data = [mil_data[i] for i in range(len(mil_data)) if partition_mask[i]]
                partition_labels = mil_labels[partition_mask]
                
                y_scores = []
                for windows in partition_data:
                    x = torch.FloatTensor(windows).unsqueeze(0).to(device)
                    score = model(x).squeeze().cpu().numpy()
                    y_scores.append(score)
                
                y_scores = np.array(y_scores)
                y_pred = (y_scores >= 0.5).astype(int)
                
                metrics = self.evaluator.compute_metrics(partition_labels, y_pred, y_scores)
                metrics['y_true'] = partition_labels
                metrics['y_pred'] = y_pred
                metrics['y_scores'] = y_scores
                
                results[partition] = metrics
        
        return results
    
    def convert_results_for_json(self, results: Dict) -> Dict:
        """Convert numpy arrays in results to lists for JSON serialization."""
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {}
            for partition, metrics in model_results.items():
                json_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        json_metrics[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_metrics[key] = float(value)
                    else:
                        json_metrics[key] = value
                json_results[model_name][partition] = json_metrics
        return json_results
    
    def create_track_comparison(self, all_results: Dict):
        """Create comparison visualizations between tracks."""
        print("[COMPARISON] Creating track comparison visualizations...")
        
        # Compare best models from each track
        comparison_data = []
        for track_name, track_results in all_results.items():
            for model_name, model_results in track_results.items():
                for partition in self.config.target_partitions:
                    if partition in model_results:
                        metrics = model_results[partition]
                        comparison_data.append({
                            'track': track_name,
                            'model': model_name,
                            'partition': partition,
                            'auroc': metrics.get('auroc', np.nan),
                            'auprc': metrics.get('auprc', np.nan),
                            'f1': metrics.get('f1', np.nan),
                            'mcc': metrics.get('mcc', np.nan)
                        })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            metrics_to_plot = ['auroc', 'auprc', 'f1', 'mcc']
            metric_names = ['AUROC', 'AUPRC', 'F1-Score', 'MCC']
            
            for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
                ax = axes[idx // 2, idx % 2]
                
                # Filter valid data
                valid_data = comparison_df.dropna(subset=[metric])
                if len(valid_data) > 0:
                    sns.boxplot(data=valid_data, x='track', y=metric, hue='partition', ax=ax)
                    ax.set_title(f'{name} by Track and Partition')
                    ax.set_ylabel(name)
                
            plt.tight_layout()
            plt.savefig(Path(self.config.root_output) / 'track_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"[COMPARISON] Comparison saved to {self.config.root_output}")


def main():
    """Main entry point."""
    print("AMR Classification Pipeline")
    print("=" * 50)
    
    # Initialize configuration
    config = Configuration()
    
    # Initialize and run pipeline
    pipeline = AMRClassificationPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()