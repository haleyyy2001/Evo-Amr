#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMR MiniRocket 综合可视化 & kNN可行性诊断

输入：
- OUTRUN: 目录中包含   <id>_mr_win2048_s1024.npz  （内有 X:[nW,nF], spans:[nW,2]）
- 同名 JSON：          <id>_mr_win2048_s1024.json  （元数据：n_windows, n_features, var_explained_topk …）
- LABELS CSV：          id,label[,species,drug]

输出 (到 OUTDIR)：
  dist_n_windows.png
  dist_mean_norm.png
  pca_pooled_by_label.png
  timeseries_window_norms_by_label.png
  class_avg_curves.png
  heatmap_sample_<id>.png
  knn_neighborhood_purity.png
  nn_distance_hist_same_vs_diff.png
  baseline_results.csv
  metadata_summary.csv
  pooled_shapes.txt
"""
import os, re, json, glob, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.covariance import LedoitWolf

# ------------------- 配置 -------------------
OUTRUN = "/insomnia001/depts/pmg/users/ht2666/mr2048_s1024"
LABELS = "/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/Kover/kover_input/clustered_3_v1-1_ampicillin_kover.csv"  # 至少包含: id,label
OUTDIR  = "/insomnia001/depts/pmg/users/ht2666/amr_viz_diag"
WIN_TAG = "_mr_win2048_s1024"
MAX_LINES = 30                 # 时序曲线最多叠加的样本
HEATMAP_EXAMPLE_PER_CLASS = 1
HEATMAP_FEATS_SUBSAMPLE = 256
CLASS_AVG_BINS = 100
K_LIST = [1,3,5,7,9]           # kNN的k
random.seed(42)
Path(OUTDIR).mkdir(parents=True, exist_ok=True)

# ------------------- 工具 -------------------
def stem_id(fp: str) -> str:
    s = Path(fp).stem
    s = re.sub(r"_mr_win\d+_s\d+$", "", s)
    return s

def pair_json(npz_path: str) -> str:
    return str(Path(npz_path).with_suffix("")) + ".json"

def load_npz(fp: str):
    z = np.load(fp)
    X = z["X"].astype(np.float32)  # [nW,nF]
    spans = z["spans"]
    return X, spans

def load_meta(json_fp: str):
    try:
        with open(json_fp, "r") as fh:
            return json.load(fh)
    except Exception:
        return {}

def pool_vec(X: np.ndarray, stats=("mean","std")) -> np.ndarray:
    vecs = []
    if "mean" in stats: vecs.append(X.mean(0))
    if "std"  in stats: vecs.append(X.std(0))
    return np.concatenate(vecs, axis=0).astype(np.float32)

def per_window_norm(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X, axis=1)

def normalized_bins(n: int, bins: int):
    if n <= 1: return np.array([0], dtype=int)
    idx = (np.linspace(0, 1 - 1e-9, n) * bins).astype(int)
    idx[idx>=bins] = bins-1
    return idx

# ------------------- 读取数据 -------------------
files = sorted(glob.glob(os.path.join(OUTRUN, f"*{WIN_TAG}.npz")))
if not files:
    raise FileNotFoundError(f"No *{WIN_TAG}.npz in {OUTRUN}")

lab = pd.read_csv(LABELS)
# Check if columns exist - handle different column names
if "genome_id" in lab.columns:
    lab["id"] = lab["genome_id"].astype(str)
if "antibiotic_label" in lab.columns:
    lab["label"] = lab["antibiotic_label"]
if not {"id","label"}.issubset(lab.columns):
    raise ValueError("labels.csv must contain at least: id,label (or genome_id,antibiotic_label)")

# Handle different possible ID column names
if "genome_id" in lab.columns:
    id2row = {str(r.genome_id): r for _, r in lab.iterrows()}
else:
    id2row = {str(r.id): r for _, r in lab.iterrows()}

rows = []
for fp in files:
    sid = stem_id(fp)
    if sid in id2row:
        r = id2row[sid]
        rows.append(dict(id=sid, npz=fp, json=pair_json(fp),
                         label=int(r.label),
                         species=str(getattr(r, "species", "")),
                         drug=str(getattr(r, "drug", "")),
                         partition=str(getattr(r, "partition_label", ""))))
df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No matching ids between OUTRUN and labels.csv")

# ------------------- 汇总元信息 + pooled -------------------
nW_list, nF_list, var_list, mean_norm_list = [], [], [], []
P_list = []

for _, row in df.iterrows():
    X, spans = load_npz(row.npz)
    meta = load_meta(row.json)
    nW_list.append(int(X.shape[0]))
    nF_list.append(int(X.shape[1]))
    var_list.append(float(meta.get("var_explained_topk", np.nan)))
    mean_norm_list.append(float(per_window_norm(X).mean()))
    P_list.append(pool_vec(X, stats=("mean","std")))

P = np.vstack(P_list)                 # [N, 2*nF]
y = df["label"].values.astype(int)

# 保存形状与元信息
with open(os.path.join(OUTDIR, "pooled_shapes.txt"), "w") as fh:
    fh.write(f"N={P.shape[0]}, pooled_dims={P.shape[1]}\n")
meta_df = pd.DataFrame({
    "id": df["id"], "label": y, "species": df["species"], "drug": df["drug"],
    "n_windows": nW_list, "n_features": nF_list, "var_explained_topk": var_list
})
meta_df.to_csv(os.path.join(OUTDIR, "metadata_summary.csv"), index=False)
print(f"Saved metadata_summary.csv to {OUTDIR}")

# ------------------- 可视化 -------------------
# 1. n_windows分布
fig = plt.figure(figsize=(6,4))
plt.hist(nW_list, bins=30, alpha=0.6)
plt.xlabel("n_windows"); plt.ylabel("count")
plt.title("Distribution of n_windows")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "dist_n_windows.png"), dpi=100)
plt.close()

# 2. mean_norm分布
fig = plt.figure(figsize=(6,4))
plt.hist(mean_norm_list, bins=30, alpha=0.6)
plt.xlabel("mean window norm"); plt.ylabel("count")
plt.title("Distribution of mean window norms")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "dist_mean_norm.png"), dpi=100)
plt.close()

# 3. PCA pooled by label
scaler = StandardScaler()
P_scaled = scaler.fit_transform(P)
pca = PCA(n_components=2)
P_pca = pca.fit_transform(P_scaled)

fig = plt.figure(figsize=(8,6))
for lbl in np.unique(y):
    mask = (y == lbl)
    plt.scatter(P_pca[mask,0], P_pca[mask,1], alpha=0.5, label=f"label={lbl}")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.title("PCA of pooled features")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "pca_pooled_by_label.png"), dpi=100)
plt.close()

# 4. 时序窗口规范按标签
fig, axes = plt.subplots(1,2, figsize=(12,5))
for lbl in [0,1]:
    ax = axes[lbl]
    mask = (y == lbl)
    subset = df[mask].sample(min(MAX_LINES, mask.sum()), random_state=42)
    for _, row in subset.iterrows():
        X, _ = load_npz(row.npz)
        norms = per_window_norm(X)
        ax.plot(norms, alpha=0.3)
    ax.set_title(f"Label={lbl} (n={mask.sum()})")
    ax.set_xlabel("window index")
    ax.set_ylabel("L2 norm")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "timeseries_window_norms_by_label.png"), dpi=100)
plt.close()

# 5. 类平均曲线
fig = plt.figure(figsize=(8,5))
for lbl in [0,1]:
    mask = (y == lbl)
    subset = df[mask]
    curves = []
    for _, row in subset.iterrows():
        X, _ = load_npz(row.npz)
        norms = per_window_norm(X)
        bins = normalized_bins(len(norms), CLASS_AVG_BINS)
        curve = np.zeros(CLASS_AVG_BINS)
        np.add.at(curve, bins, norms)
        counts = np.bincount(bins, minlength=CLASS_AVG_BINS)
        curve = curve / (counts + 1e-9)
        curves.append(curve)
    avg_curve = np.mean(curves, axis=0)
    plt.plot(avg_curve, label=f"label={lbl} (n={len(curves)})")
plt.xlabel("normalized position")
plt.ylabel("avg window norm")
plt.title("Class-average curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "class_avg_curves.png"), dpi=100)
plt.close()

# 6. 热图样本
for lbl in [0,1]:
    mask = (y == lbl)
    subset = df[mask].sample(min(HEATMAP_EXAMPLE_PER_CLASS, mask.sum()), random_state=42)
    for _, row in subset.iterrows():
        X, _ = load_npz(row.npz)
        if X.shape[1] > HEATMAP_FEATS_SUBSAMPLE:
            idx = np.random.RandomState(42).choice(X.shape[1], HEATMAP_FEATS_SUBSAMPLE, replace=False)
            X = X[:, idx]
        fig = plt.figure(figsize=(10,6))
        plt.imshow(X.T, aspect="auto", interpolation="nearest", cmap="viridis")
        plt.colorbar()
        plt.xlabel("window index")
        plt.ylabel("feature index")
        plt.title(f"Sample {row.id} (label={lbl})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"heatmap_sample_{row.id}.png"), dpi=100)
        plt.close()

# 7. kNN neighborhood purity
from sklearn.neighbors import NearestNeighbors

purities = {k: [] for k in K_LIST}
for k in K_LIST:
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(P_scaled)
    dists, indices = nn.kneighbors(P_scaled)
    for i in range(len(y)):
        neighbors = indices[i, 1:]  # exclude self
        same_label = (y[neighbors] == y[i]).mean()
        purities[k].append(same_label)

fig = plt.figure(figsize=(8,5))
for k in K_LIST:
    plt.hist(purities[k], bins=20, alpha=0.5, label=f"k={k}")
plt.xlabel("neighborhood purity")
plt.ylabel("count")
plt.title("kNN Neighborhood Purity Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "knn_neighborhood_purity.png"), dpi=100)
plt.close()

# 8. NN距离直方图：same vs diff
nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
nn.fit(P_scaled)
dists, indices = nn.kneighbors(P_scaled)
nn_dists = dists[:, 1]  # distance to nearest neighbor
same_mask = (y[indices[:, 1]] == y)
diff_mask = ~same_mask

fig = plt.figure(figsize=(8,5))
plt.hist(nn_dists[same_mask], bins=30, alpha=0.5, label="same label", density=True)
plt.hist(nn_dists[diff_mask], bins=30, alpha=0.5, label="diff label", density=True)
plt.xlabel("distance to nearest neighbor")
plt.ylabel("density")
plt.title("NN Distance Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "nn_distance_hist_same_vs_diff.png"), dpi=100)
plt.close()

# 9. 基线分类结果
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(P_scaled, y)):
    X_train, X_test = P_scaled[train_idx], P_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    ap_lr = average_precision_score(y_test, y_prob_lr)
    
    # Linear SVM
    svm = CalibratedClassifierCV(LinearSVC(max_iter=1000, random_state=42))
    svm.fit(X_train, y_train)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    auc_svm = roc_auc_score(y_test, y_prob_svm)
    ap_svm = average_precision_score(y_test, y_prob_svm)
    
    # kNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_prob_knn = knn.predict_proba(X_test)[:, 1]
    auc_knn = roc_auc_score(y_test, y_prob_knn)
    ap_knn = average_precision_score(y_test, y_prob_knn)
    
    results.append({
        "fold": fold,
        "lr_auc": auc_lr, "lr_ap": ap_lr,
        "svm_auc": auc_svm, "svm_ap": ap_svm,
        "knn_auc": auc_knn, "knn_ap": ap_knn
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTDIR, "baseline_results.csv"), index=False)

print("\n=== Baseline Results (5-fold CV) ===")
print(results_df.describe())
print(f"\nAll outputs saved to {OUTDIR}")
