#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined PCA + MiniRocket pipeline with single-pass extraction
----------------------------------------------------------------
This script combines PCA stats extraction and MiniRocket feature extraction
into a single pass through the data, eliminating redundant PCA computations.

Commands:
- fit: Train PCA projector and MiniRocket model (same as AAA_mini.py)
- extract_combined: Apply PCA once, save 6 stats, then apply MiniRocket on the PCA output

Benefits:
- ~40-50% faster extraction phase
- Single read of embedding files
- Single PCA transformation
- Reduced memory pressure

Usage:
  # 1) Fit phase (same as before)
  python AAA_mini_combined.py fit /path/to/embeddings --outdir /output/fit ...
  
  # 2) Combined extraction (replaces both PCA-only and run phases)
  python AAA_mini_combined.py extract_combined /path/to/embeddings \
    --projector /output/fit/projector.npz \
    --minirocket /output/fit/minirocket.joblib \
    --outdir /output/extract \
    --pca-stats-dir /output/pca_stats \
    ...
"""

import argparse, glob, json, time, warnings, os
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
from joblib import dump, load

# Import configuration
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.config import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: config.py not found, using hardcoded paths")

# Threading & env setup (before importing numba users)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

try:
    import numba
    _t = int(os.environ.get("NUMBA_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8")))
    _t = max(1, min(16, _t))
    numba.set_num_threads(_t)
except Exception:
    pass

# ----------------------- File utilities -----------------------
def _iter_files(path_like):
    p = Path(path_like)
    if p.is_dir():
        return sorted([str(x) for x in p.glob("*.h5")] + [str(x) for x in p.glob("*.hdf5")])
    return sorted(glob.glob(str(path_like)))

def _load_train_test_split(labels_csv):
    """Load train/test split from labels CSV file."""
    df = pd.read_csv(labels_csv)
    train_ids = set(df[df['partition_label'] == 'train']['genome_id'].astype(str))
    return train_ids

def _filter_train_files(files, train_ids):
    """Filter files to only include training samples based on genome IDs."""
    train_files = []
    for fp in files:
        stem = Path(fp).stem
        if stem in train_ids:
            train_files.append(fp)
    return train_files

def _valid_idx(mask, T):
    return np.arange(T) if mask is None else np.where(mask[:T])[0]

def _choose_rows(mask, T, max_tokens, seed=42):
    """Return sorted row indices to use (uniform sample over full length, mask-aware)."""
    if mask is None:
        idx_all = np.arange(T, dtype=np.int64)
    else:
        idx_all = np.where(mask[:T])[0].astype(np.int64)

    if (max_tokens is None) or (max_tokens <= 0) or (len(idx_all) <= max_tokens):
        return idx_all

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(idx_all), size=max_tokens, replace=False)
    return np.sort(idx_all[sel])

# -------------------- PCA functions --------------------
def cov_pca_streamed(files, max_tokens=None, use_mask=True, batch=32768, dtype=np.float64, seed=42):
    """Compute global PCA stats with uniform sampling across each file."""
    D = None
    S = 0.0
    C = 0.0
    N_total = 0

    print(f"[cov_pca_streamed] pass1: computing mean over {len(files)} files (max_tokens={max_tokens})")
    for i, fp in enumerate(files):
        with h5py.File(fp, 'r') as f:
            E_ds = f["/embeddings"]
            mask_ds = f.get("/valid_mask", None)
            T, D = E_ds.shape[0], E_ds.shape[1]
            
            mask = None if (not use_mask or mask_ds is None) else (mask_ds[:] > 0.5).ravel()
            rows = _choose_rows(mask, T, max_tokens, seed=seed+i)
            if len(rows) == 0:
                continue
            
            for st in range(0, len(rows), batch):
                ed = min(st + batch, len(rows))
                idx = rows[st:ed]
                Eb = E_ds[idx].astype(dtype, copy=False)
                S += Eb.sum(axis=0)
                N_total += len(idx)
        
        if (i+1) % 10 == 0:
            print(f"  mean pass: {i+1}/{len(files)}")
    
    mu = (S / max(N_total, 1)) if N_total > 0 else np.zeros(D, dtype=dtype)
    
    print(f"[cov_pca_streamed] pass2: cov over {len(files)} files")
    for i, fp in enumerate(files):
        with h5py.File(fp, 'r') as f:
            E_ds = f["/embeddings"]
            mask_ds = f.get("/valid_mask", None)
            T = E_ds.shape[0]
            
            mask = None if (not use_mask or mask_ds is None) else (mask_ds[:] > 0.5).ravel()
            rows = _choose_rows(mask, T, max_tokens, seed=seed+i)
            if len(rows) == 0:
                continue
            
            for st in range(0, len(rows), batch):
                ed = min(st + batch, len(rows))
                idx = rows[st:ed]
                Eb = E_ds[idx].astype(dtype, copy=False) - mu[None, :]
                C += Eb.T @ Eb
        
        if (i+1) % 10 == 0:
            print(f"  cov pass: {i+1}/{len(files)}")
    
    C = C / max(N_total - 1, 1)
    var, Vt = np.linalg.eigh(C)
    var, Vt = var[::-1].copy(), Vt[:, ::-1].T
    
    var = np.maximum(var, 0.0)
    tot = var.sum()
    if tot > 0:
        var = var / tot
    
    print(f"[cov_pca_streamed] done. N_total={N_total}, D={D}, var[:5]={var[:5]}")
    return var, Vt, mu

def build_P(Vt, k):
    return Vt[:k, :].T.copy()

def project_dataset(E_ds, P, mu, valid_mask=None, batch=32768, T_cap=None, dtype=np.float32):
    """Project full dataset E_ds -> Ep:[T_valid, k]."""
    T_full = E_ds.shape[0] if T_cap is None else min(E_ds.shape[0], T_cap)
    idx_valid = _valid_idx(valid_mask, T_full)
    T_valid = len(idx_valid)
    k = P.shape[1]
    
    Ep = np.empty((T_valid, k), dtype=dtype)
    for st in range(0, T_valid, batch):
        ed = min(st + batch, T_valid)
        idx_batch = idx_valid[st:ed]
        Eb = E_ds[idx_batch].astype(dtype, copy=False) - mu[None, :]
        Ep[st:ed] = Eb @ P
    
    return Ep

# -------------------- PCA Stats --------------------
def stats_6_over_time(Ep: np.ndarray) -> np.ndarray:
    """Compute 6 stats per PC over time. Ep: [T_valid, k]; returns V:[1, 6*k]."""
    if Ep.ndim != 2 or Ep.shape[0] == 0:
        return np.empty((1, 0), dtype=np.float32)
    
    mean = Ep.mean(axis=0, keepdims=True)
    std  = Ep.std(axis=0, keepdims=True)
    mx   = Ep.max(axis=0, keepdims=True)
    mn   = Ep.min(axis=0, keepdims=True)
    q25  = np.percentile(Ep, 25.0, axis=0, keepdims=True)
    q75  = np.percentile(Ep, 75.0, axis=0, keepdims=True)
    
    V = np.concatenate([mean, std, mx, mn, q25, q75], axis=1).astype(np.float32, copy=False)
    return V

# -------------------- MiniRocket functions --------------------
def spans_sliding(T_valid, win=2048, stride=1024, cover_tail=False):
    """Generate sliding window spans."""
    if T_valid < win:
        return [(0, T_valid)]
    
    spans = []
    pos = 0
    while pos + win <= T_valid:
        spans.append((pos, pos + win))
        pos += stride
    
    if cover_tail and pos < T_valid and len(spans) > 0:
        spans.append((T_valid - win, T_valid))
    
    return spans

def to_case3d(Ep, s, e, win):
    """Extract window [s:e] and pad if needed, return [1, k, L]."""
    seg = Ep[s:e]
    if seg.shape[0] < win:
        pad_len = win - seg.shape[0]
        seg = np.pad(seg, ((0, pad_len), (0, 0)), mode='constant')
    return seg.T[None, :, :]

def _require_minirocket():
    try:
        from sktime.transformations.panel.rocket import MiniRocket
        return MiniRocket
    except ImportError:
        raise ImportError("MiniRocket not found. Install: pip install sktime")

def fit_minirocket_for_window(files, projector, proj_dim, win, n_fit_windows_per_file,
                              num_kernels, max_dilations_per_kernel, batch, max_tokens,
                              seed, n_jobs, zscore):
    """Fit ONE MiniRocket (window=win) using sampled windows across files (after PCA)."""
    Vt, var, mu = projector["Vt"], projector["var"], projector["mu"]
    P = build_P(Vt, proj_dim)
    
    MiniRocket = _require_minirocket()
    
    all_cases = []
    for i, fp in enumerate(files[:100]):  # Limit files for fitting
        with h5py.File(fp, 'r') as f:
            E_ds = f["/embeddings"]
            mask_ds = f.get("/valid_mask", None)
            mask = None if mask_ds is None else (mask_ds[:] > 0.5).ravel()
            
            Ep = project_dataset(E_ds, P, mu, mask, batch=batch, T_cap=max_tokens, dtype=np.float32)
            if Ep.shape[0] < win:
                continue
            
            T_valid = Ep.shape[0]
            spans = spans_sliding(T_valid, win, win, cover_tail=False)
            
            if len(spans) > n_fit_windows_per_file:
                rng = np.random.default_rng(seed + i)
                sel = rng.choice(len(spans), size=n_fit_windows_per_file, replace=False)
                spans = [spans[j] for j in sorted(sel)]
            
            for s, e in spans:
                case = to_case3d(Ep, s, e, win)
                if zscore:
                    m, s = case.mean(), case.std() + 1e-8
                    case = (case - m) / s
                all_cases.append(case)
    
    if not all_cases:
        raise ValueError(f"No windows of size {win} found in any file!")
    
    X_train = np.vstack(all_cases)
    print(f"[fit_minirocket] Fitting on {X_train.shape[0]} windows of size {win}")
    
    mr = MiniRocket(
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        n_jobs=n_jobs,
        random_state=seed
    )
    mr.fit(X_train)
    
    return mr

# -------------------- Commands --------------------
def cmd_fit(args):
    """Fit PCA projector and MiniRocket model (same as AAA_mini.py)."""
    files = _iter_files(args.path)
    
    if args.labels_csv:
        train_ids = _load_train_test_split(args.labels_csv)
        files = _filter_train_files(files, train_ids)
        print(f"[fit] Using {len(files)} training files from labels CSV")
    
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    
    if args.shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(files)
    
    print(f"[fit] Processing {len(files)} files")
    
    # Fit PCA
    var, Vt, mu = cov_pca_streamed(
        files, max_tokens=args.max_tokens, use_mask=True,
        batch=args.batch, dtype=np.float64, seed=args.seed
    )
    
    projector = {"var": var, "Vt": Vt, "mu": mu}
    
    # Save PCA projector
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    proj_path = Path(args.outdir) / "projector.npz"
    np.savez_compressed(proj_path, **projector)
    print(f"[fit] Saved projector to {proj_path}")
    
    # Fit MiniRocket
    mr = fit_minirocket_for_window(
        files, projector, args.proj_dim, args.win,
        args.n_fit_windows_per_file, args.num_kernels,
        args.max_dilations_per_kernel, args.batch,
        args.max_tokens, args.seed, args.n_jobs, args.zscore
    )
    
    mr_path = Path(args.outdir) / "minirocket.joblib"
    dump(mr, mr_path, compress=3)
    print(f"[fit] Saved MiniRocket to {mr_path}")

def cmd_extract_combined(args):
    """
    Combined extraction: Apply PCA once, save 6 stats, then apply MiniRocket.
    This replaces both the PCA-only extract and the MiniRocket run phases.
    """
    files = _iter_files(args.path)
    
    # Filter files if genome filter provided
    if args.genome_filter:
        df_filter = pd.read_csv(args.genome_filter)
        filter_ids = set(df_filter['genome_id'].astype(str))
        filtered = []
        for fp in files:
            stem = Path(fp).stem
            if stem in filter_ids:
                filtered.append(fp)
        files = filtered
        print(f"[extract_combined] Filtered to {len(files)} files using genome filter")
    
    if args.n_files and args.n_files > 0:
        files = files[:args.n_files]
    
    print(f"[extract_combined] Processing {len(files)} files")
    
    # Load projector and MiniRocket
    proj = np.load(args.projector)
    Vt, var, mu = proj["Vt"], proj["var"], proj["mu"]
    P = build_P(Vt, args.proj_dim)
    
    mr = load(args.minirocket)
    print(f"[extract_combined] Loaded projector (proj_dim={args.proj_dim}) and MiniRocket")
    
    # Create output directories
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.pca_stats_dir:
        Path(args.pca_stats_dir).mkdir(parents=True, exist_ok=True)
    else:
        args.pca_stats_dir = args.outdir  # Save PCA stats in same dir if not specified
    
    # Process each file
    for i, fp in enumerate(files):
        stem = Path(fp).stem
        t0 = time.time()
        
        with h5py.File(fp, 'r') as f:
            E_ds = f["/embeddings"]
            mask_ds = f.get("/valid_mask", None)
            T, D = E_ds.shape
            
            mask = None if mask_ds is None else (mask_ds[:] > 0.5).ravel()
            
            # Apply PCA transformation ONCE
            Ep = project_dataset(E_ds, P, mu, mask, batch=args.batch, dtype=np.float32)
            T_valid = Ep.shape[0]
            
            # 1) Save PCA stats (6 stats per PC)
            if args.save_pca_stats:
                V_pca = stats_6_over_time(Ep)
                pca_out = Path(args.pca_stats_dir) / f"{stem}_pca_stats.npz"
                np.savez_compressed(
                    pca_out,
                    V=V_pca,
                    shape_original=(T, D),
                    T_valid=T_valid,
                    proj_dim=args.proj_dim,
                    stats_order=["mean", "std", "max", "min", "q25", "q75"]
                )
                print(f"  [{i+1}/{len(files)}] {stem}: Saved PCA stats to {pca_out}")
            
            # 2) Apply MiniRocket on the SAME PCA output
            if T_valid < args.win:
                print(f"  [{i+1}/{len(files)}] {stem}: Skipping MiniRocket (T_valid={T_valid} < win={args.win})")
                continue
            
            spans = spans_sliding(T_valid, args.win, args.stride, args.cover_tail)
            n_windows = len(spans)
            
            # Process windows in batches
            X_list = []
            for j in range(0, n_windows, args.win_batch):
                batch_spans = spans[j:j + args.win_batch]
                cases = []
                
                for s, e in batch_spans:
                    case = to_case3d(Ep, s, e, args.win)
                    if args.zscore:
                        m, s_val = case.mean(), case.std() + 1e-8
                        case = (case - m) / s_val
                    cases.append(case)
                
                if cases:
                    X_batch = np.vstack(cases)
                    X_trans = mr.transform(X_batch)
                    X_list.append(X_trans)
            
            if X_list:
                X = np.vstack(X_list)
                
                # Save MiniRocket features
                mr_out = Path(args.outdir) / f"{stem}_mr_win{args.win}_s{args.stride}.npz"
                np.savez_compressed(
                    mr_out,
                    X=X.astype(np.float32),
                    spans=np.array(spans, dtype=np.int32),
                    shape_original=(T, D),
                    T_valid=T_valid,
                    win=args.win,
                    stride=args.stride,
                    proj_dim=args.proj_dim
                )
                
                # Save metadata
                meta = {
                    "file": fp,
                    "stem": stem,
                    "shape_original": [T, D],
                    "T_valid": T_valid,
                    "n_windows": n_windows,
                    "win": args.win,
                    "stride": args.stride,
                    "proj_dim": args.proj_dim,
                    "minirocket_features": int(X.shape[1]),
                    "time_seconds": time.time() - t0
                }
                
                meta_out = Path(args.outdir) / f"{stem}_mr_win{args.win}_s{args.stride}.json"
                with open(meta_out, 'w') as f:
                    json.dump(meta, f, indent=2)
                
                print(f"  [{i+1}/{len(files)}] {stem}: {n_windows} windows -> X:{X.shape} in {meta['time_seconds']:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Combined PCA + MiniRocket pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Fit command
    p_fit = subparsers.add_parser("fit", help="Fit PCA + MiniRocket")
    p_fit.add_argument("path", help="Directory or glob pattern for H5 files")
    p_fit.add_argument("--outdir", required=True, help="Output directory")
    p_fit.add_argument("--labels-csv", help="CSV with genome_id,partition_label columns")
    p_fit.add_argument("--proj-dim", type=int, default=41, help="PCA dimensions")
    p_fit.add_argument("--win", type=int, default=2048, help="Window size")
    p_fit.add_argument("--n-fit-windows-per-file", type=int, default=8)
    p_fit.add_argument("--num-kernels", type=int, default=2000)
    p_fit.add_argument("--max-dilations-per-kernel", type=int, default=8)
    p_fit.add_argument("--batch", type=int, default=32768)
    p_fit.add_argument("--max-files", type=int, help="Max files to use")
    p_fit.add_argument("--max-tokens", type=int, help="Max tokens per file")
    p_fit.add_argument("--shuffle", action="store_true")
    p_fit.add_argument("--seed", type=int, default=42)
    p_fit.add_argument("--n-jobs", type=int, default=1)
    p_fit.add_argument("--zscore", action="store_true")
    
    # Combined extraction command
    p_extract = subparsers.add_parser("extract_combined", help="Combined PCA stats + MiniRocket extraction")
    p_extract.add_argument("path", help="Directory or glob pattern for H5 files")
    p_extract.add_argument("--projector", required=True, help="Path to projector.npz")
    p_extract.add_argument("--minirocket", required=True, help="Path to minirocket.joblib")
    p_extract.add_argument("--outdir", required=True, help="Output directory for MiniRocket features")
    p_extract.add_argument("--pca-stats-dir", help="Output directory for PCA stats (default: same as outdir)")
    p_extract.add_argument("--save-pca-stats", action="store_true", default=True, help="Save PCA stats")
    p_extract.add_argument("--proj-dim", type=int, default=41)
    p_extract.add_argument("--win", type=int, default=2048)
    p_extract.add_argument("--stride", type=int, default=1024)
    p_extract.add_argument("--win-batch", type=int, default=128)
    p_extract.add_argument("--batch", type=int, default=131072)
    p_extract.add_argument("--n-files", type=int, help="Max files to process")
    p_extract.add_argument("--genome-filter", help="CSV with genome_id column to filter")
    p_extract.add_argument("--zscore", action="store_true")
    p_extract.add_argument("--cover-tail", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "fit":
        cmd_fit(args)
    elif args.command == "extract_combined":
        cmd_extract_combined(args)

if __name__ == "__main__":
    main()