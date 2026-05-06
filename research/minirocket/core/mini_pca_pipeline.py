#!/insomnia001/depts/pmg/users/ht2666/mambaforge/envs/evo-amr/bin/python
# -*- coding: utf-8 -*-
"""
Generalized Windowed MiniRocket + PCA pipeline for any drug
-----------------------------------------------------------
- fit : Streamed GLOBAL PCA over many H5s, then fit ONE MiniRocket on
        projected, windowed samples (L=win, default 2048).
- run : Apply the saved projector to each H5 to:
        1. Save PCA-only features (6 stats per PC) in pca_only/
        2. Apply MiniRocket with sliding windows and save in pca_minirocket/
- pool: Aggregate window-level features into a sample-level vector via
        statistics pooling (mean/max/std). Produces V:[1, pooled_dims].

H5 expected (per file):
  /embeddings : [T, D]  float32/float64 (e.g., D=4096)
  /valid_mask : [T] or [T,1] optional (>0.5 => valid)

Install deps:
  pip install -U numpy h5py joblib sktime numba pandas

Outputs are organized as:
  /insomnia001/depts/pmg/users/ht2666/processed_embedding/
    ├── <split>-<drug>/
    │   ├── pca_only/
    │   │   └── *_pca_stats_dim{k}.npz files (6 stats per PC)
    │   ├── pca_minirocket/
    │   │   └── *_mr_win{win}_s{stride}.npz files
    │   └── projector.npz  (shared PCA metadata)
"""

import argparse, glob, json, time, warnings, os
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
from joblib import dump, load

# ---------- threading & env (set BEFORE importing numba users) ----------
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "8")

try:
    import numba
    _t = int(os.environ.get("NUMBA_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "8")))
    _t = max(1, min(16, _t))
    numba.set_num_threads(_t)
except Exception:
    pass

# ----------------------- file utils -----------------------
def _iter_files(path_like):
    p = Path(path_like)
    if p.is_dir():
        return sorted([str(x) for x in p.glob("*.h5")] + [str(x) for x in p.glob("*.hdf5")])
    return sorted(glob.glob(str(path_like)))

def _load_train_test_split(labels_csv, drug_name):
    """Load train/test split from labels CSV file for specific drug."""
    df = pd.read_csv(labels_csv)
    # CSV is already drug-specific, no need to filter by antibiotic_label
    # (antibiotic_label contains 0/1 resistance values, not drug names)
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
    """
    Return sorted row indices to use (uniform sample over full length, mask-aware).
    - If max_tokens is None or <=0 or larger than available rows, returns all valid rows.
    """
    if mask is None:
        idx_all = np.arange(T, dtype=np.int64)
    else:
        idx_all = np.where(mask[:T])[0].astype(np.int64)

    if (max_tokens is None) or (max_tokens <= 0) or (len(idx_all) <= max_tokens):
        return idx_all

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(idx_all), size=max_tokens, replace=False)
    return np.sort(idx_all[sel])

# -------------------- streamed PCA (global) --------------------
def cov_pca_streamed(files, max_tokens=None, use_mask=True, batch=32768, dtype=np.float64, seed=42):
    """
    Compute global PCA stats with uniform sampling across each file.
    - max_tokens: number of rows per file to sample uniformly (<=0 or None => no limit).
    """
    mu = None; n_tot = 0; D = None

    # pass 1: mean
    for fp in files:
        with h5py.File(fp, "r") as f:
            if "embeddings" not in f: continue
            X = f["embeddings"]; T, Df = int(X.shape[0]), int(X.shape[1])
            if D is None: D = Df
            elif D != Df: raise ValueError(f"Dim mismatch: {fp}")

            m = None
            if use_mask and "valid_mask" in f:
                m = (np.squeeze(f["valid_mask"][:]) > 0.5)

            sel_idx = _choose_rows(m, T, max_tokens, seed=seed)

            if mu is None: mu = np.zeros(D, dtype=dtype)
            for s in range(0, len(sel_idx), batch):
                sl = sel_idx[s:s+batch]
                x = np.asarray(X[sl, :], dtype=dtype, order="C")
                mu += x.sum(axis=0); n_tot += x.shape[0]
    if n_tot == 0:
        raise RuntimeError("No valid rows for PCA")
    mu /= n_tot

    # pass 2: covariance
    C = np.zeros((D, D), dtype=dtype)
    for fp in files:
        with h5py.File(fp, "r") as f:
            if "embeddings" not in f: continue
            X = f["embeddings"]; T = int(X.shape[0])

            m = None
            if use_mask and "valid_mask" in f:
                m = (np.squeeze(f["valid_mask"][:]) > 0.5)

            sel_idx = _choose_rows(m, T, max_tokens, seed=seed)

            for s in range(0, len(sel_idx), batch):
                sl = sel_idx[s:s+batch]
                x = np.asarray(X[sl, :], dtype=dtype, order="C")
                xc = x - mu
                C += xc.T @ xc
    C /= max(n_tot - 1, 1)

    # eig on symmetric C
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w = w[order].astype(np.float32)
    Vt = V[:, order].T.astype(np.float32)
    var = (w / (w.sum() + 1e-12)).astype(np.float32)
    return var, Vt, mu.astype(np.float32)

# -------------------- projection helpers --------------------
def build_P(Vt, k):
    return Vt[:k, :].T.copy()

def project_dataset(E_ds, P, mu, valid_mask=None, batch=32768, T_cap=None, dtype=np.float32):
    T = int(E_ds.shape[0]); k = int(P.shape[1])
    if T_cap: T = min(T, T_cap)
    keep_idx = _valid_idx(valid_mask, T)
    Ep = np.empty((len(keep_idx), k), dtype=dtype)
    w = 0
    for s in range(0, T, batch):
        e = min(s+batch, T)
        X = np.asarray(E_ds[s:e, :], dtype=np.float32, order="C")
        X = X - mu
        Y = X @ P
        if valid_mask is None:
            Ep[w:w+Y.shape[0]] = Y; w += Y.shape[0]
        else:
            loc = keep_idx[(keep_idx >= s) & (keep_idx < e)] - s
            if loc.size:
                Ep[w:w+loc.size] = Y[loc]; w += loc.size
    return Ep

# -------------------- windowing --------------------
def spans_sliding(T_valid, win=2048, stride=1024, cover_tail=False):
    if T_valid <= 0: return []
    if T_valid < win:
        return [(0, T_valid)]
    n = ((T_valid - win) // stride) + 1
    spans = [(i*stride, i*stride + win) for i in range(n)]
    if cover_tail:
        last_end = spans[-1][1]
        if last_end < T_valid:
            tail = (max(T_valid - win, spans[-1][0]), max(T_valid, spans[-1][1]))
            if tail != spans[-1]:
                spans.append(tail)
    return spans

def to_case3d(Ep, s, e, win):
    X = Ep[s:e]
    k = Ep.shape[1]
    if e - s < win:
        pad = np.zeros((win - (e - s), k), dtype=Ep.dtype)
        X = np.vstack([X, pad])
    return np.swapaxes(X, 0, 1)[None, :, :]

# -------------------- 6 stats over time per PC --------------------
STATS_ORDER = ['mean', 'std', 'max', 'min', 'q25', 'q75']

def stats_6_over_time(Ep: np.ndarray) -> np.ndarray:
    """Ep: [T_valid, k]; returns V:[1, 6*k] in STATS_ORDER per PC.
    """
    if Ep.ndim != 2 or Ep.shape[0] == 0:
        return np.empty((1, 0), dtype=np.float32)
    # axis=0 is time; compute per-column stats
    mean = Ep.mean(axis=0, keepdims=True)
    std  = Ep.std(axis=0, keepdims=True)
    mx   = Ep.max(axis=0,  keepdims=True)
    mn   = Ep.min(axis=0,  keepdims=True)
    q25  = np.percentile(Ep, 25.0, axis=0, keepdims=True)
    q75  = np.percentile(Ep, 75.0, axis=0, keepdims=True)
    V = np.concatenate([mean, std, mx, mn, q25, q75], axis=1).astype(np.float32, copy=False)  # [1, 6*k]
    return V

# -------------------- MiniRocket --------------------
def _require_minirocket():
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate
    except Exception as e:
        raise RuntimeError("Please install: pip install -U sktime numba") from e

def fit_minirocket_for_window(files, projector, proj_dim, win, n_fit_windows_per_file,
                              num_kernels, max_dilations_per_kernel, batch, max_tokens,
                              seed, n_jobs, zscore):
    """Fit ONE MiniRocket (window=win) using sampled windows across files (after PCA)."""
    Vt, var, mu = projector["Vt"], projector["var"], projector["mu"]
    P = build_P(Vt, proj_dim)
    samples = []
    rng = np.random.default_rng(seed)

    for fp in files:
        with h5py.File(fp, "r") as f:
            if "embeddings" not in f: continue
            X = f["embeddings"]; T = int(X.shape[0])

            vm = None
            if "valid_mask" in f:
                m = np.squeeze(f["valid_mask"][:])
                vm = (m > 0.5)

            Ep = project_dataset(X, P, mu, valid_mask=vm, batch=batch, T_cap=None)
            if zscore:
                Ep = (Ep - Ep.mean(axis=0, keepdims=True)) / (Ep.std(axis=0, keepdims=True) + 1e-8)

            T_valid = Ep.shape[0]
            spans = spans_sliding(T_valid, win=win, stride=win, cover_tail=False)
            if not spans:
                spans = [(0, T_valid)]

            choose = min(n_fit_windows_per_file, len(spans))
            idxs = np.sort(rng.choice(len(spans), size=choose, replace=False))
            for j in idxs:
                s, e = spans[j]
                Xi = to_case3d(Ep, s, e, win)
                samples.append(Xi)

    if not samples:
        raise RuntimeError("No windows collected to fit MiniRocket.")
    X_fit = np.concatenate(samples, axis=0)

    MiniRocketMultivariate = _require_minirocket()
    mr = MiniRocketMultivariate(
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=seed,
        n_jobs=1
    )
    mr.fit(X_fit)
    return mr

# -------------------- commands --------------------
def cmd_fit(args):
    files = _iter_files(args.path)
    
    # Create output directory structure
    base_outdir = Path(args.base_outdir)
    split_drug = f"{args.split}-{args.drug}"
    outdir = base_outdir / split_drug
    pca_minirocket_dir = outdir / "pca_minirocket"
    pca_minirocket_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train/test split if labels file provided
    train_files = files
    if args.labels_csv:
        train_ids = _load_train_test_split(args.labels_csv, args.drug)
        train_files = _filter_train_files(files, train_ids)
        print(f"[FIT] Using {len(train_files)} training files for {args.drug} out of {len(files)} total files")
        if not train_files:
            raise ValueError(f"No training files found for {args.drug}")
    
    # Sample training files if needed
    if args.sample_train_genomes > 0 and len(train_files) > args.sample_train_genomes:
        import random
        random.Random(args.seed).shuffle(train_files)
        train_files = train_files[:args.sample_train_genomes]
        print(f"[FIT] Sampled {args.sample_train_genomes} training genomes for PCA fitting")
    
    if args.shuffle:
        import random; random.Random(args.seed).shuffle(train_files)
    train_files = train_files[:args.max_files]
    if not train_files: raise FileNotFoundError("No training files to fit")

    t0 = time.time()

    max_tokens = None if (args.max_tokens is None or args.max_tokens <= 0) else args.max_tokens

    # Apply PCA only on training data
    var, Vt, mu = cov_pca_streamed(
        train_files, max_tokens=max_tokens, use_mask=(not args.no_mask),
        batch=args.batch, seed=args.seed
    )
    
    # Save projector in the split-drug directory
    proj_path = outdir / "projector.npz"
    np.savez_compressed(proj_path, var=var, Vt=Vt, mu=mu, 
                       split=args.split, drug=args.drug,
                       n_train_files=len(train_files))
    print(f"[FIT] projector -> {proj_path}")

    # Fit MiniRocket also only on training data
    mr = fit_minirocket_for_window(
        files=train_files,
        projector=dict(var=var, Vt=Vt, mu=mu),
        proj_dim=args.proj_dim,
        win=args.win,
        n_fit_windows_per_file=args.n_fit_windows_per_file,
        num_kernels=args.num_kernels,
        max_dilations_per_kernel=args.max_dilations_per_kernel,
        batch=args.batch,
        max_tokens=max_tokens,
        seed=args.seed,
        n_jobs=1,
        zscore=args.zscore
    )
    mr_path = pca_minirocket_dir / "minirocket.joblib"
    dump(mr, mr_path)
    print(f"[FIT] MiniRocket (win={args.win}) -> {mr_path}")
    print(f"[DONE] fit for {args.drug} in {time.time()-t0:.1f}s")

def cmd_run(args):
    files = _iter_files(args.path)
    if args.shuffle:
        import random; random.Random(args.seed).shuffle(files)
    files = files[:args.n_files]
    if not files: raise FileNotFoundError("No files to run")

    # Setup output directory structure
    base_outdir = Path(args.base_outdir)
    split_drug = f"{args.split}-{args.drug}"
    pca_only_dir = base_outdir / split_drug / "pca_only"
    pca_minirocket_dir = base_outdir / split_drug / "pca_minirocket"
    pca_only_dir.mkdir(parents=True, exist_ok=True)
    pca_minirocket_dir.mkdir(parents=True, exist_ok=True)
    
    # Load projector and minirocket
    proj_path = base_outdir / split_drug / "projector.npz"
    mr_path = pca_minirocket_dir / "minirocket.joblib"
    
    if not proj_path.exists():
        raise FileNotFoundError(f"Projector not found: {proj_path}")
    if not mr_path.exists():
        raise FileNotFoundError(f"MiniRocket model not found: {mr_path}")
    
    proj = np.load(proj_path)
    Vt, var, mu = proj["Vt"], proj["var"], proj["mu"]
    P = build_P(Vt, args.proj_dim)
    
    try:
        mr = load(mr_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load MiniRocket: {e}")

    run_max_tokens = None if (args.max_tokens is None or args.max_tokens <= 0) else args.max_tokens

    for fp in files:
        t0 = time.time()
        try:
            stem = Path(fp).stem
            
            # Check if both outputs already exist
            pca_out_npz = pca_only_dir / f"{stem}_pca_stats_dim{args.proj_dim}.npz"
            mr_base = f"{stem}_mr_win{args.win}_s{args.stride}"
            mr_out_npz = pca_minirocket_dir / f"{mr_base}.npz"
            mr_out_json = pca_minirocket_dir / f"{mr_base}.json"
            
            if (not args.overwrite) and pca_out_npz.exists() and mr_out_npz.exists() and mr_out_json.exists():
                print(f"[SKIP] {fp} -> both outputs already exist")
                continue

            with h5py.File(fp, "r") as f:
                if "embeddings" not in f:
                    warnings.warn(f"{fp}: missing /embeddings"); continue
                X = f["embeddings"]; T = int(X.shape[0])

                vm = None
                if "valid_mask" in f:
                    m = np.squeeze(f["valid_mask"][:])
                    vm = (m > 0.5)

                # Project to PCA space
                Ep = project_dataset(X, P, mu, valid_mask=vm, batch=args.batch, T_cap=run_max_tokens)
                
                # Save PCA-only features (6 stats per PC)
                if not pca_out_npz.exists() or args.overwrite:
                    V_pca = stats_6_over_time(Ep)  # [1, 6*k]
                    pca_meta = dict(
                        input_file=str(Path(fp).resolve()),
                        T_used=int(Ep.shape[0]),
                        proj_dim=int(args.proj_dim),
                        var_explained_topk=float(var[:args.proj_dim].sum()),
                        stats_order=STATS_ORDER,
                        drug=args.drug,
                        split=args.split,
                        elapsed_sec=round(time.time()-t0, 2),
                    )
                    np.savez_compressed(pca_out_npz, V=V_pca.astype(np.float32, copy=False), **pca_meta)
                    print(f"[PCA] {stem}: V.shape={V_pca.shape} -> {pca_out_npz}")
                
                # Apply z-scoring for MiniRocket if needed
                if args.zscore:
                    Ep_mr = (Ep - Ep.mean(axis=0, keepdims=True)) / (Ep.std(axis=0, keepdims=True) + 1e-8)
                else:
                    Ep_mr = Ep

            # Generate MiniRocket features
            if not mr_out_npz.exists() or args.overwrite:
                T_valid = Ep_mr.shape[0]
                spans = spans_sliding(T_valid, win=args.win, stride=args.stride, cover_tail=args.cover_tail)
                nW = len(spans)

                feats = []
                for i in range(0, nW, args.win_batch):
                    batch_spans = spans[i:i+args.win_batch]
                    Xb = np.concatenate([to_case3d(Ep_mr, s, e, args.win) for (s, e) in batch_spans], axis=0)
                    Fb = mr.transform(Xb)
                    feats.append(Fb.astype(np.float32, copy=False))
                X_win = np.vstack(feats) if feats else np.empty((0, 0), dtype=np.float32)

                np.savez_compressed(mr_out_npz,
                                    X=X_win,
                                    spans=np.asarray(spans, dtype=np.int64),
                                    win=int(args.win),
                                    stride=int(args.stride),
                                    proj_dim=int(args.proj_dim),
                                    drug=args.drug,
                                    split=args.split)
                info = dict(
                    file=str(Path(fp).resolve()),
                    T_used=int(T if run_max_tokens is None else min(T, run_max_tokens)),
                    T_valid=int(T_valid),
                    proj_dim=int(args.proj_dim),
                    var_explained_topk=float(var[:args.proj_dim].sum()),
                    n_windows=int(nW),
                    n_features=int(X_win.shape[1] if X_win.size else 0),
                    formula_n_windows="floor((T_valid-win)/stride)+1 if T_valid>=win else 1",
                    cover_tail=bool(args.cover_tail),
                    drug=args.drug,
                    split=args.split,
                    elapsed_sec=round(time.time()-t0, 2)
                )
                with open(mr_out_json, "w") as fh:
                    json.dump(info, fh, indent=2)
                print(f"[MiniRocket] {stem}: windows={nW} feats/Win={info['n_features']} -> {mr_out_npz}")
            
            print(f"[DONE] {stem} processed in {time.time()-t0:.1f}s")
        except Exception as e:
            warnings.warn(f"{fp}: {e}")

def cmd_pool(args):
    """Aggregate window-level features into a sample-level vector via stats pooling."""
    base_outdir = Path(args.base_outdir)
    split_drug = f"{args.split}-{args.drug}"
    indir = base_outdir / split_drug / "pca_minirocket"
    outdir = indir  # Save pooled results in the same directory
    
    files = sorted(indir.glob("*_mr_win*_s*.npz"))
    if not files:
        raise FileNotFoundError(f"No *_mr_win*_s*.npz (MiniRocket outputs) to pool in {indir}")
    
    stats = [s.strip() for s in args.stats.split(",") if s.strip()]
    for fp in files:
        z = np.load(fp)
        X = z["X"].astype(np.float32, copy=False)
        if X.ndim != 2 or X.shape[0] == 0:
            warnings.warn(f"{fp}: empty or invalid X; skip")
            continue
        vecs = []
        if "mean" in stats: vecs.append(X.mean(axis=0, keepdims=True))
        if "max"  in stats: vecs.append(X.max(axis=0,  keepdims=True))
        if "std"  in stats: vecs.append(X.std(axis=0,  keepdims=True))
        if not vecs:
            warnings.warn("No valid stats selected; use mean by default")
            vecs = [X.mean(axis=0, keepdims=True)]
        V = np.concatenate(vecs, axis=1).astype(np.float32, copy=False)
        base = Path(fp).stem
        out_npz = outdir / f"{base}_pooled.npz"
        np.savez_compressed(out_npz, V=V, stats=np.array(stats), 
                          drug=args.drug, split=args.split)
        print(f"[POOL] {base}: pooled_dims={V.shape[1]} -> {out_npz}")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser("Generalized Windowed MiniRocket Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # fit (GLOBAL PCA + MiniRocket @win)
    apf = sub.add_parser("fit", help="Fit global PCA + MiniRocket for window")
    apf.add_argument("path", type=str, help="Dir or glob of .h5/.hdf5")
    apf.add_argument("--base-outdir", type=str, default="/insomnia001/depts/pmg/users/ht2666/processed_embedding",
                    help="Base output directory for all processed embeddings")
    apf.add_argument("--drug", type=str, required=True, help="Drug name (e.g., ampicillin, gentamicin)")
    apf.add_argument("--split", type=str, required=True, help="Split identifier (e.g., clustered_3_v1-1)")
    apf.add_argument("--labels-csv", type=str, help="CSV file with genome_id,partition_label,antibiotic_label")
    apf.add_argument("--sample-train-genomes", type=int, default=60, 
                    help="Number of training genomes to sample for PCA fitting (0=use all)")
    apf.add_argument("--max-files", type=int, default=300)
    apf.add_argument("--max-tokens", type=int, default=200000, help="Rows per file for PCA sampling")
    apf.add_argument("--batch", type=int, default=32768)
    apf.add_argument("--no-mask", action="store_true")
    apf.add_argument("--proj-dim", type=int, default=32)
    apf.add_argument("--win", type=int, default=2048)
    apf.add_argument("--n-fit-windows-per-file", type=int, default=8)
    apf.add_argument("--num-kernels", type=int, default=2000)
    apf.add_argument("--max-dilations-per-kernel", type=int, default=8)
    apf.add_argument("--seed", type=int, default=42)
    apf.add_argument("--shuffle", action="store_true")
    apf.add_argument("--zscore", action="store_true")

    # run (windowed transform across FULL sequence)
    apr = sub.add_parser("run", help="Project + sliding MiniRocket transform")
    apr.add_argument("path", type=str)
    apr.add_argument("--base-outdir", type=str, default="/insomnia001/depts/pmg/users/ht2666/processed_embedding")
    apr.add_argument("--drug", type=str, required=True, help="Drug name")
    apr.add_argument("--split", type=str, required=True, help="Split identifier")
    apr.add_argument("--proj-dim", type=int, default=32)
    apr.add_argument("--win", type=int, default=2048)
    apr.add_argument("--stride", type=int, default=1024)
    apr.add_argument("--cover-tail", action="store_true")
    apr.add_argument("--win-batch", type=int, default=64)
    apr.add_argument("--max-tokens", type=int, default=None)
    apr.add_argument("--batch", type=int, default=32768)
    apr.add_argument("--zscore", action="store_true")
    apr.add_argument("--n-files", type=int, default=100000)
    apr.add_argument("--overwrite", action="store_true")
    apr.add_argument("--shuffle", action="store_true")
    apr.add_argument("--seed", type=int, default=42)

    # pool: stats pooling to sample-level vector
    apo = sub.add_parser("pool", help="Stats-pool window features to sample-level vector")
    apo.add_argument("--base-outdir", type=str, default="/insomnia001/depts/pmg/users/ht2666/processed_embedding")
    apo.add_argument("--drug", type=str, required=True, help="Drug name")
    apo.add_argument("--split", type=str, required=True, help="Split identifier")
    apo.add_argument("--stats", type=str, default="mean,max", help="comma-separated: mean,max,std")

    args = ap.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "pool":
        cmd_pool(args)

if __name__ == "__main__":
    main()