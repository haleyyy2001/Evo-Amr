#!/insomnia001/depts/pmg/users/ht2666/mambaforge/envs/evo-amr/bin/python
# -*- coding: utf-8 -*-
"""
SRP/GRP + MiniRocket pipeline (revised)
---------------------------------------
- fit : Build random projection (SRP/GRP) and fit MiniRocket on SRP/GRP-projected windows.
- run : Apply projector to each H5, optionally z-score per sample, slide windows, get MiniRocket features.
        Optionally save: raw SRP sequence, z-scored SRP sequence, raw MiniRocket window features.
- pool: Aggregate window features to sample-level vectors via statistics pooling (+ optional Top-k pooling).

Key changes vs original:
- Optional GaussianRandomProjection (--rp-type grp).
- SRP/GRP "fit" samples from multiple files (more robust) instead of only the first file.
- project_dataset_rp(): uniform subsampling (mask-aware) instead of naive head truncation when --max-tokens is set.
- Fit uses stride=win//2 (denser windows) to better match run-time stride (default 512).
- Top-k pooling (--topk) to preserve sparse peaks.
- Optional saving of raw SRP (Ep), z-scored SRP (Ep_z), and raw MiniRocket features (X).
- Defaults tuned for sparse/local AMR signals (more kernels, slightly smaller dilation).

H5 expected (per file):
  /embeddings : [T, D]  float32/float64 (e.g., D=4096)
  /valid_mask : [T] or [T,1] optional (>0.5 => valid)

Install deps:
  pip install -U numpy h5py joblib sktime numba pandas scikit-learn
"""

import argparse, glob, json, time, warnings, os
from pathlib import Path
import numpy as np
import h5py
import pandas as pd
from joblib import dump, load
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection

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

def _load_train_ids(labels_csv):
    """Load train genome_id set from CSV with columns: genome_id, partition_label."""
    df = pd.read_csv(labels_csv)
    return set(df[df['partition_label'] == 'train']['genome_id'].astype(str))

def _filter_train_files(files, train_ids):
    """Keep only files whose stem is in train_ids."""
    out = []
    for fp in files:
        if Path(fp).stem in train_ids:
            out.append(fp)
    return out

def _valid_idx(mask, T):
    return np.arange(T, dtype=np.int64) if mask is None else np.where(mask[:T])[0].astype(np.int64)

def _choose_rows(mask, T, max_tokens, seed=42):
    """
    Uniformly sample indices over full length (mask-aware).
    Returns sorted indices (relative to [0, T)).
    """
    if (max_tokens is None) or (max_tokens <= 0):
        return np.arange(T, dtype=np.int64) if mask is None else np.where(mask[:T])[0].astype(np.int64)
    idx_all = np.arange(T, dtype=np.int64) if mask is None else np.where(mask[:T])[0].astype(np.int64)
    if len(idx_all) <= max_tokens:
        return idx_all
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(idx_all), size=max_tokens, replace=False)
    return np.sort(idx_all[sel])

# -------------------- Random Projection (SRP/GRP) --------------------
def fit_rp_projector(files, rp_dim, rp_type="srp", max_tokens=None, use_mask=True, seed=42):
    """
    Build a random projector by probing multiple files for dimensionality and (optionally) sample rows.
    For SRP/GRP, .fit() initializes components/random matrix; it does not learn data distribution.
    """
    input_dim = None
    sample_blocks = []
    n_probe = min(800, len(files))
    for fp in files[:n_probe]:
        with h5py.File(fp, "r") as f:
            if "embeddings" not in f:
                continue
            X = f["embeddings"]
            if input_dim is None:
                input_dim = int(X.shape[1])
            T = int(X.shape[0])
            vm = None
            if use_mask and "valid_mask" in f:
                vm = (np.squeeze(f["valid_mask"][:]) > 0.5)
            sel_idx = _choose_rows(vm, T, min(2000000, max_tokens) if max_tokens else 1000, seed=seed)
            if sel_idx.size > 0:
                sample_blocks.append(np.asarray(X[sel_idx, :], dtype=np.float32, order="C"))
    if input_dim is None:
        raise ValueError("Could not determine input dimensionality from files (no /embeddings).")
    sample_data = np.vstack(sample_blocks) if sample_blocks else None
    if sample_data is None:
        raise ValueError("Could not extract sample data for RP initialization")

    if rp_type.lower() == "grp":
        rp = GaussianRandomProjection(n_components=rp_dim, random_state=seed)
    else:
        rp = SparseRandomProjection(n_components=rp_dim, density='auto', random_state=seed)

    rp.fit(sample_data)  # initializes components/random matrix
    density = getattr(rp, "density_", None)
    print(f"[RP] Built {rp_type.upper()} projector: input {input_dim} -> {rp_dim}, density={density}")
    return rp, input_dim, density

def project_dataset_rp(E_ds, rp, valid_mask=None, batch=32768, max_tokens=None, dtype=np.float32, seed=42):
    """
    Project dataset using fitted random projector.
    Uniform subsampling (mask-aware) when max_tokens is set, instead of naive head truncation.
    """
    T = int(E_ds.shape[0])
    keep_idx = _valid_idx(valid_mask, T)
    if (max_tokens is not None) and (max_tokens > 0) and (len(keep_idx) > max_tokens):
        sel = _choose_rows(None, len(keep_idx), max_tokens, seed=seed)  # uniformly on valid timeline
        keep_idx = keep_idx[sel]

    rp_dim = rp.n_components
    Ep = np.empty((len(keep_idx), rp_dim), dtype=dtype)

    # process by batches from original array, select needed indices per batch
    w = 0
    for s in range(0, T, batch):
        e = min(s + batch, T)
        loc_mask = (keep_idx >= s) & (keep_idx < e)
        if not np.any(loc_mask):
            continue
        loc = keep_idx[loc_mask] - s
        X = np.asarray(E_ds[s:e, :], dtype=np.float32, order="C")
        Y = rp.transform(X).astype(dtype, copy=False)
        Ep[w:w+loc.size] = Y[loc]
        w += loc.size
    return Ep

# -------------------- windowing --------------------
def spans_sliding(T_valid, win=1024, stride=512, cover_tail=False):
    """Generate sliding windows with given win/stride on a valid-length sequence."""
    if T_valid <= 0:
        return []
    if T_valid < win:
        return [(0, T_valid)]
    n = ((T_valid - win) // stride) + 1
    spans = [(i * stride, i * stride + win) for i in range(n)]
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
    return np.swapaxes(X, 0, 1)[None, :, :]  # [1, channels=k, length=win]

# -------------------- MiniRocket --------------------
def _require_minirocket():
    try:
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        return MiniRocketMultivariate
    except Exception as e:
        raise RuntimeError("Please install: pip install -U sktime numba") from e

def fit_minirocket_for_window(files, rp_projector, rp_dim, win, n_fit_windows_per_file,
                              num_kernels, max_dilations_per_kernel, batch, max_tokens,
                              seed, n_jobs, zscore):
    """
    Fit ONE MiniRocket (window=win) using sampled windows across files (after RP).
    Uses stride=win//2 for denser coverage to match run-time overlap better.
    """
    samples = []
    rng = np.random.default_rng(seed)

    for fp in files:
        with h5py.File(fp, "r") as f:
            if "embeddings" not in f:
                continue
            X = f["embeddings"]; T = int(X.shape[0])

            vm = None
            if "valid_mask" in f:
                m = np.squeeze(f["valid_mask"][:])
                vm = (m > 0.5)

            Ep = project_dataset_rp(X, rp_projector, valid_mask=vm, batch=batch,
                                    max_tokens=max_tokens, dtype=np.float32, seed=seed)
            if zscore:
                Ep = (Ep - Ep.mean(axis=0, keepdims=True)) / (Ep.std(axis=0, keepdims=True) + 1e-8)

            T_valid = Ep.shape[0]
            spans = spans_sliding(T_valid, win=win, stride=max(1, win//2), cover_tail=False)
            if not spans:
                spans = [(0, T_valid)]

            if n_fit_windows_per_file == 0:
                idxs = np.arange(len(spans))
            else:
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
    base_outdir = Path(args.base_outdir)
    split_drug = f"{args.split}-{args.drug}"
    outdir = base_outdir / split_drug
    srp_minirocket_dir = outdir / "srp_minirocket"
    srp_minirocket_dir.mkdir(parents=True, exist_ok=True)

    # Build training file list
    if args.fit_manifest:
        manifest_path = Path(args.fit_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Fit manifest not found: {args.fit_manifest}")
        with manifest_path.open("r") as f:
            train_files = [line.strip() for line in f if line.strip()]
        missing = [f for f in train_files if not Path(f).exists()]
        if missing:
            print(f"[WARN] {len(missing)} files missing from manifest; using the rest.")
            train_files = [f for f in train_files if Path(f).exists()]
        print(f"[FIT] Using {len(train_files)} files from balanced manifest.")
    else:
        train_files = files
        if args.labels_csv:
            train_ids = _load_train_ids(args.labels_csv)
            train_files = _filter_train_files(files, train_ids)
            print(f"[FIT] Using {len(train_files)} training files out of {len(files)} total")
            if not train_files:
                raise ValueError("No training files found.")
        if args.sample_train_genomes > 0 and len(train_files) > args.sample_train_genomes:
            import random
            random.Random(args.seed).shuffle(train_files)
            train_files = train_files[:args.sample_train_genomes]
            print(f"[FIT] Sampled {args.sample_train_genomes} training genomes for RP fitting")
        if args.shuffle:
            import random; random.Random(args.seed).shuffle(train_files)
        train_files = train_files[:args.max_files]

    if not train_files:
        raise FileNotFoundError("No training files to fit")

    t0 = time.time()
    max_tokens = None if (args.max_tokens is None or args.max_tokens <= 0) else args.max_tokens
    rp_dims = [64, 128, 256] if args.srp_dim_grid else [args.srp_dim]

    for rp_dim in rp_dims:
        print(f"\n[FIT] === RP Dim {rp_dim} ({args.rp_type.upper()}) ===")
        if args.n_fit_windows_per_file == 0:
            print(f"[FIT] Using ALL windows from each file for MiniRocket fit (stride=win//2)")
        else:
            print(f"[FIT] Expected MiniRocket windows: {len(train_files)} × {args.n_fit_windows_per_file}")

        # Fit projector
        rp_projector, input_dim, density = fit_rp_projector(
            train_files, rp_dim=rp_dim, rp_type=args.rp_type,
            max_tokens=max_tokens, use_mask=(not args.no_mask), seed=args.seed
        )

        # Save projector metadata + object
        proj_path = outdir / f"srp_projector_dim{rp_dim}.npz"  # keep name for backward compatibility
        rp_data = {
            'rp_type': args.rp_type,
            'srp_dim': rp_dim,
            'input_dim': input_dim,
            'density': density,
            'random_state': args.seed,
            'split': args.split,
            'drug': args.drug,
            'n_train_files': len(train_files)
        }
        if hasattr(rp_projector, 'components_'):
            rp_data['components'] = rp_projector.components_.astype(np.float32)
        np.savez_compressed(proj_path, **rp_data)

        projector_obj_path = outdir / f"srp_projector_obj_dim{rp_dim}.joblib"
        dump(rp_projector, projector_obj_path)
        print(f"[FIT] Projector saved -> {proj_path}")

        # Fit MiniRocket
        mr = fit_minirocket_for_window(
            files=train_files,
            rp_projector=rp_projector,
            rp_dim=rp_dim,
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
        mr_path = srp_minirocket_dir / f"minirocket_dim{rp_dim}_win{args.win}.joblib"
        dump(mr, mr_path)
        print(f"[FIT] MiniRocket saved -> {mr_path}")

    print(f"[DONE] fit for {args.drug} in {time.time()-t0:.1f}s")

def cmd_run(args):
    files = _iter_files(args.path)
    if args.shuffle:
        import random; random.Random(args.seed).shuffle(files)
    files = files[:args.n_files]
    if not files:
        raise FileNotFoundError("No files to run")

    # Dirs
    base_outdir = Path(args.base_outdir)
    split_drug = f"{args.split}-{args.drug}"
    mr_dir = base_outdir / split_drug / "srp_minirocket"
    srp_only_dir = base_outdir / split_drug / "srp_only"
    srp_z_dir = base_outdir / split_drug / "srp_z"
    mr_dir.mkdir(parents=True, exist_ok=True)
    srp_only_dir.mkdir(parents=True, exist_ok=True)
    if args.save_srp_z:
        srp_z_dir.mkdir(parents=True, exist_ok=True)

    rp_dims = [64, 128, 256] if args.srp_dim_grid else [args.srp_dim]

    for rp_dim in rp_dims:
        print(f"\n[RUN] === RP Dimension {rp_dim} ===")
        projector_obj_path = base_outdir / split_drug / f"srp_projector_obj_dim{rp_dim}.joblib"
        mr_path = mr_dir / f"minirocket_dim{rp_dim}_win{args.win}.joblib"

        if not projector_obj_path.exists():
            print(f"[SKIP] Projector not found: {projector_obj_path}")
            continue
        if not mr_path.exists():
            print(f"[SKIP] MiniRocket model not found: {mr_path}")
            continue

        try:
            rp_projector = load(projector_obj_path)
            mr = load(mr_path)
        except Exception as e:
            print(f"[ERROR] Failed to load models for dim {rp_dim}: {e}")
            continue

        max_tokens = None if (args.max_tokens is None or args.max_tokens <= 0) else args.max_tokens

        for fp in files:
            t0 = time.time()
            try:
                stem = Path(fp).stem

                # Paths
                mr_base = f"{stem}_mr_win{args.win}_s{args.stride}_dim{rp_dim}"
                mr_out_npz = mr_dir / f"{mr_base}.npz"
                mr_out_json = mr_dir / f"{mr_base}.json"

                srp_base = f"{stem}_srp_dim{rp_dim}"
                srp_out_npz = srp_only_dir / f"{srp_base}.npz"
                srp_out_json = srp_only_dir / f"{srp_base}.json"

                # Flags for skipping
                srp_exists = srp_out_npz.exists() and srp_out_json.exists()
                mr_exists = mr_out_npz.exists() and mr_out_json.exists()

                if (not args.overwrite) and srp_exists and mr_exists:
                    print(f"[SKIP] {stem} dim{rp_dim} -> SRP and MR already exist")
                    continue

                # Load embeddings & project
                with h5py.File(fp, "r") as f:
                    if "embeddings" not in f:
                        warnings.warn(f"{fp}: missing /embeddings"); continue
                    X = f["embeddings"]; T = int(X.shape[0])
                    vm = None
                    if "valid_mask" in f:
                        m = np.squeeze(f["valid_mask"][:])
                        vm = (m > 0.5)

                    Ep = project_dataset_rp(X, rp_projector, valid_mask=vm, batch=args.batch,
                                            max_tokens=max_tokens, dtype=np.float32, seed=args.seed)

                # Save raw SRP (Ep)
                if args.save_srp and (args.overwrite or not srp_exists):
                    np.savez_compressed(srp_out_npz,
                                        embeddings=Ep.astype(np.float32),
                                        srp_dim=int(rp_dim),
                                        drug=args.drug,
                                        split=args.split,
                                        T_original=int(T),
                                        T_valid=int(Ep.shape[0]))
                    srp_info = dict(
                        file=str(Path(fp).resolve()),
                        T_original=int(T),
                        T_valid=int(Ep.shape[0]),
                        srp_dim=int(rp_dim),
                        embedding_dim=int(Ep.shape[1]),
                        drug=args.drug,
                        split=args.split,
                        elapsed_sec=round(time.time()-t0, 2)
                    )
                    with open(srp_out_json, "w") as fh:
                        json.dump(srp_info, fh, indent=2)
                    print(f"[SRP] {stem} dim{rp_dim}: saved -> {srp_out_npz}")

                # z-score (per-sample) if needed
                if args.zscore:
                    mu = Ep.mean(axis=0, keepdims=True)
                    sigma = Ep.std(axis=0, keepdims=True) + 1e-8
                    Ep_mr = (Ep - mu) / sigma
                else:
                    mu = None
                    sigma = None
                    Ep_mr = Ep

                # Save z-scored SRP sequence if requested
                if args.zscore and args.save_srp_z:
                    srp_z_npz = srp_z_dir / f"{stem}_srpZ_dim{rp_dim}.npz"
                    srp_z_json = srp_z_dir / f"{stem}_srpZ_dim{rp_dim}.json"
                    np.savez_compressed(srp_z_npz,
                                        embeddings=Ep_mr.astype(np.float32),
                                        srp_dim=int(rp_dim),
                                        drug=args.drug,
                                        split=args.split,
                                        T_original=int(T),
                                        T_valid=int(Ep_mr.shape[0]))
                    zinfo = dict(
                        file=str(Path(fp).resolve()),
                        T_original=int(T),
                        T_valid=int(Ep_mr.shape[0]),
                        srp_dim=int(rp_dim),
                        zscore_applied=True
                    )
                    with open(srp_z_json, "w") as fh:
                        json.dump(zinfo, fh, indent=2)
                    print(f"[SRP-Z] {stem} dim{rp_dim}: saved -> {srp_z_npz}")

                # Generate MiniRocket features
                T_valid = Ep_mr.shape[0]
                spans = spans_sliding(T_valid, win=args.win, stride=args.stride, cover_tail=args.cover_tail)
                nW = len(spans)

                feats = []
                for i in range(0, nW, args.win_batch):
                    batch_spans = spans[i:i + args.win_batch]
                    Xb = np.concatenate([to_case3d(Ep_mr, s, e, args.win) for (s, e) in batch_spans], axis=0)
                    Fb = mr.transform(Xb)
                    feats.append(Fb.astype(np.float32, copy=False))
                X_win = np.vstack(feats) if feats else np.empty((0, 0), dtype=np.float32)

                # Save raw MiniRocket window features if requested
                if args.save_mr and (args.overwrite or not mr_exists):
                    np.savez_compressed(mr_out_npz,
                                        X=X_win,
                                        spans=np.asarray(spans, dtype=np.int64),
                                        win=int(args.win),
                                        stride=int(args.stride),
                                        srp_dim=int(rp_dim),
                                        drug=args.drug,
                                        split=args.split)
                    info = dict(
                        file=str(Path(fp).resolve()),
                        T_used=int(T if max_tokens is None else min(T, max_tokens)),
                        T_valid=int(T_valid),
                        srp_dim=int(rp_dim),
                        n_windows=int(nW),
                        n_features=int(X_win.shape[1] if X_win.size else 0),
                        formula_n_windows="floor((T_valid-win)/stride)+1 if T_valid>=win else 1",
                        cover_tail=bool(args.cover_tail),
                        drug=args.drug,
                        split=args.split,
                        elapsed_sec=round(time.time()-t0, 2)
                    )
                    if mu is not None:
                        info["zscore_applied"] = True
                        info["zscore_mean_shape"] = [int(s) for s in mu.shape]
                        info["zscore_std_shape"] = [int(s) for s in sigma.shape]
                    with open(mr_out_json, "w") as fh:
                        json.dump(info, fh, indent=2)
                    print(f"[MiniRocket] {stem} dim{rp_dim}: saved -> {mr_out_npz}")

                print(f"[DONE] {stem} dim{rp_dim} in {time.time()-t0:.1f}s")
            except Exception as e:
                warnings.warn(f"{fp} dim{rp_dim}: {e}")

def cmd_pool(args):
    """Aggregate window-level features into a sample-level vector via stats + optional Top-k pooling."""
    base_outdir = Path(args.base_outdir)
    split_drug = f"{args.split}-{args.drug}"
    indir = base_outdir / split_drug / "srp_minirocket"
    outdir = indir

    rp_dims = [64, 128, 256] if args.srp_dim_grid else [args.srp_dim]

    for rp_dim in rp_dims:
        print(f"\n[POOL] === RP Dimension {rp_dim} ===")
        files = sorted(indir.glob(f"*_mr_win*_s*_dim{rp_dim}.npz"))
        if not files:
            print(f"[SKIP] No *_mr_win*_s*_dim{rp_dim}.npz files in {indir}")
            continue

        stats = [s.strip() for s in args.stats.split(",") if s.strip()]
        topk = max(0, int(args.topk)) if hasattr(args, "topk") else 0

        for fp in files:
            z = np.load(fp)
            X = z["X"].astype(np.float32, copy=False)
            if X.ndim != 2 or X.shape[0] == 0:
                warnings.warn(f"{fp}: empty or invalid X; skip")
                continue

            vecs = []

            # Top-k pooling per feature across windows
            if topk > 0 and X.shape[0] >= topk:
                idx = np.argpartition(X, -topk, axis=0)[-topk:]
                col = np.arange(X.shape[1])[None, :].repeat(topk, 0)
                X_topk = X[idx, col]   # [topk, F]
                vecs.append(X_topk.mean(axis=0, keepdims=True))

            # Statistics pooling
            if "mean" in stats: vecs.append(X.mean(axis=0, keepdims=True))
            if "std"  in stats: vecs.append(X.std(axis=0,  keepdims=True))
            if "max"  in stats: vecs.append(X.max(axis=0,  keepdims=True))
            if "min"  in stats: vecs.append(X.min(axis=0,  keepdims=True))
            if "q25"  in stats: vecs.append(np.percentile(X, 25.0, axis=0, keepdims=True))
            if "q75"  in stats: vecs.append(np.percentile(X, 75.0, axis=0, keepdims=True))

            if not vecs:
                warnings.warn("No valid stats selected; using mean,std,max,min,q25,q75 by default")
                vecs = [
                    X.mean(axis=0, keepdims=True),
                    X.std(axis=0, keepdims=True),
                    X.max(axis=0, keepdims=True),
                    X.min(axis=0, keepdims=True),
                    np.percentile(X, 25.0, axis=0, keepdims=True),
                    np.percentile(X, 75.0, axis=0, keepdims=True),
                ]
            V = np.concatenate(vecs, axis=1).astype(np.float32, copy=False)

            base = Path(fp).stem
            if args.save_pooled:
                out_npz = outdir / f"{base}_pooled.npz"
                np.savez_compressed(out_npz, V=V, stats=np.array(stats),
                                    topk=int(topk),
                                    srp_dim=rp_dim, drug=args.drug, split=args.split)
                print(f"[POOL] {base}: pooled_dims={V.shape[1]} -> {out_npz}")
            else:
                print(f"[POOL] {base}: pooled computed (not saved; --save-pooled not set)")

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser("SRP/GRP + MiniRocket Pipeline (Revised)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # fit (RP + MiniRocket @win)
    apf = sub.add_parser("fit", help="Fit RP (SRP/GRP) + MiniRocket for window")
    apf.add_argument("path", type=str, help="Dir or glob of .h5/.hdf5")
    apf.add_argument("--base-outdir", type=str, default="/insomnia001/depts/pmg/users/ht2666/processed_embedding",
                     help="Base output directory")
    apf.add_argument("--drug", type=str, required=True, help="Drug name (e.g., gentamicin)")
    apf.add_argument("--split", type=str, required=True, help="Split identifier (e.g., clustered_3_v1-2)")
    apf.add_argument("--labels-csv", type=str, help="CSV with genome_id, partition_label, antibiotic_label")
    apf.add_argument("--fit-manifest", type=str, help="Balanced file list for fitting (overrides sampling)")
    apf.add_argument("--sample-train-genomes", type=int, default=60,
                     help="Training genomes to sample for RP init (0=use all)")
    apf.add_argument("--max-files", type=int, default=300)
    apf.add_argument("--max-tokens", type=int, default=200000, help="Rows per file for RP sampling")
    apf.add_argument("--batch", type=int, default=32768)
    apf.add_argument("--no-mask", action="store_true")
    apf.add_argument("--srp-dim", type=int, default=128, help="Target RP dimension (64/128/256)")
    apf.add_argument("--srp-dim-grid", action="store_true", help="Try all 64/128/256")
    apf.add_argument("--rp-type", type=str, default="srp", choices=["srp", "grp"],
                     help="Random projection type")
    apf.add_argument("--win", type=int, default=1024, help="Window size (tokens)")
    apf.add_argument("--n-fit-windows-per-file", type=int, default=8,
                     help="Windows sampled per file for MiniRocket fit (0=use all)")
    apf.add_argument("--num-kernels", type=int, default=6000)
    apf.add_argument("--max-dilations-per-kernel", type=int, default=6)
    apf.add_argument("--seed", type=int, default=42)
    apf.add_argument("--shuffle", action="store_true")
    apf.add_argument("--zscore", action="store_true")

    # run (project + sliding MiniRocket transform)
    apr = sub.add_parser("run", help="Project + sliding MiniRocket transform")
    apr.add_argument("path", type=str)
    apr.add_argument("--base-outdir", type=str, default="/insomnia001/depts/pmg/users/ht2666/processed_embedding")
    apr.add_argument("--drug", type=str, required=True)
    apr.add_argument("--split", type=str, required=True)
    apr.add_argument("--srp-dim", type=int, default=128)
    apr.add_argument("--srp-dim-grid", action="store_true")
    apr.add_argument("--win", type=int, default=1024)
    apr.add_argument("--stride", type=int, default=512)
    apr.add_argument("--cover-tail", action="store_true", default=True)
    apr.add_argument("--win-batch", type=int, default=64)
    apr.add_argument("--max-tokens", type=int, default=None, help="Uniformly subsample valid tokens if set")
    apr.add_argument("--batch", type=int, default=32768)
    apr.add_argument("--zscore", action="store_true")
    apr.add_argument("--n-files", type=int, default=100000)
    apr.add_argument("--overwrite", action="store_true")
    apr.add_argument("--shuffle", action="store_true")
    apr.add_argument("--seed", type=int, default=42)
    apr.add_argument("--save-srp", action="store_true", help="Save raw SRP sequence (Ep) to srp_only/")
    apr.add_argument("--save-srp-z", action="store_true", help="Save z-scored SRP sequence (Ep_z) to srp_z/ (requires --zscore)")
    apr.add_argument("--save-mr", action="store_true", help="Save raw MiniRocket window features (X) to srp_minirocket/")

    # pool (stats + optional Top-k pooling)
    apo = sub.add_parser("pool", help="Pool window features to sample-level vectors")
    apo.add_argument("--base-outdir", type=str, default="/insomnia001/depts/pmg/users/ht2666/processed_embedding")
    apo.add_argument("--drug", type=str, required=True)
    apo.add_argument("--split", type=str, required=True)
    apo.add_argument("--srp-dim", type=int, default=128)
    apo.add_argument("--srp-dim-grid", action="store_true")
    apo.add_argument("--stats", type=str, default="mean,std,max,min,q25,q75",
                     help="comma-separated: mean,std,max,min,q25,q75")
    apo.add_argument("--topk", type=int, default=0, help="Top-k pooling per feature across windows (0=disable)")
    apo.add_argument("--save-pooled", action="store_true", help="If set, save pooled vectors")

    args = ap.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "pool":
        cmd_pool(args)

if __name__ == "__main__":
    main()
