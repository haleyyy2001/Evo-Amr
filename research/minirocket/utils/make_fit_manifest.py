#!/usr/bin/env python3
"""
Balanced Fit Manifest Generator for SRP+MiniRocket Pipeline
===========================================================
Creates a deterministic, balanced manifest of exactly 60 files (30 R, 30 S)
from training samples that actually exist in the embedding directory.

This ensures SRP "fit" uses:
- Exactly 30 resistant (label=1) + 30 susceptible (label=0) files
- Only files that actually exist in the embedding directory  
- Deterministic selection with fixed random seed
- Training split only (not validation/test)

Usage:
    python make_fit_manifest.py \
      --emb-dir /path/to/embeddings \
      --labels-csv /path/to/labels.csv \
      --split-name train \
      --out-manifest /path/to/fit_manifest.txt
"""

import argparse, csv, os, re, random, sys
from pathlib import Path
import pandas as pd

def guess_id_from_path(p: Path):
    """
    Extract sample/genome ID from embedding file path.
    Adjust this function based on your filename convention.
    
    Common patterns:
      <sample_id>.h5
      <sample_id>__layer10.h5  
      <sample_id>.npz
    
    Returns the base sample ID.
    """
    # Handle common suffixes and separators
    stem = p.stem
    
    # Remove layer/embedding specific suffixes
    stem = stem.split("__")[0]  # Remove __layer10, __embeddings, etc.
    stem = stem.split("_layer")[0]  # Remove _layer10, etc.
    stem = stem.split("_emb")[0]   # Remove _emb, _embeddings, etc.
    
    return stem

def main():
    ap = argparse.ArgumentParser(description="Generate balanced fit manifest for SRP+MiniRocket")
    ap.add_argument("--emb-dir", required=True, help="Directory with embedding files")
    ap.add_argument("--labels-csv", required=True, help="CSV with genome_id,partition_label,antibiotic_label")
    ap.add_argument("--split-name", default="train", help="Split to use (default: train)")
    ap.add_argument("--out-manifest", required=True, help="Output manifest file (one path per line)")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for deterministic selection")
    ap.add_argument("--n-pos", type=int, default=30, help="Number of resistant samples (label=1)")
    ap.add_argument("--n-neg", type=int, default=30, help="Number of susceptible samples (label=0)")
    ap.add_argument("--file-glob", default="**/*.h5", help="Glob pattern for embedding files")
    args = ap.parse_args()

    emb_dir = Path(args.emb_dir)
    if not emb_dir.exists():
        sys.exit(f"[ERROR] Embedding directory not found: {emb_dir}")
    
    # Find all embedding files
    files = sorted(emb_dir.glob(args.file_glob))
    if not files:
        sys.exit(f"[ERROR] No files found under {emb_dir} with glob {args.file_glob}")
    
    print(f"[INFO] Found {len(files)} embedding files", file=sys.stderr)

    # Map file -> sample_id  
    file_rows = []
    for f in files:
        sid = guess_id_from_path(f)
        file_rows.append({"genome_id": sid, "path": str(f.resolve())})
    df_files = pd.DataFrame(file_rows)
    
    print(f"[INFO] Extracted {len(df_files)} unique sample IDs", file=sys.stderr)

    # Load labels CSV
    try:
        df_lab = pd.read_csv(args.labels_csv)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load labels CSV {args.labels_csv}: {e}")
    
    # Ensure string types for merging
    df_lab["genome_id"] = df_lab["genome_id"].astype(str)
    df_files["genome_id"] = df_files["genome_id"].astype(str)
    
    # Filter to requested split (usually 'train')
    df_lab["partition_label"] = df_lab["partition_label"].astype(str)
    target_split = str(args.split_name)
    df_lab_split = df_lab[df_lab["partition_label"] == target_split].copy()
    
    print(f"[INFO] Labels CSV has {len(df_lab_split)} samples in split '{target_split}'", file=sys.stderr)

    # Inner join: only samples that exist both in labels and as embedding files
    df = df_lab_split.merge(df_files, on="genome_id", how="inner")
    
    print(f"[INFO] Found {len(df)} samples with both labels and embedding files", file=sys.stderr)
    
    if len(df) == 0:
        sys.exit("[ERROR] No samples found with both labels and embedding files. Check genome_id matching.")

    # Check label distribution
    label_counts = df["antibiotic_label"].value_counts().sort_index()
    print(f"[INFO] Label distribution in matched samples:", file=sys.stderr)
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} samples", file=sys.stderr)

    # Balance selection: exactly n_pos resistant + n_neg susceptible
    rng = random.Random(args.seed)

    def sample_n(df_sub, n_needed, label_value):
        """Sample exactly n_needed samples from df_sub, or all if fewer available."""
        if len(df_sub) < n_needed:
            print(f"[WARN] Only {len(df_sub)} samples available for label={label_value} in split={target_split}. Will take all.", file=sys.stderr)
            return df_sub.copy()
        
        # Deterministic sample
        indices = list(range(len(df_sub)))
        rng.shuffle(indices)
        selected_indices = indices[:n_needed]
        return df_sub.iloc[selected_indices].copy()

    # Split by resistance label
    df_resistant = df[df["antibiotic_label"] == 1]  # Resistant
    df_susceptible = df[df["antibiotic_label"] == 0]  # Susceptible

    # Sample balanced sets
    df_pos_selected = sample_n(df_resistant, args.n_pos, 1)
    df_neg_selected = sample_n(df_susceptible, args.n_neg, 0)

    # Combine and shuffle final selection
    df_final = pd.concat([df_pos_selected, df_neg_selected], axis=0)
    df_final = df_final.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # Shuffle
    
    final_paths = df_final["path"].tolist()
    
    # Report final selection
    final_pos = len(df_pos_selected)
    final_neg = len(df_neg_selected)
    total_selected = len(final_paths)
    
    print(f"[INFO] Final selection: {total_selected} files ({final_pos} resistant, {final_neg} susceptible)", file=sys.stderr)
    
    if final_pos < args.n_pos or final_neg < args.n_neg:
        print(f"[WARN] Could not achieve perfect balance. Requested {args.n_pos}+{args.n_neg}, got {final_pos}+{final_neg}", file=sys.stderr)

    # Write manifest file (one path per line)
    manifest_path = Path(args.out_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with manifest_path.open("w") as f:
        for path in final_paths:
            f.write(path + "\n")

    # Write audit CSV for inspection
    audit_csv = manifest_path.with_suffix(".csv")
    df_final[["genome_id", "antibiotic_label", "partition_label", "path"]].to_csv(audit_csv, index=False)
    
    print(f"[OK] Manifest written: {manifest_path}", file=sys.stderr)
    print(f"[OK] Audit CSV written: {audit_csv}", file=sys.stderr)
    print(f"[OK] Selected {total_selected} files for balanced SRP+MiniRocket fitting")

if __name__ == "__main__":
    main()