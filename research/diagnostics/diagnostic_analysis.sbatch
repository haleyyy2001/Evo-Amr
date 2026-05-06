#!/usr/bin/env bash
#SBATCH --job-name=diagnostics_dtype_sweep
#SBATCH --partition=gpu
#SBATCH --account=pmg
#SBATCH --array=0                       # Single job for bf16 only
#SBATCH --gres=gpu:3                  # 1 GPU per task
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --signal=TERM@120              # warn 120s before kill
#SBATCH --output=/insomnia001/depts/pmg/users/ht2666/Evo_embedding/log/diagnostics_dtype_sweep_%A_%a.out
#SBATCH --error=/insomnia001/depts/pmg/users/ht2666/Evo_embedding/log/diagnostics_dtype_sweep_%A_%a.err


set -euo pipefail

# Multiple seeds on single node: bf16 only
DTYPE=bf16
SHARD_IDX=0
SHARD_COUNT=1
NUM_SEEDS=10  # Run 10 different seeds for statistical robustness

# Handle preemption & cleanup correctly (safe OUTDIR default)
OUTDIR=${OUTDIR:-/insomnia001/depts/pmg/users/ht2666/Evo_embedding/Result/_early}
trap 'echo "[TERM] Graceful stop - task ${SLURM_ARRAY_TASK_ID}"; mkdir -p "$OUTDIR"; touch "$OUTDIR/.partial"; exit 0' TERM INT HUP

# ==================== DIAGNOSTIC FEATURES ====================
# This script includes diagnostic features to help isolate issues with magic constants.
# To use them, set the following environment variables before running:
#
# 1. Disable Flash operations (to isolate flash kernel issues):
#    export EV_DISABLE_FLASH_DEPTHWISE=1  # Disables FlashDepthwiseConv1d
#    export EV_DISABLE_FLASHFFT=1         # Disables FlashFFTConv
#
# 2. Force FP32 for Hyena blocks (to isolate bf16 issues):
#    export EV_FORCE_FP32_HYENA=1         # Forces all Hyena computations to FP32
#
# 3. The script will automatically:
#    - Check for mask leakage into padded regions
#    - Print warnings if magic constants (2^23, 2^24, etc.) appear in activations
#    - Log dtype information for key parameters
#
# Example usage to test without flash operations:
#    export EV_DISABLE_FLASH_DEPTHWISE=1 EV_DISABLE_FLASHFFT=1
#    sbatch diagnos_fixed.sbatch
# ==============================================================

echo "[$(date +'%F %T')] Starting diagnostics_dtype_sweep job"
echo "[$(date +'%F %T')] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date +'%F %T')] Node: $(hostname)"
echo "[$(date +'%F %T')] CWD: $(pwd)"

# --------------------- core env ---------------------
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1337
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"
unset PYTORCH_NO_CUDA_MEMORY_CACHING 2>/dev/null || true

export TRANSFORMERS_TRUST_REMOTE_CODE=1

# Run tagging for output organization
export RUN_TAG=${RUN_TAG:-"diag_$(date +%F_%H%M%S)"}
echo "[$(date +'%F %T')] Run tag: $RUN_TAG"
echo "[$(date +'%F %T')] Array task: ${SLURM_ARRAY_TASK_ID} → dtype=$DTYPE shard=$SHARD_IDX"
# Let SLURM manage CUDA_VISIBLE_DEVICES for --gres=gpu:1

# --------------------- cache to scratch ---------------------
export EVOCACHE_ROOT=/insomnia001/depts/pmg/users/ht2666/Evo_embedding/cache
export XDG_CACHE_HOME="$EVOCACHE_ROOT/xdg"
export HF_HOME="$EVOCACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$EVOCACHE_ROOT/hf"
export TORCH_HOME="$EVOCACHE_ROOT/torch"
export TORCHINDUCTOR_CACHE_DIR="$EVOCACHE_ROOT/inductor"
export TRITON_CACHE_DIR="$EVOCACHE_ROOT/triton"
export FLASH_ATTENTION_CACHE_DIR="$EVOCACHE_ROOT/flashattn"
export CUDA_CACHE_PATH="$EVOCACHE_ROOT/cuda"
export CUDA_CACHE_MAXSIZE=$((2**30))
export PYTHONUNBUFFERED=1

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" \
         "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$FLASH_ATTENTION_CACHE_DIR" \
         "$CUDA_CACHE_PATH"

# --------------------- paths ---------------------
ROOT=/insomnia001/depts/pmg/users/ht2666/amr_pred/testtt/evo
CUSTOM=/insomnia001/depts/pmg/users/ht2666/Evo_embedding/evo_8
CFG=${ROOT}/evo_1_131k_base/config.json
WEIGHTS=${ROOT}/evo-1-8k-base
PKG=evo_1_131k_base

CENTRAL_CONTIGS=/insomnia001/depts/pmg/projects/amr_pred/data/multi-drug_aug2024/sequences/raw/nucleotide
IDS_CSV=/insomnia001/depts/pmg/users/ht2666/Evo_embedding/evo_8/data/natural_proportion_sampling/ampicillin/subsets_100_1.csv

RESULT_DIR=/insomnia001/depts/pmg/users/ht2666/Evo_embedding/Result/comprehensive
LOG_DIR=/insomnia001/depts/pmg/users/ht2666/Evo_embedding/log
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "[$(date +'%F %T')] Configuration:"
echo "  ROOT: $ROOT"
echo "  CUSTOM: $CUSTOM"
echo "  CFG: $CFG"
echo "  WEIGHTS: $WEIGHTS"
echo "  PKG: $PKG"
echo "  CENTRAL_CONTIGS: $CENTRAL_CONTIGS"
echo "  IDS_CSV: $IDS_CSV"
echo "  RESULT_DIR: $RESULT_DIR"

# --------------------- sweep knobs ---------------------
N_GENOMES=50
WINDOWS_PER_GENOME=6
CHUNK=2048
STRIDE=0
N_SYNTH=14
LAYER_SAMPLE_MAXTOK=16384
Z_THRESH=6.0
ARGMAX_LAYERS="0,8,10,16,24,31"

# --- decade bins / thresholds & representative saving ---
MAXABS_THRESH=1e9
DECADE_MIN=1     # 10^1
DECADE_MAX=9     # 10^9
PIVOT_LAYER=12
KEY_LAYERS="5,8,10,12,16,24,31"   # include attn-ish depths
SAVE_TOPK_PER_SOURCE=3
SAVE_SYNTH_KEYS=""

# --- batching & sharding ---
# FIX 2: Dynamic batch size based on dtype
BATCH_FP32=${BATCH_FP32:-2}
BATCH_MIXED=${BATCH_MIXED:-8}
SHARD_COUNT=${SHARD_COUNT:-3}       # *** 3 GPUs ***

PY=/insomnia001/depts/pmg/users/ht2666/mambaforge/envs/evo-amr/bin/python

# Create stable genome/window manifest (shared across all dtypes/shards)
SEED_SET_CSV="${RESULT_DIR}/${RUN_TAG}/_pivot_genomes_windows.csv"
if [[ ! -s "$SEED_SET_CSV" ]]; then
    echo "[$(date +'%F %T')] Creating stable genome/window manifest..."
    mkdir -p "$(dirname "$SEED_SET_CSV")"
    export SEED_SET_CSV CENTRAL_CONTIGS CHUNK N_GENOMES WINDOWS_PER_GENOME
    $PY - <<'PY'
import os, pandas as pd, numpy as np, hashlib, random, json
from pathlib import Path
CENTRAL=os.environ["CENTRAL_CONTIGS"]; CHUNK=int(os.environ["CHUNK"])
N_GENOMES=int(os.environ["N_GENOMES"]); WINDOWS_PER_GENOME=int(os.environ["WINDOWS_PER_GENOME"])
rng=random.Random(42)

# stable hash order of files
files=sorted(Path(CENTRAL).glob("*_raw.fna.gz"), key=lambda p: hashlib.sha1(str(p).encode()).hexdigest())
pick=files[:N_GENOMES]
rows=[]
for f in pick:
    gid=f.stem.replace("_raw.fna","")
    for i in range(WINDOWS_PER_GENOME):
        rows.append(dict(id=f"{gid}:{i}", genome_id=gid, win_idx=i))
pd.DataFrame(rows).to_csv(os.environ["SEED_SET_CSV"], index=False)
print("Wrote", os.environ["SEED_SET_CSV"])
PY
fi
export SEED_SET_CSV

echo "[$(date +'%F %T')] Params: N_GENOMES=$N_GENOMES WINDOWS_PER_GENOME=$WINDOWS_PER_GENOME CHUNK=$CHUNK SHARDS=$SHARD_COUNT"
echo "[$(date +'%F %T')] Stable manifest: $SEED_SET_CSV"

# Loop through multiple seeds on single node
for SEED_IDX in $(seq 0 $((NUM_SEEDS-1))); do
    echo ""
    echo "=============================================="
    echo "[$(date +'%F %T')] Starting seed ${SEED_IDX}/${NUM_SEEDS} (seed=$(( 1337 + SEED_IDX )))"
    echo "=============================================="
    
    RUN_TS=$(date +%F_%H%M%S)
    OUTDIR=${RESULT_DIR}/${RUN_TAG}/${DTYPE}_L${CHUNK}_seed${SEED_IDX}
    mkdir -p "${OUTDIR}"/{figs,metrics,hidden,tmp}

# Create run manifest
$PY - <<'PY'
import os, json, hashlib, time, platform
from pathlib import Path
m = {
    "ts": time.strftime("%F %T"),
    "dtype": os.environ["DTYPE"],
    "shard_idx": int(os.environ["SHARD_IDX"]),
    "shard_count": int(os.environ["SHARD_COUNT"]),
    "cfg_sha256": hashlib.sha256(open(os.environ["CFG"],"rb").read()).hexdigest(),
    "weights_dir": os.environ["WEIGHTS"],
    "host": platform.node(),
    "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES",""),
    "seed_set_csv": os.environ.get("SEED_SET_CSV",""),
    "run_tag": os.environ.get("RUN_TAG","")
}
outdir=Path(os.environ["OUTDIR"]); outdir.mkdir(parents=True, exist_ok=True)
open(outdir/"manifest.json","w").write(json.dumps(m, indent=2))
PY

# Update trap to include correct outdir
trap 'echo "[TERM] Graceful stop - task ${SLURM_ARRAY_TASK_ID}"; mkdir -p "$OUTDIR"; touch "$OUTDIR/.partial"; exit 0' TERM INT HUP

echo ""
echo "=============================================="
echo "[$(date +'%F %T')] Starting run for DTYPE=${DTYPE} SHARD=${SHARD_IDX}"
echo "[$(date +'%F %T')] Output directory: ${OUTDIR}"
echo "=============================================="

# Check if this shard is already completed
REAL_CSV="${OUTDIR}/metrics/${DTYPE}_real_layer_metrics_sh${SHARD_IDX}.csv"
SYNTH_CSV="${OUTDIR}/metrics/${DTYPE}_synth_layer_metrics_sh${SHARD_IDX}.csv"
if [[ -f "${OUTDIR}/.success" ]] && \
   [[ -s "$REAL_CSV" ]] && \
   [[ -s "$SYNTH_CSV" ]]; then
    echo "[SKIP] Shard ${SHARD_IDX} already completed successfully"
    exit 0
fi

DTYPE_FLAG_FP32=0
DTYPE_FLAG_BF16=0
  if [[ "$DTYPE" == "fp32" ]]; then
    DTYPE_FLAG_FP32=1
    export BATCH=${BATCH_FP32}
  elif [[ "$DTYPE" == "bf16" ]]; then
    DTYPE_FLAG_BF16=1
    export BATCH=${BATCH_MIXED}
  else
    export BATCH=${BATCH_MIXED}
  fi

  export ROOT CUSTOM CFG WEIGHTS PKG CENTRAL_CONTIGS IDS_CSV OUTDIR
  export DTYPE CHUNK STRIDE N_SYNTH N_GENOMES WINDOWS_PER_GENOME LAYER_SAMPLE_MAXTOK Z_THRESH ARGMAX_LAYERS
  export DTYPE_FLAG_FP32 DTYPE_FLAG_BF16 SEED_IDX
  export MAXABS_THRESH DECADE_MIN DECADE_MAX PIVOT_LAYER KEY_LAYERS SAVE_TOPK_PER_SOURCE SAVE_SYNTH_KEYS

# -------------------- SINGLE SHARD EXECUTION --------------------
echo "[$(date +'%F %T')] Executing shard $SHARD_IDX / $SHARD_COUNT"
echo "[$(date +'%F %T')] GPU: $CUDA_VISIBLE_DEVICES"

stdbuf -oL -eL "$PY" - <<'PY'
import os, sys, json, random, time, gzip, warnings, csv, inspect, numpy as np
warnings.filterwarnings("ignore")
from pathlib import Path
import torch, torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from contextlib import nullcontext
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional

# ========================== DIAGNOSTIC UTILITIES (for shard) ==========================
def _to_fp32(x): 
    return x.detach().to(torch.float32)

def _stats_tokenwise(H: torch.Tensor) -> Dict[str, float]:
    """H: [T, D] fp32. Returns mean token L2, max|act|, tail ratio p99.9/p95."""
    norms = H.norm(dim=-1)  # [T]
    if norms.numel() == 0:
        return dict(mean=0.0, max=0.0, tail_ratio=1.0)
    p95  = torch.quantile(norms, 0.95)
    p999 = torch.quantile(norms, 0.999) if norms.numel() >= 1000 else torch.quantile(norms, 0.99)
    return dict(
        mean=float(norms.mean().item()),
        max =float(H.abs().max().item()),
        tail_ratio=float((p999 / (p95 + 1e-12)).item())
    )

def _kurtosis(x: torch.Tensor) -> float:
    """Excess not needed; we use standardized kurtosis proxy on token norms."""
    if x.numel() == 0:
        return 0.0
    x = x - x.mean()
    m2 = (x**2).mean()
    m4 = (x**4).mean()
    return float((m4 / (m2**2 + 1e-12)).item())

def _hill_tail_index(x: torch.Tensor, k: int = 50) -> float:
    """Simple Hill estimator on absolute values (heavy-tail feel)."""
    x = x.abs().reshape(-1)
    n = x.numel()
    if n < max(10, k+1):
        return 0.0
    x_sorted = torch.sort(x)[0]
    tail = x_sorted[-k:]
    # avoid log(0)
    tail = torch.clamp(tail, min=1e-12)
    base = tail[0]
    return float((torch.log(tail / base)).mean().item())

def _print_magic_constants(tag, H):
    """Look for exact powers of two near 2^23…2^25 that commonly show up."""
    if H is None:
        return
    v = H.abs().reshape(-1)
    uniq = torch.unique(v)
    hits = []
    for tgt in (8388608.0, 10485760.0, 16777216.0):  # 2^23, ~5*2^20, 2^24
        if (uniq == tgt).any():
            hits.append(tgt)
    if hits:
        print(f"[POISON] {tag}: found constants {hits}", flush=True)

def capture_block_decomposition(model, x: torch.Tensor, lens: List[int], mask_kwargs: dict,
                                batch_index: int = 0,
                                move_to_cpu: bool = True) -> Tuple[Dict[str, List[torch.Tensor]], List[Optional[torch.nn.Module]]]:
    """Capture per-layer tensors around each block for one sample in the batch."""
    device = x.device
    B, T = x.shape
    L0 = int(lens[batch_index])
    sl = slice(0, L0)

    # prepare containers
    captures = {"pre":[], "branch":[], "mlp":[], "post":[]}
    branch_mods: List[Optional[torch.nn.Module]] = []

    handles = []
    # We will try to cover common attribute names.
    def hook_pre(module, inp, out):
        y = out[0] if isinstance(out, (tuple, list)) else out
        y = y[batch_index, sl].detach()
        captures["pre"].append(_to_fp32(y.cpu() if move_to_cpu else y))

    def hook_branch(module, inp, out):
        y = out[0] if isinstance(out, (tuple, list)) else out
        y = y[batch_index, sl].detach()
        captures["branch"].append(_to_fp32(y.cpu() if move_to_cpu else y))

    def hook_mlp(module, inp, out):
        y = out[0] if isinstance(out, (tuple, list)) else out
        y = y[batch_index, sl].detach()
        captures["mlp"].append(_to_fp32(y.cpu() if move_to_cpu else y))

    def hook_post(module, inp, out):
        y = out[0] if isinstance(out, (tuple, list)) else out
        y = y[batch_index, sl].detach()
        post_val = _to_fp32(y.cpu() if move_to_cpu else y)
        captures["post"].append(post_val)
        # Check for magic constants in post output
        li = len(captures["post"]) - 1
        _print_magic_constants(f"post L{li}", post_val)
        if len(captures["branch"]) > li:
            _print_magic_constants(f"branch L{li}", captures["branch"][li])

    # Build hooks over blocks
    try:
        for b in getattr(model, "blocks", []):
            # pre-norm output (≈ block input)
            if hasattr(b, "pre_norm"):
                handles.append(b.pre_norm.register_forward_hook(hook_pre))
            else:
                # Fallback: still want an aligned list; append None and fill later
                handles.append(b.register_forward_pre_hook(lambda m, i: None))
                captures["pre"].append(None)

            # branch module: Hyena/Attention (try multiple common names)
            branch_mod = None
            for attr in ["filter", "hyena", "hyena_layer", "attention", "attn"]:
                if hasattr(b, attr):
                    branch_mod = getattr(b, attr)
                    handles.append(branch_mod.register_forward_hook(hook_branch))
                    break
            branch_mods.append(branch_mod)

            # mlp module
            added_mlp = False
            for attr in ["mlp", "ffn", "gmlp", "parallel_mlp"]:
                if hasattr(b, attr):
                    handles.append(getattr(b, attr).register_forward_hook(hook_mlp))
                    added_mlp = True
                    break
            if not added_mlp:
                captures["mlp"].append(None)  # keep alignment

            # block output (post residual)
            handles.append(b.register_forward_hook(hook_post))

        # forward once (no AMP) to populate hooks
        with torch.inference_mode():
            _ = model(x, output_hidden_states=False, return_dict=True, **mask_kwargs)

        # If pre/mla entries got Nones due to missing attributes, patch with zeros to keep lengths aligned.
        L = len(captures["post"])
        for k in list(captures.keys()):
            if len(captures[k]) < L:
                captures[k] += [None] * (L - len(captures[k]))
            for i in range(L):
                if captures[k][i] is None:
                    # create a zero placeholder with shape of 'post' at same layer
                    ref = captures["post"][i]
                    if ref is not None:
                        z = torch.zeros_like(ref)
                        captures[k][i] = z
        return captures, branch_mods
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass

def per_layer_decomposition_table(captures: Dict[str, List[torch.Tensor]]) -> pd.DataFrame:
    """Builds a per-layer stats table with gains."""
    rows = []
    L = min(len(captures["post"]), len(captures["pre"]))
    for li in range(L):
        r = {"layer": li}
        for k in ["pre","branch","mlp","post"]:
            H = captures[k][li]
            if H is None: 
                r[f"{k}_mean"] = np.nan; r[f"{k}_max"] = np.nan; r[f"{k}_tail"] = np.nan
            else:
                st = _stats_tokenwise(H)
                r[f"{k}_mean"] = st["mean"]
                r[f"{k}_max"]  = st["max"]
                r[f"{k}_tail"] = st["tail_ratio"]
        # gains
        eps = 1e-12
        if np.isfinite(r["pre_mean"]) and np.isfinite(r["post_mean"]):
            r["gain_mean"] = r["post_mean"] / (r["pre_mean"] + eps)
        else:
            r["gain_mean"] = np.nan
        if np.isfinite(r["pre_max"]) and np.isfinite(r["post_max"]):
            r["gain_max"] = r["post_max"] / (r["pre_max"] + eps)
        else:
            r["gain_max"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)

def topk_dim_positions(H: torch.Tensor, k: int = 20) -> List[Tuple[int,int,float]]:
    """Return k (token, dim, absval) triples from H with largest magnitude."""
    if H.numel() == 0:
        return []
    A = H.abs().reshape(-1)
    k = min(k, A.numel())
    vals, idx = torch.topk(A, k)
    T, D = H.shape
    pos = (idx // D).tolist()
    dim = (idx %  D).tolist()
    return list(zip(pos, dim, vals.tolist()))

def culprits_table(captures: Dict[str, List[torch.Tensor]],
                   token_len: int,
                   layer_start: int = 0,
                   k: int = 20) -> pd.DataFrame:
    """Aggregate top-k (token,dim,absval) for 'post' per layer."""
    rows = []
    L = len(captures["post"])
    for li in range(layer_start, L):
        H = captures["post"][li]
        if H is None: 
            continue
        H = H[:token_len]
        for (t,d,v) in topk_dim_positions(H, k=k):
            rows.append(dict(layer=li, token=t, dim=d, absval=v))
    return pd.DataFrame(rows)

def tail_shape_table(captures: Dict[str, List[torch.Tensor]]) -> pd.DataFrame:
    rows = []
    L = len(captures["post"])
    for li in range(L):
        H = captures["post"][li]
        if H is None: 
            continue
        norms = H.norm(dim=-1)
        rows.append(dict(
            layer=li,
            kurtosis=_kurtosis(norms),
            hill=_hill_tail_index(norms, k=max(10, min(200, norms.numel()//50)))
        ))
    return pd.DataFrame(rows)

def plot_layer_gains(df_gain: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(10,6))
    plt.plot(df_gain["layer"], df_gain["gain_mean"], marker='o', label="gain_mean (post/pre)")
    plt.plot(df_gain["layer"], df_gain["gain_max"],  marker='s', label="gain_max (post/pre)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.xlabel("Layer"); plt.ylabel("Gain (log)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_tail_curves(df_gain: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(10,6))
    plt.plot(df_gain["layer"], df_gain["pre_tail"],    marker='.', label="pre tail")
    plt.plot(df_gain["layer"], df_gain["branch_tail"], marker='o', label="branch tail")
    plt.plot(df_gain["layer"], df_gain["mlp_tail"],    marker='^', label="mlp tail")
    plt.plot(df_gain["layer"], df_gain["post_tail"],   marker='s', label="post tail")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.xlabel("Layer"); plt.ylabel("Tail ratio (p99.9/p95)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_hot_dims(hot: pd.DataFrame, out_png: str, title: str, topn: int = 20):
    """Bar plot of topN hot dimensions aggregated over suspect layers."""
    if hot.empty:
        return
    # rank by count then vmax
    hot = hot.sort_values(["cnt","vmax"], ascending=[False, False]).head(topn)
    labels = [f"L{int(r.layer)}:d{int(r.dim)}" for _,r in hot.iterrows()]
    vals   = hot["vmax"].values
    plt.figure(figsize=(max(10, 0.35*len(labels)), 6))
    plt.bar(range(len(labels)), vals)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylabel("max |activation|"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_hot_tokens_heatmap(culprits: pd.DataFrame, out_png: str, title: str, max_token: Optional[int] = None):
    """Heatmap: token vs layer showing max|act| per (layer, token)."""
    if culprits.empty:
        return
    pivot = culprits.pivot_table(index="layer", columns="token", values="absval", aggfunc="max").fillna(0.0)
    if max_token is not None:
        pivot = pivot.loc[:, [c for c in pivot.columns if c <= max_token]]
    plt.figure(figsize=(12, 0.5*len(pivot.index)+2))
    plt.imshow(pivot.values, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.colorbar(label="max |activation|")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_smoking_gun_traj(captures: Dict[str, List[torch.Tensor]], token: int, dim: int,
                          out_png: str, title: str):
    """Single (token, dim) trajectory across layers: token L2 and that dim abs."""
    L = len(captures["post"])
    tok_norm, dim_abs = [], []
    for li in range(L):
        H = captures["post"][li]
        if H is None or token >= H.shape[0] or dim >= H.shape[1]:
            tok_norm.append(np.nan); dim_abs.append(np.nan); continue
        h = H[token]
        tok_norm.append(float(h.norm().item()))
        dim_abs.append(float(abs(h[dim].item())))
    x = np.arange(L)
    plt.figure(figsize=(10,6))
    plt.plot(x, tok_norm, marker='o', label=f"token {token} L2")
    plt.plot(x, dim_abs, marker='s', label=f"token {token}, dim {dim} |act|")
    plt.yscale("log")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.xlabel("Layer"); plt.ylabel("Value (log)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def approx_operator_gain(module: torch.nn.Module, D: int, device, dtype=torch.bfloat16, iters: int = 3) -> float:
    """Very rough power-iteration style average gain estimate for branch module."""
    if module is None:
        return float("nan")
    x = torch.randn(2, 1024, D, device=device, dtype=dtype)  # [B,T,D]
    x = torch.nn.functional.normalize(x, dim=-1)
    with torch.inference_mode():
        g = 1.0
        for _ in range(iters):
            y = module(x)
            if isinstance(y, (tuple, list)):
                y = y[0]
            num = y.float().norm(dim=-1).mean()
            den = x.float().norm(dim=-1).mean() + 1e-12
            g = float((num / den).item())
            x = torch.nn.functional.normalize(y, dim=-1).to(dtype)
    return g
# ======================== END DIAGNOSTIC UTILITIES (for shard) ========================

# ======= External No-Edit Diagnostic Tracer =======
def _now(): return time.strftime('%F %T')
def _q_abs(x, qs=(0.95,0.99,0.999)):
    xa = x.abs().reshape(-1)
    if xa.numel()==0: return {f"p{int(q*1000)/10:g}":0.0 for q in qs}
    t = torch.tensor(qs, device=xa.device, dtype=xa.dtype)
    vals = torch.quantile(xa, t, interpolation='linear')
    return {f"p{int(q*1000)/10:g}": float(v) for q,v in zip(qs, vals)}

def _hill(vals, k=50):
    v = vals.abs().reshape(-1)
    if v.numel() < max(10,k+1): return 0.0
    tail = torch.sort(v)[0][-k:].clamp_min(1e-12)
    base = tail[0]
    return float((torch.log(tail/base)).mean().item())

def _magic_hits(x):
    U = torch.unique(x.abs().reshape(-1))
    hits=[]
    for t in (8388608.0, 16777216.0):  # 2^23, 2^24
        if (U==t).any(): hits.append(t)
    return hits

class NoEditDiag:
    """
    External tracer that attaches hooks to your model to log per-layer stats.
    Toggle with env:
      EV_DIAG=1                   # enable
      EV_DIAG_DIR=/path/to/diag   # output dir
      EV_DIAG_TAG=myrun           # filename tag
      EV_DIAG_TOPK=20             # log top-K hot dims per checkpoint (0=off)
      EV_DIAG_MAXB=3              # only log first N batches (default 2)
    """
    def __init__(self, model, tag='default'):
        self.enabled = os.getenv("EV_DIAG","0") in ("1","true","TRUE","yes")
        if not self.enabled: return
        self.model = model
        self.tag = os.getenv("EV_DIAG_TAG", tag)
        self.max_batches = int(os.getenv("EV_DIAG_MAXB", "2"))
        self.topk = int(os.getenv("EV_DIAG_TOPK", "0"))
        self.root = Path(os.getenv("EV_DIAG_DIR", "./diag")); self.root.mkdir(parents=True, exist_ok=True)
        self.f_stats = open(self.root/f"layer_stats_{self.tag}.csv","a", newline="")
        self.f_hot   = open(self.root/f"hot_dims_{self.tag}.csv","a", newline="")
        self.f_meta  = open(self.root/f"meta_{self.tag}.jsonl","a")

        self.w_stats = csv.DictWriter(self.f_stats, fieldnames=[
            "ts","batch","layer","stage","T","D","mean","std","max","p95","p99","p99_9",
            "hill","dtype","device","nan","inf","magic"
        ])
        if self.f_stats.tell()==0: self.w_stats.writeheader()
        self.w_hot = csv.DictWriter(self.f_hot, fieldnames=["ts","batch","layer","stage","token","dim","absval"])
        if self.f_hot.tell()==0: self.w_hot.writeheader()

        self.hooks = []
        self.batch_idx = 0

        self._install()

    def close(self):
        if not self.enabled: return
        for h in self.hooks:
            try: h.remove()
            except: pass
        self.f_stats.close(); self.f_hot.close(); self.f_meta.close()

    def meta(self, **obj):
        if not self.enabled: return
        self.f_meta.write(json.dumps({"ts":_now(), **obj})+"\n"); self.f_meta.flush()

    @torch.no_grad()
    def _emit(self, layer, stage, x: torch.Tensor):
        if not self.enabled: return
        if self.batch_idx >= self.max_batches: return
        # Expect [B,T,D] or [T,D]; take the first sample
        if x.dim()==3: x = x[0]
        T, D = (x.shape[0], x.shape[1]) if x.dim()==2 else (1, int(x.numel()))
        norms = x.norm(dim=-1) if x.dim()==2 else x.reshape(-1)
        stats = {
            "mean": float(norms.mean().item()) if norms.numel() else 0.0,
            "std":  float(norms.std().item())  if norms.numel() else 0.0,
            "max":  float(x.abs().max().item()) if x.numel() else 0.0,
            **_q_abs(x),
            "hill": _hill(norms, k=min(200, max(20, norms.numel()//50))) if norms.numel() else 0.0,
        }
        row = dict(ts=_now(), batch=self.batch_idx, layer=layer, stage=stage, T=T, D=D,
                   dtype=str(x.dtype).replace("torch.",""), device=str(x.device),
                   nan=bool(torch.isnan(x).any().item()) if x.numel() else False,
                   inf=bool(torch.isinf(x).any().item()) if x.numel() else False,
                   magic=";".join(map(str,_magic_hits(x))), **stats)
        self.w_stats.writerow(row); self.f_stats.flush()

        if self.topk>0 and x.dim()==2 and x.numel()>0:
            A = x.abs().reshape(-1)
            k = min(self.topk, A.numel())
            vals, idx = torch.topk(A, k)
            pos = (idx // D).tolist(); dim = (idx % D).tolist()
            for t,d,v in zip(pos,dim,vals.tolist()):
                self.w_hot.writerow(dict(ts=_now(), batch=self.batch_idx, layer=layer, stage=stage,
                                         token=int(t), dim=int(d), absval=float(v)))
            self.f_hot.flush()

    # --------- hook installation (no model edits) ----------
    def _install(self):
        if not self.enabled: return

        # 0) one-time parameter dtype/device snapshot
        meta = {}
        for n,p in self.model.named_parameters():
            if any(k in n for k in ("poles","residues","norm.scale")):
                meta[f"dtype:{n}"]=str(p.dtype); meta[f"device:{n}"]=str(p.device)
        self.meta(**meta)

        # 1) Embed + initial mask (register on the first Linear/Embedding output)
        def _embed_post(mod, inp, out):
            self._emit(-1, "embed", out.detach().to(torch.float32))
        # Try to find the first block's pre-norm as "pre-block(-1)"
        try:
            emb_mod = getattr(self.model, "embedding_layer", None)
            if emb_mod is not None:
                self.hooks.append(emb_mod.register_forward_hook(lambda m,i,o: _embed_post(m,i,o)))
        except: pass

        # 2) Per-block hooks (works for both Hyena and Attention blocks)
        blocks = getattr(self.model, "blocks", [])
        for li, b in enumerate(blocks):
            # pre-norm output (≈ block input after pre_norm)
            if hasattr(b, "pre_norm"):
                self.hooks.append(b.pre_norm.register_forward_hook(
                    lambda m,i,o,li=li: self._emit(li, "pre", o.detach().to(torch.float32))
                ))
            # Attention branch: inner MHA input (pre_hook) and block output
            if hasattr(b, "inner_mha_cls"):
                self.hooks.append(b.inner_mha_cls.register_forward_pre_hook(
                    lambda m,i,li=li: self._emit(li, "mha_input(pre_norm)", i[0].detach().to(torch.float32))
                ))
                self.hooks.append(b.register_forward_hook(
                    lambda m,i,o,li=li: self._emit(li, "post(attn+mlp)", (o[0] if isinstance(o,(list,tuple)) else o).detach().to(torch.float32))
                ))
            # Hyena branch: projections, filter output, out_filter_dense, post
            if hasattr(b, "projections"):
                self.hooks.append(b.projections.register_forward_hook(
                    lambda m,i,o,li=li: self._emit(li, "proj(pre-hyena)", o.detach().to(torch.float32))
                ))
            if hasattr(b, "filter"):
                self.hooks.append(b.filter.register_forward_hook(
                    lambda m,i,o,li=li: self._emit(li, "branch(hyena)", (o[0] if isinstance(o,(list,tuple)) else o).detach().to(torch.float32))
                ))
            if hasattr(b, "out_filter_dense"):
                self.hooks.append(b.out_filter_dense.register_forward_hook(
                    lambda m,i,o,li=li: self._emit(li, "after_out_dense", o.detach().to(torch.float32))
                ))
            # MLP post path for both block types (if present)
            if hasattr(b, "mlp"):
                self.hooks.append(b.mlp.register_forward_hook(
                    lambda m,i,o,li=li: self._emit(li, "mlp_out", o.detach().to(torch.float32))
                ))
            # final block output (post residual)
            self.hooks.append(b.register_forward_hook(
                lambda m,i,o,li=li: self._emit(li, "post(block)", (o[0] if isinstance(o,(list,tuple)) else o).detach().to(torch.float32))
            ))

        # 3) Final norm
        if hasattr(self.model, "norm") and isinstance(self.model.norm, nn.Module):
            self.hooks.append(self.model.norm.register_forward_hook(
                lambda m,i,o: self._emit(len(blocks), "final_norm", o.detach().to(torch.float32))
            ))

    # call this once per batch at a convenient place
    def bump_batch(self):
        if not self.enabled: return
        self.batch_idx += 1

# Diagnostic control flag
RUN_DIAG = os.getenv("RUN_DIAG","1") == "1"

# ------------- Diagnostic flags for isolating issues -------------
# Set these env vars to disable specific optimizations for debugging
DISABLE_FLASH_DEPTHWISE = os.getenv("EV_DISABLE_FLASH_DEPTHWISE", "0") == "1"
DISABLE_FLASHFFT = os.getenv("EV_DISABLE_FLASHFFT", "0") == "1"
FORCE_FP32_HYENA = os.getenv("EV_FORCE_FP32_HYENA", "0") == "1"

if DISABLE_FLASH_DEPTHWISE:
    print("[DIAGNOSTIC] Disabling Flash Depthwise Conv", flush=True)
if DISABLE_FLASHFFT:
    print("[DIAGNOSTIC] Disabling FlashFFT Conv", flush=True)
if FORCE_FP32_HYENA:
    print("[DIAGNOSTIC] Forcing FP32 for Hyena blocks", flush=True)

# ------------- speed knobs -------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

print("[NUMERICS] TF32 matmul:", torch.backends.cuda.matmul.allow_tf32,
      " cuDNN:", torch.backends.cudnn.allow_tf32, flush=True)

# ------------ env ------------
ROOT   = os.environ["ROOT"]; CUSTOM = os.environ["CUSTOM"]
CFG    = os.environ["CFG"];  WEIGHTS= os.environ["WEIGHTS"]; PKG=os.environ["PKG"]
CENTRAL_CONTIGS = os.environ["CENTRAL_CONTIGS"]; IDS_CSV = os.environ["IDS_CSV"]
OUTDIR = Path(os.environ["OUTDIR"]); OUTDIR.mkdir(parents=True, exist_ok=True)
TMPDIR = OUTDIR/"tmp"; TMPDIR.mkdir(exist_ok=True)

DTYPE  = os.environ["DTYPE"]
CHUNK  = int(os.environ["CHUNK"]); STRIDE=int(os.environ["STRIDE"])
N_SYNTH= int(os.environ["N_SYNTH"]); N_GENOMES=int(os.environ["N_GENOMES"])
WINDOWS_PER_GENOME=int(os.environ["WINDOWS_PER_GENOME"])
LAYER_SAMPLE_MAXTOK=int(os.environ["LAYER_SAMPLE_MAXTOK"]); Z_THRESH=float(os.environ["Z_THRESH"])
ARGMAX_LAYERS=[int(x) for x in os.environ["ARGMAX_LAYERS"].split(",")]
DTYPE_FLAG_FP32=int(os.environ["DTYPE_FLAG_FP32"]); DTYPE_FLAG_BF16=int(os.environ["DTYPE_FLAG_BF16"])
BATCH=int(os.environ.get("BATCH","8"))

SHARD_IDX   = int(os.getenv("SHARD_IDX","0"))
SHARD_COUNT = int(os.getenv("SHARD_COUNT","1"))

CURRENT_SEED = 1337 + int(os.getenv("SEED_IDX", "0"))
device=torch.device("cuda"); torch.manual_seed(CURRENT_SEED); np.random.seed(CURRENT_SEED); random.seed(CURRENT_SEED)
print(f"[SEED] Using seed {CURRENT_SEED}", flush=True)

# ------------ progress heartbeat function ------------
def ts(msg):
    print(f"[{time.strftime('%F %T')}] SHARD {SHARD_IDX}/{SHARD_COUNT} :: {msg}", flush=True)

# ------------ import model ------------
# Decide RoPE patch based on cfg before importing model
sys.path.insert(0, CUSTOM)
sys.path.insert(0, ROOT)
from importlib import import_module
base_cfg_dict = json.load(open(CFG))
if base_cfg_dict.get("use_interpolated_rotary_pos_emb", False):
    import rotary_wrapper  # must precede model import

ModelMod = import_module(f"{PKG}.modified_model"); TokMod = import_module(f"{PKG}.tokenizer")
StripedHyena = getattr(ModelMod, "StripedHyena"); ByteTokenizer = getattr(TokMod, "ByteTokenizer")

# Cfg class with dict-like compatibility for model code
class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)
    
    # dict-like compatibility methods
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

def build_model():
    cfg = Cfg(dict(base_cfg_dict))
    
    # Apply diagnostic overrides if set
    if DISABLE_FLASH_DEPTHWISE:
        cfg.use_flash_depthwise = False
        print(f"[DIAGNOSTIC] Setting use_flash_depthwise=False in config", flush=True)
    if DISABLE_FLASHFFT:
        cfg.use_flashfft = False
        print(f"[DIAGNOSTIC] Setting use_flashfft=False in config", flush=True)
    
    if DTYPE_FLAG_FP32 or FORCE_FP32_HYENA:
        cfg.attn_block_dtype = torch.float32; cfg.mlp_dtype = torch.float32; cfg.hyena_block_dtype = torch.float32
        if FORCE_FP32_HYENA:
            print("[DIAGNOSTIC] Forcing all dtypes to FP32 due to FORCE_FP32_HYENA", flush=True)
    elif DTYPE_FLAG_BF16:
        # OFFICIAL PATH: Construct ALL in fp32, then convert params post-init
        cfg.attn_block_dtype = torch.float32; cfg.mlp_dtype = torch.float32; cfg.hyena_block_dtype = torch.float32
    else:
        cfg.attn_block_dtype = torch.float16; cfg.mlp_dtype = torch.float16; cfg.hyena_block_dtype = torch.float32
    
    # Use exact config values - no overrides (true HF parity)
    
    model = StripedHyena(cfg).eval()
    
    # EXACT CLASSIC EVO APPROACH: ALWAYS call to_bfloat16_except_poles_residues() 
    # This matches the force_dtype() behavior in modeling_hyena.py
    if not FORCE_FP32_HYENA and hasattr(model, "to_bfloat16_except_poles_residues"):
        model.to_bfloat16_except_poles_residues()
        print("[CLASSIC EVO] Applied to_bfloat16_except_poles_residues()", flush=True)
    elif FORCE_FP32_HYENA:
        print("[DIAGNOSTIC] Skipping bf16 conversion due to FORCE_FP32_HYENA", flush=True)
        
        # Quick dtype sanity log (cheap & very helpful)
        check_names = ["blocks.0.pre_norm.scale", "blocks.0.projections.weight", 
                       "blocks.0.filter.poles", "blocks.0.filter.residues"]
        for n, p in model.named_parameters():
            if any(n.endswith(k) for k in check_names):
                print(f"[DTYPE] {n}: {p.dtype}", flush=True)
    # For fp16: rely on dtype in cfg
    
    # Fix split checkpoint detection
    wdir = Path(WEIGHTS)
    layer_files = list(wdir.glob("layer_*.pt"))
    if layer_files:
        model.load_from_split_converted_state_dict(str(wdir))
    else:
        pts = list(wdir.glob("*.pt"))
        sd = torch.load(pts[0], map_location='cpu')
        model.load_state_dict(sd)
        del sd
    return model.to(device), cfg

tok = ByteTokenizer(); PAD = getattr(tok, "pad_id", 0)

def tokenize_single(seq):
    ids = tok.encode(seq[:CHUNK])
    L = min(len(ids), CHUNK)
    x = torch.tensor(ids[:CHUNK], dtype=torch.long, device=device)
    if L < CHUNK:
        pad = torch.full((CHUNK-L,), PAD, dtype=torch.long, device=device)
        x = torch.cat([x, pad])
    return x, L

# ------------ data helpers ------------
def load_genome_text_by_id(genome_id):
    fp = Path(CENTRAL_CONTIGS) / f"{genome_id}_raw.fna.gz"
    if not fp.exists(): return None
    seqs=[]
    with gzip.open(fp, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"): seqs.append(str(rec.seq).upper())
    return "".join(seqs) if seqs else None

def sample_windows_from_genome(genome_id, windows_per=WINDOWS_PER_GENOME, stride=STRIDE, seed=42):
    dna = load_genome_text_by_id(genome_id)
    if not dna or len(dna) < CHUNK: return []
    rng = np.random.default_rng(seed); out=[]
    if stride>0:
        for start in range(0, len(dna)-CHUNK+1, stride):
            out.append((genome_id, dna[start:start+CHUNK]))
            if len(out) >= windows_per: break
    else:
        for _ in range(windows_per):
            start = int(rng.integers(0, len(dna)-CHUNK+1))
            out.append((genome_id, dna[start:start+CHUNK]))
    return out

def load_genome_ids(n):
    """Load genome IDs from stable CSV manifest."""
    seed_csv = os.environ.get("SEED_SET_CSV", "")
    if seed_csv and Path(seed_csv).exists():
        df = pd.read_csv(seed_csv)
        unique_gids = df['genome_id'].unique()
        print(f"[INFO] Using {len(unique_gids)} genomes from stable manifest: {seed_csv}")
        return list(unique_gids[:n])
    else:
        # Fallback to original random sampling
        import random
        from pathlib import Path
        
        genome_dir = Path(CENTRAL_CONTIGS)
        genome_files = list(genome_dir.glob("*_raw.fna.gz"))
        all_gids = [f.stem.replace("_raw.fna", "") for f in genome_files]
        
        random.seed(42)
        if len(all_gids) < n:
            print(f"[WARN] Only {len(all_gids)} genomes available, using all")
            return all_gids
        else:
            selected = random.sample(all_gids, n)
            print(f"[INFO] Randomly selected {n} genomes from {len(all_gids)} available")
            return selected

def rand_dna(L, seed=0):
    rng=np.random.default_rng(seed); return "".join(np.array(list("ACGTN"))[rng.integers(0,5,size=L)])

def synthetic_sequences(n=int(os.environ["N_SYNTH"])):
    base = rand_dna(CHUNK, seed=42)
    seqs = {
        "base": base,
        "shift_1": base[1:]+base[:1],
        "shift_10": base[10:]+base[:10],
        "shift_50": base[50:]+base[:50],
        "shift_100": base[100:]+base[:100],
        "shift_500": base[500:]+base[:500],
        "shift_1000": base[1000:]+base[:1000],
        "shift_2000": base[2000:]+base[:2000],
        "random": rand_dna(CHUNK, seed=43),
        "permuted": "".join(np.array(list(base))[np.random.default_rng(9).permutation(len(base))]),
        "reverse": base[::-1],
        "all_A": "A"*CHUNK,
        "all_C": "C"*CHUNK,
        "repeat_10": (base[:10]*400)[:CHUNK],
        "repeat_100": (base[:100]*40)[:CHUNK],
    }
    return dict(list(seqs.items())[:n])

def shard_items(items):
    # items: list of (sid, seq)
    return [it for i, it in enumerate(items) if (i % SHARD_COUNT) == SHARD_IDX]

# ------------ metrics ------------
def effective_rank(X):
    # NO CAPPING: Use all tokens
    Xc = X - X.mean(0, keepdim=True)
    # Removed LAYER_SAMPLE_MAXTOK subsampling - use ALL tokens
    S = torch.linalg.svdvals(Xc)
    if S.numel()==0 or (S<=0).all(): return 0.0,0.0
    p=(S/S.sum()).clamp_min(1e-12); H=-(p*p.log()).sum()
    erank=float(torch.exp(H).item()); pr=float((S.sum()**2/(S.pow(2).sum()+1e-12)).item())
    return erank,pr

def isotropy_metrics(H):
    # NO CAPPING: Use all tokens for full T×T matrix
    Hn=F.normalize(H,dim=-1); m=Hn.mean(0,keepdim=True); cos=(Hn*m).sum(-1)
    iso=float(1 - cos.abs().mean().item())
    # Removed 4096 pair sampling - compute full T×T matrix
    C=(Hn@Hn.T)  # full T×T matrix
    return iso, float(C.mean().item()), float(C.std().item())

def percentiles_abs(x, qs=(95,99,99.9,99.99)):
    xa=x.abs().reshape(-1); t=torch.tensor([q/100 for q in qs], device=xa.device, dtype=xa.dtype)
    vals=torch.quantile(xa,t,interpolation='linear'); return {f"p{str(q).replace('.','_')}": float(v.item()) for q,v in zip(qs,vals)}

def zscore_extremes(H, z_thresh=float(os.environ["Z_THRESH"])):
    mu=H.mean(0,keepdim=True); sd=H.std(0,keepdim=True).clamp_min(1e-6); Z=(H-mu)/sd
    return float(Z.abs().max().item()), float((Z.abs()>z_thresh).float().mean().item())

def argmax_where(H):
    Ha=H.abs(); flat=int(Ha.argmax().item()); T,D=H.shape; return flat//D, flat%D

def topk_token_norms(H,k=10):
    norms=H.norm(dim=-1); v,_=torch.topk(norms,min(k,norms.numel())); return float(v.mean().item()), float(v.max().item())

# ------------ run (shard) ------------
model, cfg = build_model()

# Initialize external diagnostic tracer
diag = NoEditDiag(model, tag=f"{os.getenv('DTYPE','unk')}_sh{SHARD_IDX}")

# Dtype assertions after model build
def _peek(param_name):
    for n, p in model.named_parameters():
        if param_name in n:
            return n, p.dtype, p.device
    return None, None, None

probe = [
    "blocks.0.pre_norm.scale",   # hyena norm param
    "blocks.0.filter.poles",     # MUST stay float32
    "blocks.0.filter.residues",  # MUST stay float32
    "blocks.0.projections.weight", # bf16 in bf16 run
]

print("[DTYPES]")
for key in probe:
    n, dt, dev = _peek(key)
    if n is not None:
        print(f"  {n}: {dt} on {dev}", flush=True)

# Hard guards for critical rules
name, dt, _ = _peek("blocks.0.filter.poles")
if name: assert dt == torch.float32, f"Poles not float32: {dt}"
name, dt, _ = _peek("blocks.0.filter.residues") 
if name: assert dt == torch.float32, f"Residues not float32: {dt}"

print(f"[MODE] DTYPE={DTYPE} | AMP=disabled | TF32=False | cfg.hyena_block_dtype={getattr(cfg,'hyena_block_dtype',None)}",
      flush=True)

# HF parity: no AMP
amp_dtype = None

# build items and shard
genome_ids = load_genome_ids(N_GENOMES)
real_windows = []
for gid in genome_ids:
    real_windows += sample_windows_from_genome(gid)
real_items  = [(f"{gid}:{i}", seq) for i,(gid,seq) in enumerate(real_windows)]
synth_map   = synthetic_sequences()
synth_items = list(synth_map.items())

real_items  = shard_items(real_items)
synth_items = shard_items(synth_items)

# FIX 3: Streaming layer-by-layer processing to avoid memory explosion
def forward_and_collect(items, source_tag):
    rows=[]
    total = len(items)
    ts(f"Forward loop start [{source_tag}] total={total}")
    for i in range(0, total, BATCH):
        if i == 0 or i % max(BATCH*4, 16) == 0:
            ts(f"[{source_tag}] progress {i}/{total}")
        if (i // BATCH) % 10 == 0 and SHARD_IDX == 0:
            print(f"[{time.strftime('%F %T')}] shard {SHARD_IDX}: processed {i}/{total} items", flush=True)

        # Group batch
        batch = items[i:i+BATCH]
        seq_list = [s for _, s in batch]

        # --- tokenization ---
        toks, lens = [], []
        for s in seq_list:
            x, L = tokenize_single(s)
            toks.append(x[None, :]); lens.append(L)
        X   = torch.cat(toks, dim=0)  # [B, T]
        pos = torch.arange(CHUNK, device=device)[None, :]
        Ls  = torch.tensor(lens, device=device)[:, None]
        mask = (pos < Ls).contiguous()

        # EXACT CLASSIC EVO APPROACH: Use padding_mask directly
        # This matches modeling_hyena.py line 111: padding_mask=attention_mask
        mask_kwargs = {"padding_mask": mask}

        # --- forward with error handling ---
        try:
            with torch.inference_mode():
                out = model(X, output_hidden_states=True, return_dict=True, **mask_kwargs)
        except RuntimeError as e:
            import traceback
            print(f"[OOM] bytes_allocated={torch.cuda.memory_allocated()}, reserved={torch.cuda.memory_reserved()}", flush=True)
            print(f"[FAIL] Forward pass failed for {DTYPE} shard {SHARD_IDX}: {e}")
            traceback.print_exc()
            raise
        
        # DIAGNOSTIC CHECK 1: Verify mask is working - check for leakage in padded regions
        if "hidden_states" in out:
            HS_last = out["hidden_states"][-1]
            # positions that should be padding for each item in batch
            pad = ~mask
            # measure how much energy leaks into padding region
            leak = (HS_last.float().abs() * pad[..., None]).max().item()
            if leak > 1e-6:
                print(f"[MASK CHECK] WARNING: max |act| on PAD tokens (last layer): {leak:.4e}", flush=True)
            elif i == 0:  # only print once
                print(f"[MASK CHECK] GOOD: max |act| on PAD tokens (last layer): {leak:.4e}", flush=True)

        # === NoEditDiag tracer: mask leakage and batch advancement ===
        if isinstance(mask, torch.Tensor):
            with torch.no_grad():
                HS_last = out["hidden_states"][-1]
                leak = (HS_last.float().abs() * (~mask)[..., None]).max().item()
                diag.meta(stage="mask_check", batch=int(i//BATCH), leak_max_abs=float(leak))
        # advance batch counter for the tracer
        diag.bump_batch()

        # === DIAGNOSTIC: capture one sample through all blocks (first batch of synthetic and real) ===
        if RUN_DIAG and i == 0 and source_tag in ("synthetic", "real") and SHARD_IDX == 0:
            ts("Running diagnostic capture on first sample...")
            try:
                diag_sample = 0  # pick first item in the batch
                captures, branch_mods = capture_block_decomposition(model, X, lens, mask_kwargs, batch_index=diag_sample, move_to_cpu=True)

                # === Build decomposition table, save CSV and plots ===
                diag_dir = OUTDIR / "diagnostics"
                diag_dir.mkdir(exist_ok=True, parents=True)

                df_gain = per_layer_decomposition_table(captures)
                df_gain.to_csv(diag_dir / f"{DTYPE}_{source_tag}_layer_gain_decomposition_sh{SHARD_IDX}.csv", index=False)

                plot_layer_gains(df_gain, str(diag_dir / f"{DTYPE}_{source_tag}_layer_gain_sh{SHARD_IDX}.png"),
                                title=f"[{DTYPE}] Per-layer gain (post/pre) {source_tag} (shard {SHARD_IDX})")
                plot_tail_curves(df_gain, str(diag_dir / f"{DTYPE}_{source_tag}_tail_curves_sh{SHARD_IDX}.png"),
                                title=f"[{DTYPE}] Tail ratio by subpath (p99.9/p95) {source_tag} (shard {SHARD_IDX})")

                # === Culprits (post activations) ===
                T_eff = int(lens[diag_sample])
                df_culprits = culprits_table(captures, token_len=T_eff, layer_start=0, k=20)
                df_culprits.to_csv(diag_dir / f"{DTYPE}_{source_tag}_culprits_topk_sh{SHARD_IDX}.csv", index=False)

                # Aggregate hot dimensions over suspect layers (e.g., >= 11 from your plots)
                suspect_start = 11
                if not df_culprits.empty:
                    dim_hot = (df_culprits[df_culprits["layer"] >= suspect_start]
                              .groupby(["layer","dim"])
                              .agg(cnt=("absval","size"), vmax=("absval","max"), vmean=("absval","mean"))
                              .reset_index())
                    dim_hot.to_csv(diag_dir / f"{DTYPE}_{source_tag}_hot_dims_sh{SHARD_IDX}.csv", index=False)
                    plot_hot_dims(dim_hot, str(diag_dir / f"{DTYPE}_{source_tag}_hot_dims_sh{SHARD_IDX}.png"),
                                 title=f"[{DTYPE}] Hot dimensions (layers ≥ {suspect_start}) {source_tag} (shard {SHARD_IDX})", topn=30)

                    # Heatmap over tokens
                    plot_hot_tokens_heatmap(df_culprits, str(diag_dir / f"{DTYPE}_{source_tag}_hot_tokens_heatmap_sh{SHARD_IDX}.png"),
                                           title=f"[{DTYPE}] Token×Layer max|act| {source_tag} (shard {SHARD_IDX})",
                                           max_token=min(512, T_eff))

                # === Tail shape (kurtosis & Hill) ===
                df_tail = tail_shape_table(captures)
                df_tail.to_csv(diag_dir / f"{DTYPE}_{source_tag}_tail_shape_sh{SHARD_IDX}.csv", index=False)

                # === Smoking gun: pick worst (layer, token, dim) and plot its trajectory ===
                if not df_culprits.empty:
                    worst = df_culprits.sort_values("absval", ascending=False).iloc[0]
                    plot_smoking_gun_traj(
                        captures, token=int(worst["token"]), dim=int(worst["dim"]),
                        out_png=str(diag_dir / f"{DTYPE}_{source_tag}_smoking_gun_traj_L{int(worst['layer'])}_t{int(worst['token'])}_d{int(worst['dim'])}_sh{SHARD_IDX}.png"),
                        title=f"[{DTYPE}] Smoking gun: L{int(worst['layer'])} token {int(worst['token'])} dim {int(worst['dim'])} {source_tag} (shard {SHARD_IDX})"
                    )

                # === Optional: approximate operator gain for branch modules ===
                # (Run only for a few layers to keep it cheap)
                op_rows = []
                try:
                    D = captures["post"][0].shape[1] if captures["post"] and captures["post"][0] is not None else None
                    if D is not None:
                        dev = X.device
                        for li, mod in enumerate(branch_mods):
                            if mod is None: 
                                continue
                            if li < suspect_start - 2 or li > suspect_start + 4:
                                continue  # sample a small window around suspect zone
                            g = approx_operator_gain(mod, D=D, device=dev, dtype=torch.bfloat16, iters=3)
                            op_rows.append(dict(layer=li, approx_branch_gain=g))
                finally:
                    if op_rows:
                        pd.DataFrame(op_rows).to_csv(diag_dir / f"{DTYPE}_{source_tag}_approx_branch_gain_sh{SHARD_IDX}.csv", index=False)
                
                ts(f"Diagnostic capture complete. Results saved to {diag_dir}")
            except Exception as e:
                ts(f"Diagnostic capture failed: {e}")

        # --- Process layer-by-layer, temporarily cast to fp32, release immediately ---
        for li, HB in enumerate(out["hidden_states"]):
            HBf = HB if HB.dtype == torch.float32 else HB.to(torch.float32)

            for b, (sid, _) in enumerate(batch):
                L = lens[b]
                H = HBf[b, :L].contiguous()  # [T, D] fp32 temporary tensor

                # ===== Metrics calculation (unchanged) =====
                norms=H.norm(dim=-1); mn,sd,mx=float(norms.mean()),float(norms.std()),float(norms.max())
                er,pr=effective_rank(H); iso,mu,sdc=isotropy_metrics(H); pct=percentiles_abs(H)
                zmax,fracZ=zscore_extremes(H); t_idx,d_idx=argmax_where(H); tk_mean,tk_max=topk_token_norms(H)

                rows.append({
                    "dtype": DTYPE, "source": source_tag, "id": sid, "layer": li,
                    "mean_norm": mn, "std_norm": sd, "max_norm": mx,
                    "eff_rank": er, "part_ratio": pr,
                    "iso": iso, "cos_mu": mu, "cos_sd": sdc,
                    "max_abs": float(H.abs().max().item()),
                    "zmax": zmax, "frac_z_gt": f"gt{Z_THRESH:g}",
                    "frac_z": float(fracZ),
                    "argmax_token": int(t_idx), "argmax_dim": int(d_idx),
                    "topk_token_norm_mean": tk_mean, "topk_token_norm_max": tk_max,
                    **pct,
                })

                del H
            del HBf, HB
        del out, X, mask, toks
        torch.cuda.empty_cache()

    ts(f"Forward loop done [{source_tag}] collected {len(rows)} row(s)")
    return rows

rows_synth = forward_and_collect(synth_items, "synthetic")
rows_real  = forward_and_collect(real_items,  "real")

# write shard outputs
m = OUTDIR/"metrics"; m.mkdir(exist_ok=True, parents=True)
ts(f"Saving shard CSVs to {m}")
pd.DataFrame(rows_synth).to_csv(m/f"{DTYPE}_synth_layer_metrics_seed{os.getenv('SEED_IDX','0')}.csv", index=False)
pd.DataFrame(rows_real ).to_csv(m/f"{DTYPE}_real_layer_metrics_seed{os.getenv('SEED_IDX','0')}.csv", index=False)

ts("SHARD DONE")
PY

# Mark seed as successful
touch "${OUTDIR}/.success"
echo "[$(date +'%F %T')] Seed ${SEED_IDX} completed successfully"

echo "=============================================="
echo "[$(date +'%F %T')] SEED ${SEED_IDX} COMPLETE!"
echo "[$(date +'%F %T')] Results in: ${OUTDIR}"
echo "=============================================="

done  # End of seed loop

echo ""
echo "=============================================="
echo "[$(date +'%F %T')] ALL ${NUM_SEEDS} SEEDS COMPLETE!"
echo "[$(date +'%F %T')] Results in: ${RESULT_DIR}/${RUN_TAG}/"
echo "=============================================="
