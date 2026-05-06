#!/usr/bin/env bash
#SBATCH --job-name=evo_amr_diagnostics
#SBATCH --partition=gpu
#SBATCH --account=pmg
#SBATCH --array=0                       # Single job for bf16 only
#SBATCH --gres=gpu:1                    # 1 GPU per task
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --signal=TERM@120              # warn 120s before kill

set -euo pipefail

# ==================== PORTABLE EVO-AMR DIAGNOSTICS ====================
# This is a portable version of the diagnostic script that uses the 
# configuration system to be directory-agnostic
# ====================================================================

# Detect script location and set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
export EVO_AMR_BASE_DIR="$BASE_DIR"

# Load configuration
echo "[$(date +'%F %T')] Loading Evo-AMR configuration from: $BASE_DIR"
export PYTHONPATH="$BASE_DIR:$BASE_DIR/config:$PYTHONPATH"

# Initialize configuration and get paths
CONFIG_PY="$BASE_DIR/config/config_manager.py"
if [[ ! -f "$CONFIG_PY" ]]; then
    echo "ERROR: Configuration manager not found at: $CONFIG_PY"
    exit 1
fi

# Get paths and environment from configuration
eval "$(/insomnia001/depts/pmg/users/ht2666/mambaforge/envs/evo-amr/bin/python - <<'EOF'
import sys
sys.path.insert(0, "/insomnia001/depts/pmg/users/ht2666/Evo_amr/config")
from config_manager import ConfigManager

config = ConfigManager("/insomnia001/depts/pmg/users/ht2666/Evo_amr")

# Set up environment variables
env_vars = config.setup_environment()
for k, v in env_vars.items():
    print(f"export {k}='{v}'")

# Set up paths
print(f"export ROOT='{config.get_model_path('evo_1_131k_base')}'")
print(f"export CUSTOM='{config.get_model_path('evo_8')}'")
print(f"export CFG='{config.get_model_path('evo_1_131k_base')}/config.json'")
print(f"export WEIGHTS='{config.get_model_path('evo_1_8k_base')}'")  # Use 8k weights
print(f"export PKG='evo_1_131k_base'")
print(f"export RESULT_DIR='{config.get_output_path('diagnostics')}'")
print(f"export LOG_DIR='{config.get_output_path('logs')}'")

# Data paths - these should be configured in paths.yaml
try:
    print(f"export CENTRAL_CONTIGS='{config.get_path('data', 'sequences')}'")
    print(f"export IDS_CSV='{config.get_model_path('evo_8')}/data/natural_proportion_sampling/ampicillin/subsets_100_1.csv'")
except:
    print("# WARNING: Data paths not configured - please set manually in paths.yaml")
    print("export CENTRAL_CONTIGS='/path/to/your/data/sequences/raw/nucleotide'")
    print("export IDS_CSV='$CUSTOM/data/natural_proportion_sampling/ampicillin/subsets_100_1.csv'")

# SLURM config
slurm_config = config.get_slurm_config()
print(f"export SLURM_LOG_DIR='{slurm_config['log_dir']}'")
EOF
)"

# Ensure required directories exist
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$SLURM_LOG_DIR"

# Update SLURM output/error paths to use portable location
#SBATCH --output=${SLURM_LOG_DIR}/evo_amr_diagnostics_%A_%a.out
#SBATCH --error=${SLURM_LOG_DIR}/evo_amr_diagnostics_%A_%a.err

echo "[$(date +'%F %T')] Starting Evo-AMR diagnostics job"
echo "[$(date +'%F %T')] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date +'%F %T')] Node: $(hostname)"
echo "[$(date +'%F %T')] Base directory: $BASE_DIR"
echo "[$(date +'%F %T')] Results directory: $RESULT_DIR"

# Multiple seeds on single node: bf16 only
DTYPE=bf16
SHARD_IDX=0
SHARD_COUNT=1
NUM_SEEDS=10  # Run 10 different seeds for statistical robustness

# Handle preemption & cleanup correctly
trap 'echo "[TERM] Graceful stop - task ${SLURM_ARRAY_TASK_ID}"; exit 0' TERM INT HUP

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
# ==============================================================

# --------------------- core env ---------------------
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=1337
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"
unset PYTORCH_NO_CUDA_MEMORY_CACHING 2>/dev/null || true

# Run tagging for output organization
export RUN_TAG=${RUN_TAG:-"diag_$(date +%F_%H%M%S)"}
echo "[$(date +'%F %T')] Run tag: $RUN_TAG"

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
BATCH_FP32=${BATCH_FP32:-2}
BATCH_MIXED=${BATCH_MIXED:-8}

# Python executable
PY=/insomnia001/depts/pmg/users/ht2666/mambaforge/envs/evo-amr/bin/python

# Create stable genome/window manifest (shared across all dtypes/seeds)
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

echo "[$(date +'%F %T')] Params: N_GENOMES=$N_GENOMES WINDOWS_PER_GENOME=$WINDOWS_PER_GENOME CHUNK=$CHUNK"
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
    "seed_idx": int(os.environ["SEED_IDX"]),
    "num_seeds": int(os.environ["NUM_SEEDS"]),
    "cfg_sha256": hashlib.sha256(open(os.environ["CFG"],"rb").read()).hexdigest(),
    "weights_dir": os.environ["WEIGHTS"],
    "host": platform.node(),
    "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES",""),
    "seed_set_csv": os.environ.get("SEED_SET_CSV",""),
    "run_tag": os.environ.get("RUN_TAG",""),
    "base_dir": os.environ.get("EVO_AMR_BASE_DIR","")
}
outdir=Path(os.environ["OUTDIR"]); outdir.mkdir(parents=True, exist_ok=True)
open(outdir/"manifest.json","w").write(json.dumps(m, indent=2))
PY

echo ""
echo "=============================================="
echo "[$(date +'%F %T')] Starting run for DTYPE=${DTYPE} SEED=${SEED_IDX}"
echo "[$(date +'%F %T')] Output directory: ${OUTDIR}"
echo "=============================================="

# Check if this seed run is already completed
REAL_CSV="${OUTDIR}/metrics/${DTYPE}_real_layer_metrics_seed${SEED_IDX}.csv"
SYNTH_CSV="${OUTDIR}/metrics/${DTYPE}_synth_layer_metrics_seed${SEED_IDX}.csv"
if [[ -f "${OUTDIR}/.success" ]] && \
   [[ -s "$REAL_CSV" ]] && \
   [[ -s "$SYNTH_CSV" ]]; then
    echo "[SKIP] Seed ${SEED_IDX} already completed successfully"
    continue
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

# -------------------- SINGLE SEED EXECUTION --------------------
echo "[$(date +'%F %T')] Executing seed $SEED_IDX with GPU: $CUDA_VISIBLE_DEVICES"

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

# Set up paths for imports
sys.path.insert(0, os.environ["CUSTOM"])
sys.path.insert(0, os.environ["ROOT"])

# Load configuration for current seed
CURRENT_SEED = 1337 + int(os.getenv("SEED_IDX", "0"))
torch.manual_seed(CURRENT_SEED); np.random.seed(CURRENT_SEED); random.seed(CURRENT_SEED)
print(f"[SEED] Using seed {CURRENT_SEED}", flush=True)

device=torch.device("cuda")

# ------------ import model ------------
from importlib import import_module
base_cfg_dict = json.load(open(os.environ["CFG"]))
if base_cfg_dict.get("use_interpolated_rotary_pos_emb", False):
    import rotary_wrapper  # must precede model import

ModelMod = import_module(f"{os.environ['PKG']}.modified_model")
TokMod = import_module(f"{os.environ['PKG']}.tokenizer")
StripedHyena = getattr(ModelMod, "StripedHyena")
ByteTokenizer = getattr(TokMod, "ByteTokenizer")

# ------------ speed knobs -------------
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

# ------------ progress heartbeat function ------------
def ts(msg):
    print(f"[{time.strftime('%F %T')}] SEED {os.getenv('SEED_IDX', '0')} :: {msg}", flush=True)

# ------------ Diagnostic flags for isolating issues -------------
DISABLE_FLASH_DEPTHWISE = os.getenv("EV_DISABLE_FLASH_DEPTHWISE", "0") == "1"
DISABLE_FLASHFFT = os.getenv("EV_DISABLE_FLASHFFT", "0") == "1"
FORCE_FP32_HYENA = os.getenv("EV_FORCE_FP32_HYENA", "0") == "1"

if DISABLE_FLASH_DEPTHWISE:
    print("[DIAGNOSTIC] Disabling Flash Depthwise Conv", flush=True)
if DISABLE_FLASHFFT:
    print("[DIAGNOSTIC] Disabling FlashFFT Conv", flush=True)
if FORCE_FP32_HYENA:
    print("[DIAGNOSTIC] Forcing FP32 for Hyena blocks", flush=True)

# Cfg class with dict-like compatibility for model code
class Cfg:
    def __init__(self, d):
        self.__dict__.update(d)
    
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
    
    # Fix split checkpoint detection
    wdir = Path(WEIGHTS)
    layer_files = list(wdir.glob("layer_*.pt"))
    if layer_files:
        model.load_from_split_converted_state_dict(str(wdir))
    else:
        pts = list(wdir.glob("*.pt"))
        if pts:
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
    Xc = X - X.mean(0, keepdim=True)
    S = torch.linalg.svdvals(Xc)
    if S.numel()==0 or (S<=0).all(): return 0.0,0.0
    p=(S/S.sum()).clamp_min(1e-12); H=-(p*p.log()).sum()
    erank=float(torch.exp(H).item()); pr=float((S.sum()**2/(S.pow(2).sum()+1e-12)).item())
    return erank,pr

def isotropy_metrics(H):
    Hn=F.normalize(H,dim=-1); m=Hn.mean(0,keepdim=True); cos=(Hn*m).sum(-1)
    iso=float(1 - cos.abs().mean().item())
    C=(Hn@Hn.T)
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

# ------------ run (seed) ------------
model, cfg = build_model()

print(f"[MODE] DTYPE={DTYPE} | AMP=disabled | TF32=False | cfg.hyena_block_dtype={getattr(cfg,'hyena_block_dtype',None)}",
      flush=True)

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

# Streaming layer-by-layer processing to avoid memory explosion
def forward_and_collect(items, source_tag):
    rows=[]
    total = len(items)
    ts(f"Forward loop start [{source_tag}] total={total}")
    for i in range(0, total, BATCH):
        if i == 0 or i % max(BATCH*4, 16) == 0:
            ts(f"[{source_tag}] progress {i}/{total}")

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
            print(f"[FAIL] Forward pass failed for {DTYPE} seed {os.getenv('SEED_IDX', '0')}: {e}")
            traceback.print_exc()
            raise
        
        # Check mask leakage into padded regions
        if "hidden_states" in out:
            HS_last = out["hidden_states"][-1]
            pad = ~mask
            leak = (HS_last.float().abs() * pad[..., None]).max().item()
            if leak > 1e-6:
                print(f"[MASK CHECK] WARNING: max |act| on PAD tokens (last layer): {leak:.4e}", flush=True)
            elif i == 0:
                print(f"[MASK CHECK] GOOD: max |act| on PAD tokens (last layer): {leak:.4e}", flush=True)

        # --- Process layer-by-layer, temporarily cast to fp32, release immediately ---
        for li, HB in enumerate(out["hidden_states"]):
            HBf = HB if HB.dtype == torch.float32 else HB.to(torch.float32)

            for b, (sid, _) in enumerate(batch):
                L = lens[b]
                H = HBf[b, :L].contiguous()  # [T, D] fp32 temporary tensor

                # ===== Metrics calculation =====
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

# write seed outputs
m = OUTDIR/"metrics"; m.mkdir(exist_ok=True, parents=True)
ts(f"Saving seed CSVs to {m}")
pd.DataFrame(rows_synth).to_csv(m/f"{DTYPE}_synth_layer_metrics_seed{os.getenv('SEED_IDX','0')}.csv", index=False)
pd.DataFrame(rows_real ).to_csv(m/f"{DTYPE}_real_layer_metrics_seed{os.getenv('SEED_IDX','0')}.csv", index=False)

ts("SEED DONE")
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