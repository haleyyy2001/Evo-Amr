#!/usr/bin/env python3
"""
Evo-1-8K ULTRA-OPTIMIZED Processor - FIXED + EXTRACT_FINAL/TOKEN_NORM/POOL
==============================================================================
Critical Bug Fixes (kept from your version):
- HDF5 resize with proper maxshape
- CUDA Graph AMP context consistency
- Thread-safe tokenizer (thread-local instances)
- O(1) deque for futures
- Proper GPU detection
- Atomic file writing
- Enhanced NaN handling

New Features in this revision:
- --extract_final : take final RMSNorm output (last_hidden_state)
- --token_norm {none,l2,ln} : per-token normalization on GPU
- --pool {none,mean,max,meanmax} : per-window pooling (I/O-light mode)
    * none    -> per-token writing (original behavior)
    * mean    -> 1 vector per 4k window
    * max     -> 1 vector per 4k window
    * meanmax -> 2H vector per 4k window (mean||max)
==============================================================================
"""

# Import rotary wrapper BEFORE any model imports
import sys
import os
# Add current directory to path for rotary_wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rotary_wrapper  # patches flash_attn if needed

# ────────────────────────────────────────────────────────────────────────
# Standard library
# ────────────────────────────────────────────────────────────────────────
import argparse
import csv
import gc
import gzip
import logging
import os
import shutil
import subprocess
import tempfile
from tempfile import NamedTemporaryFile
import threading
import time
import traceback
import warnings
import queue
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import multiprocessing as mp

# ────────────────────────────────────────────────────────────────────────
# Third-party
# ────────────────────────────────────────────────────────────────────────
import h5py
import numpy as np
from Bio import SeqIO
import yaml

# ╭──────────────────────────────────────────────────────────────────────╮
# │                           CONFIGURATION                               │
# ╰──────────────────────────────────────────────────────────────────────╯

@dataclass
class Config:
    """Ultra-optimized configuration for maximum performance"""

    # Cache and data directories (will be overridden by config.yaml)
    CENTRAL_CONTIGS: str = "/path/to/your/sequences/raw/nucleotide"
    LOCAL_GENOME_DIR: str = "/path/to/your/genomes"
    EMBEDDINGS_DIR: str = "/path/to/your/embeddings_output"
    CSV_FILE: str = "/path/to/your/genome_list.csv"

    # Model configuration (relative paths will be resolved)
    MODEL_CONFIG_PATH: str = "../evo/evo_1_131k_base/config.json"
    MODEL_WEIGHTS_PATH: str = "../evo/evo-1-8k-base"
    CUSTOM_MODEL_PATH: str = "."

    # Processing parameters
    CHUNK_SIZE: int = 4000
    MAX_TOKENS: int = 4000
    BATCH_SIZE: int = 8
    GENOMES_PER_GPU_BATCH: int = 10
    NUM_GPUS: int = 0

    # Threading for tokenization
    TOKENIZER_THREADS: int = 4
    TOKENIZER_QUEUE_SIZE: int = 20

    # Memory settings
    MIN_FREE_MEMORY_GB: float = 6.0
    MEMORY_FRACTION: float = 0.92
    MAX_GENOME_SIZE_MB: int = 1200

    # System settings
    MAX_RETRIES: int = 3
    CLEANUP_INTERVAL: int = 1
    SEQUENTIAL_MODE: bool = False
    MEMORY_DEBUG: bool = False
    STORAGE_DEBUG: bool = False

    # Optimization settings
    USE_MIXED_PRECISION: bool = True
    USE_BFLOAT16: bool = True
    USE_CUDA_GRAPHS: bool = False
    USE_STATEFUL: bool = False
    EXTRACT_LAYER: int = 16           # used if EXTRACT_FINAL is False
    USE_ASYNC_WRITER: bool = True
    USE_DUAL_STREAMS: bool = True
    AUTO_TUNE_BATCH: bool = True
    COMPRESSION: str = "lzf"
    COMPRESSION_LEVEL: int = None
    BLOCK_SIZE: int = 4000

    # Diagnostic flags
    NO_MASK_TEST: bool = False
    EDGE_REPEAT_PAD: bool = False
    LOG_UNIQUE_IDS: bool = False
    USE_FP32: bool = False

    # Normalization settings
    RAW_EMBEDDINGS: bool = True
    NORMALIZE_EMBEDDINGS: bool = False
    CLIP_MIN: float = -10.0
    CLIP_MAX: float = 10.0
    LAYER_NORM_EPS: float = 1e-5
    MAX_NORM: float = 10.0

    # Precision and checksum
    USE_FLOAT16_STORAGE: bool = False   # on-disk dtype; recommend fp16 over bf16 for h5py
    USE_FLETCHER32: bool = False
    USE_SHUFFLE: bool = True

    # NEW: extraction & postprocessing flags
    EXTRACT_FINAL: bool = False                   # if True: use final RMSNorm (last_hidden_state)
    TOKEN_NORM: str = "l2"                        # "none" | "l2" | "ln"
    POOL: str = "none"                            # "none" | "mean" | "max" | "meanmax"

    @classmethod
    def from_yaml(cls, yaml_path: str = None):
        """Load configuration from YAML file"""
        if yaml_path is None:
            # Default config path relative to script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(script_dir, 'config.yaml')
        
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Create config instance
            config = cls()
            
            # Update with YAML values
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Make paths absolute if they're relative
            script_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Convert relative paths to absolute
            path_fields = ['MODEL_CONFIG_PATH', 'MODEL_WEIGHTS_PATH', 'CUSTOM_MODEL_PATH']
            for field in path_fields:
                path_value = getattr(config, field)
                if not os.path.isabs(path_value):
                    setattr(config, field, os.path.abspath(os.path.join(script_dir, path_value)))
            
            return config
        else:
            print(f"Warning: Config file not found at {yaml_path}. Using default values.")
            print("Please create config.yaml file with your paths.")
            return cls()

# ╭──────────────────────────────────────────────────────────────────────╮
# │                    LOGGING & UTILITY HELPERS                          │
# ╰──────────────────────────────────────────────────────────────────────╯

def setup_logging(suffix: str = "") -> logging.Logger:
    log_dir = Path(os.getcwd()) / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = log_dir / f"evo_ultra_fixed_{ts}{suffix}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(fname)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger = logging.getLogger(f"EvoUltraFixed{suffix}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_gpu_count() -> int:
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_lines = [line for line in result.stdout.split('\n') if line.strip() and 'GPU' in line]
            count = len(gpu_lines)
            if count == 0:
                raise RuntimeError("nvidia-smi found no GPUs")
            return count
    except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
        raise RuntimeError(f"No GPUs detected or nvidia-smi unavailable: {e}")

def validate_gpu_ids(gpu_ids: List[int]) -> List[int]:
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            available_gpus = []
            for line in result.stdout.split('\n'):
                if 'GPU' in line and ':' in line:
                    try:
                        gpu_id = int(line.split(':')[0].split()[-1])
                        available_gpus.append(gpu_id)
                    except (ValueError, IndexError):
                        continue
            if not available_gpus:
                raise RuntimeError("No GPUs found in nvidia-smi output")
            valid_gpus = [gpu for gpu in gpu_ids if gpu in available_gpus]
            if not valid_gpus:
                raise RuntimeError(f"Requested GPUs {gpu_ids} not available. Available: {available_gpus}")
            return valid_gpus
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(f"Cannot validate GPUs: {e}")

# ╭──────────────────────────────────────────────────────────────────────╮
# │                    THREADED TOKENIZER                                 │
# ╰──────────────────────────────────────────────────────────────────────╯

class ThreadedTokenizer:
    """High-performance threaded tokenizer with thread-local storage"""

    def __init__(self, tokenizer_class, chunk_size: int, cfg: Config, torch_module, logger):
        self.tokenizer_class = tokenizer_class
        self.chunk_size = chunk_size
        self.cfg = cfg
        self.torch = torch_module
        self.logger = logger
        max_threads = os.cpu_count() or 4
        actual_threads = min(cfg.TOKENIZER_THREADS, max_threads)
        self.executor = ThreadPoolExecutor(max_workers=actual_threads)
        self.stop_event = threading.Event()
        self.logger.info(f"ThreadedTokenizer: {actual_threads} threads")

    def _get_tokenizer(self):
        import threading
        if not hasattr(self, '_tls'):
            self._tls = threading.local()
        if not hasattr(self._tls, 'tokenizer'):
            self._tls.tokenizer = self.tokenizer_class()
        return self._tls.tokenizer

    def _tokenize_chunk(self, chunk: str):
        tokenizer = self._get_tokenizer()
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(chunk)
        else:
            ids = tokenizer(chunk)
        ids = self.torch.tensor(ids, dtype=self.torch.long)
        valid_len = min(ids.numel(), self.chunk_size)
        if ids.size(0) > self.chunk_size:
            ids = ids[:self.chunk_size]
        elif ids.size(0) < self.chunk_size:
            if getattr(self.cfg, 'EDGE_REPEAT_PAD', False):
                last = ids[-1].item() if ids.numel() > 0 else 0
                pad = self.torch.full((self.chunk_size - ids.size(0),), last, dtype=self.torch.long)
            else:
                pad_val = getattr(self, 'pad_id', 0)
                pad = self.torch.full((self.chunk_size - ids.size(0),), pad_val, dtype=self.torch.long)
            ids = self.torch.cat([ids, pad])
        return ids, valid_len

    def _tokenize_chunk_with_id(self, idx: int, chunk: str):
        tokens, valid_len = self._tokenize_chunk(chunk)
        return idx, tokens, valid_len

    def start_tokenizing(self, dna_chunks: List[Tuple[int, str]]) -> deque:
        futures = deque()
        for idx, chunk in dna_chunks:
            if self.stop_event.is_set():
                break
            future = self.executor.submit(self._tokenize_chunk_with_id, idx, chunk)
            futures.append(future)
        return futures

    def get_batch(self, futures: deque, batch_size: int):
        ids, batch, lens = [], [], []
        for _ in range(min(batch_size, len(futures))):
            if futures:
                future = futures.popleft()
                try:
                    idx, tokens, valid_len = future.result(timeout=40.0)
                    ids.append(idx)
                    batch.append(tokens)
                    lens.append(valid_len)
                except Exception as e:
                    self.logger.error(f"Tokenization error: {e}")
        if batch:
            return ids, self.torch.stack(batch), self.torch.tensor(lens, dtype=self.torch.long)
        return None, None, None

    def shutdown(self):
        self.stop_event.set()
        self.executor.shutdown(wait=True)

# ╭──────────────────────────────────────────────────────────────────────╮
# │                    ASYNC HDF5 WRITER                                  │
# ╰──────────────────────────────────────────────────────────────────────╯

class AsyncHDF5Writer:
    """Ultra-optimized async writer with atomic file operations"""

    def __init__(self, output_path: Path, expected_tokens: int, hidden_dim: int,
                 cfg: Config, logger: logging.Logger, genome_id: str = None):
        self.output_path = output_path
        self.temp_path = output_path.with_name(output_path.name + '.tmp')
        self.cfg = cfg
        self.logger = logger
        self.genome_id = genome_id
        self.write_queue = queue.Queue(maxsize=30)
        self.writer_thread = None
        self.stop_event = threading.Event()
        self.error_event = threading.Event()
        self.h5_file = None
        self.dataset = None
        self.current_pos = 0
        self.expected_tokens = expected_tokens
        self.hidden_dim = hidden_dim
        self.write_buffer = []
        self.buffer_size = 0
        self.total_written = 0

        self._init_file()
        self.writer_thread = threading.Thread(target=self._writer_loop)
        self.writer_thread.start()

    def _init_file(self):
        self.h5_file = h5py.File(self.temp_path, 'w', libver='latest')
        chunk_rows = min(self.cfg.BLOCK_SIZE, self.expected_tokens)
        chunk_shape = (chunk_rows, self.hidden_dim)
        storage_dtype = 'float16' if self.cfg.USE_FLOAT16_STORAGE else 'float32'
        try:
            self.dataset = self.h5_file.create_dataset(
                'embeddings',
                shape=(self.expected_tokens, self.hidden_dim),
                maxshape=(None, self.hidden_dim),
                dtype=storage_dtype,
                compression=self.cfg.COMPRESSION if self.cfg.COMPRESSION != 'none' else None,
                compression_opts=self.cfg.COMPRESSION_LEVEL if self.cfg.COMPRESSION == 'gzip' else None,
                chunks=chunk_shape,
                shuffle=self.cfg.USE_SHUFFLE,
                fletcher32=self.cfg.USE_FLETCHER32 
            )
            self.logger.info(f"HDF5 initialized with {self.cfg.COMPRESSION} compression")
        except (TypeError, ValueError) as e:
            self.logger.warning(f"LZF not available ({e}), using gzip level 1")
            self.dataset = self.h5_file.create_dataset(
                'embeddings',
                shape=(self.expected_tokens, self.hidden_dim),
                maxshape=(None, self.hidden_dim),
                dtype=storage_dtype,
                compression='gzip',
                compression_opts=1,
                chunks=chunk_shape,
                shuffle=self.cfg.USE_SHUFFLE,
                fletcher32=self.cfg.Use_FLETCHER32 if hasattr(self.cfg,'Use_FLETCHER32') else self.cfg.USE_FLETCHER32
            )

    def _writer_loop(self):
        while not self.stop_event.is_set() or not self.write_queue.empty() or self.write_buffer:
            try:
                try:
                    data, position = self.write_queue.get(timeout=0.1)
                    self.write_buffer.append((data, position))
                    self.buffer_size += data.shape[0]
                except queue.Empty:
                    if self.write_buffer and self.stop_event.is_set():
                        self._flush_buffer()
                    continue
                if self.buffer_size >= self.cfg.BLOCK_SIZE or self.stop_event.is_set():
                    self._flush_buffer()
            except Exception as e:
                self.logger.error(f"Writer thread error: {e}")
                self.error_event.set()
                break

    def _flush_buffer(self):
        if not self.write_buffer:
            return
        self.write_buffer.sort(key=lambda x: x[1])
        for data, position in self.write_buffer:
            end_pos = position + data.shape[0]
            self.dataset[position:end_pos] = data
            self.current_pos = max(self.current_pos, end_pos)
            self.total_written += data.shape[0]
        if self.total_written % (self.cfg.BLOCK_SIZE * 2) == 0:
            self.h5_file.flush()
        self.write_buffer = []
        self.buffer_size = 0

    def write(self, data: np.ndarray, position: int):
        if self.error_event.is_set():
            raise RuntimeError("Writer thread encountered an error")
        self.write_queue.put((data, position))

    def finalize(self) -> Path:
        try:
            self.stop_event.set()
            if self.writer_thread:
                self.writer_thread.join(timeout=60)
            if self.error_event.is_set():
                raise RuntimeError("Writer thread had errors")
            if self.write_buffer:
                self._flush_buffer()
            self.h5_file.flush()
            if self.current_pos < self.expected_tokens:
                self.dataset.resize(self.current_pos, axis=0)
            self.h5_file.attrs['total_tokens'] = self.current_pos
            self.h5_file.attrs['hidden_dim'] = self.hidden_dim
            if self.genome_id:
                self.h5_file.attrs['genome_id'] = self.genome_id
            self.h5_file.attrs['compression'] = self.cfg.COMPRESSION
            self.h5_file.attrs['complete'] = True
            self.h5_file.attrs['ultra_optimized_fixed'] = True
            self.h5_file.close()
            os.replace(str(self.temp_path), str(self.output_path))
            return self.output_path
        finally:
            try:
                if os.path.exists(self.temp_path):
                    os.remove(self.temp_path)
            except Exception:
                pass

# ╭──────────────────────────────────────────────────────────────────────╮
# │                 MEMORY PROFILER                                       │
# ╰──────────────────────────────────────────────────────────────────────╯

class MemoryProfiler:
    def __init__(self, device: str, logger: logging.Logger, min_free: float):
        import torch
        self.torch, self.device, self.logger, self.min_free = torch, device, logger, min_free
        self.dev_id = 0

    def _stats(self) -> Dict[str, float]:
        try:
            if not self.torch.cuda.is_available():
                return {}
            if hasattr(self.torch.cuda, 'mem_get_info'):
                free_bytes, total_bytes = self.torch.cuda.mem_get_info(self.dev_id)
                free = free_bytes / 1_073_741_824
                tot = total_bytes / 1_073_741_824
            else:
                tot = self.torch.cuda.get_device_properties(self.dev_id).total_memory / 1_073_741_824
                r = self.torch.cuda.memory_reserved(self.dev_id) / 1_073_741_824
                free = tot - r
            a = self.torch.cuda.memory_allocated(self.dev_id) / 1_073_741_824
            r = self.torch.cuda.memory_reserved(self.dev_id) / 1_073_741_824
            return {"allocated": a, "reserved": r, "free": free, "total": tot, "util": 100 * a / tot}
        except Exception:
            return {}

    def okay(self, need_gb: float) -> bool:
        s = self._stats()
        return s and s["free"] >= need_gb

    def log(self, title: str) -> None:
        s = self._stats()
        if s:
            self.logger.info(f"GPU {title}: {s['allocated']:.2f}GB/{s['total']:.1f}GB ({s['util']:.1f}% util)")

    def reset_peaks(self):
        try:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats(self.dev_id)
        except Exception:
            pass

    def peaks(self) -> Dict[str, float]:
        try:
            alloc = self.torch.cuda.max_memory_allocated(self.dev_id) / 1_073_741_824
            rsrv = self.torch.cuda.max_memory_reserved(self.dev_id) / 1_073_741_824
            return {"peak_alloc_GB": alloc, "peak_reserved_GB": rsrv}
        except Exception:
            return {}

    def log_peaks(self, title: str = "PEAK"):
        p = self.peaks()
        if p:
            self.logger.info(f"[{title}] peak_alloc={p['peak_alloc_GB']:.2f} GB | peak_reserved={p['peak_reserved_GB']:.2f} GB")

# ╭──────────────────────────────────────────────────────────────────────╮
# │                  ULTRA-OPTIMIZED GPU ENCODER                          │
# ╰──────────────────────────────────────────────────────────────────────╯

class UltraGPUEncoder:
    """Maximum performance encoder with all bug fixes + new extraction options"""

    def __init__(self, gpu_id: int, cfg: Config, logger: logging.Logger, retry_count: int = 0):
        self.cfg, self.logger = cfg, logger
        self.device_id = gpu_id
        self.retry_count = retry_count
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        import torch
        self.torch = torch
        if not self.torch.cuda.is_available():
            raise RuntimeError("CUDA not available / No GPU detected")
        self.device = 'cuda:0'
        self.torch.backends.cuda.matmul.allow_tf32 = True
        self.torch.backends.cudnn.allow_tf32 = True
        self.torch.set_float32_matmul_precision('high')
        self.torch.cuda.set_device(0)

        if hasattr(self.torch.cuda, 'set_per_process_memory_fraction'):
            try:
                self.torch.cuda.set_per_process_memory_fraction(cfg.MEMORY_FRACTION, 0)
            except RuntimeError:
                pass

        self.mp = MemoryProfiler(self.device, logger, cfg.MIN_FREE_MEMORY_GB)

        self.model = None
        self.tokenizer_class = None
        self.tokenizer = None
        self.layer_norm = None
        self.cuda_graph = None
        self.graph_input = None
        self.graph_mask = None
        self.graph_output_last_hidden = None
        self.graph_failed = False
        self.hidden_dim = None
        self.prefetch_stream = None
        self.pos_template = None
        self._ln_no_affine = None

        try:
            self._load_model()
            if cfg.AUTO_TUNE_BATCH:
                self._autotune_batch()
            if cfg.USE_CUDA_GRAPHS and self.retry_count == 0:
                try:
                    self._setup_cuda_graph()
                except Exception as e:
                    self.logger.warning(f"CUDA graph setup failed: {e}. Continuing without graphs.")
                    self.graph_failed = True
                    self.cuda_graph = None
                    self.torch.cuda.empty_cache()
                    self.torch.cuda.synchronize()
                    gc.collect()
            if cfg.USE_DUAL_STREAMS:
                self.prefetch_stream = self.torch.cuda.Stream(device=self.device)
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self):
        try:
            if hasattr(self, 'cuda_graph') and self.cuda_graph:
                del self.cuda_graph
                self.cuda_graph = None
            if hasattr(self, 'graph_input'):
                del self.graph_input
            if hasattr(self, 'graph_mask'):
                del self.graph_mask
            if hasattr(self, 'graph_output_last_hidden'):
                del self.graph_output_last_hidden
            if hasattr(self, 'tokenizer') and self.tokenizer:
                self.tokenizer.shutdown()
                del self.tokenizer
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'torch'):
                self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize()
            gc.collect()
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Cleanup error (non-critical): {e}")

    def _positional_probe(self, amp_dtype):
        import torch
        L = min(256, self.cfg.CHUNK_SIZE)
        x = torch.randint(1, 128, (1, L), device=self.device, dtype=torch.long)
        kw = {}
        lens = torch.tensor([L], device=self.device)
        valid = (torch.arange(L, device=self.device).unsqueeze(0) < lens.unsqueeze(1))
        if hasattr(self, '_mask_sig'):
            if "attention_mask" in self._mask_sig:
                kw["attention_mask"] = valid
            elif "padding_mask" in self._mask_sig:
                kw["padding_mask"] = valid
        with torch.inference_mode():
            if amp_dtype:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    y1 = self.model(x, return_dict=True, **kw)["last_hidden_state"][0]
                    y2 = self.model(x.roll(shifts=1, dims=1), return_dict=True, **kw)["last_hidden_state"][0]
            else:
                y1 = self.model(x, return_dict=True, **kw)["last_hidden_state"][0]
                y2 = self.model(x.roll(shifts=1, dims=1), return_dict=True, **kw)["last_hidden_state"][0]
        delta = (y1 - y2).abs().mean().item()
        self.logger.info(f"[POS_PROBE] mean|Δ| after circular shift = {delta:.6f}")
        if delta < 1e-4:
            self.logger.error("🚨 POS ENCODING LOOKS INACTIVE (Δ≈0). Check rotary_wrapper / model config.")
        return delta

    def _load_model(self) -> None:
        self.mp.log("BEFORE model-load")
        import json
        from types import SimpleNamespace
        # Ensure custom model path is in sys.path
        if self.cfg.CUSTOM_MODEL_PATH not in sys.path:
            sys.path.insert(0, self.cfg.CUSTOM_MODEL_PATH)
        import rotary_wrapper
        from evo_1_131k_base.modified_model import StripedHyena
        from evo_1_131k_base.tokenizer import ByteTokenizer

        self.tokenizer_class = ByteTokenizer
        if hasattr(self.tokenizer_class, 'pad_id'):
            self.pad_id = int(getattr(self.tokenizer_class, 'pad_id'))
        else:
            self.pad_id = 0
        self.logger.info(f"Using PAD_ID: {self.pad_id}")

        class ConfigAdapter(SimpleNamespace):
            def get(self, key, default=None): return getattr(self, key, default)
            def __getitem__(self, key): return getattr(self, key)
            def __contains__(self, key): return hasattr(self, key)

        with open(self.cfg.MODEL_CONFIG_PATH, 'r') as f:
            config_dict = json.load(f)
        self.model_config = ConfigAdapter(**config_dict)

        import torch as _torch
        setattr(self.model_config, 'use_flash_attn', True)
        setattr(self.model_config, 'attn_block_dtype', _torch.bfloat16)
        setattr(self.model_config, 'mlp_dtype', _torch.bfloat16)
        setattr(self.model_config, 'hyena_block_dtype', _torch.float32)

        self.hidden_dim = getattr(self.model_config, 'hidden_size', 4096)
        self.logger.info(f"Detected model hidden_dim: {self.hidden_dim}")

        self.tokenizer = ThreadedTokenizer(self.tokenizer_class, self.cfg.CHUNK_SIZE, self.cfg, self.torch, self.logger)
        self.tokenizer.pad_id = self.pad_id

        self.model = StripedHyena(self.model_config)

        weights_path = Path(self.cfg.MODEL_WEIGHTS_PATH)
        if weights_path.is_dir():
            layer_files = list(weights_path.glob("layer_*.pt"))
            if layer_files:
                self.model.load_from_split_converted_state_dict(str(weights_path))
            else:
                pt_files = list(weights_path.glob("*.pt"))
                if pt_files:
                    state_dict = self.torch.load(pt_files[0], map_location='cpu')
                    self.model.load_state_dict(state_dict)
        else:
            state_dict = self.torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict)

        try:
            tokenizer_instance = self.tokenizer_class()
            probe_tokens = tokenizer_instance.encode("ACGTNACGTN")
            self.logger.info(f"PAD_ID = {self.pad_id}, probe token set: {set(probe_tokens)}")
            if self.pad_id in set(probe_tokens):
                self.logger.warning(f"🚨 PAD_ID {self.pad_id} collides with real tokens!")
            else:
                self.logger.info("✅ No PAD collision detected")
        except Exception as e:
            self.logger.warning(f"Could not check PAD collision: {e}")

        self.logger.info(f"PAD embedding zeroing disabled (using length-based masks instead)")
        self.model = self.model.to(self.device)
        self.model.eval()

        import inspect
        self._mask_sig = set(inspect.signature(self.model.forward).parameters.keys())
        self.logger.info(f"[FWD SIG] {self._mask_sig}")

        amp_dtype = self.torch.bfloat16 if self.cfg.USE_BFLOAT16 and not getattr(self.cfg, 'USE_FP32', False) else None
        self._positional_probe(amp_dtype)

        first_param = next(self.model.parameters())
        self.logger.info(f"[PRECISION] Model running in: {first_param.dtype}")

        total_params = sum(p.numel() for p in self.model.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in self.model.parameters())
        self.logger.info(f"[WEIGHTS] total={total_params:,} nonzero={nonzero_params:,} ({100*nonzero_params/total_params:.2f}% nonzero)")

        probe_param = None
        for name, param in self.model.named_parameters():
            if "projections.weight" in name or "Wqkv.weight" in name:
                probe_param = param
                probe_name = name
                break
        if probe_param is not None:
            param_std = probe_param.detach().float().std().item()
            param_mean = probe_param.detach().float().mean().item()
            self.logger.info(f"[WEIGHTS] probe '{probe_name}' std={param_std:.6f} mean={param_mean:.6f}")
            if param_std < 1e-4:
                self.logger.error("❌ CRITICAL: Model weights appear collapsed!")
        else:
            self.logger.warning("No suitable probe parameter found for weight verification")

        for p in self.model.parameters():
            p.requires_grad = False

        if self.cfg.NORMALIZE_EMBEDDINGS:
            self.layer_norm = self.torch.nn.LayerNorm(self.hidden_dim, eps=self.cfg.LAYER_NORM_EPS, elementwise_affine=False).to(self.device)
            if self.cfg.USE_BFLOAT16:
                self.layer_norm = self.layer_norm.to(self.torch.bfloat16)

        self._debug_model_behavior()
        self.mp.log("AFTER model-load")

    def _debug_model_behavior(self):
        try:
            with self.torch.no_grad():
                x = self.torch.full((1, min(128, self.cfg.CHUNK_SIZE)),
                                    fill_value=self.pad_id, dtype=self.torch.long, device=self.device)
                pos = self.torch.arange(x.shape[1], device=self.device).unsqueeze(0)
                lens = self.torch.tensor([x.shape[1]], device=self.device)
                mask = (pos < lens.unsqueeze(1))
                out = self.model(x, padding_mask=mask, return_dict=True)
                h = out['last_hidden_state']
                sim = self.torch.cosine_similarity(h[0, :32], h[0, 32:64], dim=-1).mean().item()
                self.logger.info(f"[debug] mean cosine(0..31 vs 32..63): {sim:.3f}")
        except Exception as e:
            self.logger.warning(f"[debug] model probe failed: {e}")

    def _autotune_batch(self):
        self.logger.info(f"Auto-tuning batch size (retry={self.retry_count})...")
        base_batch = self.cfg.BATCH_SIZE
        if self.retry_count > 0:
            reduction_factor = 0.75 ** self.retry_count
            b_high = max(int(base_batch * 2 * reduction_factor), 4)
            self.logger.info(f"Reducing max batch size due to retry: {b_high}")
        else:
            b_high = max(base_batch * 2, 4)
        b_low = 1
        ok = 1
        if getattr(self.cfg, 'USE_FP32', False):
            amp_dtype = self.torch.float32
        else:
            amp_dtype = self.torch.bfloat16 if self.cfg.USE_BFLOAT16 else self.torch.float16

        while b_low <= b_high:
            b_mid = (b_low + b_high) // 2
            try:
                self.mp.reset_peaks()
                x = self.torch.full((b_mid, self.cfg.CHUNK_SIZE), fill_value=self.pad_id, dtype=self.torch.long, device=self.device)
                with self.torch.inference_mode(), self.torch.cuda.amp.autocast(dtype=amp_dtype):
                    _ = self.model(x, return_dict=True)
                pk = self.mp.peaks()
                self.logger.info(f"[AUTOTUNE] batch={b_mid} peak_alloc={pk.get('peak_alloc_GB', 0):.2f} GB | peak_reserved={pk.get('peak_reserved_GB', 0):.2f} GB")
                ok = b_mid
                b_low = b_mid + 1
                del x
                self.torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.torch.cuda.empty_cache()
                    b_high = b_mid - 1
                else:
                    raise
        if self.cfg.USE_CUDA_GRAPHS and not self.graph_failed and self.retry_count == 0:
            safe_batch = max(1, int(ok * 0.5))
            self.logger.info(f"Applying CUDA graph safety margin: {ok} -> {safe_batch}")
            self.cfg.BATCH_SIZE = safe_batch
        else:
            self.cfg.BATCH_SIZE = ok
        self.logger.info(f"Final auto-tuned batch size: {self.cfg.BATCH_SIZE}")

    def _setup_cuda_graph(self):
        self.logger.info(f"Setting up CUDA graph for batch={self.cfg.BATCH_SIZE}")
        if not self.mp.okay(8.0):
            self.logger.warning("Insufficient memory for CUDA graph, skipping")
            self.graph_failed = True
            return
        if getattr(self.cfg, 'USE_FP32', False):
            amp_dtype = self.torch.float32
        else:
            amp_dtype = self.torch.bfloat16 if self.cfg.USE_BFLOAT16 else self.torch.float16

        self.graph_input = self.torch.full((self.cfg.BATCH_SIZE, self.cfg.CHUNK_SIZE),
                                           fill_value=self.pad_id, dtype=self.torch.long, device=self.device)
        self.graph_mask = self.torch.ones((self.cfg.BATCH_SIZE, self.cfg.CHUNK_SIZE), dtype=self.torch.bool, device=self.device)

        for _ in range(3):
            with self.torch.inference_mode(), self.torch.cuda.amp.autocast(dtype=amp_dtype):
                _ = self.model(self.graph_input, padding_mask=self.graph_mask, return_dict=True)

        try:
            self.cuda_graph = self.torch.cuda.CUDAGraph()
            with self.torch.cuda.graph(self.cuda_graph):
                with self.torch.inference_mode(), self.torch.cuda.amp.autocast(dtype=amp_dtype):
                    out = self.model(self.graph_input, padding_mask=self.graph_mask, return_dict=True, output_hidden_states=True)
                    # --- CHOOSE H (graph path) ---
                    if self.cfg.EXTRACT_FINAL:
                        h = out['last_hidden_state']
                    elif 'hidden_states' in out and self.cfg.EXTRACT_LAYER is not None and self.cfg.EXTRACT_LAYER >= 0:
                        layer_idx = min(self.cfg.EXTRACT_LAYER, len(out['hidden_states']) - 1)
                        h = out['hidden_states'][layer_idx]
                    else:
                        h = out['last_hidden_state']
                    # Per-token normalization inside graph capture
                    if self.cfg.TOKEN_NORM != "none":
                        if self.cfg.TOKEN_NORM == "l2":
                            h = self.torch.nn.functional.normalize(h, p=2, dim=-1, eps=1e-6)
                        elif self.cfg.TOKEN_NORM == "ln":
                            if self._ln_no_affine is None or self._ln_no_affine.normalized_shape[0] != h.shape[-1]:
                                self._ln_no_affine = self.torch.nn.LayerNorm(h.shape[-1], elementwise_affine=False, eps=1e-6).to(h.device)
                            h = self._ln_no_affine(h)
                    self.graph_output_last_hidden = h
            self.logger.info("CUDA graph captured successfully with consistent AMP context")
        except Exception as e:
            self.logger.error(f"CUDA graph capture failed: {e}")
            self.cuda_graph = None
            self.graph_failed = True
            if hasattr(self, 'graph_input'):
                del self.graph_input
                self.graph_input = None
            if hasattr(self, 'graph_output_last_hidden'):
                del self.graph_output_last_hidden
                self.graph_output_last_hidden = None
            self.torch.cuda.empty_cache()
            self.torch.cuda.synchronize()
            raise

    def _normalize_hidden_states(self, h):
        if self.cfg.RAW_EMBEDDINGS:
            return h
        h = self.torch.clamp(h, min=self.cfg.CLIP_MIN, max=self.cfg.CLIP_MAX)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        if self.torch.isnan(h).any() or self.torch.isinf(h).any():
            h = self.torch.nan_to_num(h, nan=0.0, posinf=self.cfg.CLIP_MAX, neginf=self.cfg.CLIP_MIN)
        return h

    def encode(self, dna: str, genome_id: str = None) -> Optional[str]:
        dna = dna.upper()
        if len(dna) < 100:
            return None

        self.mp.reset_peaks()

        inference_params = None
        if getattr(self.cfg, 'USE_STATEFUL', False):
            if hasattr(self.model, 'initialize_inference_params'):
                inference_params = self.model.initialize_inference_params()
                self.logger.info("[STATEFUL] Initialized global position tracking - processing sequentially")
                original_use_cuda_graphs = self.cfg.USE_CUDA_GRAPHS
                self.cfg.USE_CUDA_GRAPHS = False
            else:
                self.logger.warning("[STATEFUL] Model doesn't support stateful inference!")

        if not self.mp.okay(6.0):
            self.torch.cuda.empty_cache()
            gc.collect()
            if not self.mp.okay(6.0):
                return None

        if genome_id:
            output_path = Path(self.cfg.EMBEDDINGS_DIR) / f"{genome_id}.h5"
        else:
            with NamedTemporaryFile(suffix='.h5', dir=self.cfg.EMBEDDINGS_DIR, delete=False) as tf:
                output_path = Path(tf.name)

        # --- dataset sizing depends on pooling mode ---
        total_chunks = (len(dna) + self.cfg.CHUNK_SIZE - 1) // self.cfg.CHUNK_SIZE
        if self.cfg.POOL in ("mean", "max"):
            expected_rows = total_chunks
            out_dim = self.hidden_dim
        elif self.cfg.POOL == "meanmax":
            expected_rows = total_chunks
            out_dim = self.hidden_dim * 2
        else:  # "none" -> per-token
            expected_rows = total_chunks * self.cfg.CHUNK_SIZE
            out_dim = self.hidden_dim

        writer = AsyncHDF5Writer(output_path, expected_rows, out_dim, self.cfg, self.logger, genome_id)

        self.logger.info(f"Processing {len(dna):,} bp in ~{total_chunks} chunks")

        dna_chunks = []
        for i in range(0, len(dna), self.cfg.CHUNK_SIZE):
            chunk = dna[i: i + self.cfg.CHUNK_SIZE]
            if len(chunk) >= 50:
                chunk_idx = i // self.cfg.CHUNK_SIZE
                dna_chunks.append((chunk_idx, chunk))

        tokenizer_futures = self.tokenizer.start_tokenizing(dna_chunks)

        chunks_processed = 0
        if self.pos_template is None or self.pos_template.shape[1] != self.cfg.CHUNK_SIZE:
            self.pos_template = self.torch.arange(self.cfg.CHUNK_SIZE, device=self.device).unsqueeze(0)

        tokenizer_time = 0
        h2d_time = 0
        forward_time = 0
        post_norm_time = 0

        try:
            if getattr(self.cfg, 'USE_FP32', False):
                amp_dtype = None
                self.logger.info("[FP32_TEST] Running in FP32 mode (no AMP)")
            else:
                amp_dtype = self.torch.bfloat16 if self.cfg.USE_BFLOAT16 else self.torch.float16

            while tokenizer_futures:
                t0 = time.time()
                ids, batch_tensor, lens = self.tokenizer.get_batch(tokenizer_futures, self.cfg.BATCH_SIZE)
                if batch_tensor is None:
                    break
                tokenizer_time += time.time() - t0

                actual_batch_size = batch_tensor.shape[0]
                if self.cfg.USE_CUDA_GRAPHS and actual_batch_size < self.cfg.BATCH_SIZE:
                    padding = self.torch.full(
                        (self.cfg.BATCH_SIZE - actual_batch_size, self.cfg.CHUNK_SIZE),
                        self.pad_id, dtype=self.torch.long
                    ).pin_memory()
                    batch_tensor = self.torch.cat([batch_tensor, padding])
                    pad_rows = self.cfg.BATCH_SIZE - actual_batch_size
                    lens = self.torch.cat([lens, self.torch.zeros(pad_rows, dtype=self.torch.long)], dim=0)
                    valid_count = actual_batch_size
                else:
                    valid_count = actual_batch_size

                t0 = time.time()
                if self.cfg.USE_DUAL_STREAMS and self.prefetch_stream:
                    with self.torch.cuda.stream(self.prefetch_stream):
                        batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                        lens = lens.to(self.device, non_blocking=True)
                    self.torch.cuda.current_stream().wait_stream(self.prefetch_stream)
                else:
                    batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                    lens = lens.to(self.device, non_blocking=True)
                valid_mask = (self.pos_template < lens.unsqueeze(1))
                mask_kwargs = {}
                if "attention_mask" in self._mask_sig:
                    mask_kwargs["attention_mask"] = valid_mask
                elif "padding_mask" in self._mask_sig:
                    mask_kwargs["padding_mask"] = valid_mask
                if lens.eq(self.cfg.CHUNK_SIZE).all():
                    mask_kwargs = {}
                if chunks_processed < 5 or chunks_processed % 100 == 0:
                    self.logger.info(f"[MASK] lens min/max/mean: {lens.min().item()}/{lens.max().item()}/{lens.float().mean().item():.1f}")
                    if mask_kwargs:
                        self.logger.info(f"[MASK] using keys={list(mask_kwargs.keys())}")
                h2d_time += time.time() - t0

                t0 = time.time()
                with self.torch.inference_mode():
                    if self.cfg.USE_CUDA_GRAPHS and self.cuda_graph and not self.graph_failed:
                        try:
                            self.graph_input.copy_(batch_tensor)
                            if mask_kwargs:
                                if "attention_mask" in mask_kwargs:
                                    self.graph_mask.copy_(mask_kwargs["attention_mask"])
                                elif "padding_mask" in mask_kwargs:
                                    self.graph_mask.copy_(mask_kwargs["padding_mask"])
                            self.cuda_graph.replay()
                            h = self.graph_output_last_hidden
                        except Exception as e:
                            self.logger.warning(f"CUDA graph replay failed, falling back: {e}")
                            self.graph_failed = True
                            self.cuda_graph = None
                            if getattr(self.cfg, 'NO_MASK_TEST', False):
                                outputs = self.model(batch_tensor, return_dict=True, output_hidden_states=True)
                            else:
                                outputs = self.model(batch_tensor, return_dict=True, output_hidden_states=True, **mask_kwargs)

                            # --- CHOOSE H (fallback path) ---
                            if self.cfg.EXTRACT_FINAL:
                                h = outputs['last_hidden_state']
                                if chunks_processed == 0:
                                    self.logger.info("[EXTRACT] Using final RMSNorm (last_hidden_state)")
                            elif 'hidden_states' in outputs and self.cfg.EXTRACT_LAYER is not None and self.cfg.EXTRACT_LAYER >= 0:
                                layer_idx = min(self.cfg.EXTRACT_LAYER, len(outputs['hidden_states']) - 1)
                                h = outputs['hidden_states'][layer_idx]
                                if chunks_processed == 0:
                                    self.logger.info(f"[EXTRACT] Using layer {layer_idx}/{len(outputs['hidden_states']) - 1} (pre-final RMSNorm)")
                            else:
                                h = outputs['last_hidden_state']
                                if chunks_processed == 0:
                                    self.logger.info("[EXTRACT] Fallback to last_hidden_state (final RMSNorm)")

                            # --- TOKEN NORM (fallback path) ---
                            if self.cfg.TOKEN_NORM != "none":
                                if self.cfg.TOKEN_NORM == "l2":
                                    h = self.torch.nn.functional.normalize(h, p=2, dim=-1, eps=1e-6)
                                elif self.cfg.TOKEN_NORM == "ln":
                                    if self._ln_no_affine is None or self._ln_no_affine.normalized_shape[0] != h.shape[-1]:
                                        self._ln_no_affine = self.torch.nn.LayerNorm(h.shape[-1], elementwise_affine=False, eps=1e-6).to(h.device)
                                    h = self._ln_no_affine(h)
                    else:
                        if getattr(self.cfg, 'NO_MASK_TEST', False):
                            if chunks_processed == 0:
                                self.logger.info("[NO_MASK_TEST] Running without attention mask")
                            if amp_dtype:
                                with self.torch.cuda.amp.autocast(dtype=amp_dtype):
                                    outputs = self.model(batch_tensor, return_dict=True, output_hidden_states=True)
                            else:
                                outputs = self.model(batch_tensor, return_dict=True, output_hidden_states=True)
                        else:
                            if amp_dtype:
                                with self.torch.cuda.amp.autocast(dtype=amp_dtype):
                                    outputs = self.model(batch_tensor, return_dict=True, output_hidden_states=True, **mask_kwargs)
                            else:
                                outputs = self.model(batch_tensor, return_dict=True, output_hidden_states=True, **mask_kwargs)

                        # --- CHOOSE H (normal path) ---
                        if self.cfg.EXTRACT_FINAL:
                            h = outputs['last_hidden_state']
                            if chunks_processed == 0:
                                self.logger.info("[EXTRACT] Using final RMSNorm (last_hidden_state)")
                        elif 'hidden_states' in outputs and self.cfg.EXTRACT_LAYER is not None and self.cfg.EXTRACT_LAYER >= 0:
                            layer_idx = min(self.cfg.EXTRACT_LAYER, len(outputs['hidden_states']) - 1)
                            h = outputs['hidden_states'][layer_idx]
                            if chunks_processed == 0:
                                self.logger.info(f"[EXTRACT] Using layer {layer_idx}/{len(outputs['hidden_states']) - 1} (pre-final RMSNorm)")
                        else:
                            h = outputs['last_hidden_state']
                            if chunks_processed == 0:
                                self.logger.info("[EXTRACT] Fallback to last_hidden_state (final RMSNorm)")

                        # --- TOKEN NORM (normal path) ---
                        if self.cfg.TOKEN_NORM != "none":
                            if self.cfg.TOKEN_NORM == "l2":
                                h = self.torch.nn.functional.normalize(h, p=2, dim=-1, eps=1e-6)
                            elif self.cfg.TOKEN_NORM == "ln":
                                if self._ln_no_affine is None or self._ln_no_affine.normalized_shape[0] != h.shape[-1]:
                                    self._ln_no_affine = self.torch.nn.LayerNorm(h.shape[-1], elementwise_affine=False, eps=1e-6).to(h.device)
                                h = self._ln_no_affine(h)

                forward_time += time.time() - t0

                t0 = time.time()
                with self.torch.inference_mode():
                    if self.cfg.NORMALIZE_EMBEDDINGS:
                        h = self._normalize_hidden_states(h)
                post_norm_time += time.time() - t0

                # --- WRITE ---
                h = h[:valid_count]

                for local_idx in range(valid_count):
                    L = int(lens[local_idx].item())
                    if L <= 0:
                        continue
                    chunk_idx = ids[local_idx]

                    if self.cfg.POOL == "none":
                        # per-token writing
                        chunk_h = h[local_idx, :L]                  # [L, H]
                        hidden_np = chunk_h.cpu().float().numpy()
                        if self.cfg.USE_FLOAT16_STORAGE:
                            hidden_np = hidden_np.astype(np.float16, copy=False)
                        abs_position = chunk_idx * self.cfg.CHUNK_SIZE
                        writer.write(hidden_np.copy(), abs_position)
                    else:
                        # pooled writing: one row per 4k window
                        win = h[local_idx, :L]                      # [L, H]
                        if self.cfg.POOL == "mean":
                            vec = win.mean(dim=0)                   # [H]
                        elif self.cfg.POOL == "max":
                            vec = win.amax(dim=0)                   # [H]
                        else:  # "meanmax"
                            v_mean = win.mean(dim=0)                # [H]
                            v_max  = win.amax(dim=0)                # [H]
                            vec = self.torch.cat([v_mean, v_max], dim=-1)  # [2H]
                        vec_np = vec.cpu().float().numpy()
                        if self.cfg.USE_FLOAT16_STORAGE:
                            vec_np = vec_np.astype(np.float16, copy=False)
                        writer.write(vec_np[None, :].copy(), chunk_idx)

                chunks_processed += valid_count

                if chunks_processed % 100 == 0:
                    self.torch.cuda.empty_cache()
                if chunks_processed % 200 == 0:
                    total_time = tokenizer_time + h2d_time + forward_time + post_norm_time
                    if total_time > 0:
                        self.logger.info(f"Progress: {chunks_processed}/{total_chunks} | "
                                         f"Tokenizer: {100*tokenizer_time/total_time:.1f}% | "
                                         f"H2D: {100*h2d_time/total_time:.1f}% | "
                                         f"Forward: {100*forward_time/total_time:.1f}% | "
                                         f"PostNorm: {100*post_norm_time/total_time:.1f}%")

            final_path = writer.finalize()
            total_time = tokenizer_time + h2d_time + forward_time + post_norm_time
            if total_time > 0:
                self.logger.info(f"Performance breakdown - "
                                 f"Tokenizer: {tokenizer_time:.1f}s ({100*tokenizer_time/total_time:.1f}%) | "
                                 f"H2D: {h2d_time:.1f}s ({100*h2d_time/total_time:.1f}%) | "
                                 f"Forward: {forward_time:.1f}s ({100*forward_time/total_time:.1f}%) | "
                                 f"PostNorm: {post_norm_time:.1f}s ({100*post_norm_time/total_time:.1f}%)")
            self.mp.log_peaks(title=f"PEAK {genome_id or ''}".strip())

            if 'original_use_cuda_graphs' in locals():
                self.cfg.USE_CUDA_GRAPHS = original_use_cuda_graphs

            return str(final_path)

        except Exception as e:
            self.logger.error(f"Encoding error: {e}")
            writer.stop_event.set()
            return None

    def __del__(self):
        self._cleanup()

# ╭──────────────────────────────────────────────────────────────────────╮
# │                     GENOME PROCESSOR                                  │
# ╰──────────────────────────────────────────────────────────────────────╯

class GenomeProcessor:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg, self.logger = cfg, logger
        self.out_dir = Path(cfg.EMBEDDINGS_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _completed(self) -> Set[str]:
        done = set()
        for f in self.out_dir.glob("*.h5"):
            try:
                with h5py.File(f, "r") as h:
                    if h.attrs.get("complete", False):
                        done.add(f.stem)
            except Exception:
                self.logger.debug(f"Potentially corrupt file (keeping for debug): {f}")
        return done

    def _load_fasta(self, gid: str) -> Optional[str]:
        path = Path(self.cfg.CENTRAL_CONTIGS) / f"{gid}_raw.fna.gz"
        if not path.exists():
            return None
        if path.stat().st_size / 1_048_576 > self.cfg.MAX_GENOME_SIZE_MB:
            return None
        seqs = []
        try:
            with gzip.open(path, "rt") as fh:
                for rec in SeqIO.parse(fh, "fasta"):
                    seqs.append(str(rec.seq).upper())
        except Exception:
            return None
        return "".join(seqs) if seqs else None

    def run(self, gids: List[str], gpu: int, encoder=None) -> Tuple[int, int]:
        if encoder is None:
            try:
                encoder = UltraGPUEncoder(gpu, self.cfg, self.logger)
            except Exception as e:
                self.logger.error(f"Failed to initialize encoder: {e}")
                return 0, len(gids)

        ok = fail = 0
        for i, gid in enumerate(gids):
            try:
                self.logger.info(f"Processing {gid} ({i+1}/{len(gids)})")
                dna = self._load_fasta(gid)
                if not dna:
                    fail += 1
                    continue
                start_time = time.time()
                result_path = encoder.encode(dna, genome_id=gid)
                if result_path and Path(result_path).exists():
                    try:
                        with h5py.File(result_path, 'r+') as h:
                            if 'embeddings' in h:
                                shape = h['embeddings'].shape
                                h.attrs['genome_id'] = gid
                                h.attrs['sequence_length'] = len(dna)
                                h.attrs['embedding_shape_0'] = shape[0]
                                h.attrs['embedding_shape_1'] = shape[1]
                        elapsed = time.time() - start_time
                        file_size_mb = Path(result_path).stat().st_size / (1024**2)
                        self.logger.info(f"✅ {gid}: {file_size_mb:.1f}MB in {elapsed:.1f}s")
                        ok += 1
                    except Exception as e:
                        self.logger.error(f"File verification failed for {gid}: {e}")
                        fail += 1
                else:
                    self.logger.error(f"❌ Encoding failed for {gid}")
                    fail += 1
            except Exception as e:
                self.logger.error(f"Error processing {gid}: {e}")
                fail += 1
        return ok, fail

# ╭──────────────────────────────────────────────────────────────────────╮
# │                 BATCH WORKER                                          │
# ╰──────────────────────────────────────────────────────────────────────╯

def _worker(payload: Dict[str, Any]) -> Tuple[int, int]:
    for key in list(os.environ.keys()):
        if key.startswith('CUDA_'):
            del os.environ[key]
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    gpu, gids, cfg_dict = payload['gpu'], payload['gids'], payload['cfg']
    cfg = Config()
    cfg.__dict__.update(cfg_dict)
    logger = setup_logging(f"_gpu{gpu}")
    logger.info(f"Ultra-Fixed worker on GPU {gpu} - {len(gids)} genomes")
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        encoder = None
        try:
            encoder = UltraGPUEncoder(gpu, cfg, logger, retry_count=retry_count)
            gp = GenomeProcessor(cfg, logger)
            return gp.run(gids, gpu, encoder)
        except Exception as e:
            logger.error(f"Worker error (attempt {retry_count + 1}/{max_retries}): {e}")
            if encoder:
                encoder._cleanup()
                encoder = None
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg or "memory" in error_msg:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying with more conservative settings (retry {retry_count})...")
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    time.sleep(5)
                    continue
            return 0, len(gids)
    logger.error(f"Worker failed after {max_retries} attempts")
    return 0, len(gids)

# ╭──────────────────────────────────────────────────────────────────────╮
# │                             MAIN                                      │
# ╰──────────────────────────────────────────────────────────────────────╯

def main() -> None:
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    p = argparse.ArgumentParser("Evo-1-8K ULTRA-OPTIMIZED encoder (FIXED)")
    p.add_argument('--config', default=None, help='Path to YAML config file (default: config.yaml in script dir)')
    p.add_argument('--csv_file', default=None, help='Override CSV file path from config')
    p.add_argument('--limit', type=int)
    p.add_argument('--gpu_ids', default="")
    p.add_argument('--num_gpus', type=int, default=0)
    p.add_argument('--sequential', action='store_true')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--chunk_size', type=int, default=4000)
    p.add_argument('--status', action='store_true')

    # Model paths
    p.add_argument('--config_path', default="")
    p.add_argument('--model_weights_path', default="")
    p.add_argument('--custom_model_path', default="")

    # Memory settings
    p.add_argument('--memory_fraction', type=float, default=0.92)
    p.add_argument('--min_free_memory', type=float, default=6.0)

    # Optimization flags
    p.add_argument('--raw_embeddings', action='store_true')
    p.add_argument('--no_cuda_graphs', action='store_true')
    p.add_argument('--no_async_writer', action='store_true')
    p.add_argument('--no_mixed_precision', action='store_true')
    p.add_argument('--no_dual_streams', action='store_true')
    p.add_argument('--no_auto_tune', action='store_true')
    p.add_argument('--compression', default='lzf', choices=['gzip', 'lzf', 'none'])
    p.add_argument('--use_float32', action='store_true')
    p.add_argument('--enable_fletcher32', action='store_true')
    p.add_argument('--tokenizer_threads', type=int, default=4)

    # Diagnostic flags
    p.add_argument('--stateful', action='store_true', help='Use stateful inference for global position tracking')
    p.add_argument('--extract_layer', type=int, default=16, help='Layer index (pre-final RMSNorm). Ignored if --extract_final is set.')
    p.add_argument('--use_fp32', action='store_true', help='Force FP32 (no AMP/bf16) for diagnostic')
    p.add_argument('--no_mask_test', action='store_true', help='Run without attention/padding mask')
    p.add_argument('--edge_repeat_pad', action='store_true', help='Repeat last token instead of PAD 0')
    p.add_argument('--log_unique_ids', action='store_true', help='Log unique token IDs per batch')

    # NEW: extraction & postproc
    p.add_argument('--extract_final', action='store_true',
                   help='Use final RMSNorm output (last_hidden_state). Overrides --extract_layer.')
    p.add_argument('--token_norm', choices=['none','l2','ln'], default='l2',
                   help='Per-token normalization before pooling/writing. Default: l2.')
    p.add_argument('--pool', choices=['none','mean','max','meanmax'], default='none',
                   help='Window-level pooling. none=per-token writing (original).')

    args = p.parse_args()

    # Load configuration from YAML
    cfg = Config.from_yaml(args.config)
    
    # Override config values with command line arguments
    if args.csv_file:
        cfg.CSV_FILE = args.csv_file
    cfg.CHUNK_SIZE = args.chunk_size
    cfg.MAX_TOKENS = args.chunk_size
    cfg.BATCH_SIZE = args.batch_size
    cfg.SEQUENTIAL_MODE = args.sequential
    cfg.NUM_GPUS = args.num_gpus
    cfg.TOKENIZER_THREADS = args.tokenizer_threads

    if args.config_path:
        cfg.MODEL_CONFIG_PATH = os.path.abspath(args.config_path)
    if args.model_weights_path:
        cfg.MODEL_WEIGHTS_PATH = os.path.abspath(args.model_weights_path)
    if args.custom_model_path:
        cfg.CUSTOM_MODEL_PATH = os.path.abspath(args.custom_model_path)

    cfg.MEMORY_FRACTION = args.memory_fraction
    cfg.MIN_FREE_MEMORY_GB = args.min_free_memory
    cfg.RAW_EMBEDDINGS = args.raw_embeddings
    cfg.USE_CUDA_GRAPHS = not args.no_cuda_graphs
    cfg.USE_ASYNC_WRITER = not args.no_async_writer
    cfg.USE_MIXED_PRECISION = not args.no_mixed_precision
    cfg.USE_DUAL_STREAMS = not args.no_dual_streams
    cfg.AUTO_TUNE_BATCH = not args.no_auto_tune
    cfg.COMPRESSION = args.compression
    cfg.USE_FLOAT16_STORAGE = not args.use_float32
    cfg.USE_FLETCHER32 = args.enable_fletcher32

    cfg.USE_STATEFUL = args.stateful
    cfg.EXTRACT_LAYER = args.extract_layer
    cfg.USE_FP32 = args.use_fp32
    cfg.NO_MASK_TEST = args.no_mask_test
    cfg.EDGE_REPEAT_PAD = args.edge_repeat_pad
    cfg.LOG_UNIQUE_IDS = args.log_unique_ids

    # NEW
    cfg.EXTRACT_FINAL = args.extract_final
    cfg.TOKEN_NORM   = args.token_norm
    cfg.POOL         = args.pool

    logger = setup_logging()
    logger.info("Evo-1-8K ULTRA-OPTIMIZED Processor (FIXED + EXTRACT_FINAL/TOKEN_NORM/POOL)")

    if cfg.USE_FP32:
        cfg.USE_MIXED_PRECISION = False
        cfg.USE_BFLOAT16 = False
        logger.info("[DIAGNOSTIC] FP32 mode enabled - disabled AMP and bf16")

    if cfg.NO_MASK_TEST:
        logger.info("[DIAGNOSTIC] No-mask test mode enabled")
    if cfg.EDGE_REPEAT_PAD:
        logger.info("[DIAGNOSTIC] Edge-repeat padding enabled")
    if cfg.LOG_UNIQUE_IDS:
        logger.info("[DIAGNOSTIC] Token diversity logging enabled")

    try:
        import torch
        if not torch.cuda.is_available():
            logger.error("CUDA not available - cannot proceed")
            return
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.error("No CUDA GPUs detected - cannot proceed")
            return
        logger.info(f"Early GPU check passed: {gpu_count} GPUs available")
    except Exception as e:
        logger.error(f"Early GPU check failed: {e}")
        return

    try:
        with open(cfg.CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            gids = []
            for row in reader:
                genome_id = None
                for key in row.keys():
                    if key.strip().lower() in ['genome_id', 'genomeid', 'genome', 'id']:
                        genome_id = row[key].strip()
                        break
                if genome_id:
                    gids.append(genome_id)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    if args.limit:
        gids = gids[:args.limit]

    gp = GenomeProcessor(cfg, logger)
    done = gp._completed()
    todo = [g for g in gids if g not in done]

    logger.info(f"Dataset: {len(gids)} total, {len(done)} done, {len(todo)} pending")
    if args.status:
        return

    try:
        if args.gpu_ids:
            gpus = [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]
        else:
            gpu_count = get_gpu_count() if cfg.NUM_GPUS == 0 else cfg.NUM_GPUS
            gpus = list(range(gpu_count))
        gpus = validate_gpu_ids(gpus)
        logger.info(f"Using GPUs: {gpus}")
    except RuntimeError as e:
        logger.error(f"GPU validation failed: {e}")
        return

    start = time.time()
    if cfg.SEQUENTIAL_MODE or not todo:
        ok, fail = gp.run(todo, gpus[0] if gpus else 0)
    else:
        batches = []
        ptr = 0
        while ptr < len(todo):
            for gpu in gpus:
                if ptr >= len(todo):
                    break
                batch = todo[ptr: ptr + cfg.GENOMES_PER_GPU_BATCH]
                if batch:
                    batches.append({'gpu': gpu, 'gids': batch, 'cfg': cfg.__dict__.copy()})
                    ptr += cfg.GENOMES_PER_GPU_BATCH
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
            futures = {executor.submit(_worker, batch): i for i, batch in enumerate(batches)}
            for future in as_completed(futures):
                try:
                    s, f = future.result()
                    ok += s
                    fail += f
                except Exception as e:
                    logger.error(f"Batch failed: {e}")
                    fail += len(batches[futures[future]]['gids'])

    dt = time.time() - start
    logger.info("═" * 70)
    logger.info("ULTRA-OPTIMIZED PROCESSING COMPLETE (FIXED)")
    logger.info(f"Time: {dt/3600:.2f} hours")
    logger.info(f"Results: {ok} success / {fail} failures")
    if ok > 0:
        logger.info(f"Throughput: {ok/(dt/3600):.1f} genomes/hour")

if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    main()
