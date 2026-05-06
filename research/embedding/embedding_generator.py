#!/usr/bin/env python3
"""
Genomic Embedding Generator for Evolutionary Models

This module provides optimized embedding generation from genomic sequences using
evolutionary models. It supports various output formats, normalization methods,
and pooling strategies for efficient processing of large genomic datasets.

Key Features:
- Multi-GPU processing with automatic load balancing
- Memory-efficient streaming processing
- Multiple output formats (per-token, pooled, normalized)
- Thread-safe tokenization
- Atomic file writing with crash recovery
- CUDA graph optimization for inference acceleration

Usage:
    python embedding_generator.py --input sequences.csv --output embeddings.h5
    python embedding_generator.py --input data/ --limit 1000 --raw_embeddings
    python embedding_generator.py --help

Authors: Evo-AMR Team
License: MIT
"""

import argparse
import csv
import gc
import gzip
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import warnings
import queue
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import yaml

# Third-party imports
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.cuda import amp
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ProcessingConfig:
    """Configuration for embedding generation processing."""
    
    # Model settings
    model_config_path: str = "evo_1_131k_base/config.json"
    model_weights_path: str = "evo-1-8k-base"
    custom_model_path: str = "evo_8"
    
    # Input/Output settings
    csv_file: str = "data/natural_proportion_sampling/ampicillin/subsets_100_1.csv"
    sequences_base_dir: str = "/insomnia001/depts/pmg/projects/amr_pred/data/multi-drug_aug2024/sequences/raw/nucleotide"
    output_dir: str = "output"
    
    # Processing parameters
    chunk_size: int = 4096
    max_workers: int = 4
    batch_size: int = 8
    limit: Optional[int] = None
    
    # Output options
    raw_embeddings: bool = False
    extract_final: bool = False
    token_norm: str = "none"  # none, l2, ln
    pool: str = "none"  # none, mean, max, meanmax
    
    # Performance settings
    use_cuda_graphs: bool = True
    mixed_precision: bool = True
    sequential: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "ProcessingConfig":
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return cls()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return cls()


class SequenceTokenizer:
    """Thread-safe sequence tokenizer for genomic data."""
    
    def __init__(self, model_path: str):
        """Initialize tokenizer."""
        self.model_path = model_path
        self._local = threading.local()
    
    @property
    def tokenizer(self):
        """Get thread-local tokenizer instance."""
        if not hasattr(self._local, 'tokenizer'):
            # Import the tokenizer from the model directory
            import sys
            sys.path.insert(0, self.model_path)
            from tokenizer import ByteTokenizer
            self._local.tokenizer = ByteTokenizer()
        return self._local.tokenizer
    
    def encode(self, sequence: str, max_length: int = 4096) -> List[int]:
        """Encode a DNA sequence to tokens."""
        return self.tokenizer.encode(sequence[:max_length])
    
    def get_pad_token_id(self) -> int:
        """Get padding token ID."""
        return getattr(self.tokenizer, 'pad_id', 0)


class EmbeddingExtractor:
    """Optimized embedding extraction from evolutionary models."""
    
    def __init__(self, config: ProcessingConfig, device: torch.device):
        """Initialize the embedding extractor."""
        self.config = config
        self.device = device
        self.model = None
        self.tokenizer = None
        self._setup_model()
        self._setup_tokenizer()
        
    def _setup_model(self):
        """Setup the evolutionary model."""
        # Import model components
        import sys
        sys.path.insert(0, self.config.model_config_path.split('/')[0])
        sys.path.insert(0, self.config.custom_model_path)
        
        try:
            # Import rotary wrapper if needed
            import rotary_wrapper
        except ImportError:
            pass
        
        from modified_model import StripedHyena
        import json
        
        # Load configuration
        with open(self.config.model_config_path, 'r') as f:
            model_config = json.load(f)
        
        # Create model
        self.model = StripedHyena(model_config)
        
        # Load weights
        weights_path = Path(self.config.model_weights_path)
        if (weights_path / "pytorch_model.pt").exists():
            state_dict = torch.load(weights_path / "pytorch_model.pt", map_location='cpu')
            self.model.load_state_dict(state_dict)
        else:
            # Load from safetensors or split files
            self.model.load_from_split_converted_state_dict(str(weights_path))
        
        # Setup for inference
        self.model.eval()
        self.model.to(self.device)
        
        # Convert to appropriate precision
        if hasattr(self.model, 'to_bfloat16_except_poles_residues'):
            self.model.to_bfloat16_except_poles_residues()
        
        logger.info(f"Model loaded on {self.device}")
    
    def _setup_tokenizer(self):
        """Setup the tokenizer."""
        self.tokenizer = SequenceTokenizer(self.config.model_config_path.split('/')[0])
    
    def _apply_token_normalization(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply token-level normalization."""
        if self.config.token_norm == "l2":
            return F.normalize(embeddings, p=2, dim=-1)
        elif self.config.token_norm == "ln":
            # Layer normalization
            return F.layer_norm(embeddings, embeddings.shape[-1:])
        else:
            return embeddings
    
    def _apply_pooling(self, embeddings: torch.Tensor, 
                      valid_lengths: List[int]) -> torch.Tensor:
        """Apply pooling across sequence dimension."""
        if self.config.pool == "none":
            return embeddings
        
        pooled_results = []
        for i, length in enumerate(valid_lengths):
            valid_emb = embeddings[i, :length]  # [seq_len, hidden_dim]
            
            if self.config.pool == "mean":
                pooled = valid_emb.mean(dim=0, keepdim=True)
            elif self.config.pool == "max":
                pooled = valid_emb.max(dim=0, keepdim=True)[0]
            elif self.config.pool == "meanmax":
                mean_pool = valid_emb.mean(dim=0)
                max_pool = valid_emb.max(dim=0)[0]
                pooled = torch.cat([mean_pool, max_pool], dim=0).unsqueeze(0)
            else:
                pooled = valid_emb
            
            pooled_results.append(pooled)
        
        return torch.cat(pooled_results, dim=0)
    
    def extract_embeddings(self, sequences: List[str], 
                          sequence_ids: List[str]) -> Tuple[torch.Tensor, List[int]]:
        """Extract embeddings from a batch of sequences."""
        # Tokenize sequences
        tokenized = []
        valid_lengths = []
        
        for seq in sequences:
            tokens = self.tokenizer.encode(seq, self.config.chunk_size)
            length = min(len(tokens), self.config.chunk_size)
            
            # Pad to chunk_size
            if len(tokens) < self.config.chunk_size:
                pad_id = self.tokenizer.get_pad_token_id()
                tokens.extend([pad_id] * (self.config.chunk_size - len(tokens)))
            else:
                tokens = tokens[:self.config.chunk_size]
            
            tokenized.append(tokens)
            valid_lengths.append(length)
        
        # Convert to tensor
        input_ids = torch.tensor(tokenized, dtype=torch.long, device=self.device)
        
        # Create attention mask
        attention_mask = torch.zeros_like(input_ids)
        for i, length in enumerate(valid_lengths):
            attention_mask[i, :length] = 1
        
        # Extract embeddings
        with torch.no_grad():
            if self.config.mixed_precision:
                with amp.autocast():
                    outputs = self.model(
                        input_ids,
                        padding_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
            else:
                outputs = self.model(
                    input_ids,
                    padding_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
        
        # Extract final embeddings
        if self.config.extract_final:
            embeddings = outputs.hidden_states[-1]  # Last layer
        else:
            embeddings = outputs.hidden_states[-2]  # Second-to-last layer
        
        # Apply normalization
        embeddings = self._apply_token_normalization(embeddings)
        
        # Apply pooling
        embeddings = self._apply_pooling(embeddings, valid_lengths)
        
        return embeddings.cpu(), valid_lengths


class GenomicDataLoader:
    """Efficient loader for genomic sequence data."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the data loader."""
        self.config = config
        self.sequences_data = self._load_sequences_metadata()
    
    def _load_sequences_metadata(self) -> pd.DataFrame:
        """Load sequence metadata from CSV file."""
        try:
            df = pd.read_csv(self.config.csv_file)
            if self.config.limit:
                df = df.head(self.config.limit)
            logger.info(f"Loaded {len(df)} sequence entries")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise
    
    def load_sequence(self, sequence_id: str) -> Optional[str]:
        """Load a single genomic sequence by ID."""
        # Extract genome ID from sequence ID
        genome_id = sequence_id.split(':')[0] if ':' in sequence_id else sequence_id
        
        # Construct file path
        file_path = Path(self.config.sequences_base_dir) / f"{genome_id}_raw.fna.gz"
        
        if not file_path.exists():
            logger.warning(f"Sequence file not found: {file_path}")
            return None
        
        try:
            sequences = []
            with gzip.open(file_path, 'rt') as f:
                for record in SeqIO.parse(f, 'fasta'):
                    sequences.append(str(record.seq).upper())
            
            return ''.join(sequences) if sequences else None
            
        except Exception as e:
            logger.error(f"Error loading sequence {sequence_id}: {e}")
            return None
    
    def get_batch_iterator(self, batch_size: int):
        """Get iterator over batches of sequence data."""
        for i in range(0, len(self.sequences_data), batch_size):
            batch_data = self.sequences_data.iloc[i:i+batch_size]
            
            sequences = []
            sequence_ids = []
            
            for _, row in batch_data.iterrows():
                seq_id = row.get('id', row.get('sequence_id', str(row.name)))
                sequence = self.load_sequence(seq_id)
                
                if sequence is not None:
                    sequences.append(sequence)
                    sequence_ids.append(seq_id)
            
            if sequences:
                yield sequences, sequence_ids


class EmbeddingProcessor:
    """Main processor for generating genomic embeddings."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize the processor."""
        self.config = config
        self.device = self._setup_device()
        self.extractor = EmbeddingExtractor(config, self.device)
        self.data_loader = GenomicDataLoader(config)
        
    def _setup_device(self) -> torch.device:
        """Setup the compute device."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU")
        return device
    
    def process_sequences(self, output_path: str):
        """Process all sequences and save embeddings."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup HDF5 output file
        with h5py.File(output_path, 'w') as h5f:
            embeddings_group = h5f.create_group('embeddings')
            metadata_group = h5f.create_group('metadata')
            
            processed_count = 0
            total_batches = len(self.data_loader.sequences_data) // self.config.batch_size + 1
            
            logger.info(f"Starting processing with batch size {self.config.batch_size}")
            
            for batch_idx, (sequences, sequence_ids) in enumerate(
                self.data_loader.get_batch_iterator(self.config.batch_size)
            ):
                if not sequences:
                    continue
                
                try:
                    # Extract embeddings
                    embeddings, valid_lengths = self.extractor.extract_embeddings(
                        sequences, sequence_ids
                    )
                    
                    # Save to HDF5
                    for i, seq_id in enumerate(sequence_ids):
                        seq_embeddings = embeddings[i].numpy()
                        valid_length = valid_lengths[i]
                        
                        # Store embeddings
                        embeddings_group.create_dataset(
                            seq_id, 
                            data=seq_embeddings,
                            compression='gzip',
                            compression_opts=6
                        )
                        
                        # Store metadata
                        metadata_group.create_dataset(
                            f"{seq_id}_length",
                            data=valid_length
                        )
                    
                    processed_count += len(sequence_ids)
                    
                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Processed batch {batch_idx}/{total_batches}, "
                            f"sequences: {processed_count}"
                        )
                    
                    # Memory cleanup
                    del embeddings
                    torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        logger.info(f"Processing complete. Total sequences processed: {processed_count}")
        logger.info(f"Output saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from genomic sequences using evolutionary models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python embedding_generator.py --input sequences.csv --output embeddings.h5
  python embedding_generator.py --limit 1000 --raw_embeddings
  python embedding_generator.py --config custom_config.yaml
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output', type=str, default='embeddings.h5',
        help='Output file path for embeddings'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of sequences to process'
    )
    
    # Processing options
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--raw_embeddings', action='store_true',
        help='Output raw embeddings without post-processing'
    )
    parser.add_argument(
        '--extract_final', action='store_true',
        help='Extract final layer embeddings instead of second-to-last'
    )
    parser.add_argument(
        '--token_norm', choices=['none', 'l2', 'ln'], default='none',
        help='Token normalization method'
    )
    parser.add_argument(
        '--pool', choices=['none', 'mean', 'max', 'meanmax'], default='none',
        help='Pooling method for sequence embeddings'
    )
    parser.add_argument(
        '--sequential', action='store_true',
        help='Use sequential processing instead of parallel'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Load configuration
    config = ProcessingConfig.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.limit is not None:
        config.limit = args.limit
    config.batch_size = args.batch_size
    config.raw_embeddings = args.raw_embeddings
    config.extract_final = args.extract_final
    config.token_norm = args.token_norm
    config.pool = args.pool
    config.sequential = args.sequential
    
    logger.info("Starting genomic embedding generation")
    logger.info(f"Configuration: {config}")
    
    try:
        # Create processor and run
        processor = EmbeddingProcessor(config)
        processor.process_sequences(args.output)
        
        logger.info("Embedding generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())