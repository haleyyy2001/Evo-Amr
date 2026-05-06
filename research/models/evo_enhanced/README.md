# Evo-1-8K ULTRA-OPTIMIZED Processor

This is a refactored version of the Evo embedding processor that uses configurable paths instead of hardcoded absolute paths, making it portable and shareable.

## Setup

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -U numpy h5py joblib Bio torch transformers pyyaml
   ```

2. **Configure paths**: Edit `config.yaml` to set your specific paths:
   ```yaml
   # Data paths - customize these for your setup
   CENTRAL_CONTIGS: "/path/to/your/sequences/raw/nucleotide"  # Directory containing genome FASTA files
   EMBEDDINGS_DIR: "/path/to/your/embeddings_output"  # Where to save H5 embedding files
   CSV_FILE: "/path/to/your/genome_list.csv"  # CSV file with genome IDs
   ```

3. **Verify model paths**: The model paths are set to be relative to the script location:
   - `../evo/evo_1_131k_base/config.json` 
   - `../evo/evo-1-8k-base`
   - `.` (current directory for custom model code)

## Usage

### Basic usage with config file:
```bash
python AAA_evo_ultra_fast_fixed.py --limit 10
```

### Override CSV file:
```bash
python AAA_evo_ultra_fast_fixed.py --csv_file /path/to/different/genomes.csv --limit 10
```

### Use a different config file:
```bash
python AAA_evo_ultra_fast_fixed.py --config /path/to/custom/config.yaml --limit 10
```

### Multi-GPU processing:
```bash
python AAA_evo_ultra_fast_fixed.py --gpu_ids "0,1,2,3" --limit 100
```

## Configuration File

The `config.yaml` file contains all configurable parameters:

- **Data paths**: Set these to match your data location
- **Model paths**: Relative paths to the Evo model files
- **Processing parameters**: Batch sizes, chunk sizes, etc.
- **Memory settings**: GPU memory management
- **Optimization flags**: CUDA graphs, mixed precision, etc.

## Key Changes

1. **Portable paths**: No more hardcoded absolute paths
2. **Configuration file**: All settings in one YAML file
3. **Relative model paths**: Model files are found relative to script location
4. **Command line overrides**: Can override config file settings via arguments

## Directory Structure

```
evo_8/
├── AAA_evo_ultra_fast_fixed.py  # Main processor script
├── config.yaml                   # Configuration file  
├── rotary_wrapper.py             # Rotary position encoding wrapper
├── evo_1_131k_base/              # Model code directory
│   ├── modified_model.py
│   ├── tokenizer.py
│   └── config.json
└── ../evo/                       # Model weights directory
    ├── evo_1_131k_base/
    └── evo-1-8k-base/
```

## Troubleshooting

1. **Config file not found**: If you get a warning about config.yaml not found, create it using the template provided.

2. **Model files not found**: Check that the relative paths in config.yaml point to the correct model directories.

3. **Import errors**: Make sure all dependencies are installed and the model code directory structure is correct.

4. **CUDA errors**: Adjust memory settings in config.yaml based on your GPU capabilities.