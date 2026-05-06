"""
Input validation utilities for Evo-AMR pipeline
==============================================

Provides comprehensive validation for input data, configuration files,
and system requirements.
"""

import re
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import pandas as pd
import numpy as np


class ValidationError(Exception):
    """Custom validation exception with helpful error messages."""
    
    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        """
        Initialize validation error.
        
        Args:
            message: Error description
            suggestions: List of troubleshooting suggestions
        """
        self.message = message
        self.suggestions = suggestions or []
        
        full_message = f"\n❌ Validation Error: {message}"
        if self.suggestions:
            full_message += "\n\n💡 Suggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                full_message += f"\n   {i}. {suggestion}"
        
        super().__init__(full_message)


def validate_input_file(file_path: Union[str, Path], 
                       expected_format: str = 'csv',
                       required_columns: Optional[List[str]] = None) -> Path:
    """
    Validate input file exists and has correct format.
    
    Args:
        file_path: Path to input file
        expected_format: Expected file format ('csv', 'h5', 'fasta')
        required_columns: Required columns for CSV files
        
    Returns:
        Path object to validated file
        
    Raises:
        ValidationError: If file validation fails
        
    Examples:
        >>> validate_input_file("data.csv", "csv", ["sequence_id", "sequence"])
        PosixPath('data.csv')
    """
    file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        raise ValidationError(
            f"Input file not found: {file_path}",
            [
                f"Check if the path is correct: {file_path.absolute()}",
                "Ensure the file hasn't been moved or deleted",
                "Check file permissions"
            ]
        )
    
    # Check file extension
    if expected_format == 'csv' and file_path.suffix.lower() not in ['.csv', '.tsv']:
        raise ValidationError(
            f"Expected CSV file, got: {file_path.suffix}",
            [
                "Ensure file has .csv or .tsv extension",
                "Convert file to CSV format if needed"
            ]
        )
    
    # Validate CSV structure
    if expected_format == 'csv':
        try:
            # Read first few rows to check structure
            df = pd.read_csv(file_path, nrows=5)
            
            # Check required columns
            if required_columns:
                missing = set(required_columns) - set(df.columns)
                if missing:
                    raise ValidationError(
                        f"CSV missing required columns: {missing}",
                        [
                            f"Required columns: {required_columns}",
                            f"Found columns: {list(df.columns)}",
                            "Check column names for typos or case sensitivity",
                            "Ensure the CSV header is present"
                        ]
                    )
            
            # Check if file is empty
            if df.empty:
                raise ValidationError(
                    "CSV file is empty or contains only headers",
                    [
                        "Ensure the file contains data rows",
                        "Check if data was properly written to the file"
                    ]
                )
                
        except pd.errors.ParserError as e:
            raise ValidationError(
                f"Invalid CSV format: {e}",
                [
                    "Check for malformed CSV (unclosed quotes, inconsistent delimiters)",
                    "Try opening the file in a text editor to inspect format",
                    "Consider using different delimiter (tab vs comma)"
                ]
            )
    
    return file_path


def validate_sequence(seq: str, 
                     seq_type: str = 'dna',
                     min_length: int = 1,
                     max_length: Optional[int] = None) -> bool:
    """
    Validate biological sequence.
    
    Args:
        seq: Sequence string
        seq_type: Type of sequence ('dna', 'rna', 'protein')
        min_length: Minimum allowed length
        max_length: Maximum allowed length (None for no limit)
        
    Returns:
        True if sequence is valid
        
    Raises:
        ValidationError: If sequence validation fails
        
    Examples:
        >>> validate_sequence("ATCGATCG", "dna")
        True
        >>> validate_sequence("INVALID", "dna")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValidationError: Invalid DNA sequence...
    """
    if not isinstance(seq, str):
        raise ValidationError(
            f"Sequence must be string, got {type(seq)}",
            ["Convert sequence to string format"]
        )
    
    # Check length
    if len(seq) < min_length:
        raise ValidationError(
            f"Sequence too short: {len(seq)} < {min_length}",
            [f"Ensure sequence has at least {min_length} characters"]
        )
    
    if max_length and len(seq) > max_length:
        raise ValidationError(
            f"Sequence too long: {len(seq)} > {max_length}",
            [f"Trim sequence to max {max_length} characters"]
        )
    
    # Define valid characters
    valid_chars = {
        'dna': set('ATCGN'),
        'rna': set('AUCGN'),
        'protein': set('ACDEFGHIKLMNPQRSTVWY')
    }
    
    if seq_type not in valid_chars:
        raise ValidationError(
            f"Unknown sequence type: {seq_type}",
            [f"Supported types: {list(valid_chars.keys())}"]
        )
    
    # Check for invalid characters
    seq_upper = seq.upper()
    invalid_chars = set(seq_upper) - valid_chars[seq_type]
    
    if invalid_chars:
        raise ValidationError(
            f"Invalid {seq_type.upper()} sequence contains: {invalid_chars}",
            [
                f"Valid {seq_type.upper()} characters: {sorted(valid_chars[seq_type])}",
                "Check for typos or non-standard characters",
                "Consider using 'N' for ambiguous nucleotides"
            ]
        )
    
    return True


def validate_config(config: Dict[str, Any], 
                   required_keys: List[str],
                   path_keys: Optional[List[str]] = None) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: Keys that must be present
        path_keys: Keys that should contain valid paths
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValidationError: If configuration validation fails
    """
    # Check required keys
    missing_keys = set(required_keys) - set(config.keys())
    if missing_keys:
        raise ValidationError(
            f"Configuration missing required keys: {missing_keys}",
            [
                f"Required keys: {required_keys}",
                "Check configuration file format",
                "Ensure all required sections are present"
            ]
        )
    
    # Validate paths
    if path_keys:
        for key in path_keys:
            if key in config:
                path_value = config[key]
                if isinstance(path_value, str):
                    path_obj = Path(path_value)
                    if not path_obj.exists():
                        raise ValidationError(
                            f"Path in config['{key}'] does not exist: {path_value}",
                            [
                                "Check if path is correct",
                                "Create directory if it should exist",
                                "Update configuration with correct path"
                            ]
                        )
    
    return True


def validate_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and memory.
    
    Returns:
        Dictionary with GPU information
        
    Examples:
        >>> gpu_info = validate_gpu_availability()
        >>> 'available' in gpu_info
        True
    """
    try:
        import torch
        
        gpu_info = {
            'available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': []
        }
        
        if gpu_info['available']:
            for i in range(gpu_info['device_count']):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_free': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                }
                gpu_info['devices'].append(device_info)
        
        return gpu_info
        
    except ImportError:
        return {
            'available': False,
            'device_count': 0,
            'devices': [],
            'error': 'PyTorch not available'
        }


def validate_dependencies() -> Dict[str, bool]:
    """
    Check if required dependencies are installed.
    
    Returns:
        Dictionary mapping package names to availability
    """
    required_packages = [
        'torch',
        'transformers', 
        'accelerate',
        'biopython',
        'pandas',
        'numpy',
        'scikit-learn',
        'h5py',
        'yaml'
    ]
    
    availability = {}
    for package in required_packages:
        try:
            __import__(package)
            availability[package] = True
        except ImportError:
            availability[package] = False
    
    return availability