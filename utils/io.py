"""
I/O utilities for Evo-AMR pipeline
=================================

Safe file I/O operations with error handling and validation.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Union, Dict, Any, List, Optional
import pandas as pd
import h5py
import numpy as np
from .validation import ValidationError


def ensure_directory(directory: Union[str, Path], 
                    create_parents: bool = True) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        directory: Directory path
        create_parents: Whether to create parent directories
        
    Returns:
        Path object to directory
        
    Raises:
        ValidationError: If directory cannot be created
    """
    dir_path = Path(directory)
    
    try:
        dir_path.mkdir(parents=create_parents, exist_ok=True)
        return dir_path
    except PermissionError:
        raise ValidationError(
            f"Permission denied creating directory: {dir_path}",
            [
                "Check write permissions for parent directory",
                "Run with appropriate user permissions",
                "Choose a different output location"
            ]
        )
    except OSError as e:
        raise ValidationError(
            f"Failed to create directory {dir_path}: {e}",
            [
                "Check if path is valid",
                "Ensure sufficient disk space",
                "Check for filesystem limitations"
            ]
        )


def safe_load_csv(file_path: Union[str, Path], 
                 required_columns: Optional[List[str]] = None,
                 **kwargs) -> pd.DataFrame:
    """
    Safely load CSV file with validation.
    
    Args:
        file_path: Path to CSV file
        required_columns: Required column names
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValidationError: If file loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(
            f"CSV file not found: {file_path}",
            ["Check file path", "Ensure file exists"]
        )
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValidationError(
                    f"CSV missing required columns: {missing}",
                    [
                        f"Required: {required_columns}",
                        f"Found: {list(df.columns)}"
                    ]
                )
        
        return df
        
    except Exception as e:
        raise ValidationError(
            f"Failed to load CSV {file_path}: {e}",
            [
                "Check file format",
                "Ensure proper encoding (UTF-8)",
                "Check for corrupted file"
            ]
        )


def safe_load_h5(file_path: Union[str, Path], 
                 dataset_name: Optional[str] = None) -> Union[h5py.File, np.ndarray]:
    """
    Safely load HDF5 file.
    
    Args:
        file_path: Path to HDF5 file
        dataset_name: Specific dataset to load (None for file handle)
        
    Returns:
        HDF5 file handle or numpy array
        
    Raises:
        ValidationError: If file loading fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(
            f"HDF5 file not found: {file_path}",
            ["Check file path", "Ensure file exists"]
        )
    
    try:
        h5_file = h5py.File(file_path, 'r')
        
        if dataset_name:
            if dataset_name not in h5_file:
                h5_file.close()
                raise ValidationError(
                    f"Dataset '{dataset_name}' not found in {file_path}",
                    [
                        f"Available datasets: {list(h5_file.keys())}",
                        "Check dataset name spelling"
                    ]
                )
            return h5_file[dataset_name][:]
        
        return h5_file
        
    except Exception as e:
        raise ValidationError(
            f"Failed to load HDF5 {file_path}: {e}",
            [
                "Check file format",
                "Ensure file is not corrupted",
                "Check HDF5 library version compatibility"
            ]
        )


def atomic_write(content: Any, 
                file_path: Union[str, Path],
                write_func: callable) -> None:
    """
    Atomic file writing to prevent corruption.
    
    Args:
        content: Content to write
        file_path: Target file path
        write_func: Function to write content (should accept content and file path)
        
    Raises:
        ValidationError: If writing fails
    """
    file_path = Path(file_path)
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    
    try:
        # Write to temporary file
        write_func(content, temp_path)
        
        # Atomic move
        shutil.move(str(temp_path), str(file_path))
        
    except Exception as e:
        # Clean up temp file if it exists
        if temp_path.exists():
            temp_path.unlink()
        
        raise ValidationError(
            f"Failed to write file {file_path}: {e}",
            [
                "Check write permissions",
                "Ensure sufficient disk space",
                "Check if file is not locked by another process"
            ]
        )


def backup_file(file_path: Union[str, Path], 
               backup_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Create backup of existing file.
    
    Args:
        file_path: File to backup
        backup_dir: Directory for backup (default: same directory)
        
    Returns:
        Path to backup file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValidationError(
            f"Cannot backup non-existent file: {file_path}",
            ["Check if file exists before backup"]
        )
    
    if backup_dir:
        backup_dir = Path(backup_dir)
        ensure_directory(backup_dir)
        backup_path = backup_dir / f"{file_path.stem}.backup{file_path.suffix}"
    else:
        backup_path = file_path.with_suffix(f".backup{file_path.suffix}")
    
    try:
        shutil.copy2(str(file_path), str(backup_path))
        return backup_path
    except Exception as e:
        raise ValidationError(
            f"Failed to create backup: {e}",
            [
                "Check write permissions",
                "Ensure sufficient disk space"
            ]
        )