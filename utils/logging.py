"""
Logging utilities for Evo-AMR pipeline
=====================================

Centralized logging configuration for consistent log formatting across modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(name: str,
                log_file: Optional[Union[str, Path]] = None,
                level: int = logging.INFO,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    
    Args:
        name: Logger name (typically __name__)
        log_file: Optional log file path
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
        
    Examples:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger = setup_logger(__name__, "pipeline.log", logging.DEBUG)
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    return logger


def get_progress_logger(name: str) -> logging.Logger:
    """
    Get logger optimized for progress reporting.
    
    Args:
        name: Logger name
        
    Returns:
        Logger with progress-friendly formatting
    """
    return setup_logger(
        name,
        format_string='%(asctime)s - %(message)s'
    )


class ProgressLogger:
    """Context manager for progress logging with automatic cleanup."""
    
    def __init__(self, name: str, total_items: int, log_interval: int = 100):
        """
        Initialize progress logger.
        
        Args:
            name: Process name
            total_items: Total number of items to process
            log_interval: Log every N items
        """
        self.name = name
        self.total_items = total_items
        self.log_interval = log_interval
        self.logger = get_progress_logger(name)
        self.processed = 0
    
    def __enter__(self):
        self.logger.info(f"Starting {self.name}: {self.total_items} items")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed {self.name}: {self.processed}/{self.total_items} items")
        else:
            self.logger.error(f"Failed {self.name}: {exc_val}")
    
    def update(self, increment: int = 1):
        """Update progress counter."""
        self.processed += increment
        
        if self.processed % self.log_interval == 0 or self.processed == self.total_items:
            percentage = (self.processed / self.total_items) * 100
            self.logger.info(f"Progress: {self.processed}/{self.total_items} ({percentage:.1f}%)")
    
    def log_milestone(self, message: str):
        """Log a milestone message."""
        self.logger.info(f"Milestone: {message}")


# Pre-configured loggers for common use cases
def get_pipeline_logger(module_name: str) -> logging.Logger:
    """Get logger for pipeline modules."""
    return setup_logger(
        f"evo_amr.{module_name}",
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_debug_logger(module_name: str, debug_file: str = "debug.log") -> logging.Logger:
    """Get debug logger with file output."""
    return setup_logger(
        f"evo_amr.{module_name}",
        log_file=debug_file,
        level=logging.DEBUG,
        format_string='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )