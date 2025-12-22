"""
Logging configuration for EarthQuake Building Sim.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (optional)
        console: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("earthquake_sim")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional). If None, returns root earthquake_sim logger.
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"earthquake_sim.{name}")
    return logging.getLogger("earthquake_sim")


# Default logger setup
_default_logger = None


def init_default_logger():
    """Initialize the default logger if not already done."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


# Quick access logger
logger = get_logger()
