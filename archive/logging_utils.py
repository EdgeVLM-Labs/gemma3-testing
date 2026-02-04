"""Logging utilities for consistent output formatting."""

import logging
import sys
from typing import Optional


def setup_logging(
    verbosity: int = 0,
    log_file: Optional[str] = None,
    name: str = "gemma3n"
) -> logging.Logger:
    """
    Set up logging with consistent formatting.

    Args:
        verbosity: Logging level (0=WARNING, 1=INFO, 2=DEBUG)
        log_file: Optional file path to write logs to
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Map verbosity to logging levels
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    level = level_map.get(verbosity, logging.DEBUG)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
