"""
utils/logger.py — Centralised logging configuration.

Usage:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.debug("Extracted entities: %s", entities)
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    from config import settings

    log_level = getattr(logging, settings.logging.level.upper(), logging.DEBUG)
    log_file = settings.logging.log_file

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger