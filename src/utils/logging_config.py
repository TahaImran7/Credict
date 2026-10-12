"""
Project-wide logging configuration.

Usage:
    from src.utils.logging_config import get_logger
    log = get_logger(__name__)
    log.info("...")
"""

import logging
import sys

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once. Subsequent calls are ignored."""
    global _configured
    if _configured:
        return

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATEFMT,
        stream=sys.stdout,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, ensuring root configuration is applied."""
    setup_logging()
    return logging.getLogger(name)