"""
Central path resolution for CreditLens.

Every path used by the project is derived from PROJECT_ROOT, which is
computed from this file's location. This guarantees portability: the project
can be cloned to any directory on any OS without editing code.
"""

from pathlib import Path

# src/utils/paths.py -> parents[0]=utils, [1]=src, [2]=CreditLens
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Top-level directories
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
SAVED_MODELS_DIR: Path = MODELS_DIR / "saved"
EVALUATION_DIR: Path = MODELS_DIR / "evaluation"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
UI_DIR: Path = PROJECT_ROOT / "ui"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Convenient combined path for the dataset file
DEFAULT_DATASET_PATH: Path = RAW_DATA_DIR / "default of credit card clients.xls"


def ensure_directories() -> None:
    """Create runtime directories if they do not exist. Safe to call repeatedly."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAVED_MODELS_DIR,
        EVALUATION_DIR,
        DOCS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)