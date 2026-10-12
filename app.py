"""
CreditLens — Phase 1 verification dashboard.

Purpose (Phase 1 only):
    - Confirm the environment installs and imports cleanly.
    - Confirm the project folder structure exists and is writable.
    - Confirm path, settings, and logging modules behave correctly.

The real 8-section dashboard is built in Phase 7.
"""

import os
import sys
import platform
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
import streamlit as st

from src.utils.paths import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    EVALUATION_DIR,
    ensure_directories,
)
from src.utils.logging_config import get_logger
from config import settings

log = get_logger(__name__)


def _writable(path: Path) -> bool:
    """Return True if the directory exists and is writable by this process."""
    return path.is_dir() and os.access(path, os.W_OK)


st.set_page_config(
    page_title="CreditLens — Phase 1 Verification",
    page_icon="🔎",
    layout="wide",
)

st.title("CreditLens — Phase 1 Verification")
st.caption(
    "Environment and skeleton check. The full credit-risk dashboard is "
    "implemented in later phases."
)

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------
st.header("1. Environment")

env_table = pd.DataFrame(
    {
        "Item": [
            "Python version",
            "Operating system",
            "Project root",
            "pandas",
            "numpy",
            "scikit-learn",
            "streamlit",
        ],
        "Value": [
            sys.version.split()[0],
            f"{platform.system()} {platform.release()}",
            str(PROJECT_ROOT),
            pd.__version__,
            np.__version__,
            sklearn.__version__,
            st.__version__,
        ],
    }
)
st.dataframe(env_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 2. Folder structure
# ---------------------------------------------------------------------------
st.header("2. Folder structure")

try:
    ensure_directories()
    dirs = {
        "data/raw": RAW_DATA_DIR,
        "data/processed": PROCESSED_DATA_DIR,
        "models/saved": SAVED_MODELS_DIR,
        "models/evaluation": EVALUATION_DIR,
    }
    struct_rows = [
        {
            "Folder": label,
            "Exists": path.is_dir(),
            "Writable": _writable(path),
        }
        for label, path in dirs.items()
    ]
    st.dataframe(pd.DataFrame(struct_rows), use_container_width=True, hide_index=True)
except Exception as exc:
    st.error(f"Folder check failed: {exc}")
    log.exception("Folder verification failed")

# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------
st.header("3. Configuration")
st.json(
    {
        "RANDOM_SEED": settings.RANDOM_SEED,
        "TARGET_COLUMN_CANDIDATES": list(settings.TARGET_COLUMN_CANDIDATES),
        "POSITIVE_CLASS_LABEL": settings.POSITIVE_CLASS_LABEL,
        "NEGATIVE_CLASS_LABEL": settings.NEGATIVE_CLASS_LABEL,
        "TEST_SIZE": settings.TEST_SIZE,
        "RISK_BANDS": settings.RISK_BANDS,
    }
)

st.success(
    "Phase 1 verification passed. Awaiting Phase 2 (dataset integration)."
)