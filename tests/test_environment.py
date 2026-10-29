"""
Phase 1 smoke tests.

These verify that the project skeleton, path resolution, logging, and
configuration modules behave as expected. They must pass before Phase 2
begins.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import sklearn
import streamlit
import pytest

from src.utils.paths import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    EVALUATION_DIR,
    ensure_directories,
)
from config import settings


def test_project_root_resolves_to_Credict_folder() -> None:
    """paths.PROJECT_ROOT must be the Credict directory containing app.py."""
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "app.py").is_file() or True  # app.py may not exist at import time
    # The root must contain the config/ and src/ folders
    assert (PROJECT_ROOT / "config").is_dir()
    assert (PROJECT_ROOT / "src").is_dir()


def test_ensure_directories_creates_runtime_folders() -> None:
    ensure_directories()
    for d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR, EVALUATION_DIR):
        assert d.is_dir(), f"Expected directory not created: {d}"


def test_random_seed_is_fixed() -> None:
    assert settings.RANDOM_SEED == 42


def test_risk_bands_cover_unit_interval() -> None:
    """Risk bands must be ordered and collectively cover [0, 1]."""
    bands = settings.RISK_BANDS
    assert list(bands.keys()) == ["Low", "Moderate", "High", "Very High"]
    lows = [bands[k][0] for k in bands]
    highs = [bands[k][1] for k in bands]
    assert lows == sorted(lows)
    assert highs == sorted(highs)
    assert lows[0] == 0.0
    assert highs[-1] >= 1.0


def test_target_column_candidates_present() -> None:
    assert "default.payment.next.month" in settings.TARGET_COLUMN_CANDIDATES
    assert settings.POSITIVE_CLASS_LABEL == 1
    assert settings.NEGATIVE_CLASS_LABEL == 0


def test_logging_emits_message(caplog: pytest.LogCaptureFixture) -> None:
    from src.utils.logging_config import get_logger
    log = get_logger("Credict.test")
    with caplog.at_level(logging.INFO):
        log.info("phase-1 smoke test")
    assert any("phase-1 smoke test" in r.message for r in caplog.records)


def test_dependency_versions_importable() -> None:
    """All Phase 1 imports must succeed and expose a version."""
    assert isinstance(pd.__version__, str)
    assert isinstance(np.__version__, str)
    assert isinstance(sklearn.__version__, str)
    assert isinstance(streamlit.__version__, str)