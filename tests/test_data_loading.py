"""
Phase 2 — dataset loading and validation tests.

Tests that require the raw dataset are skipped with a clear message if the
file is not present, so the suite does not fail on a clean checkout.
"""

import pandas as pd
import pytest

from config import settings
from src.data.loader import load_raw_dataset, _canonicalize_target
from src.data.validation import validate_dataset
from src.utils.paths import RAW_DATA_DIR


def _dataset_available() -> bool:
    return (RAW_DATA_DIR / settings.DATASET_FILENAME).exists()


requires_dataset = pytest.mark.skipif(
    not _dataset_available(),
    reason="Raw dataset not present. Run download_dataset() first.",
)


def test_canonicalize_target_handles_spaces_variant():
    df = pd.DataFrame({"default payment next month": [0, 1]})
    out = _canonicalize_target(df)
    assert settings.TARGET_COLUMN in out.columns


def test_canonicalize_target_handles_dots_variant():
    df = pd.DataFrame({"default.payment.next.month": [0, 1]})
    out = _canonicalize_target(df)
    assert settings.TARGET_COLUMN in out.columns


def test_canonicalize_target_raises_when_missing():
    df = pd.DataFrame({"some_other_column": [0, 1]})
    with pytest.raises(KeyError):
        _canonicalize_target(df)


@requires_dataset
def test_raw_dataset_shape():
    df = load_raw_dataset()
    assert df.shape[0] == 30000
    assert df.shape[1] == 25


@requires_dataset
def test_raw_dataset_target_values():
    df = load_raw_dataset()
    assert settings.TARGET_COLUMN in df.columns
    assert set(df[settings.TARGET_COLUMN].unique()) <= {0, 1}


@requires_dataset
def test_raw_dataset_has_all_features():
    df = load_raw_dataset()
    for col in settings.FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature: {col}"


@requires_dataset
def test_validation_report_ok():
    df = load_raw_dataset()
    report = validate_dataset(df)
    assert report.ok, report.summary()
    assert report.target_present
    assert report.n_rows == 30000