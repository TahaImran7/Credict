"""
Phase 4 — prediction interface tests.
"""

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.prediction.predictor import (
    DEFAULT_MODEL_NAME,
    assign_risk_band,
    predict_batch,
    predict_batch_with_bands,
    predict_single,
    _prepare_record,
)
from src.utils.paths import SAVED_MODELS_DIR


def _real_models_present() -> bool:
    return (SAVED_MODELS_DIR / "random_forest.joblib").exists()


requires_models = pytest.mark.skipif(
    not _real_models_present(),
    reason="Real models not trained yet. Run: python -m scripts.train_models",
)


# ---------------------------------------------------------------------------
# Risk-band assignment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,expected", [
    (0.00, "Low"),
    (0.05, "Low"),
    (0.199, "Low"),
    (0.20, "Moderate"),
    (0.35, "Moderate"),
    (0.499, "Moderate"),
    (0.50, "High"),
    (0.75, "High"),
    (0.799, "High"),
    (0.80, "Very High"),
    (0.95, "Very High"),
    (1.00, "Very High"),
])
def test_assign_risk_band(p, expected):
    assert assign_risk_band(p) == expected


def test_assign_risk_band_rejects_out_of_range():
    with pytest.raises(ValueError):
        assign_risk_band(-0.01)
    with pytest.raises(ValueError):
        assign_risk_band(1.01)


# ---------------------------------------------------------------------------
# Record preparation
# ---------------------------------------------------------------------------

def _baseline_record() -> dict:
    return {
        "LIMIT_BAL": 50000,
        "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 1000, "BILL_AMT2": 1000, "BILL_AMT3": 1000,
        "BILL_AMT4": 1000, "BILL_AMT5": 1000, "BILL_AMT6": 1000,
        "PAY_AMT1": 500, "PAY_AMT2": 500, "PAY_AMT3": 500,
        "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500,
    }


def test_prepare_record_from_dict():
    df = _prepare_record(_baseline_record())
    assert df.shape == (1, 23)
    assert list(df.columns) == list(settings.FEATURE_COLUMNS)


def test_prepare_record_from_series():
    s = pd.Series(_baseline_record())
    df = _prepare_record(s)
    assert df.shape == (1, 23)


def test_prepare_record_rejects_multi_row_dataframe():
    df = pd.DataFrame([_baseline_record(), _baseline_record()])
    with pytest.raises(ValueError):
        _prepare_record(df)


def test_prepare_record_rejects_missing_column():
    bad = _baseline_record()
    bad.pop("LIMIT_BAL")
    with pytest.raises(KeyError):
        _prepare_record(bad)


# ---------------------------------------------------------------------------
# Prediction against the real model
# ---------------------------------------------------------------------------

@requires_models
def test_predict_single_probability_in_range():
    result = predict_single(_baseline_record())
    assert 0.0 <= result.probability <= 1.0
    assert result.model_name == DEFAULT_MODEL_NAME
    assert result.risk_band in settings.RISK_BANDS
    assert result.predicted_class in (0, 1)
    assert result.timestamp_utc  # non-empty ISO string


@requires_models
def test_predict_single_deterministic():
    a = predict_single(_baseline_record())
    b = predict_single(_baseline_record())
    assert abs(a.probability - b.probability) < 1e-9


@requires_models
def test_predict_batch_matches_single():
    rec = _baseline_record()
    batch = predict_batch(pd.DataFrame([rec, rec]))
    single = predict_single(rec).probability
    assert len(batch) == 2
    assert np.allclose(batch, single)


@requires_models
def test_predict_batch_with_bands_columns():
    rec = _baseline_record()
    out = predict_batch_with_bands(pd.DataFrame([rec, rec]))
    assert list(out.columns) == ["probability", "risk_band"]
    assert out["probability"].between(0, 1).all()