"""
Phase 5 — portfolio analytics tests.
"""

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.analytics.portfolio import (
    RISK_BAND_ORDER,
    default_rate_by_category,
    default_rate_by_pay_status,
    mean_features_by_outcome,
    model_risk_bands_for,
    overview_snapshot,
    portfolio_summary,
    probability_distribution_stats,
    risk_band_counts,
    risk_band_default_rates,
)
from src.data import preprocessing as pp
from src.utils.paths import RAW_DATA_DIR


def _dataset_available() -> bool:
    return (RAW_DATA_DIR / settings.DATASET_FILENAME).exists()


def _loaded_df() -> pd.DataFrame:
    from src.data.loader import load_raw_dataset
    return pp.clean_dataset(load_raw_dataset())


# ---------------------------------------------------------------------------
# portfolio_summary
# ---------------------------------------------------------------------------

def test_portfolio_summary_basic():
    df = pd.DataFrame({settings.TARGET_COLUMN: [0, 1, 0, 0, 1]})
    s = portfolio_summary(df)
    assert s["n_customers"] == 5
    assert s["n_defaulters"] == 2
    assert s["n_non_defaulters"] == 3
    assert s["default_rate"] == pytest.approx(0.4)


def test_portfolio_summary_requires_target():
    df = pd.DataFrame({"X": [1, 2]})
    with pytest.raises(KeyError):
        portfolio_summary(df)


# ---------------------------------------------------------------------------
# probability stats / bands
# ---------------------------------------------------------------------------

def test_probability_distribution_stats_keys():
    s = probability_distribution_stats([0.1, 0.2, 0.3, 0.4, 0.5])
    for k in ("n", "mean", "median", "std", "min", "max", "p25", "p75", "p90"):
        assert k in s
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(0.3)


def test_risk_band_counts_order_and_shares():
    probs = [0.05, 0.15, 0.30, 0.60, 0.85, 0.90]
    df = risk_band_counts(probs)
    assert df["risk_band"].tolist() == list(RISK_BAND_ORDER)
    assert df["count"].sum() == 6
    assert df["share"].sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# default rates by category
# ---------------------------------------------------------------------------

def test_default_rate_by_category_reports_low_confidence():
    df = pd.DataFrame({
        "EDUCATION": [1] * 50 + [2] * 5,
        settings.TARGET_COLUMN: [0] * 40 + [1] * 10 + [0] * 5,
    })
    out = default_rate_by_category(df, "EDUCATION", min_count=30)
    assert set(out["category"]) == {1, 2}
    cat2 = out[out["category"] == 2].iloc[0]
    assert bool(cat2["low_confidence"]) is True
    cat1 = out[out["category"] == 1].iloc[0]
    assert bool(cat1["low_confidence"]) is False


def test_default_rate_by_category_requires_column():
    df = pd.DataFrame({settings.TARGET_COLUMN: [0, 1]})
    with pytest.raises(KeyError):
        default_rate_by_category(df, "EDUCATION")


@pytest.mark.skipif(not _dataset_available(), reason="Dataset not present.")
def test_default_rate_by_pay_status_long_form():
    df = _loaded_df()
    out = default_rate_by_pay_status(df)
    assert set(out["pay_col"]) == {"PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}
    assert out["n"].sum() == 30000 * 6


# ---------------------------------------------------------------------------
# feature vs. outcome
# ---------------------------------------------------------------------------

def test_mean_features_by_outcome_ratio():
    df = pd.DataFrame({
        "LIMIT_BAL": [100000, 200000, 300000, 400000],
        "AGE": [30, 40, 50, 60],
        settings.TARGET_COLUMN: [0, 0, 1, 1],
    })
    out = mean_features_by_outcome(df, features=("LIMIT_BAL", "AGE"))
    lim = out[out["feature"] == "LIMIT_BAL"].iloc[0]
    assert lim["mean_non_default"] == 150000.0
    assert lim["mean_default"] == 350000.0
    assert lim["ratio_default_to_non"] == pytest.approx(350000 / 150000)


# ---------------------------------------------------------------------------
# risk-band observed default rate
# ---------------------------------------------------------------------------

def test_risk_band_default_rates():
    probs = [0.05, 0.10, 0.30, 0.40, 0.60, 0.90]
    y =     [0,    0,    1,    0,    1,    1]
    out = risk_band_default_rates(probs, y)
    assert out["risk_band"].tolist() == list(RISK_BAND_ORDER)
    # Low band: 2 records, 0 defaults
    low = out[out["risk_band"] == "Low"].iloc[0]
    assert low["n"] == 2
    assert low["n_defaults"] == 0
    # Very High band: 1 record, 1 default
    vh = out[out["risk_band"] == "Very High"].iloc[0]
    assert vh["n"] == 1
    assert vh["n_defaults"] == 1


def test_risk_band_default_rates_shape_mismatch():
    with pytest.raises(ValueError):
        risk_band_default_rates([0.1, 0.2], [0])


# ---------------------------------------------------------------------------
# overview snapshot
# ---------------------------------------------------------------------------

def test_overview_snapshot_without_probabilities():
    df = pd.DataFrame({settings.TARGET_COLUMN: [0, 1, 0]})
    snap = overview_snapshot(df)
    assert "portfolio" in snap
    assert "probability_stats" not in snap


def test_overview_snapshot_with_probabilities():
    df = pd.DataFrame({settings.TARGET_COLUMN: [0, 1, 0, 1]})
    probs = [0.1, 0.6, 0.2, 0.8]
    snap = overview_snapshot(df, probs)
    assert "portfolio" in snap
    assert "probability_stats" in snap
    assert "risk_bands" in snap
    assert isinstance(snap["risk_bands"], list)


# ---------------------------------------------------------------------------
# Real-data integration (skipped without artefacts)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _dataset_available(), reason="Dataset not present.")
def test_portfolio_summary_real_data():
    df = _loaded_df()
    s = portfolio_summary(df)
    assert s["n_customers"] == 30000
    assert 0.20 < s["default_rate"] < 0.24   # dataset is ~22.12%