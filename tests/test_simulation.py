"""
Phase 5 — what-if simulator tests.
"""

import pandas as pd
import pytest

from config import settings
from src.simulation.what_if import (
    PERMITTED_FEATURES,
    DISCLAIMER,
    ScenarioChange,
    WhatIfResult,
    simulate,
    simulate_grid,
    validate_changes,
)
from src.utils.paths import SAVED_MODELS_DIR


def _real_models_present() -> bool:
    return (SAVED_MODELS_DIR / "random_forest.joblib").exists()


requires_models = pytest.mark.skipif(
    not _real_models_present(),
    reason="Real models not trained yet. Run: python -m scripts.train_models",
)


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_changes_accepts_permitted_feature():
    out = validate_changes({"LIMIT_BAL": 75000})
    assert out == {"LIMIT_BAL": 75000.0}


def test_validate_changes_rejects_unknown_feature():
    with pytest.raises(ValueError, match="not permitted"):
        validate_changes({"NOT_A_FEATURE": 1})


def test_validate_changes_rejects_out_of_range():
    with pytest.raises(ValueError):
        validate_changes({"AGE": 5})
    with pytest.raises(ValueError):
        validate_changes({"PAY_0": 99})


def test_validate_changes_rejects_bad_category():
    with pytest.raises(ValueError):
        validate_changes({"SEX": 3})
    with pytest.raises(ValueError):
        validate_changes({"EDUCATION": 9})


def test_validate_changes_rejects_empty_dict():
    with pytest.raises(ValueError):
        validate_changes({})


def test_permitted_features_include_all_23():
    assert len(PERMITTED_FEATURES) == 23
    assert set(PERMITTED_FEATURES) == set(settings.FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@requires_models
def test_simulate_no_effective_change_returns_same_probability():
    """Passing a value equal to the current value should produce zero diff."""
    rec = _baseline_record()
    result, _ = simulate(rec, {"LIMIT_BAL": rec["LIMIT_BAL"]})
    assert result.difference_pp == pytest.approx(0.0, abs=1e-9)
    assert result.changes == []


@requires_models
def test_simulate_does_not_mutate_input():
    rec = _baseline_record()
    before = dict(rec)
    simulate(rec, {"LIMIT_BAL": 100000})
    assert rec == before


@requires_models
def test_simulate_returns_valid_result():
    rec = _baseline_record()
    result, warnings = simulate(rec, {"LIMIT_BAL": 200000})
    assert isinstance(result, WhatIfResult)
    assert 0.0 <= result.original_probability <= 1.0
    assert 0.0 <= result.modified_probability <= 1.0
    assert result.original_band in settings.RISK_BANDS
    assert result.modified_band in settings.RISK_BANDS
    assert len(result.changes) == 1
    assert isinstance(result.changes[0], ScenarioChange)
    assert result.changes[0].feature == "LIMIT_BAL"
    assert result.disclaimer == DISCLAIMER
    assert isinstance(warnings, list)


@requires_models
def test_simulate_multiple_changes_records_all():
    rec = _baseline_record()
    changes = {"LIMIT_BAL": 200000, "PAY_0": 2, "PAY_AMT1": 0}
    result, _ = simulate(rec, changes)
    feature_set = {c.feature for c in result.changes}
    assert feature_set == set(changes.keys())


@requires_models
def test_simulate_difference_pp_sign_matches_direction():
    rec = _baseline_record()
    result, _ = simulate(rec, {"PAY_0": 3})
    # Sign of difference_pp must match direction property
    if result.difference_pp > 0.5:
        assert result.direction == "increased"
    elif result.difference_pp < -0.5:
        assert result.direction == "decreased"
    else:
        assert result.direction == "negligible"


@requires_models
def test_simulate_grid_shape_and_monotone_values():
    rec = _baseline_record()
    df = simulate_grid(rec, "LIMIT_BAL", [10000, 50000, 200000])
    assert list(df.columns) == ["value", "probability", "risk_band"]
    assert len(df) == 3
    assert df["value"].tolist() == [10000.0, 50000.0, 200000.0]
    assert df["probability"].between(0, 1).all()
    assert df["risk_band"].isin(settings.RISK_BANDS).all()


@requires_models
def test_simulate_grid_rejects_unpermitted_feature():
    rec = _baseline_record()
    with pytest.raises(ValueError):
        simulate_grid(rec, "NOT_A_FEATURE", [1, 2, 3])