"""
Phase 4 — SHAP explainability tests.
"""

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.explainability.shap_explainer import (
    aggregate_by_original_feature,
    clear_explainer_cache,
    explain_single,
    global_importance,
    original_feature_from_expanded,
)
from src.prediction.predictor import DEFAULT_MODEL_NAME
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
# Name aggregation (pure functions)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expanded,expected", [
    ("num__LIMIT_BAL", "LIMIT_BAL"),
    ("num__AGE", "AGE"),
    ("cat__SEX_1", "SEX"),
    ("cat__EDUCATION_2", "EDUCATION"),
    ("cat__MARRIAGE_3", "MARRIAGE"),
    ("cat__PAY_0_2", "PAY_0"),
    ("cat__PAY_6_-1", "PAY_6"),
])
def test_original_feature_from_expanded(expanded, expected):
    assert original_feature_from_expanded(expanded) == expected


def test_aggregate_sums_one_hot_columns_back_to_original():
    names = ["num__LIMIT_BAL", "cat__SEX_1", "cat__SEX_2", "cat__PAY_0_0", "cat__PAY_0_1"]
    values = np.array([0.3, 0.1, 0.2, -0.4, -0.1])
    orig, agg = aggregate_by_original_feature(names, values)
    assert orig == ["LIMIT_BAL", "SEX", "PAY_0"]
    assert agg[0] == pytest.approx(0.3)
    assert agg[1] == pytest.approx(0.3)
    assert agg[2] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Individual explanation
# ---------------------------------------------------------------------------

@requires_models
def test_explain_single_shape_and_probability_consistency():
    clear_explainer_cache()
    exp = explain_single(_baseline_record(), model_name=DEFAULT_MODEL_NAME)
    assert len(exp.original_features) > 0
    assert len(exp.original_features) == len(exp.original_shap_values)
    assert len(exp.original_features) == len(exp.original_input_values)
    assert len(exp.expanded_features) == len(exp.expanded_shap_values)
    assert 0.0 <= exp.probability <= 1.0


@requires_models
def test_explain_single_probability_matches_predictor():
    from src.prediction.predictor import predict_single
    clear_explainer_cache()
    exp = explain_single(_baseline_record())
    p = predict_single(_baseline_record()).probability
    assert abs(exp.probability - p) < 1e-9


@requires_models
def test_explain_single_top_contributors_sorted_by_abs():
    clear_explainer_cache()
    exp = explain_single(_baseline_record())
    top = exp.top_contributors(n=5)
    assert len(top) == 5
    abs_vals = [abs(t["shap_value"]) for t in top]
    assert abs_vals == sorted(abs_vals, reverse=True)
    for t in top:
        assert t["direction"] in ("increases risk", "decreases risk")


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------

@requires_models
def test_global_importance_shape_and_order():
    from src.data import preprocessing as pp
    clear_explainer_cache()
    _, X_test, _, _ = pp.load_splits()
    df = global_importance(X_test, max_samples=200)
    assert list(df.columns) == ["feature", "mean_abs_shap"]
    assert len(df) == 23
    assert df["mean_abs_shap"].between(0, np.inf).all()
    # Sorted descending
    vals = df["mean_abs_shap"].to_list()
    assert vals == sorted(vals, reverse=True)
    # Every original feature appears exactly once
    assert set(df["feature"]) == set(settings.FEATURE_COLUMNS)