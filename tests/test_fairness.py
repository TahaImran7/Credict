"""
Phase 6 — fairness audit tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.fairness.audit import (
    AuditResult,
    age_bands,
    audit_groups,
    audit_standard_groups,
    flatten_audit_results,
)
from config import settings


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def _toy_data():
    # Group A: 100 records, 20% prevalence, model good
    # Group B: 100 records, 20% prevalence, model worse
    rng = np.random.default_rng(0)
    y_true = np.array([1] * 20 + [0] * 80 + [1] * 20 + [0] * 80)
    # Group A: perfect (predict y_true)
    pred_a = y_true[:100].copy()
    # Group B: flip 10 positives to negatives and 10 negatives to positives
    pred_b = y_true[100:].copy()
    pred_b[:10] = 0
    pred_b[20:30] = 1
    y_pred = np.concatenate([pred_a, pred_b])
    groups = np.array(["A"] * 100 + ["B"] * 100)
    return y_true, y_pred, groups


def test_audit_groups_metrics_shape_and_columns():
    y_true, y_pred, groups = _toy_data()
    res = audit_groups(y_true, y_pred, groups, group_col="grp")
    assert isinstance(res, AuditResult)
    assert set(res.metrics["group"]) == {"A", "B"}
    for col in ("n", "n_positive", "prevalence", "tpr", "fpr",
                "precision", "f1", "roc_auc"):
        assert col in res.metrics.columns


def test_audit_groups_metric_values_are_correct():
    y_true, y_pred, groups = _toy_data()
    res = audit_groups(y_true, y_pred, groups, group_col="grp")
    a = res.metrics[res.metrics["group"] == "A"].iloc[0]
    b = res.metrics[res.metrics["group"] == "B"].iloc[0]
    # Group A: perfect classifier → TPR=1, FPR=0
    assert a["tpr"] == pytest.approx(1.0)
    assert a["fpr"] == pytest.approx(0.0)
    assert a["precision"] == pytest.approx(1.0)
    # Group B: 10 of 20 positives classified correctly → TPR=0.5
    # 10 of 80 negatives misclassified → FPR=10/80=0.125
    assert b["tpr"] == pytest.approx(0.5)
    assert b["fpr"] == pytest.approx(10 / 80)
    # Prevalence is data-only and equal
    assert a["prevalence"] == pytest.approx(0.2)
    assert b["prevalence"] == pytest.approx(0.2)


def test_audit_groups_differences_from_reference():
    y_true, y_pred, groups = _toy_data()
    res = audit_groups(y_true, y_pred, groups, group_col="grp",
                       reference_group="A")
    assert res.reference_group == "A"
    a_diff = res.differences[res.differences["group"] == "A"].iloc[0]
    b_diff = res.differences[res.differences["group"] == "B"].iloc[0]
    assert a_diff["tpr_minus_ref"] == pytest.approx(0.0)
    assert b_diff["tpr_minus_ref"] == pytest.approx(-0.5)


def test_audit_groups_defaults_reference_to_largest():
    y_true, y_pred, groups = _toy_data()
    # Make A larger
    y_true2 = np.concatenate([y_true, [0, 1, 0]])
    y_pred2 = np.concatenate([y_pred, [0, 0, 1]])
    groups2 = np.concatenate([groups, ["A", "A", "A"]])
    res = audit_groups(y_true2, y_pred2, groups2, group_col="grp")
    assert res.reference_group == "A"


def test_audit_groups_handles_group_with_no_positives():
    y_true = np.array([0, 0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 1, 0])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    res = audit_groups(y_true, y_pred, groups, group_col="grp")
    a = res.metrics[res.metrics["group"] == "A"].iloc[0]
    assert a["n_positive"] == 0
    assert np.isnan(a["tpr"])
    assert np.isnan(a["precision"]) or a["precision"] != a["precision"]


def test_audit_groups_warns_about_small_groups():
    y_true = np.array([0] * 50 + [1] * 5 + [0] * 5)
    y_pred = np.array([0] * 50 + [1] * 5 + [0] * 5)
    groups = np.array(["A"] * 50 + ["B"] * 10)
    res = audit_groups(y_true, y_pred, groups, group_col="grp", min_group_size=30)
    assert "B" in res.small_groups
    assert any("fewer than 30" in w for w in res.warnings)


def test_audit_groups_length_mismatch_raises():
    with pytest.raises(ValueError):
        audit_groups([0, 1], [0], [1, 1], group_col="grp")


def test_audit_groups_reference_not_present_raises():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    groups = ["A", "A", "B", "B"]
    with pytest.raises(ValueError):
        audit_groups(y_true, y_pred, groups, group_col="grp",
                     reference_group="Z")


# ---------------------------------------------------------------------------
# Age bands
# ---------------------------------------------------------------------------

def test_age_bands_creates_expected_labels():
    ages = pd.Series([18, 25, 35, 45, 55, 65, 80])
    bands = age_bands(ages)
    assert bands.tolist() == ["<30", "<30", "30-39", "40-49", "50-59", "60+", "60+"]


# ---------------------------------------------------------------------------
# Multi-dimension audit
# ---------------------------------------------------------------------------

def test_audit_standard_groups_returns_four_results():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(20, 70, n),
        settings.TARGET_COLUMN: rng.integers(0, 2, n),
    })
    y_pred = rng.integers(0, 2, n)
    y_prob = rng.random(n)
    results = audit_standard_groups(df, y_pred, y_prob)
    assert set(results.keys()) == {"SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"}
    for name, res in results.items():
        assert res.group_col == name
        assert len(res.metrics) >= 2


def test_flatten_audit_results_produces_two_frames():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(20, 70, n),
        settings.TARGET_COLUMN: rng.integers(0, 2, n),
    })
    y_pred = rng.integers(0, 2, n)
    y_prob = rng.random(n)
    results = audit_standard_groups(df, y_pred, y_prob)
    metrics_df, diff_df = flatten_audit_results(results)
    assert {"audit", "group", "reference_group", "n"} <= set(metrics_df.columns)
    assert {"audit", "group", "reference_group"} <= set(diff_df.columns)
    assert len(metrics_df) == len(diff_df)