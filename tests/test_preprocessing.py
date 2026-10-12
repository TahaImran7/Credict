"""
Phase 2 — cleaning, splitting, and preprocessing tests.
"""

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.data import preprocessing as pp
from src.utils.paths import RAW_DATA_DIR


def _dataset_available() -> bool:
    return (RAW_DATA_DIR / settings.DATASET_FILENAME).exists()


requires_dataset = pytest.mark.skipif(
    not _dataset_available(),
    reason="Raw dataset not present. Run download_dataset() first.",
)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_clean_merges_education_and_marriage_codes():
    df = pd.DataFrame({
        "EDUCATION": [0, 1, 2, 3, 4, 5, 6],
        "MARRIAGE":  [0, 1, 2, 3, 0, 1, 2],
    })
    out = pp.clean_dataset(df)
    assert set(out["EDUCATION"].unique()) == {1, 2, 3, 4}
    assert set(out["MARRIAGE"].unique()) == {1, 2, 3}


def test_clean_does_not_modify_input():
    df = pd.DataFrame({"EDUCATION": [0], "MARRIAGE": [0]})
    _ = pp.clean_dataset(df)
    assert df["EDUCATION"].iloc[0] == 0
    assert df["MARRIAGE"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Feature / target split
# ---------------------------------------------------------------------------

def test_split_features_target_returns_correct_order():
    df = pd.DataFrame({c: [0] for c in settings.FEATURE_COLUMNS})
    df[settings.TARGET_COLUMN] = [0]
    X, y = pp.split_features_target(df)
    assert list(X.columns) == list(settings.FEATURE_COLUMNS)
    assert y.name == settings.TARGET_COLUMN


def test_split_features_target_raises_on_missing_column():
    df = pd.DataFrame({"LIMIT_BAL": [1000], settings.TARGET_COLUMN: [0]})
    with pytest.raises(KeyError):
        pp.split_features_target(df)


# ---------------------------------------------------------------------------
# Preprocessor (synthetic — runs even without the dataset)
# ---------------------------------------------------------------------------

def test_preprocessor_fits_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "LIMIT_BAL": rng.normal(100000, 20000, n),
        "AGE": rng.integers(21, 70, n),
        **{f"BILL_AMT{i}": rng.normal(0, 10000, n) for i in range(1, 7)},
        **{f"PAY_AMT{i}": rng.normal(0, 5000, n) for i in range(1, 7)},
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        **{f"PAY_{i}": rng.integers(-2, 10, n) for i in [0, 2, 3, 4, 5, 6]},
    })
    X = df[list(settings.FEATURE_COLUMNS)]
    pre = pp.build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == n
    # All features should be numeric and finite.
    assert np.isfinite(Xt).all()


def test_preprocessor_handles_unseen_category_values():
    """handle_unknown='ignore' must not raise on a value absent from fit."""
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame({
        "LIMIT_BAL": rng.normal(100000, 20000, n),
        "AGE": rng.integers(21, 70, n),
        **{f"BILL_AMT{i}": rng.normal(0, 10000, n) for i in range(1, 7)},
        **{f"PAY_AMT{i}": rng.normal(0, 5000, n) for i in range(1, 7)},
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        **{f"PAY_{i}": rng.integers(-2, 10, n) for i in [0, 2, 3, 4, 5, 6]},
    })
    X = df[list(settings.FEATURE_COLUMNS)]
    pre = pp.build_preprocessor().fit(X)

    X_new = X.copy()
    X_new.loc[X_new.index[0], "EDUCATION"] = 99   # never seen at fit
    X_new.loc[X_new.index[1], "PAY_0"] = -99
    # Should not raise:
    pre.transform(X_new)


# ---------------------------------------------------------------------------
# End-to-end split (requires the real dataset)
# ---------------------------------------------------------------------------

@requires_dataset
def test_full_pipeline_split_and_fit(tmp_path, monkeypatch):
    from src.data.loader import load_raw_dataset

    df = pp.clean_dataset(load_raw_dataset())
    X, y = pp.split_features_target(df)
    X_tr, X_te, y_tr, y_te = pp.make_train_test_split(X, y)

    assert len(X_tr) + len(X_te) == len(X)
    assert abs(len(X_te) / len(X) - settings.TEST_SIZE) < 0.01

    # Stratification: positive rate should be preserved within tolerance.
    p_train = y_tr.mean()
    p_test = y_te.mean()
    assert abs(p_train - p_test) < 0.01

    pre = pp.build_preprocessor()
    Xt_train = pre.fit_transform(X_tr)
    Xt_test = pre.transform(X_te)
    assert Xt_train.shape[1] == Xt_test.shape[1]


@requires_dataset
def test_split_is_reproducible():
    from src.data.loader import load_raw_dataset

    df = pp.clean_dataset(load_raw_dataset())
    X, y = pp.split_features_target(df)
    X_tr1, X_te1, y_tr1, _ = pp.make_train_test_split(X, y)
    X_tr2, X_te2, y_tr2, _ = pp.make_train_test_split(X, y)
    assert X_tr1.equals(X_tr2)
    assert X_te1.equals(X_te2)
    assert y_tr1.equals(y_tr2)