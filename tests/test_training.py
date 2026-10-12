"""
Phase 3 — training, evaluation, persistence, and comparison tests.

The heavy tests (real GridSearchCV on the real dataset) are skipped unless
model artefacts are already present. The fast tests use synthetic data and
tiny grids to verify behaviour without a long runtime.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import settings
from src.training.evaluate import (
    ModelEvaluation,
    evaluate_model,
    load_evaluation,
    save_evaluation,
)
from src.training.model_comparison import (
    compare_models,
    load_comparison,
    save_comparison,
    select_best_model,
)
from src.training.train import (
    build_logistic_pipeline,
    build_random_forest_pipeline,
    load_model,
    model_path,
    save_model,
)
from src.utils.paths import EVALUATION_DIR, SAVED_MODELS_DIR


# ---------------------------------------------------------------------------
# Synthetic fixture (fast — no file I/O, no real training loop)
# ---------------------------------------------------------------------------

def _synthetic_xy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
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
    # Simple learnable signal so models are not random.
    logit = (
        -1.5
        + 0.00002 * X["LIMIT_BAL"]
        - 0.3 * (X["PAY_0"] > 0).astype(int)
        + rng.normal(0, 0.5, n)
    )
    p = 1 / (1 + np.exp(-logit))
    y = pd.Series(rng.binomial(1, p), name=settings.TARGET_COLUMN)
    return X, y


@pytest.fixture(scope="module")
def fitted_lr():
    X, y = _synthetic_xy()
    pipe = build_logistic_pipeline()
    pipe.fit(X, y)
    return pipe, X, y


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def test_logistic_pipeline_has_expected_steps():
    pipe = build_logistic_pipeline()
    assert list(pipe.named_steps.keys()) == ["pre", "clf"]


def test_random_forest_pipeline_has_expected_steps():
    pipe = build_random_forest_pipeline()
    assert list(pipe.named_steps.keys()) == ["pre", "clf"]


def test_lr_pipeline_can_fit_and_predict(fitted_lr):
    pipe, X, _ = fitted_lr
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_model_returns_expected_metrics(fitted_lr):
    pipe, X, y = fitted_lr
    ev = evaluate_model(pipe, X, y, "synthetic_lr")
    for metric in ("accuracy", "precision", "recall", "f1",
                   "roc_auc", "pr_auc", "brier"):
        assert metric in ev.metrics
        assert 0.0 <= ev.metrics[metric] <= 1.0
    assert len(ev.confusion_matrix) == 2
    assert sum(sum(row) for row in ev.confusion_matrix) == len(y)
    assert ev.n_samples == len(y)
    assert ev.n_positive == int(y.sum())


def test_evaluation_json_round_trip(tmp_path, fitted_lr):
    pipe, X, y = fitted_lr
    ev = evaluate_model(pipe, X, y, "synthetic_lr")
    path = tmp_path / "eval.json"
    save_evaluation(ev, path)
    ev2 = load_evaluation(path)
    assert ev2.model_name == ev.model_name
    assert ev2.metrics == ev.metrics
    assert ev2.confusion_matrix == ev.confusion_matrix


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _fake_evaluation(name: str, pr_auc: float, roc_auc: float) -> ModelEvaluation:
    return ModelEvaluation(
        model_name=name,
        threshold=0.5,
        n_samples=100,
        n_positive=22,
        n_negative=78,
        metrics={
            "accuracy": 0.8,
            "precision": 0.5,
            "recall": 0.6,
            "f1": 0.55,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier": 0.15,
        },
        confusion_matrix=[[70, 8], [9, 13]],
        roc_curve={"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
        pr_curve={"precision": [0.5], "recall": [0.6]},
        calibration={"mean_predicted": [0.2], "fraction_positive": [0.2]},
    )


def test_compare_models_shape():
    df = compare_models([
        _fake_evaluation("a", 0.5, 0.7),
        _fake_evaluation("b", 0.6, 0.75),
    ])
    assert list(df.index) == ["a", "b"]
    assert "pr_auc" in df.columns
    assert "roc_auc" in df.columns


def test_select_best_model_uses_pr_auc_first():
    df = compare_models([
        _fake_evaluation("a", 0.5, 0.9),   # lower PR-AUC
        _fake_evaluation("b", 0.6, 0.7),   # higher PR-AUC
    ])
    assert select_best_model(df) == "b"


def test_select_best_model_tiebreak_on_roc_auc():
    df = compare_models([
        _fake_evaluation("a", 0.6, 0.7),
        _fake_evaluation("b", 0.6, 0.8),   # same PR-AUC, higher ROC-AUC
    ])
    assert select_best_model(df) == "b"


def test_comparison_csv_round_trip(tmp_path):
    df = compare_models([_fake_evaluation("a", 0.5, 0.7)])
    path = tmp_path / "cmp.csv"
    save_comparison(df, path)
    df2 = load_comparison(path)
    assert list(df2.index) == ["a"]
    assert "pr_auc" in df2.columns


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_load_model(tmp_path, monkeypatch, fitted_lr):
    pipe, X, _ = fitted_lr
    monkeypatch.setattr("src.training.train.SAVED_MODELS_DIR", tmp_path)
    save_model(pipe, "synthetic_lr")
    loaded = load_model("synthetic_lr")
    proba_orig = pipe.predict_proba(X)
    proba_loaded = loaded.predict_proba(X)
    assert np.allclose(proba_orig, proba_loaded)


# ---------------------------------------------------------------------------
# Real artefacts (skipped unless training has been run)
# ---------------------------------------------------------------------------

def _real_models_present() -> bool:
    return (
        (SAVED_MODELS_DIR / "logistic_regression.joblib").exists()
        and (SAVED_MODELS_DIR / "random_forest.joblib").exists()
    )


@pytest.mark.skipif(
    not _real_models_present(),
    reason="Real models not trained yet. Run: python -m scripts.train_models",
)
def test_real_models_predict_in_unit_interval():
    from src.data import preprocessing as pp
    _, X_test, _, _ = pp.load_splits()
    for name in ("logistic_regression", "random_forest"):
        model = load_model(name)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.skipif(
    not (EVALUATION_DIR / "model_comparison.csv").exists(),
    reason="Model comparison not generated yet.",
)
def test_real_model_comparison_file_exists():
    df = load_comparison(EVALUATION_DIR / "model_comparison.csv")
    assert {"logistic_regression", "random_forest"} <= set(df.index)
    assert df["roc_auc"].between(0, 1).all()
    assert df["pr_auc"].between(0, 1).all()