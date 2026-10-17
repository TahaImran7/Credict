"""
Cached data and model access for the CreditLens UI.

Every loader returns either a model (via st.cache_resource) or a DataFrame
(via st.cache_data). Cache keys include file mtime so retraining or
re-exporting automatically invalidates the relevant entries.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from config import settings
from src.data.loader import load_raw_dataset
from src.data import preprocessing as pp
from src.training.evaluate import ModelEvaluation, load_evaluation
from src.training.model_comparison import load_comparison
from src.training.train import load_model as _load_model_direct
from src.utils.paths import (
    EVALUATION_DIR,
    RAW_DATA_DIR,
    SAVED_MODELS_DIR,
)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_model_cached(name: str, _mtime: float):
    return _load_model_direct(name)


def get_model(name: str):
    path = SAVED_MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' is missing. Run: python -m scripts.train_models"
        )
    return _load_model_cached(name, _mtime(path))


# ---------------------------------------------------------------------------
# Splits and raw data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_splits_cached(_k1: float, _k2: float, _k3: float, _k4: float):
    return pp.load_splits()


def get_splits():
    keys = (
        _mtime(pp.X_TRAIN_PATH), _mtime(pp.X_TEST_PATH),
        _mtime(pp.Y_TRAIN_PATH), _mtime(pp.Y_TEST_PATH),
    )
    return _load_splits_cached(*keys)


@st.cache_data(show_spinner=False)
def _load_raw_cached(_mtime_value: float):
    df = load_raw_dataset()
    return pp.clean_dataset(df)


def get_raw_data() -> pd.DataFrame:
    raw_path = RAW_DATA_DIR / settings.DATASET_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset missing at {raw_path}. "
            "Run: python -c \"from src.data.loader import download_dataset; download_dataset()\""
        )
    return _load_raw_cached(_mtime(raw_path))


# ---------------------------------------------------------------------------
# Evaluation artefacts
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_eval_cached(name: str, _mtime_value: float) -> dict:
    ev = load_evaluation(EVALUATION_DIR / f"{name}_eval.json")
    # Return as a plain dict so Streamlit can hash it
    return {
        "model_name": ev.model_name,
        "threshold": ev.threshold,
        "n_samples": ev.n_samples,
        "n_positive": ev.n_positive,
        "n_negative": ev.n_negative,
        "metrics": ev.metrics,
        "confusion_matrix": ev.confusion_matrix,
        "roc_curve": ev.roc_curve,
        "pr_curve": ev.pr_curve,
        "calibration": ev.calibration,
    }


def get_evaluation(name: str) -> dict:
    path = EVALUATION_DIR / f"{name}_eval.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation for '{name}' is missing. "
            "Run: python -m scripts.train_models"
        )
    return _load_eval_cached(name, _mtime(path))


@st.cache_data(show_spinner=False)
def _load_comparison_cached(_mtime_value: float) -> pd.DataFrame:
    return load_comparison(EVALUATION_DIR / "model_comparison.csv")


def get_comparison() -> pd.DataFrame:
    path = EVALUATION_DIR / "model_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Model comparison file missing. Run: python -m scripts.train_models"
        )
    return _load_comparison_cached(_mtime(path))


@st.cache_data(show_spinner=False)
def _load_global_importance_cached(_mtime_value: float) -> pd.DataFrame:
    return pd.read_csv(EVALUATION_DIR / "global_importance.csv")


def get_global_importance() -> pd.DataFrame:
    path = EVALUATION_DIR / "global_importance.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Global importance file missing. "
            "Run: python -m scripts.build_dashboard_artifacts"
        )
    return _load_global_importance_cached(_mtime(path))


@st.cache_data(show_spinner=False)
def _load_test_predictions_cached(_mtime_value: float) -> pd.DataFrame:
    return pd.read_csv(EVALUATION_DIR / "test_predictions.csv")


def get_test_predictions() -> pd.DataFrame:
    path = EVALUATION_DIR / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Test predictions missing. "
            "Run: python -m scripts.build_dashboard_artifacts"
        )
    return _load_test_predictions_cached(_mtime(path))


@st.cache_data(show_spinner=False)
def _load_fairness_metrics_cached(_m: float, _d: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(EVALUATION_DIR / "fairness_metrics.csv")
    d = pd.read_csv(EVALUATION_DIR / "fairness_differences.csv")
    return m, d


def get_fairness() -> tuple[pd.DataFrame, pd.DataFrame]:
    mp = EVALUATION_DIR / "fairness_metrics.csv"
    dp = EVALUATION_DIR / "fairness_differences.csv"
    if not (mp.exists() and dp.exists()):
        raise FileNotFoundError(
            "Fairness files missing. Run: python -m scripts.run_fairness_audit"
        )
    m, d = _load_fairness_metrics_cached(_mtime(mp), _mtime(dp))

    if "n" in m.columns:
        m = m[m["n"].notna()].reset_index(drop=True)

    if "small_group" in m.columns:
        m["small_group"] = (
            m["small_group"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )

    return m, d


def clear_all_caches() -> None:
    """Clear every CreditLens cache. Call from the sidebar."""
    _load_model_cached.clear()
    _load_splits_cached.clear()
    _load_raw_cached.clear()
    _load_eval_cached.clear()
    _load_comparison_cached.clear()
    _load_global_importance_cached.clear()
    _load_test_predictions_cached.clear()
    _load_fairness_metrics_cached.clear()