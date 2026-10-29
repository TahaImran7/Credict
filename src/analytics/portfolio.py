"""
Portfolio-level analytics for Credict.

All functions return plain pandas objects (DataFrames / Series) so they can
be tested without a UI and plotted later. No Plotly imports here.

Descriptive functions operate on whichever labelled dataset the caller
provides. Model-derived risk bands use the persisted Random Forest, but only
when the caller explicitly asks for them.
"""

import numpy as np
import pandas as pd

from config import settings
from src.prediction.predictor import predict_batch_with_bands, DEFAULT_MODEL_NAME
from src.utils.logging_config import get_logger

log = get_logger(__name__)

RISK_BAND_ORDER: tuple[str, ...] = ("Low", "Moderate", "High", "Very High")


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------

def portfolio_summary(df: pd.DataFrame) -> dict:
    """
    High-level counts and default rate for a labelled dataframe containing
    the target column.
    """
    if settings.TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{settings.TARGET_COLUMN}' is required for "
            "portfolio_summary()."
        )
    n = int(len(df))
    positives = int(df[settings.TARGET_COLUMN].sum())
    return {
        "n_customers": n,
        "n_defaulters": positives,
        "n_non_defaulters": n - positives,
        "default_rate": positives / n if n else float("nan"),
    }


def probability_distribution_stats(probabilities: pd.Series | np.ndarray) -> dict:
    """Basic descriptive statistics for a set of probabilities."""
    p = np.asarray(probabilities, dtype=float)
    return {
        "n": int(p.size),
        "mean": float(np.mean(p)),
        "median": float(np.median(p)),
        "std": float(np.std(p)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "p25": float(np.percentile(p, 25)),
        "p75": float(np.percentile(p, 75)),
        "p90": float(np.percentile(p, 90)),
    }


def risk_band_counts(probabilities: pd.Series | np.ndarray) -> pd.DataFrame:
    """
    Counts and shares of records per risk band, ordered Low -> Very High.
    """
    p = np.asarray(probabilities, dtype=float)
    bands = pd.Series([_band_for(x) for x in p], name="risk_band")
    counts = bands.value_counts().reindex(RISK_BAND_ORDER, fill_value=0)
    shares = counts / counts.sum() if counts.sum() else counts
    out = pd.DataFrame({
        "risk_band": counts.index,
        "count": counts.values,
        "share": shares.values,
    })
    return out


def _band_for(p: float) -> str:
    for band, (low, high) in settings.RISK_BANDS.items():
        if band == "Very High":
            if low <= p <= 1.0:
                return band
        elif low <= p < high:
            return band
    return "Very High"


# ---------------------------------------------------------------------------
# Category-level default rates
# ---------------------------------------------------------------------------

def default_rate_by_category(
    df: pd.DataFrame,
    category_col: str,
    min_count: int = 30,
) -> pd.DataFrame:
    """
    Default rate for each value of a categorical feature.
    Rows with fewer than `min_count` records are flagged (rate reported but
    flagged as low-confidence).
    """
    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not in dataframe.")
    if settings.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{settings.TARGET_COLUMN}' is required.")

    grouped = (
        df.groupby(category_col, dropna=False)[settings.TARGET_COLUMN]
          .agg(["count", "sum", "mean"])
          .rename(columns={"count": "n", "sum": "n_defaults", "mean": "default_rate"})
          .reset_index()
          .rename(columns={category_col: "category"})
    )
    grouped["low_confidence"] = grouped["n"] < min_count
    return grouped.sort_values("category").reset_index(drop=True)


def default_rate_by_pay_status(
    df: pd.DataFrame,
    pay_cols: tuple[str, ...] = ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"),
) -> pd.DataFrame:
    """
    Stacked long-form default rates for all repayment-status columns.
    Columns: pay_col, category, n, n_defaults, default_rate, low_confidence.
    """
    frames = []
    for col in pay_cols:
        if col not in df.columns:
            continue
        sub = default_rate_by_category(df, col)
        sub.insert(0, "pay_col", col)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=[
            "pay_col", "category", "n", "n_defaults",
            "default_rate", "low_confidence",
        ])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Feature vs. outcome summaries
# ---------------------------------------------------------------------------

def mean_features_by_outcome(
    df: pd.DataFrame,
    features: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Mean of each numeric feature split by the target value.
    Useful as a quick portfolio-level signal table.
    """
    if settings.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{settings.TARGET_COLUMN}' is required.")
    if features is None:
        features = settings.NUMERIC_FEATURES
    rows = []
    grouped = df.groupby(settings.TARGET_COLUMN)
    for f in features:
        if f not in df.columns:
            continue
        means = grouped[f].mean()
        rows.append({
            "feature": f,
            "mean_non_default": float(means.get(0, np.nan)),
            "mean_default": float(means.get(1, np.nan)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["ratio_default_to_non"] = (
            out["mean_default"] / out["mean_non_default"].replace(0, np.nan)
        )
    return out


# ---------------------------------------------------------------------------
# Model-derived risk over the portfolio
# ---------------------------------------------------------------------------

def model_risk_bands_for(
    X: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """
    Run the model over every row in X and return probability + band.
    """
    return predict_batch_with_bands(X, model_name=model_name)


def risk_band_default_rates(
    probabilities: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    For each risk band, count records, count actual defaults, and compute the
    observed default rate. Requires a labelled dataset.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(y_true, dtype=int)
    if p.shape != y.shape:
        raise ValueError("probabilities and y_true must have the same length.")
    bands = pd.Series([_band_for(x) for x in p])
    df = pd.DataFrame({"risk_band": bands, "y": y})
    out = (
        df.groupby("risk_band")["y"]
          .agg(["count", "sum", "mean"])
          .rename(columns={"count": "n", "sum": "n_defaults", "mean": "observed_default_rate"})
          .reindex(RISK_BAND_ORDER, fill_value=0)
          .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# Combined snapshot (used by the Overview page)
# ---------------------------------------------------------------------------

def overview_snapshot(
    df: pd.DataFrame,
    probabilities: pd.Series | np.ndarray | None = None,
) -> dict:
    """
    Return a compact dict of headline metrics for the Overview dashboard.
    If probabilities are provided, includes risk-band breakdown.
    """
    snap: dict = {"portfolio": portfolio_summary(df)}
    if probabilities is not None:
        snap["probability_stats"] = probability_distribution_stats(probabilities)
        snap["risk_bands"] = risk_band_counts(probabilities).to_dict(orient="records")
    return snap