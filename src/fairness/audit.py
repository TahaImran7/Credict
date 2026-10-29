"""
Group-level fairness audit for Credict.

This module is for educational model auditing. It does not certify legal
compliance and does not prove or disprove discrimination. Differences in
observed outcomes across groups may reflect historical bias, dataset
composition, sample size, or genuine differences in the underlying
population — not the model alone.

All metrics are computed on a labelled dataset (y_true must be known).
Individual prediction explanations belong in src.explainability; group-level
audits belong here. The two answer different questions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import settings
from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Metrics computed per group. `prevalence` is a property of the data;
# the rest describe the model's behaviour on that group.
GROUP_METRICS: tuple[str, ...] = (
    "n",
    "n_positive",
    "n_negative",
    "prevalence",
    "tp",
    "fp",
    "tn",
    "fn",
    "tpr",
    "fpr",
    "precision",
    "npv",
    "f1",
    "accuracy",
    "roc_auc",
)

# Metric columns whose differences from a reference group are meaningful.
DIFF_METRICS: tuple[str, ...] = (
    "prevalence",
    "tpr",
    "fpr",
    "precision",
    "npv",
    "f1",
    "accuracy",
    "roc_auc",
)


@dataclass
class AuditResult:
    group_col: str
    reference_group: object
    threshold: float
    min_group_size: int
    metrics: pd.DataFrame          # one row per group
    differences: pd.DataFrame      # metrics minus reference-group values
    warnings: list[str]

    @property
    def small_groups(self) -> list:
        if "small_group" not in self.metrics.columns:
            return []
        return self.metrics.loc[self.metrics["small_group"], "group"].tolist()


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator or NaN when denominator is zero."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _group_metrics_for_slice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = tp + tn + fp + fn
    n_pos = tp + fn
    n_neg = tn + fp

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)      # == tpr
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) \
        if not (np.isnan(precision) or np.isnan(recall)) else float("nan")

    roc_auc = float("nan")
    if y_prob is not None and n_pos > 0 and n_neg > 0:
        from sklearn.metrics import roc_auc_score
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": _safe_div(n_pos, n),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": recall,
        "fpr": _safe_div(fp, fp + tn),
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "accuracy": _safe_div(tp + tn, n),
        "roc_auc": roc_auc,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def audit_groups(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    group_col: str,
    y_prob: pd.Series | np.ndarray | None = None,
    reference_group: object | None = None,
    threshold: float = 0.5,
    min_group_size: int = 30,
) -> AuditResult:
    """
    Compute per-group classification metrics and differences from a
    reference group.

    Parameters
    ----------
    y_true, y_pred, groups : array-like, equal length
    group_col : str
        Name of the grouping column (used for display and column naming).
    y_prob : optional probabilities, used only for per-group ROC-AUC.
    reference_group : optional
        Which group value to use as the reference for difference columns.
        Defaults to the largest group.
    threshold : decision threshold already applied to y_pred (informational).
    min_group_size : groups with fewer records are flagged as small_group.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups)

    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"groups={len(groups)}."
        )

    y_prob_arr = None if y_prob is None else np.asarray(y_prob, dtype=float)

    rows = []
    for g in pd.unique(groups):
        mask = groups == g
        metrics = _group_metrics_for_slice(
            y_true[mask], y_pred[mask],
            None if y_prob_arr is None else y_prob_arr[mask],
        )
        metrics["group"] = g
        rows.append(metrics)

    df = pd.DataFrame(rows)
    # Place group as first column; order remaining columns per GROUP_METRICS.
    ordered_cols = ["group"] + [m for m in GROUP_METRICS if m in df.columns]
    df = df[ordered_cols].reset_index(drop=True)

    # Small-group flags
    df["small_group"] = df["n"] < min_group_size

    # Choose reference group
    if reference_group is None:
        reference_group = df.loc[df["n"].idxmax(), "group"]
    ref_row = df.loc[df["group"] == reference_group]
    if ref_row.empty:
        raise ValueError(
            f"reference_group={reference_group!r} not found in groups "
            f"{df['group'].tolist()}"
        )

    # Differences from reference (only where both values are defined)
    diff_cols = {}
    for metric in DIFF_METRICS:
        if metric not in df.columns:
            continue
        diff_cols[f"{metric}_minus_ref"] = df[metric] - ref_row[metric].iloc[0]
    diff_df = pd.concat([df[["group"]], pd.DataFrame(diff_cols)], axis=1)

    # Warnings about statistical reliability
    warnings: list[str] = []
    small = df.loc[df["small_group"], "group"].tolist()
    if small:
        warnings.append(
            f"Groups with fewer than {min_group_size} records: {small}. "
            "Their metrics may be unstable."
        )
    no_pos = df.loc[df["n_positive"] == 0, "group"].tolist()
    if no_pos:
        warnings.append(
            f"Groups with no positive (default) cases: {no_pos}. "
            "TPR, precision, and F1 are undefined for these groups."
        )
    no_neg = df.loc[df["n_negative"] == 0, "group"].tolist()
    if no_neg:
        warnings.append(
            f"Groups with no negative cases: {no_neg}. "
            "FPR and NPV are undefined for these groups."
        )

    log.info(
        "Audited %d groups on '%s'; reference=%r; warnings=%d",
        len(df), group_col, reference_group, len(warnings),
    )
    return AuditResult(
        group_col=group_col,
        reference_group=reference_group,
        threshold=float(threshold),
        min_group_size=int(min_group_size),
        metrics=df,
        differences=diff_df,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Convenience: age bands
# ---------------------------------------------------------------------------

DEFAULT_AGE_BINS: tuple[int, ...] = (0, 30, 40, 50, 60, 200)
DEFAULT_AGE_LABELS: tuple[str, ...] = ("<30", "30-39", "40-49", "50-59", "60+")


def age_bands(
    ages: pd.Series,
    bins: tuple[int, ...] = DEFAULT_AGE_BINS,
    labels: tuple[str, ...] = DEFAULT_AGE_LABELS,
) -> pd.Series:
    """Bin ages into labelled bands. right=False → [low, high)."""
    return pd.cut(ages, bins=list(bins), labels=list(labels), right=False)


# ---------------------------------------------------------------------------
# Audit across several grouping columns at once
# ---------------------------------------------------------------------------

def audit_standard_groups(
    df: pd.DataFrame,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
    group_specs: dict[str, pd.Series] | None = None,
    min_group_size: int = 30,
) -> dict[str, AuditResult]:
    """
    Run audit_groups for SEX, EDUCATION, MARRIAGE, and age bands by default.

    Parameters
    ----------
    df : labelled dataframe containing settings.TARGET_COLUMN and the
         grouping columns (SEX, EDUCATION, MARRIAGE, AGE).
    y_pred : predicted class labels, aligned to df.index.
    y_prob : optional predicted probabilities.
    group_specs : optional dict mapping name -> group Series.
        If None, uses {"SEX": df["SEX"], "EDUCATION": df["EDUCATION"],
                       "MARRIAGE": df["MARRIAGE"], "AGE_BAND": age_bands(df["AGE"])}.
    """
    if settings.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{settings.TARGET_COLUMN}' is required.")
    y_true = df[settings.TARGET_COLUMN]

    if group_specs is None:
        group_specs = {}
        if "SEX" in df.columns:
            group_specs["SEX"] = df["SEX"]
        if "EDUCATION" in df.columns:
            group_specs["EDUCATION"] = df["EDUCATION"]
        if "MARRIAGE" in df.columns:
            group_specs["MARRIAGE"] = df["MARRIAGE"]
        if "AGE" in df.columns:
            group_specs["AGE_BAND"] = age_bands(df["AGE"])

    results: dict[str, AuditResult] = {}
    for name, groups in group_specs.items():
        results[name] = audit_groups(
            y_true=y_true, y_pred=y_pred, groups=groups,
            group_col=name, y_prob=y_prob,
            min_group_size=min_group_size,
        )
    return results


def flatten_audit_results(
    results: dict[str, AuditResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine per-group metrics from multiple audit dimensions into two tables:
    one of metrics, one of differences from reference.
    """
    metrics_frames = []
    diff_frames = []
    for name, res in results.items():
        m = res.metrics.copy()
        m.insert(0, "audit", name)
        m.insert(2, "reference_group", res.reference_group)
        metrics_frames.append(m)

        d = res.differences.copy()
        d.insert(0, "audit", name)
        d.insert(2, "reference_group", res.reference_group)
        diff_frames.append(d)

    metrics_df = pd.concat(metrics_frames, ignore_index=True) if metrics_frames \
        else pd.DataFrame()
    diff_df = pd.concat(diff_frames, ignore_index=True) if diff_frames \
        else pd.DataFrame()
    return metrics_df, diff_df


# ---------------------------------------------------------------------------
# Interpretation text (kept factual; no claims of fairness or discrimination)
# ---------------------------------------------------------------------------

FAIRNESS_LIMITATION: str = (
    "Group-level metrics describe how the model behaves on the evaluated "
    "sample. Differences between groups may reflect the historical data, "
    "sample sizes, dataset composition, or real differences in the "
    "underlying population. They do not by themselves prove or disprove "
    "discrimination, and they do not certify legal compliance."
)