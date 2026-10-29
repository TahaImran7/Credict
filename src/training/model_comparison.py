"""
Model comparison utilities for CreditLens.

Handles the two-stage selection:
  Stage 1: Pick best base model family (LR vs RF) by PR-AUC then ROC-AUC.
  Stage 2: If a calibrated variant of the winning family exists AND improves
           Brier score by a meaningful margin, prefer the calibrated variant.

This makes the "selected model" consistent with what the UI actually uses
(see src.prediction.predictor.DEFAULT_MODEL_NAME).
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.training.evaluate import ModelEvaluation
from src.utils.logging_config import get_logger

log = get_logger(__name__)

COMPARISON_METRICS: tuple[str, ...] = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier",
)

# Minimum Brier improvement (relative) required to prefer the calibrated model
MIN_BRIER_IMPROVEMENT: float = 0.05   # 5%


def compare_models(evaluations: Iterable[ModelEvaluation]) -> pd.DataFrame:
    rows = []
    for ev in evaluations:
        row = {
            "model": ev.model_name,
            "n_test": ev.n_samples,
            "n_positive": ev.n_positive,
            "threshold": ev.threshold,
        }
        for m in COMPARISON_METRICS:
            row[m] = ev.metrics.get(m, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def select_best_model(
    df: pd.DataFrame,
    primary_metric: str = "pr_auc",
    secondary_metric: str = "roc_auc",
) -> str:
    """
    Documented selection rule for BASE model families:
      1. Primary: PR-AUC
      2. Tie-break: ROC-AUC
      3. Final tie-break: alphabetically first model name
    """
    if primary_metric not in df.columns or secondary_metric not in df.columns:
        raise KeyError(
            f"Metrics {primary_metric!r} or {secondary_metric!r} not in comparison."
        )
    sorted_df = df.sort_values(by=[primary_metric, secondary_metric], ascending=False)
    best = sorted_df.index[0]
    log.info(
        "Select_best_model: '%s' by %s=%.4f, %s=%.4f",
        best, primary_metric, sorted_df.loc[best, primary_metric],
        secondary_metric, sorted_df.loc[best, secondary_metric],
    )
    return best


def select_final_model(
    df: pd.DataFrame,
    base_models: tuple[str, ...] = ("logistic_regression", "random_forest"),
    calibration_suffix: str = "_calibrated",
    min_brier_improvement: float = MIN_BRIER_IMPROVEMENT,
) -> tuple[str, dict]:
    """
    Two-stage selection with optional calibration preference.

    Returns
    -------
    (final_model_name, rationale_dict)
    """
    base_df = df.loc[[m for m in base_models if m in df.index]]
    if base_df.empty:
        raise ValueError(
            f"No base models found. Expected one of {base_models} in "
            f"comparison index {df.index.tolist()}."
        )
    best_base = select_best_model(base_df)

    rationale = {
        "best_base": best_base,
        "base_pr_auc": float(df.loc[best_base, "pr_auc"]),
        "base_roc_auc": float(df.loc[best_base, "roc_auc"]),
        "base_brier": float(df.loc[best_base, "brier"]),
    }

    cal_name = f"{best_base}{calibration_suffix}"
    if cal_name not in df.index:
        rationale["final"] = best_base
        rationale["reason"] = "no calibrated variant available"
        return best_base, rationale

    cal_brier = float(df.loc[cal_name, "brier"])
    improvement = (rationale["base_brier"] - cal_brier) / rationale["base_brier"]
    cal_pr = float(df.loc[cal_name, "pr_auc"])
    cal_roc = float(df.loc[cal_name, "roc_auc"])

    rationale.update({
        "calibrated_brier": cal_brier,
        "calibrated_pr_auc": cal_pr,
        "calibrated_roc_auc": cal_roc,
        "brier_improvement_pct": improvement * 100,
        "pr_auc_drop": rationale["base_pr_auc"] - cal_pr,
        "roc_auc_drop": rationale["base_roc_auc"] - cal_roc,
    })

    if improvement >= min_brier_improvement:
        log.info(
            "Preferring calibrated '%s' over raw '%s' (Brier +%.1f%% improvement).",
            cal_name, best_base, improvement * 100,
        )
        rationale["final"] = cal_name
        rationale["reason"] = (
            f"calibrated variant improves Brier by {improvement*100:.1f}% "
            f"(>= {min_brier_improvement*100:.0f}% threshold)"
        )
        return cal_name, rationale

    log.info(
        "Keeping raw '%s' — calibrated Brier improvement %.1f%% below %.0f%% threshold.",
        best_base, improvement * 100, min_brier_improvement * 100,
    )
    rationale["final"] = best_base
    rationale["reason"] = (
        f"calibrated Brier improvement {improvement*100:.1f}% "
        f"below {min_brier_improvement*100:.0f}% threshold"
    )
    return best_base, rationale


def save_comparison(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    log.info("Saved model comparison to %s", path)
    return path


def load_comparison(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="model")