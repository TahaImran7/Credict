"""
Model comparison utilities for CreditLens.

Consumes ModelEvaluation objects from src.training.evaluate and produces a
tabular comparison plus a documented selection recommendation.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.training.evaluate import ModelEvaluation
from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Metrics surfaced in the comparison table (order matters for display).
COMPARISON_METRICS: tuple[str, ...] = (
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier",
)


def compare_models(evaluations: Iterable[ModelEvaluation]) -> pd.DataFrame:
    """
    Return a DataFrame with one row per model and one column per metric,
    plus test-set size and positive count for reference.
    """
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
    df = pd.DataFrame(rows).set_index("model")
    return df


def select_best_model(
    df: pd.DataFrame,
    primary_metric: str = "pr_auc",
    secondary_metric: str = "roc_auc",
) -> str:
    """
    Documented selection rule:
      1. Primary: PR-AUC — appropriate for imbalanced credit-default data.
      2. Tie-break: ROC-AUC.
      3. Final tie-break: alphabetically first model name (deterministic).

    Returns the selected model name.
    """
    if primary_metric not in df.columns or secondary_metric not in df.columns:
        raise KeyError(
            f"Metrics {primary_metric!r} or {secondary_metric!r} not in comparison."
        )
    sorted_df = df.sort_values(
        by=[primary_metric, secondary_metric],
        ascending=False,
    )
    best = sorted_df.index[0]
    log.info(
        "Selected model '%s' by %s=%.4f, %s=%.4f",
        best,
        primary_metric,
        sorted_df.loc[best, primary_metric],
        secondary_metric,
        sorted_df.loc[best, secondary_metric],
    )
    return best


def save_comparison(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    log.info("Saved model comparison to %s", path)
    return path


def load_comparison(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="model")