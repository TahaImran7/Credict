"""
Evaluation metrics, curves, and calibration data for CreditLens.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from src.utils.logging_config import get_logger

log = get_logger(__name__)


def _to_python(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class ModelEvaluation:
    model_name: str
    threshold: float
    n_samples: int
    n_positive: int
    n_negative: int
    metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    roc_curve: dict[str, list[float]] = field(default_factory=dict)
    pr_curve: dict[str, list[float]] = field(default_factory=dict)
    calibration: dict[str, list[float]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(_to_python(asdict(self)), indent=2)


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
    threshold: float | None = None,
) -> ModelEvaluation:
    """
    Compute classification metrics, ROC/PR curves, and calibration data.

    If `threshold` is None, uses `model.threshold` when available (for
    CalibratedPipeline instances), otherwise falls back to 0.5.
    """
    if threshold is None:
        threshold = float(getattr(model, "threshold", 0.5))

    y_true = np.asarray(y_test).astype(int)
    y_prob = np.asarray(model.predict_proba(X_test))[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="quantile"
    )

    evaluation = ModelEvaluation(
        model_name=model_name,
        threshold=float(threshold),
        n_samples=int(len(y_true)),
        n_positive=int(y_true.sum()),
        n_negative=int((y_true == 0).sum()),
        metrics=metrics,
        confusion_matrix=cm.tolist(),
        roc_curve={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        pr_curve={"precision": prec.tolist(), "recall": rec.tolist()},
        calibration={
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        },
    )

    log.info(
        "Evaluated %s (threshold=%.3f): ROC-AUC=%.4f, PR-AUC=%.4f, Brier=%.4f",
        model_name, threshold,
        metrics["roc_auc"], metrics["pr_auc"], metrics["brier"],
    )
    return evaluation


def save_evaluation(evaluation: ModelEvaluation, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(evaluation.to_json(), encoding="utf-8")
    log.info("Saved evaluation to %s", path)
    return path


def load_evaluation(path: Path) -> ModelEvaluation:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelEvaluation(**data)