"""
SHAP-based explanations for Credict.

Uses the RAW Random Forest classifier (not the calibrated wrapper) for
TreeExplainer, which requires a tree-based estimator. If the model name
points to a calibrated pipeline, it is transparently unwrapped.

SHAP values describe contributions to the model's INTERNAL score (before
isotonic calibration). The isotonic mapping is monotonic, so:
- Feature ranking and relative importance are unchanged.
- Signs (risk-increasing vs risk-decreasing) are unchanged.
- Only the absolute magnitudes are rescaled relative to the display
  probability.

The `probability` returned in ShapExplanation is the calibrated
probability from the full pipeline.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from config import settings
from src.prediction.predictor import DEFAULT_MODEL_NAME, _prepare_record
from src.training.calibrate import unwrap
from src.training.train import load_model
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ShapExplanation:
    model_name: str
    base_value: float
    probability: float
    expanded_features: list[str] = field(default_factory=list)
    expanded_shap_values: list[float] = field(default_factory=list)
    original_features: list[str] = field(default_factory=list)
    original_shap_values: list[float] = field(default_factory=list)
    original_input_values: list = field(default_factory=list)

    def top_contributors(self, n: int = 5) -> list[dict]:
        pairs = list(zip(self.original_features, self.original_shap_values,
                         self.original_input_values))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return [
            {
                "feature": name,
                "shap_value": float(sv),
                "input_value": val,
                "direction": "increases risk" if sv > 0 else "decreases risk",
            }
            for name, sv, val in pairs[:n]
        ]


def _positive_class_shap(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _positive_class_base(expected_value) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        return float(expected_value[1])
    return float(expected_value)


def original_feature_from_expanded(expanded: str) -> str:
    if expanded.startswith("num__"):
        return expanded[len("num__"):]
    if expanded.startswith("cat__"):
        rest = expanded[len("cat__"):]
        matches = [c for c in settings.CATEGORICAL_FEATURES
                   if rest == c or rest.startswith(c + "_")]
        if matches:
            return max(matches, key=len)
        return rest
    return expanded


def aggregate_by_original_feature(
    expanded_names: list[str],
    expanded_values: np.ndarray,
) -> tuple[list[str], list[float]]:
    order: list[str] = []
    total: dict[str, float] = {}
    for name, val in zip(expanded_names, expanded_values):
        orig = original_feature_from_expanded(name)
        if orig not in total:
            order.append(orig)
            total[orig] = 0.0
        total[orig] += float(val)
    return order, [total[k] for k in order]


@lru_cache(maxsize=4)
def _get_explainer(model_name: str) -> tuple[object, object, list[str]]:
    """
    Cache (raw_pipeline, TreeExplainer, expanded_feature_names) per model.
    CalibratedPipeline wrappers are transparently unwrapped.
    """
    model = load_model(model_name)
    pipeline = unwrap(model)
    if "pre" not in pipeline.named_steps or "clf" not in pipeline.named_steps:
        raise ValueError(
            f"Model '{model_name}' is not a Pipeline with 'pre' and 'clf'."
        )
    clf = pipeline.named_steps["clf"]
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception as exc:
        raise RuntimeError(
            f"SHAP TreeExplainer is not compatible with the '{model_name}' "
            f"classifier ({type(clf).__name__}). Details: {exc}"
        ) from exc
    expanded_names = list(pipeline.named_steps["pre"].get_feature_names_out())
    return pipeline, explainer, expanded_names


def clear_explainer_cache() -> None:
    _get_explainer.cache_clear()


def explain_single(
    record: dict | pd.Series | pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> ShapExplanation:
    pipeline, explainer, expanded_names = _get_explainer(model_name)
    X_row = _prepare_record(record)
    pre = pipeline.named_steps["pre"]
    X_transformed = pre.transform(X_row)

    sv_raw = explainer.shap_values(X_transformed)
    sv = _positive_class_shap(sv_raw)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    sv_row = sv[0]

    base_value = _positive_class_base(explainer.expected_value)

    # Calibrated probability from the full model (may be calibrated or raw)
    full_model = load_model(model_name)
    probability = float(full_model.predict_proba(X_row)[0, 1])

    orig_names, orig_values = aggregate_by_original_feature(expanded_names, sv_row)
    input_values = [X_row.iloc[0][name] for name in orig_names]

    return ShapExplanation(
        model_name=model_name,
        base_value=base_value,
        probability=probability,
        expanded_features=expanded_names,
        expanded_shap_values=sv_row.tolist(),
        original_features=orig_names,
        original_shap_values=orig_values,
        original_input_values=input_values,
    )


def global_importance(
    X_sample: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    max_samples: int = 500,
    random_state: int = settings.RANDOM_SEED,
) -> pd.DataFrame:
    pipeline, explainer, expanded_names = _get_explainer(model_name)
    if len(X_sample) > max_samples:
        X_used = X_sample.sample(max_samples, random_state=random_state)
    else:
        X_used = X_sample
    X_ordered = X_used[list(settings.FEATURE_COLUMNS)]
    X_transformed = pipeline.named_steps["pre"].transform(X_ordered)

    sv_raw = explainer.shap_values(X_transformed)
    sv = _positive_class_shap(sv_raw)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    mean_abs_expanded = np.abs(sv).mean(axis=0)
    orig_names, orig_values = aggregate_by_original_feature(
        expanded_names, mean_abs_expanded,
    )
    df = pd.DataFrame(
        {"feature": orig_names, "mean_abs_shap": orig_values}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    log.info(
        "Computed global SHAP importance on %d samples; top feature: %s",
        len(X_used), df.iloc[0]["feature"],
    )
    return df