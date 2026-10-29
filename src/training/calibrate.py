"""
Probability calibration for CreditLens.

The Random Forest is trained with balanced class weights, which improves
discrimination on imbalanced data but inflates predicted probabilities
(mean predicted ~0.35 vs actual base rate ~0.22). Calibration corrects this
by learning a monotonic mapping from the model's internal scores to
observed frequencies.

Approach: isotonic regression fitted on 5-fold out-of-fold predictions.
- Discrimination is largely preserved (ROC-AUC drop is small, due to ties).
- Probabilities align with observed rates (Brier score improves).
- The raw 0.5 threshold is mapped to the calibrated threshold that produces
  the same classification rate, so confusion matrix and downstream
  classification metrics are unchanged.

Files:
  models/saved/random_forest.joblib             (raw, used for SHAP)
  models/saved/random_forest_calibrated.joblib  (calibrated, for prediction)
"""

import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from config import settings
from src.utils.logging_config import get_logger
from src.utils.paths import SAVED_MODELS_DIR

log = get_logger(__name__)


class CalibratedPipeline(BaseEstimator, ClassifierMixin):
    """
    A fitted sklearn Pipeline wrapped with an isotonic calibrator.

    Attributes
    ----------
    pipeline : fitted Pipeline (raw). Used internally and by SHAP.
    calibrator : fitted IsotonicRegression
    threshold : calibrated probability threshold equivalent to raw 0.5.
                Preserves the original confusion matrix at threshold 0.5
                on the raw model.
    """

    def __init__(self, pipeline, calibrator, threshold):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = float(threshold)

    def predict_proba(self, X):
        raw_full = self.pipeline.predict_proba(X)
        if raw_full.ndim != 2 or raw_full.shape[1] != 2:
            raise ValueError(
                f"Underlying pipeline returned predict_proba with shape "
                f"{raw_full.shape}; expected (n, 2). The pipeline may have "
                f"been trained on single-class data."
            )
        raw = raw_full[:, 1]
        cal = self.calibrator.predict(raw.reshape(-1, 1))
        cal = np.clip(np.asarray(cal, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)


def unwrap(model):
    """Return the raw sklearn Pipeline from a CalibratedPipeline if needed."""
    if isinstance(model, CalibratedPipeline):
        return model.pipeline
    return model


def calibrate_pipeline(
    fitted_pipeline: Pipeline,
    X_train,
    y_train,
    n_folds: int = 5,
    seed: int = settings.RANDOM_SEED,
) -> CalibratedPipeline:
    """
    Fit an isotonic calibrator on k-fold out-of-fold predictions.

    Uses StratifiedKFold so folds preserve class balance. No leakage: each
    fold's predictions come from a model that did not see that fold.
    """
    y_arr = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)

    # --- Defensive checks -------------------------------------------
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        raise ValueError(
            f"calibrate_pipeline requires at least 2 classes in y, "
            f"got {len(classes)}. Unique values: {classes.tolist()}"
        )
    if counts.min() < n_folds:
        raise ValueError(
            f"Each class must have at least n_folds={n_folds} samples for "
            f"StratifiedKFold. Class counts: "
            f"{dict(zip(classes.tolist(), counts.tolist()))}"
        )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X_train))

    log.info("Computing %d-fold out-of-fold predictions for calibration...", n_folds)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_arr), start=1):
        X_tr = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
        y_tr = y_arr[train_idx]
        X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]

        fold_pipeline = clone(fitted_pipeline)
        fold_pipeline.fit(X_tr, y_tr)

        fold_proba_full = fold_pipeline.predict_proba(X_val)
        if fold_proba_full.ndim != 2 or fold_proba_full.shape[1] != 2:
            raise ValueError(
                f"Fold {fold_idx} trained on single-class data — "
                f"predict_proba shape is {fold_proba_full.shape}."
            )
        oof_preds[val_idx] = fold_proba_full[:, 1]
        log.info("  Fold %d/%d complete", fold_idx, n_folds)

    log.info("Fitting isotonic calibrator on out-of-fold predictions...")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_preds.reshape(-1, 1), y_arr)

    threshold_cal = float(calibrator.predict([[0.5]])[0])
    log.info("Calibrated threshold for raw 0.5 is %.4f", threshold_cal)

    return CalibratedPipeline(
        pipeline=fitted_pipeline,
        calibrator=calibrator,
        threshold=threshold_cal,
    )


def save_calibrated_model(model: CalibratedPipeline,
                          name: str = "random_forest_calibrated"):
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVED_MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    log.info("Saved calibrated model to %s", path)
    return path


def load_calibrated_model(name: str = "random_forest_calibrated") -> CalibratedPipeline:
    path = SAVED_MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrated model not found at {path}. "
            "Run: python -m scripts.train_models"
        )
    return joblib.load(path)