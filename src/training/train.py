"""
Model training for CreditLens.

Two pipelines are built and tuned with cross-validation on the training set:
  - Logistic Regression (interpretable baseline)
  - Random Forest (advanced tree-based model)

The full pipeline (preprocessor + classifier) is saved as a single joblib
artefact so that prediction, explanation, and simulation call the same
object and cannot drift apart.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from config import settings
from src.data.preprocessing import build_preprocessor
from src.utils.logging_config import get_logger
from src.utils.paths import EVALUATION_DIR, SAVED_MODELS_DIR

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=settings.RANDOM_SEED,
        )),
    ])


def build_random_forest_pipeline() -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=settings.RANDOM_SEED,
        )),
    ])


# ---------------------------------------------------------------------------
# Hyperparameter grids (kept small per spec 4.2)
# ---------------------------------------------------------------------------

LOGISTIC_GRID: dict[str, list[Any]] = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__class_weight": [None, "balanced"],
}

RANDOM_FOREST_GRID: dict[str, list[Any]] = {
    "clf__max_depth": [None, 12],
    "clf__min_samples_leaf": [1, 5],
    "clf__class_weight": ["balanced"],  # fixed; imbalance matters here
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=settings.RANDOM_SEED,
    )


def _tune(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=_cv_splitter(),
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    log.info("Tuning %s with %d candidates...", model_name, _grid_size(param_grid))
    grid.fit(X_train, y_train)
    log.info(
        "%s best params: %s | best CV ROC-AUC: %.4f",
        model_name,
        grid.best_params_,
        grid.best_score_,
    )
    return grid


def _grid_size(grid: dict[str, list[Any]]) -> int:
    n = 1
    for values in grid.values():
        n *= len(values)
    return n


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    return _tune(
        build_logistic_pipeline(),
        LOGISTIC_GRID,
        X_train,
        y_train,
        "logistic_regression",
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    return _tune(
        build_random_forest_pipeline(),
        RANDOM_FOREST_GRID,
        X_train,
        y_train,
        "random_forest",
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def model_path(name: str) -> Path:
    return SAVED_MODELS_DIR / f"{name}.joblib"


def save_model(pipeline: Pipeline, name: str) -> Path:
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(name)
    joblib.dump(pipeline, path)
    log.info("Saved model '%s' to %s", name, path)
    return path


def load_model(name: str) -> Pipeline:
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}. Train it first."
        )
    return joblib.load(path)


def save_cv_results(grid: GridSearchCV, name: str) -> Path:
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATION_DIR / f"{name}_cv_results.csv"
    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(path, index=False)
    log.info("Saved CV results to %s", path)
    return path