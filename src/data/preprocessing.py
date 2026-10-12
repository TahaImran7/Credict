"""
Cleaning, splitting, and preprocessing for CreditLens.

The preprocessor is a single fitted ColumnTransformer that is shared by
training, prediction, and what-if simulation. Fitting is done on training
data only to prevent leakage.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import settings
from src.utils.logging_config import get_logger
from src.utils.paths import PROCESSED_DATA_DIR, SAVED_MODELS_DIR

log = get_logger(__name__)

PREPROCESSOR_PATH: Path = SAVED_MODELS_DIR / "preprocessor.joblib"
X_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "X_train.parquet"
X_TEST_PATH: Path = PROCESSED_DATA_DIR / "X_test.parquet"
Y_TRAIN_PATH: Path = PROCESSED_DATA_DIR / "y_train.parquet"
Y_TEST_PATH: Path = PROCESSED_DATA_DIR / "y_test.parquet"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply documented category merges.

    - EDUCATION: 0, 5, 6 -> 4 (others)
    - MARRIAGE:  0       -> 3 (others)

    The raw file on disk is never modified.
    """
    df = df.copy()
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace(settings.EDUCATION_MAP)
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace(settings.MARRIAGE_MAP)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X with features in the fixed canonical order, and y as int."""
    missing = [c for c in settings.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    if settings.TARGET_COLUMN not in df.columns:
        raise KeyError(f"Missing target column: {settings.TARGET_COLUMN}")
    X = df[list(settings.FEATURE_COLUMNS)].copy()
    y = df[settings.TARGET_COLUMN].astype(int).copy()
    return X, y


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = settings.TEST_SIZE,
    seed: int = settings.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split with a fixed seed."""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Unfitted preprocessor:
      - StandardScaler for numeric features
      - OneHotEncoder(handle_unknown='ignore') for categorical features
    """
    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(settings.NUMERIC_FEATURES)),
            ("cat", categorical_pipe, list(settings.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_preprocessor(X_train: pd.DataFrame, save: bool = True) -> ColumnTransformer:
    """Fit on training data only, optionally persist to disk."""
    pre = build_preprocessor()
    pre.fit(X_train)
    if save:
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pre, PREPROCESSOR_PATH)
        log.info("Saved fitted preprocessor to %s", PREPROCESSOR_PATH)
    return pre


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {path}. Run fit_preprocessor() first."
        )
    return joblib.load(path)


def save_splits(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(X_TRAIN_PATH, index=False)
    X_test.to_parquet(X_TEST_PATH, index=False)
    y_train.to_frame().to_parquet(Y_TRAIN_PATH, index=False)
    y_test.to_frame().to_parquet(Y_TEST_PATH, index=False)
    log.info("Saved train/test splits to %s", PROCESSED_DATA_DIR)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_parquet(X_TRAIN_PATH)
    X_test = pd.read_parquet(X_TEST_PATH)
    y_train = pd.read_parquet(Y_TRAIN_PATH)[settings.TARGET_COLUMN]
    y_test = pd.read_parquet(Y_TEST_PATH)[settings.TARGET_COLUMN]
    return X_train, X_test, y_train, y_test