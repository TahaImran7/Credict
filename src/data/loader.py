"""
Dataset download and loading for Credict.

Handles the fact that different UCI mirrors expose the target column under
different names, and that the .xls workbook contains a legacy numbering row
above the real header row.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from config import settings
from src.utils.logging_config import get_logger
from src.utils.paths import RAW_DATA_DIR, ensure_directories

log = get_logger(__name__)


def download_dataset(dest_dir: Path = RAW_DATA_DIR, force: bool = False) -> Path:
    """
    Download the UCI .xls file to dest_dir if it is not already present.

    Parameters
    ----------
    dest_dir : Path
        Directory where the file should be saved. Defaults to data/raw/.
    force : bool
        If True, re-download even when the file already exists.

    Returns
    -------
    Path
        Absolute path to the downloaded file.
    """
    ensure_directories()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / settings.DATASET_FILENAME

    if dest.exists() and not force:
        log.info("Dataset already present at %s", dest)
        return dest

    log.info("Downloading dataset from %s", settings.DATASET_URL)
    urllib.request.urlretrieve(settings.DATASET_URL, dest)
    log.info("Saved dataset to %s", dest)
    return dest


def _canonicalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whichever target variant is present to settings.TARGET_COLUMN."""
    for candidate in settings.TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            if candidate != settings.TARGET_COLUMN:
                log.info(
                    "Renaming target column '%s' -> '%s'",
                    candidate,
                    settings.TARGET_COLUMN,
                )
                df = df.rename(columns={candidate: settings.TARGET_COLUMN})
            return df
    raise KeyError(
        f"No target column found. Expected one of "
        f"{settings.TARGET_COLUMN_CANDIDATES}, but columns are: {list(df.columns)}"
    )


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Read the raw dataset from disk with the correct header row and a
    canonicalized target column name.

    Does NOT clean or transform the data.
    """
    if path is None:
        path = RAW_DATA_DIR / settings.DATASET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Call download_dataset() first."
        )
    df = pd.read_excel(path, header=settings.DATASET_HEADER_ROW)
    df = _canonicalize_target(df)
    log.info("Loaded raw dataset with shape %s", df.shape)
    return df