"""
One-shot Phase 2 pipeline:
  download -> load -> validate -> clean -> split -> fit preprocessor -> save artefacts.

Run from the project root:
    python -m scripts.prepare_data
"""

from config import settings
from src.data.loader import download_dataset, load_raw_dataset
from src.data import preprocessing as pp
from src.data.validation import validate_dataset
from src.utils.logging_config import get_logger

log = get_logger(__name__)


def main() -> None:
    log.info("=== Credict Phase 2: dataset preparation ===")

    path = download_dataset()
    log.info("Using dataset file: %s", path)

    df = load_raw_dataset()
    report = validate_dataset(df)
    print(report.summary())
    if not report.ok:
        raise SystemExit("Dataset validation failed; see report above.")

    df_clean = pp.clean_dataset(df)
    X, y = pp.split_features_target(df_clean)
    X_tr, X_te, y_tr, y_te = pp.make_train_test_split(X, y)

    log.info("Split sizes: train=%d, test=%d", len(X_tr), len(X_te))
    log.info(
        "Positive-class rate: train=%.4f, test=%.4f",
        float(y_tr.mean()), float(y_te.mean()),
    )

    pp.fit_preprocessor(X_tr, save=True)
    pp.save_splits(X_tr, X_te, y_tr, y_te)

    log.info("Phase 2 preparation complete.")


if __name__ == "__main__":
    main()