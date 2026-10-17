"""Overview dashboard page."""

import streamlit as st

from src.analytics.portfolio import (
    portfolio_summary, probability_distribution_stats,
    risk_band_counts, risk_band_default_rates,
)
from ui import components as c
from ui import charts
from ui import _cache


def render() -> None:
    c.page_title(
        "CreditLens Overview",
        "Portfolio-level risk summary, model status, and headline metrics.",
    )

    try:
        comparison = _cache.get_comparison()
        predictions = _cache.get_test_predictions()
        raw = _cache.get_raw_data()
    except FileNotFoundError as exc:
        c.error_panel(str(exc), "python -m scripts.train_models")
        return

    summary = portfolio_summary(raw)
    stats = probability_distribution_stats(predictions["probability"].values)

    best_model = comparison["pr_auc"].idxmax()
    best_roc = comparison.loc[best_model, "roc_auc"]
    best_pr = comparison.loc[best_model, "pr_auc"]

    c.kpi_row([
        ("Portfolio customers", f"{summary['n_customers']:,}", "Historical dataset"),
        ("Actual default rate", f"{summary['default_rate']:.2%}",
         f"{summary['n_defaulters']:,} defaults"),
        ("Selected model", str(best_model).replace("_", " ").title(),
         "By PR-AUC then ROC-AUC"),
        ("Test-set ROC-AUC", f"{best_roc:.3f}", f"PR-AUC {best_pr:.3f}"),
    ])

    st.markdown("")
    col_left, col_right = st.columns(2)
    with col_left:
        band_counts = risk_band_counts(predictions["probability"].values)
        st.plotly_chart(charts.risk_band_bar(band_counts),
                        use_container_width=True, config={"displayModeBar": False})
    with col_right:
        band_rates = risk_band_default_rates(
            predictions["probability"].values, predictions["y_true"].values)
        st.plotly_chart(charts.default_rate_by_band_bar(band_rates),
                        use_container_width=True, config={"displayModeBar": False})

    c.section_header(
        "Predicted probability distribution",
        "Model output on the held-out test set. The distribution reflects the "
        "class-balanced Random Forest.",
    )
    st.plotly_chart(charts.probability_histogram(predictions["probability"].values),
                    use_container_width=True, config={"displayModeBar": False})

    c.section_header("Model comparison snapshot")
    display = comparison.reset_index().rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC", "brier": "Brier",
        "n_test": "Test n", "n_positive": "Positives", "threshold": "Threshold",
    })
    display["Model"] = display["Model"].str.replace("_", " ").str.title()
    c.data_table(display, precision=4)

    c.disclaimer(
        "All values are drawn from actual training runs and the held-out test "
        "set. Predicted probabilities are model estimates, not guarantees."
    )