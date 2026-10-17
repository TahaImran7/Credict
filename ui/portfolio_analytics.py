"""Portfolio analytics page."""

import streamlit as st

from src.analytics.portfolio import (
    default_rate_by_category, default_rate_by_pay_status,
    mean_features_by_outcome, portfolio_summary,
    probability_distribution_stats, risk_band_counts, risk_band_default_rates,
)
from ui import components as c
from ui import charts
from ui import _cache


def render() -> None:
    c.page_title(
        "Portfolio Analytics",
        "Descriptive and model-derived views across the customer base.",
    )

    try:
        raw = _cache.get_raw_data()
        preds = _cache.get_test_predictions()
    except FileNotFoundError as exc:
        c.error_panel(str(exc))
        return

    tabs = st.tabs([
        "Summary", "Predicted risk bands", "Repayment status", "Financial patterns",
    ])

    with tabs[0]:
        summary = portfolio_summary(raw)
        stats = probability_distribution_stats(preds["probability"].values)
        c.kpi_row([
            ("Historical customers", f"{summary['n_customers']:,}", None),
            ("Historical default rate", f"{summary['default_rate']:.2%}",
             f"{summary['n_defaulters']:,} defaults"),
            ("Mean predicted probability", f"{stats['mean']:.3f}", None),
            ("Median predicted probability", f"{stats['median']:.3f}", None),
        ])
        st.markdown("")
        c.info_panel(
            "Historical figures reflect the labelled UCI dataset. Model "
            "estimates are produced by the selected Random Forest on the "
            "held-out test set.",
        )
        st.markdown("")
        c.section_header("Predicted probability distribution")
        st.plotly_chart(charts.probability_histogram(preds["probability"].values),
                        use_container_width=True, config={"displayModeBar": False})

    with tabs[1]:
        band_counts = risk_band_counts(preds["probability"].values)
        band_rates = risk_band_default_rates(
            preds["probability"].values, preds["y_true"].values)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(charts.risk_band_bar(band_counts),
                            use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(charts.default_rate_by_band_bar(band_rates),
                            use_container_width=True, config={"displayModeBar": False})
        st.markdown("**Band breakdown**")
        c.data_table(band_counts, precision=4)
        st.markdown("**Observed default rate per band**")
        c.data_table(band_rates, precision=4)
        c.disclaimer(
            "Risk bands are analytical thresholds defined in the configuration "
            "file. They are not a regulatory classification."
        )

    with tabs[2]:
        pay_stats = default_rate_by_pay_status(raw)
        pay_cols = sorted(pay_stats["pay_col"].unique())
        selected = st.selectbox("Repayment status column", pay_cols, key="pa_pay")
        sub = pay_stats[pay_stats["pay_col"] == selected]
        st.plotly_chart(
            charts.category_default_rate_bar(sub, selected,
                                             f"Default rate by {selected}"),
            use_container_width=True, config={"displayModeBar": False},
        )
        c.data_table(sub, precision=4)
        c.disclaimer(
            "Rows flagged as low confidence have fewer than 30 records."
        )

    with tabs[3]:
        sub_tabs = st.tabs(["Mean by outcome", "By education", "By marital status"])
        with sub_tabs[0]:
            means = mean_features_by_outcome(raw)
            c.data_table(means, precision=2)
            c.disclaimer("Descriptive means only. No significance testing applied.")
        with sub_tabs[1]:
            edu = default_rate_by_category(raw, "EDUCATION")
            st.plotly_chart(
                charts.category_default_rate_bar(edu, "EDUCATION",
                                                 "Default rate by education"),
                use_container_width=True, config={"displayModeBar": False},
            )
            c.data_table(edu, precision=4)
        with sub_tabs[2]:
            mar = default_rate_by_category(raw, "MARRIAGE")
            st.plotly_chart(
                charts.category_default_rate_bar(mar, "MARRIAGE",
                                                 "Default rate by marital status"),
                use_container_width=True, config={"displayModeBar": False},
            )
            c.data_table(mar, precision=4)