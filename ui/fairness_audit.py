"""Fairness audit page."""

import pandas as pd
import streamlit as st

from src.reporting.report_generator import (
    fairness_report_csv, fairness_report_json,
)
from src.fairness.audit import FAIRNESS_LIMITATION
from ui import components as c
from ui import charts, styles
from ui import _cache


def render() -> None:
    c.page_title(
        "Fairness Audit",
        "Group-level model behaviour on the held-out test set. Descriptive "
        "only — does not certify compliance or prove discrimination.",
    )

    try:
        metrics_df, diffs_df = _cache.get_fairness()
    except FileNotFoundError as exc:
        c.error_panel(str(exc), "python -m scripts.run_fairness_audit")
        return

    c.info_panel(
        "A customer's individual risk explanation and a group-level model "
        "audit answer different questions. This page is the latter.",
    )

    # ---- Pick a dimension -----------------------------------------------
    audits = sorted(metrics_df["audit"].unique())
    audit = st.selectbox("Audit dimension", audits, key="fair_audit")
    sub_metrics = metrics_df[metrics_df["audit"] == audit].copy()
    sub_diffs = diffs_df[diffs_df["audit"] == audit].copy()

    reference = sub_metrics["reference_group"].iloc[0] if len(sub_metrics) else None
    c.info_panel(
        f"Reference group: <b>{reference}</b>. Differences are computed as "
        f"group value minus reference value.",
        kind="info",
    )

    # ---- Metrics table --------------------------------------------------
    c.section_header("Group metrics")
    display_cols = [
        "group", "n", "n_positive", "n_negative", "prevalence",
        "tpr", "fpr", "precision", "npv", "f1", "accuracy", "roc_auc",
        "small_group",
    ]
    present = [col for col in display_cols if col in sub_metrics.columns]
    st.dataframe(sub_metrics[present].round(4), use_container_width=True,
                 hide_index=True)

    # ---- Metric selection & charts --------------------------------------
    c.section_header("Group comparison")
    available_metrics = [m for m in (
        "tpr", "fpr", "precision", "npv", "f1", "accuracy", "roc_auc",
    ) if m in sub_metrics.columns]
    metric = st.selectbox("Metric to compare", available_metrics, key="fair_metric")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            charts.fairness_group_bar(sub_metrics, metric, audit, reference),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col2:
        diff_chart = charts.fairness_difference_bar(sub_diffs, metric, audit)
        if diff_chart is not None:
            st.plotly_chart(diff_chart, use_container_width=True,
                            config={"displayModeBar": False})

    # ---- Differences table ----------------------------------------------
    c.section_header("Differences from reference")
    st.dataframe(sub_diffs.round(4), use_container_width=True, hide_index=True)

    # ---- Warnings -------------------------------------------------------
    warnings = []
    if "small_group" in sub_metrics.columns:
        small = sub_metrics[sub_metrics["small_group"]]["group"].tolist()
        if small:
            warnings.append(
                f"Groups with fewer than {int(30)} records: {small}. "
                "Their metrics may be unstable."
            )
    no_pos = sub_metrics[sub_metrics["n_positive"] == 0]["group"].tolist() \
        if "n_positive" in sub_metrics.columns else []
    if no_pos:
        warnings.append(
            f"Groups with no default cases: {no_pos}. TPR and precision are "
            "undefined for these groups."
        )
    for w in warnings:
        c.info_panel(w, kind="warning")

    # ---- Downloads ------------------------------------------------------
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download metrics CSV",
            data=fairness_report_csv(metrics_df, diffs_df),
            file_name="creditlens_fairness_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download fairness JSON",
            data=fairness_report_json(metrics_df, diffs_df, warnings,
                                     limitation_text=FAIRNESS_LIMITATION),
            file_name="creditlens_fairness_report.json",
            mime="application/json",
            use_container_width=True,
        )

    c.disclaimer(FAIRNESS_LIMITATION)