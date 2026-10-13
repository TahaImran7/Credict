"""What-if simulator page."""

import pandas as pd
import streamlit as st

from src.prediction.predictor import DEFAULT_MODEL_NAME
from src.reporting.report_generator import whatif_report_csv
from src.simulation.what_if import (
    PERMITTED_FEATURES, simulate, validate_changes,
)
from ui import components as c
from ui import styles
from ui import _cache


# Features that make sense to vary via slider (numeric) vs. select (category)
_NUMERIC = {
    "LIMIT_BAL", "AGE",
    *{f"BILL_AMT{i}" for i in range(1, 7)},
    *{f"PAY_AMT{i}" for i in range(1, 7)},
}
_CATEGORICAL = {"SEX", "EDUCATION", "MARRIAGE"}
_PAY_STATUS = {"PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}


# ---------------------------------------------------------------------------
# Helpers — must be defined at module level, BEFORE render() uses them.
# ---------------------------------------------------------------------------

def _load_baseline() -> dict | None:
    mode = st.radio(
        "Baseline customer",
        options=["Use a manual baseline", "Load from dataset"],
        horizontal=True,
        label_visibility="collapsed",
        key="sim_source",
    )
    if mode == "Load from dataset":
        try:
            raw = _cache.get_raw_data()
        except FileNotFoundError as exc:
            c.error_panel(str(exc))
            return None
        idx = st.number_input(
            "Dataset row index", min_value=0, max_value=len(raw) - 1,
            value=0, step=1, key="sim_idx",
        )
        row = raw.iloc[int(idx)]
        return {col: row[col] for col in (
            "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
            "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
            "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        )}
    st.caption(
        "Edit the baseline record below. Only the fields you select later "
        "will change."
    )
    return c.record_input_form("sim_base")


def _change_widget(feature: str, current) -> object:
    """Return the new value widget for a selected feature."""
    if feature in _PAY_STATUS:
        return st.selectbox(
            f"New value for {feature}",
            options=list(c.PAY_STATUS_OPTIONS.keys()),
            format_func=lambda k: c.PAY_STATUS_OPTIONS[k],
            index=list(c.PAY_STATUS_OPTIONS.keys()).index(int(current)),
            key=f"sim_chg_{feature}",
        )
    if feature == "SEX":
        return st.selectbox(
            "New value for SEX",
            options=list(c.SEX_OPTIONS.keys()),
            format_func=lambda k: c.SEX_OPTIONS[k],
            index=list(c.SEX_OPTIONS.keys()).index(int(current)),
            key="sim_chg_SEX",
        )
    if feature == "EDUCATION":
        return st.selectbox(
            "New value for EDUCATION",
            options=list(c.EDUCATION_OPTIONS.keys()),
            format_func=lambda k: c.EDUCATION_OPTIONS[k],
            index=list(c.EDUCATION_OPTIONS.keys()).index(int(current)),
            key="sim_chg_EDUCATION",
        )
    if feature == "MARRIAGE":
        return st.selectbox(
            "New value for MARRIAGE",
            options=list(c.MARRIAGE_OPTIONS.keys()),
            format_func=lambda k: c.MARRIAGE_OPTIONS[k],
            index=list(c.MARRIAGE_OPTIONS.keys()).index(int(current)),
            key="sim_chg_MARRIAGE",
        )
    if feature == "LIMIT_BAL":
        return st.number_input(
            "New LIMIT_BAL", min_value=0, max_value=1_000_000,
            value=int(current), step=1000, key="sim_chg_LIMIT_BAL",
        )
    if feature == "AGE":
        return st.number_input(
            "New AGE", min_value=18, max_value=100,
            value=int(current), key="sim_chg_AGE",
        )
    if feature.startswith("BILL_AMT"):
        return st.number_input(
            f"New {feature}", min_value=-1_000_000, max_value=10_000_000,
            value=int(current), step=100, key=f"sim_chg_{feature}",
        )
    if feature.startswith("PAY_AMT"):
        return st.number_input(
            f"New {feature}", min_value=0, max_value=10_000_000,
            value=int(current), step=100, key=f"sim_chg_{feature}",
        )
    return current


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render() -> None:
    c.page_title(
        "What-if Simulator",
        "Explore how changing one or more customer inputs affects the "
        "model's estimate.",
    )

    try:
        _cache.get_model(DEFAULT_MODEL_NAME)
    except FileNotFoundError as exc:
        c.error_panel(str(exc), "python -m scripts.train_models")
        return

    baseline = _load_baseline()
    if baseline is None:
        return

    st.markdown("---")
    c.section_header("Select features to change")
    selected = st.multiselect(
        "Permitted features",
        options=list(PERMITTED_FEATURES),
        default=[],
        help="Only features relevant to the model can be varied.",
    )

    if not selected:
        c.info_panel(
            "Select at least one feature above to define a scenario."
        )
        return

    st.markdown("**New values**")
    changes: dict = {}
    for feat in selected:
        changes[feat] = _change_widget(feat, baseline[feat])

    # Validate before running
    try:
        validate_changes(changes)
    except ValueError as exc:
        c.info_panel(f"Invalid change: {exc}", kind="danger")
        return

    run = st.button("Run scenario", type="primary")

    if run:
        with st.spinner("Running scenario..."):
            try:
                result, warnings = simulate(
                    baseline, changes, model_name=DEFAULT_MODEL_NAME,
                )
                st.session_state["sim_result"] = result
                st.session_state["sim_warnings"] = warnings
            except Exception as exc:
                c.info_panel(f"Simulation failed: {exc}", kind="danger")
                return

    if "sim_result" not in st.session_state:
        return

    result = st.session_state["sim_result"]
    warnings = st.session_state.get("sim_warnings", [])

    st.markdown("---")
    c.section_header("Scenario result")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            c.kpi_card(
                "Original probability",
                f"{result.original_probability:.2%}",
                result.original_band,
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            c.kpi_card(
                "Modified probability",
                f"{result.modified_probability:.2%}",
                result.modified_band,
            ),
            unsafe_allow_html=True,
        )
    with col3:
        direction_label = {
            "increased": "Risk increased",
            "decreased": "Risk decreased",
            "negligible": "Effectively unchanged",
        }.get(result.direction, result.direction)
        st.markdown(
            c.kpi_card(
                "Difference",
                f"{result.difference_pp:+.2f} pp",
                direction_label,
            ),
            unsafe_allow_html=True,
        )

    # ---- Changes table --------------------------------------------------
    if result.changes:
        st.markdown("**Changes applied**")
        rows = [
            {"Feature": ch.feature, "Original": ch.original, "New": ch.new}
            for ch in result.changes
        ]
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    # ---- Warnings -------------------------------------------------------
    for w in warnings:
        c.info_panel(w, kind="warning")

    # ---- Disclaimer -----------------------------------------------------
    c.disclaimer(result.disclaimer)

    # ---- Download -------------------------------------------------------
    st.download_button(
        "Download scenario CSV",
        data=whatif_report_csv(result),
        file_name=(
            f"creditlens_whatif_"
            f"{result.modified_band.replace(' ', '_')}.csv"
        ),
        mime="text/csv",
    )