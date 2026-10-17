"""Reusable UI components for the CreditLens dashboard."""

from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from ui import styles


def get_palette() -> dict:
    theme = st.session_state.get("theme", "light")
    return styles.get_palette(theme)


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="cl-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="cl-section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def page_title(title: str, description: str) -> None:
    p = get_palette()
    st.markdown(
        f"""
        <div style="margin-bottom: 24px;">
            <h1 style="font-size: 30px; font-weight: 700; color: {p['NAVY']};
                       margin: 0 0 6px 0; letter-spacing: -0.02em;">{title}</h1>
            <p style="color: {p['MUTED']}; font-size: 15px; margin: 0;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, hint: str | None = None) -> str:
    p = get_palette()
    hint_html = f'<div class="cl-kpi-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="cl-card" style="border-top: 2px solid {p["ACCENT_SOFT"]};">'
        f'<div class="cl-kpi-label">{label}</div>'
        f'<div class="cl-kpi-value">{value}</div>'
        f'{hint_html}'
        f'</div>'
    )


def kpi_row(items: Iterable[tuple[str, str, str | None]]) -> None:
    items = list(items)
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        with col:
            st.markdown(kpi_card(label, value, hint), unsafe_allow_html=True)


def risk_band_badge(band: str) -> str:
    p = get_palette()
    colour = {
        "Low": p["RISK_LOW"], "Moderate": p["RISK_MODERATE"],
        "High": p["RISK_HIGH"], "Very High": p["RISK_VERY_HIGH"],
    }.get(band, p["MUTED"])
    return f'<span class="cl-band" style="background:{colour};">{band}</span>'


def probability_block(probability: float, band: str) -> str:
    pct = probability * 100
    return (
        f'<div class="cl-card" style="text-align:center; padding: 28px 22px;">'
        f'<div class="cl-kpi-label">Estimated default probability (next month)</div>'
        f'<div class="cl-probability-big">{pct:.1f}'
        f'<span class="cl-probability-pct">%</span></div>'
        f'<div style="margin-top: 10px;">{risk_band_badge(band)}</div>'
        f'</div>'
    )


def info_panel(text: str, kind: str = "info") -> None:
    p = get_palette()
    tints = {
        "info": (p["INFO"], p["INFO_SOFT"]),
        "success": (p["SUCCESS"], p["SUCCESS_SOFT"]),
        "warning": (p["WARNING"], p["WARNING_SOFT"]),
        "danger": (p["DANGER"], p["DANGER_SOFT"]),
    }
    colour, bg = tints.get(kind, tints["info"])
    st.markdown(
        f'<div style="border-left: 4px solid {colour}; background: {bg}; '
        f'padding: 12px 16px; border-radius: 8px; color: {p["NAVY"]}; '
        f'font-size: 14px; line-height: 1.55; margin: 8px 0;">{text}</div>',
        unsafe_allow_html=True,
    )


def disclaimer(text: str) -> None:
    st.markdown(f'<div class="cl-note">{text}</div>', unsafe_allow_html=True)


def empty_state(title: str, message: str, command: str | None = None) -> None:
    p = get_palette()
    st.markdown(
        f"""
        <div class="cl-card" style="text-align: center; padding: 40px 24px;">
            <div style="font-size: 18px; font-weight: 600; color: {p['NAVY']};
                        margin-bottom: 8px;">{title}</div>
            <div style="color: {p['MUTED']}; font-size: 14px; margin-bottom: 12px;">{message}</div>
            {f'<code style="background:{p["SURFACE_HOVER"]}; padding:6px 12px; '
             f'border-radius:6px; color:{p["SLATE"]}; font-size:13px;">{command}</code>'
             if command else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def error_panel(message: str, command: str | None = None) -> None:
    empty_state("Could not load data", message, command)


def _format_cell(val, precision: int) -> str:
    """Format a single cell for HTML table display."""
    if pd.isna(val):
        return "—"
    if isinstance(val, (bool, np.bool_)):
        return "Yes" if bool(val) else "No"
    if isinstance(val, (int, np.integer)):
        return f"{int(val):,}"
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if f == int(f) and abs(f) < 1e12:
            return f"{int(f):,}"
        return f"{f:.{precision}f}"
    return str(val)


def data_table(df: pd.DataFrame, precision: int = 4) -> None:
    """
    Display-only table rendered as HTML so text colours, borders, and
    backgrounds work identically in both light and dark themes.

    Replaces st.dataframe (canvas, not CSS-styleable) and st.table (no
    hide_index argument in the current Streamlit version).
    """
    if df.empty:
        st.info("No data to display.")
        return

    headers = "".join(f"<th>{col}</th>" for col in df.columns)
    rows_html = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{_format_cell(val, precision)}</td>" for val in row
        )
        rows_html.append(f"<tr>{cells}</tr>")

    st.markdown(
        f'<div class="cl-table-wrap">'
        f'<table class="cl-table">'
        f'<thead><tr>{headers}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_table(df: pd.DataFrame, precision: int = 4) -> None:
    """Alias for data_table — kept for backward compatibility."""
    data_table(df, precision=precision)


PAY_STATUS_OPTIONS = {
    -2: "No consumption (-2)", -1: "Paid in full (-1)",
     0: "Revolving credit (0)", 1: "1 month delay (1)",
     2: "2 months delay (2)", 3: "3 months delay (3)",
     4: "4 months delay (4)", 5: "5 months delay (5)",
     6: "6 months delay (6)", 7: "7 months delay (7)",
     8: "8 months delay (8)", 9: "9+ months delay (9)",
}
SEX_OPTIONS = {1: "Male", 2: "Female"}
EDUCATION_OPTIONS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Others"}
MARRIAGE_OPTIONS = {1: "Married", 2: "Single", 3: "Others"}


def _default_record() -> dict:
    return {
        "LIMIT_BAL": 50000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 35,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 1000, "BILL_AMT2": 1000, "BILL_AMT3": 1000,
        "BILL_AMT4": 1000, "BILL_AMT5": 1000, "BILL_AMT6": 1000,
        "PAY_AMT1": 500, "PAY_AMT2": 500, "PAY_AMT3": 500,
        "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500,
    }


def record_input_form(key_prefix: str, defaults: dict | None = None,
                      disabled_features: set[str] | None = None) -> dict:
    if defaults is None:
        defaults = _default_record()
    disabled_features = disabled_features or set()

    result: dict = {}
    tabs = st.tabs(["Personal & Credit", "Repayment Status", "Bill Amounts", "Payment Amounts"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            result["LIMIT_BAL"] = st.number_input(
                "Credit limit (NT$)", min_value=0, max_value=1_000_000,
                value=int(defaults.get("LIMIT_BAL", 50000)), step=1000,
                key=f"{key_prefix}_LIMIT_BAL", disabled="LIMIT_BAL" in disabled_features,
            )
            result["SEX"] = st.selectbox(
                "Sex (as recorded)", options=list(SEX_OPTIONS.keys()),
                format_func=lambda k: SEX_OPTIONS[k],
                index=list(SEX_OPTIONS.keys()).index(int(defaults.get("SEX", 1))),
                key=f"{key_prefix}_SEX", disabled="SEX" in disabled_features,
            )
            result["EDUCATION"] = st.selectbox(
                "Education", options=list(EDUCATION_OPTIONS.keys()),
                format_func=lambda k: EDUCATION_OPTIONS[k],
                index=list(EDUCATION_OPTIONS.keys()).index(int(defaults.get("EDUCATION", 2))),
                key=f"{key_prefix}_EDUCATION", disabled="EDUCATION" in disabled_features,
            )
        with c2:
            result["AGE"] = st.number_input(
                "Age (years)", min_value=18, max_value=100,
                value=int(defaults.get("AGE", 35)),
                key=f"{key_prefix}_AGE", disabled="AGE" in disabled_features,
            )
            result["MARRIAGE"] = st.selectbox(
                "Marital status", options=list(MARRIAGE_OPTIONS.keys()),
                format_func=lambda k: MARRIAGE_OPTIONS[k],
                index=list(MARRIAGE_OPTIONS.keys()).index(int(defaults.get("MARRIAGE", 2))),
                key=f"{key_prefix}_MARRIAGE", disabled="MARRIAGE" in disabled_features,
            )

    with tabs[1]:
        st.caption("Repayment status codes reflect the number of months a payment was delayed.")
        c1, c2 = st.columns(2)
        with c1:
            for col in ("PAY_0", "PAY_2", "PAY_3"):
                result[col] = st.selectbox(
                    col, options=list(PAY_STATUS_OPTIONS.keys()),
                    format_func=lambda k: PAY_STATUS_OPTIONS[k],
                    index=list(PAY_STATUS_OPTIONS.keys()).index(int(defaults.get(col, 0))),
                    key=f"{key_prefix}_{col}", disabled=col in disabled_features,
                )
        with c2:
            for col in ("PAY_4", "PAY_5", "PAY_6"):
                result[col] = st.selectbox(
                    col, options=list(PAY_STATUS_OPTIONS.keys()),
                    format_func=lambda k: PAY_STATUS_OPTIONS[k],
                    index=list(PAY_STATUS_OPTIONS.keys()).index(int(defaults.get(col, 0))),
                    key=f"{key_prefix}_{col}", disabled=col in disabled_features,
                )

    with tabs[2]:
        st.caption("Bill statement amount for each of the six historical months (NT$).")
        c1, c2 = st.columns(2)
        with c1:
            for col in ("BILL_AMT1", "BILL_AMT2", "BILL_AMT3"):
                result[col] = st.number_input(
                    col, min_value=-1_000_000, max_value=10_000_000,
                    value=int(defaults.get(col, 0)), step=100,
                    key=f"{key_prefix}_{col}", disabled=col in disabled_features,
                )
        with c2:
            for col in ("BILL_AMT4", "BILL_AMT5", "BILL_AMT6"):
                result[col] = st.number_input(
                    col, min_value=-1_000_000, max_value=10_000_000,
                    value=int(defaults.get(col, 0)), step=100,
                    key=f"{key_prefix}_{col}", disabled=col in disabled_features,
                )

    with tabs[3]:
        st.caption("Amount paid toward the previous statement (NT$).")
        c1, c2 = st.columns(2)
        with c1:
            for col in ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3"):
                result[col] = st.number_input(
                    col, min_value=0, max_value=10_000_000,
                    value=int(defaults.get(col, 0)), step=100,
                    key=f"{key_prefix}_{col}", disabled=col in disabled_features,
                )
        with c2:
            for col in ("PAY_AMT4", "PAY_AMT5", "PAY_AMT6"):
                result[col] = st.number_input(
                    col, min_value=0, max_value=10_000_000,
                    value=int(defaults.get(col, 0)), step=100,
                    key=f"{key_prefix}_{col}", disabled=col in disabled_features,
                )

    return result