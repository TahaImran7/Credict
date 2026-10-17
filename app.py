"""
CreditLens — main application entry point.
Run: streamlit run app.py
"""

import streamlit as st

from ui import styles
from ui import _cache
from ui import (
    overview, risk_assessment, simulator, model_performance,
    explainability, portfolio_analytics, fairness_audit, reports,
)

st.set_page_config(
    page_title="CreditLens",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme state ----------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"


def _toggle_theme() -> None:
    st.session_state.theme = (
        "dark" if st.session_state.theme == "light" else "light"
    )


# --- Inject the correct stylesheet ---------------------------------------
styles.inject(st.session_state.theme)

# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    p = styles.get_palette(st.session_state.theme)

    st.markdown(
        f"""
        <div style="padding: 10px 6px 22px 6px;">
            <div style="
                font-size: 22px;
                font-weight: 700;
                color: {p['SIDEBAR_TEXT']};
                letter-spacing: -0.02em;
            ">CreditLens</div>
            <div style="
                color: {p['SIDEBAR_TEXT_MUTED']};
                font-size: 12px;
                margin-top: 4px;
                letter-spacing: 0.02em;
            ">Explainable AI Credit Risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Theme toggle
    theme_label = "Switch to Dark Mode" if st.session_state.theme == "light" \
        else "Switch to Light Mode"
    st.button(theme_label, on_click=_toggle_theme, use_container_width=True)

    st.markdown("---")

    PAGES = {
        "Overview": overview.render,
        "Risk Assessment": risk_assessment.render,
        "What-if Simulator": simulator.render,
        "Model Performance": model_performance.render,
        "Explainability": explainability.render,
        "Portfolio Analytics": portfolio_analytics.render,
        "Fairness Audit": fairness_audit.render,
        "Reports": reports.render,
    }

    page = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")

    with st.expander("Session"):
        if st.button("Clear cached data", use_container_width=True):
            _cache.clear_all_caches()
            st.success("Caches cleared.")
        st.caption(
            "Clears in-memory caches. The application does not retrain "
            "models automatically."
        )

    st.caption("Academic decision-support prototype. Not a lending authority.")

# --- Dispatch -------------------------------------------------------------
try:
    PAGES[page]()
except FileNotFoundError as exc:
    st.error(f"Missing artefact: {exc}")
    st.info(
        "Run the phase scripts in this order:\n"
        "1. `python -c \"from src.data.loader import download_dataset; download_dataset()\"`\n"
        "2. `python -m scripts.prepare_data`\n"
        "3. `python -m scripts.train_models`\n"
        "4. `python -m scripts.build_dashboard_artifacts`\n"
        "5. `python -m scripts.run_fairness_audit`"
    )
except Exception as exc:
    st.error(f"Unexpected error: {exc}")
    with st.expander("Details"):
        import traceback
        st.code(traceback.format_exc())