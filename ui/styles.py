"""
CreditLens visual design system.

Single source of truth for colours, spacing, and the injected stylesheet.
Import this module once per page (app.py calls inject() exactly once).
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = "#0F172A"
NAVY_SOFT = "#1E293B"
SLATE = "#334155"
MUTED = "#64748B"
LIGHT = "#94A3B8"
BORDER = "#E2E8F0"
BORDER_STRONG = "#CBD5E1"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F8FAFC"
SURFACE_HOVER = "#F1F5F9"

ACCENT = "#0F766E"
ACCENT_HOVER = "#14B8A6"
ACCENT_SOFT = "#CCFBF1"

SUCCESS = "#059669"
INFO = "#0284C7"
WARNING = "#D97706"
DANGER = "#B91C1C"

RISK_COLORS = {
    "Low": SUCCESS,
    "Moderate": INFO,
    "High": WARNING,
    "Very High": DANGER,
}

CHART_SEQUENCE = [
    ACCENT, "#1E40AF", "#7C3AED", WARNING, DANGER,
    "#0891B2", "#65A30D", "#DB2777",
]

FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

# ---------------------------------------------------------------------------
# Injected stylesheet
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: {FONT_STACK};
    color: {NAVY};
}}

/* Hide top header decoration but keep the sidebar toggle */
[data-testid="stHeader"] {{
    background: transparent;
    height: 0;
}}
[data-testid="stToolbar"] {{ right: 1rem; }}
[data-testid="stDecoration"] {{ display: none; }}
footer {{ visibility: hidden; }}

/* Main container breathing room */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {SURFACE_ALT};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stRadio > label {{
    display: none;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {{
    padding: 8px 12px;
    border-radius: 8px;
    transition: background 120ms ease;
    cursor: pointer;
    font-weight: 500;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {{
    background: {SURFACE_HOVER};
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] {{
    background: {ACCENT_SOFT};
    color: {ACCENT};
}}

/* Buttons */
.stButton > button, .stDownloadButton > button {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 8px 18px;
    transition: background 120ms ease, transform 80ms ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: {ACCENT_HOVER};
    color: white;
}}
.stButton > button:active, .stDownloadButton > button:active {{
    transform: translateY(1px);
}}
.stButton > button[kind="secondary"] {{
    background: {SURFACE};
    color: {NAVY};
    border: 1px solid {BORDER_STRONG};
}}
.stButton > button[kind="secondary"]:hover {{
    background: {SURFACE_HOVER};
    color: {NAVY};
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    height: 40px;
    padding: 0 16px;
    color: {MUTED};
    font-weight: 500;
    background: transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
    border-radius: 8px !important;
}}
[data-baseweb="select"] > div {{
    border-color: {BORDER_STRONG} !important;
    border-radius: 8px !important;
}}
[data-baseweb="select"] > div:focus-within {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_SOFT} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* Metrics — we don't use st.metric much, but override anyway */
[data-testid="stMetricValue"] {{
    color: {NAVY};
    font-weight: 600;
}}
[data-testid="stMetricLabel"] {{
    color: {MUTED};
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* Expander */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {SURFACE};
}}

/* Alerts — softened */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border-left-width: 4px;
}}

/* Divider */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 20px 0;
}}

/* ---------- CreditLens custom components ---------- */

.cl-section-title {{
    color: {NAVY};
    font-size: 20px;
    font-weight: 600;
    border-left: 3px solid {ACCENT};
    padding-left: 12px;
    margin: 20px 0 12px;
    line-height: 1.3;
}}
.cl-section-subtitle {{
    color: {MUTED};
    font-size: 14px;
    margin: -6px 0 16px;
    padding-left: 15px;
}}

.cl-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
}}

.cl-kpi-label {{
    color: {MUTED};
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}}
.cl-kpi-value {{
    color: {NAVY};
    font-size: 26px;
    font-weight: 600;
    line-height: 1.15;
    margin: 0;
}}
.cl-kpi-hint {{
    color: {MUTED};
    font-size: 12px;
    margin-top: 6px;
}}

.cl-band {{
    display: inline-block;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

.cl-probability-big {{
    font-size: 48px;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
    margin: 8px 0;
}}
.cl-probability-pct {{
    font-size: 24px;
    font-weight: 500;
    color: {MUTED};
}}

.cl-note {{
    color: {MUTED};
    font-size: 12px;
    line-height: 1.55;
    border-left: 3px solid {BORDER_STRONG};
    padding: 6px 12px;
    margin: 8px 0;
    background: {SURFACE_ALT};
    border-radius: 4px;
}}

.cl-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: {SURFACE_HOVER};
    color: {SLATE};
    font-size: 12px;
    font-weight: 500;
    margin-right: 6px;
}}
</style>
"""


def inject() -> None:
    """Inject the CreditLens stylesheet. Idempotent per session."""
    st.markdown(_CSS, unsafe_allow_html=True)