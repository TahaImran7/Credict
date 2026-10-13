"""
CreditLens visual design system.

Palette: dark navy sidebar + soft off-white gradient canvas + white cards.
Accent: teal (#0D9488). No emoji. No Streamlit default colours visible.

Other modules import named constants from here and reference them in inline
HTML and Plotly. Do not hard-code colours anywhere else.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Sidebar palette (dark)
# ---------------------------------------------------------------------------
SIDEBAR_BG = "#0B1729"
SIDEBAR_BG_ALT = "#101E36"
SIDEBAR_HOVER = "#16294A"
SIDEBAR_TEXT = "#E2E8F0"
SIDEBAR_TEXT_MUTED = "#94A3B8"
SIDEBAR_ACCENT = "#5EEAD4"
SIDEBAR_DIVIDER = "#1E3055"

# ---------------------------------------------------------------------------
# Main canvas
# ---------------------------------------------------------------------------
BG_TOP = "#F8FAFC"
BG_BOTTOM = "#EEF2F7"
BG_WASH_A = "rgba(13, 148, 136, 0.07)"   # teal glow
BG_WASH_B = "rgba(59, 130, 246, 0.055)"  # blue glow

SURFACE = "#FFFFFF"
SURFACE_ALT = "#F8FAFC"
SURFACE_SOFT = "#F1F5F9"
SURFACE_HOVER = "#EEF2F7"

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
NAVY = "#0B1729"           # primary text
NAVY_SOFT = "#1E293B"      # slightly softer primary
SLATE = "#334155"          # strong secondary
MUTED = "#64748B"          # muted secondary
LIGHT = "#94A3B8"          # very muted

# ---------------------------------------------------------------------------
# Borders & shadows
# ---------------------------------------------------------------------------
BORDER = "#E6EAF2"
BORDER_STRONG = "#CBD5E1"
SHADOW_SM = "0 1px 2px rgba(15, 23, 42, 0.04)"
SHADOW_MD = "0 2px 6px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.03)"

# ---------------------------------------------------------------------------
# Accent (teal)
# ---------------------------------------------------------------------------
ACCENT = "#0D9488"
ACCENT_HOVER = "#14B8A6"
ACCENT_SOFT = "#CCFBF1"
ACCENT_DEEP = "#0F766E"

# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------
SUCCESS = "#10B981"
INFO = "#3B82F6"
WARNING = "#F59E0B"
DANGER = "#DC2626"

# Soft tinted backgrounds for info panels
SUCCESS_SOFT = "rgba(16, 185, 129, 0.10)"
INFO_SOFT = "rgba(59, 130, 246, 0.10)"
WARNING_SOFT = "rgba(245, 158, 11, 0.12)"
DANGER_SOFT = "rgba(220, 38, 38, 0.10)"

# ---------------------------------------------------------------------------
# Risk bands
# ---------------------------------------------------------------------------
RISK_COLORS = {
    "Low": "#10B981",         # emerald
    "Moderate": "#3B82F6",    # azure
    "High": "#F59E0B",        # amber
    "Very High": "#DC2626",   # crimson
}

# Chart categorical sequence — reads well on white
CHART_SEQUENCE = [
    ACCENT, "#1D4ED8", "#7C3AED", "#0891B2", "#65A30D",
    "#DB2777", "#EA580C", "#4F46E5",
]

FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

# ---------------------------------------------------------------------------
# Injected stylesheet
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {{
    font-family: {FONT_STACK};
    color: {NAVY};
    -webkit-font-smoothing: antialiased;
}}

/* Main canvas gets a soft radial gradient wash so it is never flat white. */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(1200px 600px at 8% -10%, {BG_WASH_A}, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, {BG_WASH_B}, transparent 55%),
        linear-gradient(180deg, {BG_TOP} 0%, {BG_BOTTOM} 100%);
    background-attachment: fixed;
}}

/* Streamlit chrome cleanup */
[data-testid="stHeader"] {{
    background: transparent;
    height: 0;
    z-index: 5;
}}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* Main container */
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}}

/* ---------- Sidebar (dark navy) ---------- */
[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG};
    border-right: 1px solid {SIDEBAR_DIVIDER};
    color: {SIDEBAR_TEXT};
}}
[data-testid="stSidebar"] * {{
    color: {SIDEBAR_TEXT};
}}
[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {{
    display: flex;
    align-items: center;
    padding: 9px 14px;
    margin: 1px 0;
    border-radius: 8px;
    transition: background 130ms ease, color 130ms ease;
    cursor: pointer;
    font-weight: 500;
    font-size: 14px;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {{
    background: {SIDEBAR_HOVER};
}}
/* Hide the actual radio circle */
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {{
    display: none !important;
}}
/* Selected item — teal accent bar + soft background */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] {{
    background: {SIDEBAR_BG_ALT};
    box-shadow: inset 3px 0 0 {SIDEBAR_ACCENT};
    font-weight: 600;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] * {{
    color: {SIDEBAR_ACCENT} !important;
}}

/* Sidebar divider */
[data-testid="stSidebar"] hr {{
    border-top: 1px solid {SIDEBAR_DIVIDER};
    margin: 14px 0;
}}

/* Sidebar buttons (Session tools) — dark-styled secondary */
[data-testid="stSidebar"] .stButton > button {{
    background: {SIDEBAR_BG_ALT};
    color: {SIDEBAR_TEXT};
    border: 1px solid {SIDEBAR_DIVIDER};
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {SIDEBAR_HOVER};
    border-color: {SIDEBAR_ACCENT};
    color: {SIDEBAR_ACCENT};
}}

/* Sidebar expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    border: 1px solid {SIDEBAR_DIVIDER} !important;
    background: {SIDEBAR_BG_ALT};
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {{
    color: {SIDEBAR_TEXT};
}}

/* Sidebar SVGs (collapse arrow, expander icons) — light */
[data-testid="stSidebar"] svg {{
    fill: {SIDEBAR_TEXT} !important;
    color: {SIDEBAR_TEXT} !important;
}}

/* The collapse button when sidebar is expanded */
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] {{
    background: {SIDEBAR_BG_ALT} !important;
    border: 1px solid {SIDEBAR_DIVIDER} !important;
    color: {SIDEBAR_TEXT} !important;
}}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {{
    background: {SIDEBAR_HOVER} !important;
}}

/* The expand button when sidebar is collapsed (sits on main canvas) */
[data-testid="collapsedControl"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="collapsedControl"] svg {{
    fill: {NAVY} !important;
    color: {NAVY} !important;
}}

/* ---------- Buttons (main area) ---------- */
.main .stButton > button,
.main .stDownloadButton > button {{
    background: {ACCENT};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 8px 18px;
    box-shadow: {SHADOW_SM};
    transition: background 120ms ease, transform 80ms ease, box-shadow 120ms ease;
}}
.main .stButton > button:hover,
.main .stDownloadButton > button:hover {{
    background: {ACCENT_HOVER};
    color: white;
    box-shadow: {SHADOW_MD};
}}
.main .stButton > button:active,
.main .stDownloadButton > button:active {{
    transform: translateY(1px);
}}
.main .stButton > button[kind="secondary"] {{
    background: {SURFACE};
    color: {NAVY};
    border: 1px solid {BORDER_STRONG};
    box-shadow: none;
}}
.main .stButton > button[kind="secondary"]:hover {{
    background: {SURFACE_HOVER};
    color: {NAVY};
    border-color: {ACCENT};
}}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    height: 42px;
    padding: 0 18px;
    color: {MUTED};
    font-weight: 500;
    background: transparent;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {NAVY};
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT_DEEP} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* ---------- Inputs ---------- */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {{
    border-radius: 8px !important;
    background: {SURFACE} !important;
}}
[data-baseweb="select"] > div, [data-baseweb="input"] {{
    border-color: {BORDER_STRONG} !important;
    border-radius: 8px !important;
    background: {SURFACE} !important;
}}
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_SOFT} !important;
}}
label[data-testid="stWidgetLabel"] {{
    color: {SLATE};
    font-size: 13px;
    font-weight: 500;
}}

/* Radio (main area) */
.main [data-baseweb="radio"] div[role="radio"] {{ accent-color: {ACCENT}; }}

/* Multiselect chips */
[data-baseweb="tag"] {{
    background: {ACCENT_SOFT} !important;
    color: {ACCENT_DEEP} !important;
    border-radius: 6px !important;
}}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
    background: {SURFACE};
    box-shadow: {SHADOW_SM};
}}

/* ---------- Expander (main) ---------- */
.main [data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    background: {SURFACE};
    box-shadow: {SHADOW_SM};
}}
.main [data-testid="stExpander"] summary {{
    color: {NAVY};
    font-weight: 500;
}}

/* ---------- Alerts ---------- */
[data-testid="stAlert"] {{
    border-radius: 10px;
    border-left-width: 4px;
    background: {SURFACE};
}}

/* ---------- Divider ---------- */
hr {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 20px 0;
}}

/* ---------- Plotly chart card feel ---------- */
.main .js-plotly-plot {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 8px;
    box-shadow: {SHADOW_SM};
}}

/* ==============================================================
   CreditLens custom components
   ============================================================== */

.cl-section-title {{
    color: {NAVY};
    font-size: 19px;
    font-weight: 600;
    border-left: 3px solid {ACCENT};
    padding-left: 12px;
    margin: 24px 0 10px;
    line-height: 1.3;
    letter-spacing: -0.01em;
}}
.cl-section-subtitle {{
    color: {MUTED};
    font-size: 13.5px;
    margin: -4px 0 16px;
    padding-left: 15px;
    line-height: 1.55;
}}

.cl-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: {SHADOW_MD};
}}

.cl-kpi-label {{
    color: {MUTED};
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
}}
.cl-kpi-value {{
    color: {NAVY};
    font-size: 26px;
    font-weight: 700;
    line-height: 1.15;
    margin: 0;
    letter-spacing: -0.02em;
}}
.cl-kpi-hint {{
    color: {MUTED};
    font-size: 12px;
    margin-top: 6px;
    line-height: 1.4;
}}

.cl-band {{
    display: inline-block;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 700;
    color: white;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

.cl-probability-big {{
    font-size: 52px;
    font-weight: 700;
    color: {NAVY};
    line-height: 1;
    margin: 10px 0 4px;
    letter-spacing: -0.03em;
}}
.cl-probability-pct {{
    font-size: 26px;
    font-weight: 500;
    color: {MUTED};
    margin-left: 2px;
}}

.cl-note {{
    color: {SLATE};
    font-size: 12.5px;
    line-height: 1.6;
    border-left: 3px solid {BORDER_STRONG};
    padding: 10px 14px;
    margin: 10px 0;
    background: {SURFACE_ALT};
    border-radius: 6px;
}}

.cl-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: {SURFACE_SOFT};
    color: {SLATE};
    font-size: 12px;
    font-weight: 500;
    margin-right: 6px;
}}

/* Scrollbar (subtle, matches theme) */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER_STRONG}; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: {LIGHT}; }}
</style>
"""


def inject() -> None:
    """Inject the CreditLens stylesheet. Safe to call once per session."""
    st.markdown(_CSS, unsafe_allow_html=True)