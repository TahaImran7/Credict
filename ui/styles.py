"""
CreditLens visual design system — light and dark themes.
"""

import streamlit as st

LIGHT = {
    "SIDEBAR_BG": "#0B1729", "SIDEBAR_BG_ALT": "#101E36", "SIDEBAR_HOVER": "#16294A",
    "SIDEBAR_TEXT": "#E2E8F0", "SIDEBAR_TEXT_MUTED": "#94A3B8",
    "SIDEBAR_ACCENT": "#5EEAD4", "SIDEBAR_DIVIDER": "#1E3055",
    "BG_TOP": "#F8FAFC", "BG_BOTTOM": "#EEF2F7",
    "BG_WASH_A": "rgba(13, 148, 136, 0.07)", "BG_WASH_B": "rgba(59, 130, 246, 0.055)",
    "SURFACE": "#FFFFFF", "SURFACE_ALT": "#F8FAFC",
    "SURFACE_SOFT": "#F1F5F9", "SURFACE_HOVER": "#EEF2F7",
    "NAVY": "#0B1729", "NAVY_SOFT": "#1E293B", "SLATE": "#334155",
    "MUTED": "#64748B", "LIGHT": "#94A3B8",
    "BORDER": "#E6EAF2", "BORDER_STRONG": "#CBD5E1",
    "ACCENT": "#0D9488", "ACCENT_HOVER": "#14B8A6",
    "ACCENT_SOFT": "#CCFBF1", "ACCENT_DEEP": "#0F766E",
    "SUCCESS": "#10B981", "INFO": "#3B82F6", "WARNING": "#F59E0B", "DANGER": "#DC2626",
    "SUCCESS_SOFT": "rgba(16, 185, 129, 0.10)",
    "INFO_SOFT": "rgba(59, 130, 246, 0.10)",
    "WARNING_SOFT": "rgba(245, 158, 11, 0.12)",
    "DANGER_SOFT": "rgba(220, 38, 38, 0.10)",
    "RISK_LOW": "#10B981", "RISK_MODERATE": "#3B82F6",
    "RISK_HIGH": "#F59E0B", "RISK_VERY_HIGH": "#DC2626",
}

DARK = {
    "SIDEBAR_BG": "#050A14", "SIDEBAR_BG_ALT": "#0A1628", "SIDEBAR_HOVER": "#122040",
    "SIDEBAR_TEXT": "#E2E8F0", "SIDEBAR_TEXT_MUTED": "#8899B0",
    "SIDEBAR_ACCENT": "#5EEAD4", "SIDEBAR_DIVIDER": "#152340",
    "BG_TOP": "#0A1220", "BG_BOTTOM": "#0E1526",
    "BG_WASH_A": "rgba(94, 234, 212, 0.04)", "BG_WASH_B": "rgba(96, 165, 250, 0.035)",
    "SURFACE": "#111C2E", "SURFACE_ALT": "#152238",
    "SURFACE_SOFT": "#1A2A44", "SURFACE_HOVER": "#1E3050",
    "NAVY": "#F1F5F9", "NAVY_SOFT": "#E2E8F0", "SLATE": "#CBD5E1",
    "MUTED": "#94A3B8", "LIGHT": "#64748B",
    "BORDER": "#1E3050", "BORDER_STRONG": "#2A4060",
    "ACCENT": "#14B8A6", "ACCENT_HOVER": "#2DD4BF",
    "ACCENT_SOFT": "rgba(20, 184, 166, 0.15)", "ACCENT_DEEP": "#5EEAD4",
    "SUCCESS": "#34D399", "INFO": "#60A5FA", "WARNING": "#FBBF24", "DANGER": "#F87171",
    "SUCCESS_SOFT": "rgba(52, 211, 153, 0.12)",
    "INFO_SOFT": "rgba(96, 165, 250, 0.12)",
    "WARNING_SOFT": "rgba(251, 191, 36, 0.14)",
    "DANGER_SOFT": "rgba(248, 113, 113, 0.12)",
    "RISK_LOW": "#34D399", "RISK_MODERATE": "#60A5FA",
    "RISK_HIGH": "#FBBF24", "RISK_VERY_HIGH": "#F87171",
}


def get_palette(theme: str = "light") -> dict:
    return DARK if theme == "dark" else LIGHT


SIDEBAR_BG = LIGHT["SIDEBAR_BG"]
SIDEBAR_BG_ALT = LIGHT["SIDEBAR_BG_ALT"]
SIDEBAR_HOVER = LIGHT["SIDEBAR_HOVER"]
SIDEBAR_TEXT = LIGHT["SIDEBAR_TEXT"]
SIDEBAR_TEXT_MUTED = LIGHT["SIDEBAR_TEXT_MUTED"]
SIDEBAR_ACCENT = LIGHT["SIDEBAR_ACCENT"]
SIDEBAR_DIVIDER = LIGHT["SIDEBAR_DIVIDER"]
BG_TOP = LIGHT["BG_TOP"]; BG_BOTTOM = LIGHT["BG_BOTTOM"]
BG_WASH_A = LIGHT["BG_WASH_A"]; BG_WASH_B = LIGHT["BG_WASH_B"]
SURFACE = LIGHT["SURFACE"]; SURFACE_ALT = LIGHT["SURFACE_ALT"]
SURFACE_SOFT = LIGHT["SURFACE_SOFT"]; SURFACE_HOVER = LIGHT["SURFACE_HOVER"]
NAVY = LIGHT["NAVY"]; NAVY_SOFT = LIGHT["NAVY_SOFT"]
SLATE = LIGHT["SLATE"]; MUTED = LIGHT["MUTED"]; LIGHT_TEXT = LIGHT["LIGHT"]
BORDER = LIGHT["BORDER"]; BORDER_STRONG = LIGHT["BORDER_STRONG"]
ACCENT = LIGHT["ACCENT"]; ACCENT_HOVER = LIGHT["ACCENT_HOVER"]
ACCENT_SOFT = LIGHT["ACCENT_SOFT"]; ACCENT_DEEP = LIGHT["ACCENT_DEEP"]
SUCCESS = LIGHT["SUCCESS"]; INFO = LIGHT["INFO"]
WARNING = LIGHT["WARNING"]; DANGER = LIGHT["DANGER"]
SUCCESS_SOFT = LIGHT["SUCCESS_SOFT"]; INFO_SOFT = LIGHT["INFO_SOFT"]
WARNING_SOFT = LIGHT["WARNING_SOFT"]; DANGER_SOFT = LIGHT["DANGER_SOFT"]
RISK_COLORS = {
    "Low": LIGHT["RISK_LOW"], "Moderate": LIGHT["RISK_MODERATE"],
    "High": LIGHT["RISK_HIGH"], "Very High": LIGHT["RISK_VERY_HIGH"],
}
CHART_SEQUENCE = [
    ACCENT, "#1D4ED8", "#7C3AED", "#0891B2", "#65A30D",
    "#DB2777", "#EA580C", "#4F46E5",
]
FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"


def _build_css(p: dict) -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: {FONT_STACK};
    color: {p['NAVY']};
    -webkit-font-smoothing: antialiased;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(1200px 600px at 8% -10%, {p['BG_WASH_A']}, transparent 60%),
        radial-gradient(900px 500px at 100% 0%, {p['BG_WASH_B']}, transparent 55%),
        linear-gradient(180deg, {p['BG_TOP']} 0%, {p['BG_BOTTOM']} 100%);
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{ background: transparent; height: 0; z-index: 5; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

.main .block-container {{
    padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;
}}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {{
    background: {p['SIDEBAR_BG']};
    border-right: 1px solid {p['SIDEBAR_DIVIDER']};
}}
[data-testid="stSidebar"] * {{ color: {p['SIDEBAR_TEXT']}; }}
[data-testid="stSidebar"] .stRadio > label {{ display: none; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {{ gap: 2px; }}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {{
    display: flex; align-items: center; padding: 9px 14px; margin: 1px 0;
    border-radius: 8px; cursor: pointer; font-weight: 500; font-size: 14px;
    transition: background 130ms ease, color 130ms ease;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {{
    background: {p['SIDEBAR_HOVER']};
}}
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {{
    display: none !important;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] {{
    background: {p['SIDEBAR_BG_ALT']};
    box-shadow: inset 3px 0 0 {p['SIDEBAR_ACCENT']};
    font-weight: 600;
}}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] * {{
    color: {p['SIDEBAR_ACCENT']} !important;
}}
[data-testid="stSidebar"] hr {{
    border-top: 1px solid {p['SIDEBAR_DIVIDER']};
    margin: 14px 0;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    border: 1px solid {p['SIDEBAR_DIVIDER']} !important;
    background: {p['SIDEBAR_BG_ALT']};
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary * {{
    color: {p['SIDEBAR_TEXT']} !important;
}}
[data-testid="stSidebar"] svg {{
    fill: {p['SIDEBAR_TEXT']} !important;
    color: {p['SIDEBAR_TEXT']} !important;
}}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebar"] [data-testid="baseButton-headerNoPadding"] {{
    background: {p['SIDEBAR_BG_ALT']} !important;
    border: 1px solid {p['SIDEBAR_DIVIDER']} !important;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
    color: {p['SIDEBAR_TEXT_MUTED']} !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {{
    color: {p['SIDEBAR_TEXT']} !important;
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{
    color: {p['SIDEBAR_ACCENT']} !important;
}}
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stAlert"] * {{
    background: {p['SIDEBAR_BG_ALT']} !important;
    color: {p['SIDEBAR_TEXT']} !important;
}}
[data-testid="collapsedControl"] {{
    background: {p['SURFACE']} !important;
    border: 1px solid {p['BORDER']} !important;
    border-radius: 8px !important;
}}
[data-testid="collapsedControl"] svg {{
    fill: {p['NAVY']} !important;
    color: {p['NAVY']} !important;
}}

/* ---------- TEXT ---------- */
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] a {{
    color: {p['NAVY']} !important;
}}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {{
    color: {p['MUTED']} !important;
}}
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] * {{
    color: {p['NAVY']} !important;
}}
.main code, .main pre {{
    color: {p['NAVY']} !important;
    background: {p['SURFACE_ALT']} !important;
    border-radius: 4px;
}}
.main code {{ padding: 1px 5px; }}
[data-testid="stAlert"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] a {{
    color: {p['NAVY']} !important;
}}
[data-testid="stAlert"] {{ background: {p['SURFACE']} !important; }}

.main [data-testid="stExpander"] {{
    border: 1px solid {p['BORDER']} !important;
    border-radius: 10px !important;
    background: {p['SURFACE']};
}}
.main [data-testid="stExpander"] summary,
.main [data-testid="stExpander"] summary * {{
    color: {p['NAVY']} !important;
}}
.main [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {{
    color: {p['NAVY']} !important;
}}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {p['BORDER']};
}}
.stTabs [data-baseweb="tab"] {{
    height: 42px; padding: 0 18px; font-weight: 500; background: transparent;
}}
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] p {{
    color: {p['MUTED']} !important;
}}
.stTabs [data-baseweb="tab"]:hover span,
.stTabs [data-baseweb="tab"]:hover p {{
    color: {p['NAVY']} !important;
}}
.stTabs [aria-selected="true"] {{
    border-bottom: 2px solid {p['ACCENT']} !important;
}}
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] p {{
    color: {p['ACCENT_DEEP']} !important;
}}

/* ---------- INPUTS ---------- */
.main input,
.stTextInput input,
.stNumberInput input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-baseweb="input"] input {{
    color: {p['NAVY']} !important;
    -webkit-text-fill-color: {p['NAVY']} !important;
    caret-color: {p['NAVY']} !important;
    background: {p['SURFACE']} !important;
}}
[data-baseweb="input"] {{
    border-color: {p['BORDER_STRONG']} !important;
    border-radius: 8px !important;
    background: {p['SURFACE']} !important;
}}
[data-baseweb="input"]:focus-within {{
    border-color: {p['ACCENT']} !important;
    box-shadow: 0 0 0 3px {p['ACCENT_SOFT']} !important;
}}
[data-baseweb="select"] > div {{
    border-color: {p['BORDER_STRONG']} !important;
    border-radius: 8px !important;
    background: {p['SURFACE']} !important;
}}
[data-baseweb="select"] > div:focus-within {{
    border-color: {p['ACCENT']} !important;
    box-shadow: 0 0 0 3px {p['ACCENT_SOFT']} !important;
}}
[data-baseweb="select"] input {{
    color: {p['NAVY']} !important;
    -webkit-text-fill-color: {p['NAVY']} !important;
    background: transparent !important;
}}
[data-baseweb="select"] div,
[data-baseweb="select"] span {{
    color: {p['NAVY']} !important;
}}
[data-baseweb="select"] svg {{
    fill: {p['NAVY']} !important;
    color: {p['NAVY']} !important;
}}
label[data-testid="stWidgetLabel"],
label[data-testid="stWidgetLabel"] *,
label[data-testid="stWidgetLabel"] p {{
    color: {p['SLATE']} !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}}
[data-testid="stNumberInputStepUp"],
[data-testid="stNumberInputStepDown"] {{
    background: {p['SURFACE_ALT']} !important;
    border: 1px solid {p['BORDER']} !important;
}}
[data-testid="stNumberInputStepUp"] svg,
[data-testid="stNumberInputStepDown"] svg {{
    fill: {p['NAVY']} !important;
    color: {p['NAVY']} !important;
}}
.main [data-baseweb="radio"] div[role="radio"] {{
    accent-color: {p['ACCENT']};
}}
.main [role="radiogroup"] label,
.main [role="radiogroup"] label * {{
    color: {p['NAVY']} !important;
}}
[data-baseweb="tag"] {{
    background: {p['ACCENT_SOFT']} !important;
    border-radius: 6px !important;
}}
[data-baseweb="tag"],
[data-baseweb="tag"] * {{
    color: {p['ACCENT_DEEP']} !important;
    -webkit-text-fill-color: {p['ACCENT_DEEP']} !important;
}}
[data-baseweb="slider"] [role="slider"] * {{
    color: {p['NAVY']} !important;
}}
[data-baseweb="popover"],
[data-baseweb="popover"] [role="listbox"] {{
    background: {p['SURFACE']} !important;
    border: 1px solid {p['BORDER']} !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {{
    background: {p['SURFACE']} !important;
    color: {p['NAVY']} !important;
}}
[data-baseweb="popover"] [role="option"] span,
[data-baseweb="popover"] li *,
[data-baseweb="menu"] li * {{
    color: {p['NAVY']} !important;
    -webkit-text-fill-color: {p['NAVY']} !important;
}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] li[aria-selected="true"] {{
    background: {p['SURFACE_HOVER']} !important;
}}

/* ---------- Custom HTML table ---------- */
.cl-table-wrap {{
    border: 1px solid {p['BORDER']};
    border-radius: 12px;
    overflow-x: auto;
    background: {p['SURFACE']};
    margin: 10px 0;
}}
.cl-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: {FONT_STACK};
}}
.cl-table thead th {{
    background: {p['SURFACE_ALT']} !important;
    color: {p['NAVY']} !important;
    font-weight: 600;
    text-align: left;
    padding: 10px 14px !important;
    border-bottom: 1px solid {p['BORDER']};
    white-space: nowrap;
}}
.cl-table tbody td {{
    background: {p['SURFACE']} !important;
    color: {p['SLATE']} !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid {p['BORDER']};
    white-space: nowrap;
}}
.cl-table tbody tr:last-child td {{ border-bottom: none; }}
.cl-table tbody tr:hover td {{ background: {p['SURFACE_HOVER']} !important; }}

[data-testid="stDataFrame"] {{
    border: 1px solid {p['BORDER']} !important;
    border-radius: 10px;
    overflow: hidden;
    background: {p['SURFACE']} !important;
}}

hr {{ border: none; border-top: 1px solid {p['BORDER']}; margin: 20px 0; }}

.main .js-plotly-plot {{
    background: {p['SURFACE']};
    border: 1px solid {p['BORDER']};
    border-radius: 12px;
    padding: 8px;
}}

/* ---------- CreditLens custom components ---------- */
.cl-section-title {{
    color: {p['NAVY']} !important;
    font-size: 19px; font-weight: 600;
    border-left: 3px solid {p['ACCENT']};
    padding-left: 12px; margin: 24px 0 10px;
    line-height: 1.3; letter-spacing: -0.01em;
}}
.cl-section-subtitle {{
    color: {p['MUTED']} !important;
    font-size: 13.5px; margin: -4px 0 16px;
    padding-left: 15px; line-height: 1.55;
}}
.cl-card {{
    background: {p['SURFACE']};
    border: 1px solid {p['BORDER']};
    border-radius: 12px;
    padding: 18px 20px;
}}
.cl-kpi-label {{
    color: {p['MUTED']} !important;
    font-size: 11.5px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
    margin-bottom: 6px;
}}
.cl-kpi-value {{
    color: {p['NAVY']} !important;
    font-size: 26px; font-weight: 700;
    line-height: 1.15; margin: 0; letter-spacing: -0.02em;
}}
.cl-kpi-hint {{
    color: {p['MUTED']} !important;
    font-size: 12px; margin-top: 6px; line-height: 1.4;
}}
.cl-band {{
    display: inline-block; padding: 5px 12px; border-radius: 6px;
    font-size: 11.5px; font-weight: 700;
    color: white !important;
    letter-spacing: 0.06em; text-transform: uppercase;
}}
.cl-probability-big {{
    font-size: 52px; font-weight: 700;
    color: {p['NAVY']} !important;
    line-height: 1; margin: 10px 0 4px; letter-spacing: -0.03em;
}}
.cl-probability-pct {{
    font-size: 26px; font-weight: 500;
    color: {p['MUTED']} !important;
    margin-left: 2px;
}}
.cl-note {{
    color: {p['SLATE']} !important;
    font-size: 12.5px; line-height: 1.6;
    border-left: 3px solid {p['BORDER_STRONG']};
    padding: 10px 14px; margin: 10px 0;
    background: {p['SURFACE_ALT']};
    border-radius: 6px;
}}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {p['BORDER_STRONG']}; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: {p['LIGHT']}; }}

/* ============================================================
   BUTTONS — placed LAST so they win the cascade.
   Every known DOM variant + every known wrapper class covered.
   ============================================================ */

/* -- MAIN AREA: every button → teal + white text ---------- */
.stApp button[kind="primary"],
.stApp button[kind="secondary"],
.stApp button[data-testid="stBaseButton-primary"],
.stApp button[data-testid="stBaseButton-secondary"],
.stApp button[data-testid="baseButton-primary"],
.stApp button[data-testid="baseButton-secondary"],
.stApp .stButton > button,
.stApp .stButton > div > button,
.stApp [data-testid="stButton"] > button,
.stApp [data-testid="stButton"] > div > button,
.stApp .stDownloadButton > button,
.stApp [data-testid="stDownloadButton"] > button,
.stApp [data-testid="stDownloadButton"] button {{
    background: {p['ACCENT']} !important;
    background-color: {p['ACCENT']} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}}

.stApp button[kind="primary"] *,
.stApp button[kind="secondary"] *,
.stApp button[data-testid="stBaseButton-primary"] *,
.stApp button[data-testid="stBaseButton-secondary"] *,
.stApp button[data-testid="baseButton-primary"] *,
.stApp button[data-testid="baseButton-secondary"] *,
.stApp .stButton > button *,
.stApp .stButton > div > button *,
.stApp [data-testid="stButton"] > button *,
.stApp [data-testid="stButton"] > div > button *,
.stApp .stDownloadButton > button *,
.stApp [data-testid="stDownloadButton"] > button *,
.stApp [data-testid="stDownloadButton"] button *,
.stApp [data-testid="stDownloadButton"] [data-testid="stMarkdownContainer"] *,
.stApp [data-testid="stDownloadButton"] [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stDownloadButton"] [data-testid="stMarkdownContainer"] span {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: transparent !important;
    background-color: transparent !important;
}}

.stApp button[kind="primary"]:hover,
.stApp button[kind="secondary"]:hover,
.stApp button[data-testid="stBaseButton-primary"]:hover,
.stApp button[data-testid="stBaseButton-secondary"]:hover,
.stApp .stButton > button:hover,
.stApp [data-testid="stButton"] > button:hover,
.stApp .stDownloadButton > button:hover,
.stApp [data-testid="stDownloadButton"] > button:hover,
.stApp [data-testid="stDownloadButton"] button:hover {{
    background: {p['ACCENT_HOVER']} !important;
    background-color: {p['ACCENT_HOVER']} !important;
}}

/* -- SIDEBAR OVERRIDE: back to dark navy ---------------- */
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"],
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[data-testid="baseButton-primary"],
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stButton"] > button,
[data-testid="stSidebar"] .stDownloadButton > button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button,
[data-testid="stSidebar"] [data-testid="stDownloadButton"] button {{
    background: {p['SIDEBAR_BG_ALT']} !important;
    background-color: {p['SIDEBAR_BG_ALT']} !important;
    color: {p['SIDEBAR_TEXT']} !important;
    border: 1px solid {p['SIDEBAR_DIVIDER']} !important;
}}

[data-testid="stSidebar"] button *,
[data-testid="stSidebar"] button[kind="primary"] *,
[data-testid="stSidebar"] button[kind="secondary"] *,
[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] *,
[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] *,
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] [data-testid="stButton"] > button * {{
    color: {p['SIDEBAR_TEXT']} !important;
    -webkit-text-fill-color: {p['SIDEBAR_TEXT']} !important;
    background: transparent !important;
    background-color: transparent !important;
}}

[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] button[kind="primary"]:hover,
[data-testid="stSidebar"] button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {p['SIDEBAR_HOVER']} !important;
    background-color: {p['SIDEBAR_HOVER']} !important;
    border-color: {p['SIDEBAR_ACCENT']} !important;
}}

[data-testid="stSidebar"] button:hover * {{
    color: {p['SIDEBAR_ACCENT']} !important;
    -webkit-text-fill-color: {p['SIDEBAR_ACCENT']} !important;
}}
</style>
"""


def inject(theme: str = "light") -> None:
    st.markdown(_build_css(get_palette(theme)), unsafe_allow_html=True)