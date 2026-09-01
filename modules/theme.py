"""
theme.py
--------
Catchy, vibrant, user-friendly wellness palette (v2 — richer & more polished):
  background : layered gradient (lavender -> mint -> sky)
  text       : deep indigo/charcoal for readability
  accent     : energetic violet-to-teal gradient
  positive   : fresh green
  caution    : warm amber
  warning    : coral red
  neutral    : soft lilac card background
Adds: custom font, glass-style cards with hover lift, animated gradient
buttons, styled inputs/tabs/expanders, custom scrollbar. Still calm
enough for a health/wellness context — no harsh neons, strong contrast.
"""

import streamlit as st

COLORS = {
    "bg": "#F4F1FB",
    "text": "#241E33",
    "accent": "#7C4DFF",
    "accent_dark": "#5B2FE0",
    "accent_soft": "#22C1B8",
    "card": "#FFFFFF",
    "card_border": "#E6DEFB",
    "positive": "#2FB673",
    "caution": "#F2A93B",
    "warning": "#F0563E",
    "neutral": "#EFE9FE",
}


def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: radial-gradient(circle at 15% 0%, #EFE9FE 0%, transparent 45%),
                    radial-gradient(circle at 85% 10%, #DFF7EF 0%, transparent 50%),
                    linear-gradient(160deg, {COLORS['bg']} 0%, #EAF7F4 60%, #F0F4FE 100%);
        color: {COLORS['text']};
    }}

    header[data-testid="stHeader"] {{
        background: transparent;
        border-bottom: 1px solid {COLORS['card_border']};
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['neutral']} 0%, #E4F6F1 100%);
        border-right: 1px solid {COLORS['card_border']};
    }}
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] label {{
        font-weight: 600;
    }}

    h1, h2, h3, h4 {{
        background: linear-gradient(90deg, {COLORS['accent_dark']}, {COLORS['accent_soft']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -0.015em;
    }}

    p, span, label, .stMarkdown {{
        color: {COLORS['text']};
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent_soft']});
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.5em;
        font-weight: 700;
        letter-spacing: 0.01em;
        box-shadow: 0 6px 16px rgba(124, 77, 255, 0.32);
        transition: all 0.2s ease;
    }}
    .stButton>button:hover {{
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(124, 77, 255, 0.42);
        color: white;
    }}
    .stButton>button:active {{
        transform: translateY(0px) scale(0.98);
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(6px);
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 4px 16px rgba(124, 77, 255, 0.1);
    }}
    div[data-testid="stMetricValue"] {{
        color: {COLORS['accent_dark']};
        font-weight: 800;
    }}

    /* Cards */
    .mrs-card {{
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(8px);
        border: 1px solid {COLORS['card_border']};
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(124, 77, 255, 0.1);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }}
    .mrs-card:hover {{
        box-shadow: 0 8px 26px rgba(124, 77, 255, 0.16);
        transform: translateY(-1px);
    }}

    /* Badges */
    .mrs-badge {{
        display:inline-block; padding: 5px 13px; border-radius: 999px;
        font-size: 0.78em; font-weight: 700; letter-spacing:.02em;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .mrs-badge-demo {{ background:linear-gradient(90deg,#FEF0D9,#FCE2B2); color:#9A6A0E; }}
    .mrs-badge-research {{ background:linear-gradient(90deg,#DFF7EF,#CFF3E6); color:{COLORS['accent_dark']}; }}
    .mrs-badge-warning {{ background:linear-gradient(90deg,#FDE1DB,#FCCFC6); color:{COLORS['warning']}; }}
    .mrs-badge-local {{ background:linear-gradient(90deg,#EFE9FE,#E3D9FD); color:{COLORS['accent_dark']}; }}

    /* Inputs */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stTextArea textarea,
    .stSelectbox>div>div {{
        border-radius: 10px !important;
        border: 1px solid {COLORS['card_border']} !important;
        background: rgba(255,255,255,0.9) !important;
    }}
    .stTextInput>div>div>input:focus,
    .stTextArea textarea:focus {{
        border-color: {COLORS['accent']} !important;
        box-shadow: 0 0 0 3px rgba(124, 77, 255, 0.15) !important;
    }}

    /* Sliders */
    .stSlider [data-baseweb="slider"] div[role="slider"] {{
        background-color: {COLORS['accent']} !important;
        box-shadow: 0 0 0 6px rgba(124, 77, 255, 0.15);
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        background: {COLORS['neutral']};
        font-weight: 600;
        padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent_soft']}) !important;
        color: white !important;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background: {COLORS['neutral']};
        border-radius: 10px;
        font-weight: 600;
    }}

    /* Progress bar */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent_soft']});
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {COLORS['accent']}, {COLORS['accent_soft']});
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


def mode_badge(text, kind="research"):
    cls = {"demo": "mrs-badge-demo", "research": "mrs-badge-research",
           "warning": "mrs-badge-warning", "local": "mrs-badge-local"}.get(kind, "mrs-badge-research")
    st.markdown(f'<span class="mrs-badge {cls}">{text}</span>', unsafe_allow_html=True)


def card_open():
    st.markdown('<div class="mrs-card">', unsafe_allow_html=True)


def card_close():
    st.markdown('</div>', unsafe_allow_html=True)