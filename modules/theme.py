"""
theme.py
--------
Catchy, vibrant, user-friendly wellness palette:
  background : soft gradient (lavender -> mint)
  text       : deep indigo/charcoal for readability
  accent     : energetic violet-to-teal gradient
  positive   : fresh green
  caution    : warm amber
  warning    : coral red
  neutral    : soft lilac card background
Bright enough to feel welcoming and modern, still calm enough for a
health/wellness context (avoids harsh neons, keeps good contrast).
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
    .stApp {{
        background: linear-gradient(160deg, {COLORS['bg']} 0%, #EAF7F4 100%);
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
    h1, h2, h3, h4 {{
        background: linear-gradient(90deg, {COLORS['accent_dark']}, {COLORS['accent_soft']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -0.01em;
    }}
    .stButton>button {{
        background: linear-gradient(90deg, {COLORS['accent']}, {COLORS['accent_soft']});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55em 1.4em;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(124, 77, 255, 0.28);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(124, 77, 255, 0.4);
        background: linear-gradient(90deg, {COLORS['accent_dark']}, {COLORS['accent_soft']});
        color: white;
    }}
    div[data-testid="stMetric"] {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 10px rgba(124, 77, 255, 0.08);
    }}
    .mrs-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 3px 14px rgba(124, 77, 255, 0.08);
    }}
    .mrs-badge {{
        display:inline-block; padding: 4px 12px; border-radius: 14px;
        font-size: 0.78em; font-weight: 700; letter-spacing:.02em;
    }}
    .mrs-badge-demo {{ background:#FEF0D9; color:#9A6A0E; }}
    .mrs-badge-research {{ background:#DFF7EF; color:{COLORS['accent_dark']}; }}
    .mrs-badge-warning {{ background:#FDE1DB; color:{COLORS['warning']}; }}
    .mrs-badge-local {{ background:#EFE9FE; color:{COLORS['accent_dark']}; }}
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