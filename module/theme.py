"""
theme.py
--------
Professional, restrained healthcare/research palette:
  background : off-white (#F7F6F3)
  text       : charcoal (#2B2E33)
  accent     : muted teal/green (#3C7A6B) - used sparingly
  positive   : #4C8C6B (green)
  caution    : #B98A2E (amber)
  warning    : #B04A3C (red)
  neutral    : #E7E4DE (card background)
Explicitly NOT blue-heavy, per mentor feedback.
"""

import streamlit as st

COLORS = {
    "bg": "#F7F6F3",
    "text": "#2B2E33",
    "accent": "#3C7A6B",
    "accent_dark": "#2A5A50",
    "card": "#FFFFFF",
    "card_border": "#E4E1DA",
    "positive": "#3F7D5C",
    "caution": "#B98A2E",
    "warning": "#B0473A",
    "neutral": "#EDEBE6",
}


def inject_theme():
    st.markdown(f"""
    <style>
    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
    }}
    header[data-testid="stHeader"] {{
        background: {COLORS['bg']};
        border-bottom: 1px solid {COLORS['card_border']};
    }}
    section[data-testid="stSidebar"] {{
        background: {COLORS['neutral']};
        border-right: 1px solid {COLORS['card_border']};
    }}
    h1, h2, h3, h4 {{
        color: {COLORS['accent_dark']} !important;
        font-weight: 600 !important;
    }}
    .stButton>button {{
        background: {COLORS['accent']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5em 1.2em;
    }}
    .stButton>button:hover {{
        background: {COLORS['accent_dark']};
        color: white;
    }}
    div[data-testid="stMetric"] {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 8px;
        padding: 10px;
    }}
    .mrs-card {{
        background: {COLORS['card']};
        border: 1px solid {COLORS['card_border']};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }}
    .mrs-badge {{
        display:inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.78em; font-weight: 600; letter-spacing:.02em;
    }}
    .mrs-badge-demo {{ background:#F1E6C9; color:#7A5A12; }}
    .mrs-badge-research {{ background:#DCEAE3; color:{COLORS['accent_dark']}; }}
    .mrs-badge-warning {{ background:#F3D9D4; color:{COLORS['warning']}; }}
    .mrs-badge-local {{ background:#EAEAEA; color:#555; }}
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
