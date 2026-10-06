import os
import re
import hmac
import random
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
import certifi                                              # [CHANGED] TLS fix for MongoDB Atlas
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


from modules.config import APP_NAME, APP_TAGLINE, QTABLE_DIR
from modules.theme import inject_theme, mode_badge, card_open, card_close, COLORS
from pymongo import MongoClient, ReturnDocument            # [CHANGED] ReturnDocument added
from pymongo.errors import PyMongoError
from modules import psychology as psy
from modules import physiological as physio
from modules import dataset as ds
from modules import models as mdl
from modules import recommender as rec
from modules import safety as saf
from modules import bias as bias_mod
from modules import validation as val
from modules import evidence as ev

st.set_page_config(page_title=APP_NAME, layout="wide")
inject_theme()

# --------------------------------------------------------------
# Session-state defaults
# --------------------------------------------------------------
for key, default in [
    ("verified", False), ("username", None), ("user_email", None),
    ("editing_profile", False), ("profile_doc", None), ("profile_user", None),
    ("recs", []), ("got_recs", False), ("pool", pd.DataFrame()),
    ("session_number", 1), ("session_finished", False),
    ("page", "Dashboard"),
    ("full_baseline_this_session", False),                  # [NEW] True only in the session where the full questionnaires were answered
    ("login_doc_id", None),                                 # [NEW] _id of this session's login_history row (for check-out)
    ("wesad_context", None),                                # [NEW] physiological context chosen from WESAD research mode
    ("admin_ok", False),                                    # [NEW] admin page unlocked
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --------------------------------------------------------------
# [NEW] Short-form state check-in for RETURNING users (8 items)
#   PHQ-4 : Kroenke, Spitzer, Williams & Lowe (2009), Psychosomatics 50(6):613-621
#           (validated in the general population: Lowe et al., 2010, J Affect Disord 122:86-95)
#   PSS-4 : Cohen, Kamarck & Mermelstein (1983), J Health Soc Behav 24:385-396
# Traits (TIPI) and quality of life (WHOQOL-BREF) are stable, so the saved
# first-session baseline is reused instead of being asked again.
# --------------------------------------------------------------
PHQ4_ITEMS = [
    "Feeling nervous, anxious or on edge",
    "Not being able to stop or control worrying",
    "Feeling down, depressed or hopeless",
    "Little interest or pleasure in doing things",
]
PHQ4_OPTIONS = {0: "Not at all", 1: "Several days",
                2: "More than half the days", 3: "Nearly every day"}
PSS4_ITEMS = [
    "felt that you were unable to control the important things in your life?",
    "felt confident about your ability to handle your personal problems?",   # reverse-scored
    "felt that things were going your way?",                                  # reverse-scored
    "felt difficulties were piling up so high that you could not overcome them?",
]
PSS4_REVERSED = {1, 2}
PSS4_OPTIONS = {0: "Never", 1: "Almost never", 2: "Sometimes",
                3: "Fairly often", 4: "Very often"}
SHORT_FORM_NOTE = (
    "Short check-in: PHQ-4 (Kroenke et al., 2009) + PSS-4 (Cohen et al., 1983). "
    "Scores are rescaled to the DASS-21 0-42 range for the recommender; this rescaling "
    "is an approximation, not a validated conversion."
)


def score_short_checkin(phq4, pss4):
    """phq4: 4 ints (0-3); pss4: 4 raw ints (0-4). Returns DASS-scale sub-scores."""
    anxiety_raw = phq4[0] + phq4[1]        # GAD-2  (0-6)
    depression_raw = phq4[2] + phq4[3]     # PHQ-2  (0-6)
    pss_total = sum((4 - v) if i in PSS4_REVERSED else v for i, v in enumerate(pss4))  # 0-16
    return {
        "anxiety": anxiety_raw * 7.0,
        "depression": depression_raw * 7.0,
        "stress": pss_total * (42.0 / 16.0),
        "phq4_total": int(sum(phq4)),
        "pss4_total": int(pss_total),
    }


def phq4_band(total):
    return ("normal" if total <= 2 else "mild" if total <= 5