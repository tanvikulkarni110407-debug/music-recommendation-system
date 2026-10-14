"""
MuSync - research prototype (Streamlit + MongoDB Atlas)

REQUIRED STREAMLIT SECRETS  (App -> Settings -> Secrets).  NOTHING secret lives in this file.
-----------------------------------------------------------------------------------------
MONGODB_URI        = "mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority"
MONGODB_DATABASE   = "musync"
ADMIN_PASSWORD     = "choose-a-strong-password"

requirements.txt must contain (besides what you already have):
    pymongo[srv]  dnspython  certifi  openpyxl  pyserial
"""
import os
import re
import io
import json
import time
import uuid
import hmac
import ssl
import hashlib
import secrets
import random
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
import certifi                                              # TLS fix for MongoDB Atlas
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Safe import for serial communication with NodeMCU/ESP32
try:
    import serial
except ImportError:
    serial = None


from modules.config import APP_NAME, APP_TAGLINE, QTABLE_DIR
from modules.theme import inject_theme, mode_badge, card_open, card_close, COLORS
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError, DuplicateKeyError
from modules import psychology as psy
from modules import physiological as physio
from modules import dataset as ds
from modules import models as mdl
from modules import recommender as rec
from modules import safety as saf
from modules import bias as bias_mod
from modules import validation as val
from modules import evidence as ev

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="collapsed")
inject_theme()

# --------------------------------------------------------------
# NodeMCU / ESP32 Therapeutic Pulse Generator
# --------------------------------------------------------------
class NodeMCUPulseGenerator:
    """Maps track tempo + self-reported stress to a pulse frequency (20-150 Hz) and
    optionally streams it to a NodeMCU / ESP32 over a serial port.

    * calculate_*  : pure maths, no I/O - safe to call on every Streamlit rerun.
    * send_pulse   : performs serial I/O - call ONLY from an explicit user action.
    Falls back to simulation if pyserial is missing or the board is unreachable.
    """
    MIN_FREQ_HZ = 20
    MAX_FREQ_HZ = 150
    MIN_BPM = 60.0
    MAX_BPM = 180.0
    HIGH_STRESS_THRESHOLD = 60      # stress > 60/100 -> grounding scale
    HIGH_STRESS_SCALE = 0.7

    def __init__(self, port="COM3", baudrate=115200):
        self.port = port
        self.baudrate = int(baudrate)
        self.max_freq = self.MAX_FREQ_HZ
        self.min_freq = self.MIN_FREQ_HZ

    @property
    def serial_available(self):
        return serial is not None

    def configure(self, port=None, baudrate=None):
        if port:
            self.port = str(port).strip()
        if baudrate:
            self.baudrate = int(baudrate)

    @staticmethod
    def _to_float(value, default):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float(default)
        return v if np.isfinite(v) else float(default)

    def calculate_safe_frequency(self, bpm, stress_level):
        """BPM 60-180 -> 20-150 Hz (linear); stress > 60 -> x0.7; result clamped to [20, 150]."""
        bpm_val = self._to_float(bpm, 90.0)
        stress_val = self._to_float(stress_level, 40.0)      # note: 0 is a valid stress value
        bpm_val = min(max(bpm_val, self.MIN_BPM), self.MAX_BPM)
        span = self.max_freq - self.min_freq
        raw = self.min_freq + ((bpm_val - self.MIN_BPM) / (self.MAX_BPM - self.MIN_BPM)) * span
        if stress_val > self.HIGH_STRESS_THRESHOLD:
            raw *= self.HIGH_STRESS_SCALE
        return int(max(self.min_freq, min(self.max_freq, round(raw))))

    @staticmethod
    def period_ms(freq_hz):
        """Period (ms) = 1000 / Frequency (Hz)."""
        return round(1000.0 / freq_hz, 2) if freq_hz and freq_hz > 0 else 0.0

    def calculate_pulse(self, bpm, stress_level):
        """Return (freq_hz, period_ms) without touching any hardware."""
        freq = self.calculate_safe_frequency(bpm, stress_level)
        return freq, self.period_ms(freq)

    def send_pulse(self, bpm, stress_level):
        """Compute and transmit f"{freq}\\n". Returns (freq, period_ms, mode, status_text)
        where mode is 'hardware' or 'simulation'."""
        freq, period = self.calculate_pulse(bpm, stress_level)
        if serial is None:
            return freq, period, "simulation", (
                f"🧪 Safe simulation mode: {freq} Hz computed, nothing sent (pyserial not installed).")
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                time.sleep(0.1)
                ser.write(f"{freq}\n".encode("utf-8"))
                ser.flush()
            return freq, period, "hardware", (
                f"✅ Dispatched to NodeMCU/ESP32 on {self.port} @ {self.baudrate} baud: {freq} Hz.")
        except Exception as exc:                              # port busy / missing / permission
            return freq, period, "simulation", (
                f"🧪 Safe simulation mode: {freq} Hz computed (hardware unreachable on {self.port}: {exc}).")


pulse_hardware = NodeMCUPulseGenerator(port="COM3", baudrate=115200)

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
NAME_MIN_LEN = 2                      # shortest accepted name
NAME_MAX_LEN = 40                     # longest accepted name

# Marathi translations for the full baseline questionnaires.
TIPI_MARATHI = ['मी स्वतःला मोकळा-मिळणारा, उत्साही मानतो.', 'मी स्वतःला टीकात्मक, वाद घालणारा मानतो.', 'मी स्वतःला विश्वासार्ह, स्वअनुशासित मानतो.', 'मी स्वतःला चिंताग्रस्त, सहज अस्वस्थ होणारा मानतो.', 'मी स्वतःला नवीन अनुभवांसाठी खुले, गुंतागुंतीचा मानतो.', 'मी स्वतःला राखीव, शांत मानतो.', 'मी स्वतःला सहकार्यशील, उबदार स्वभावाचा मानतो.', 'मी स्वतःला असंघटित, निष्काळजी मानतो.', 'मी स्वतःला शांत, भावनिकदृष्ट्या स्थिर मानतो.', 'मी स्वतःला पारंपरिक, कल्पनाशक्ती कमी असलेला मानतो.']
DASS_MARATHI = ['मला शांत होणे कठीण वाटत होते.', 'माझ्या तोंडात कोरडेपणा जाणवत होता.', 'मला कोणताही सकारात्मक अनुभव घेता येत नव्हता.', 'मला श्वास घेणे कठीण वाटत होते.', 'काहीतरी करण्याची प्रेरणा मिळवणे कठीण वाटत होते.', 'मी परिस्थितींवर अतिप्रतिक्रिया देत होतो.', 'माझे हात थरथरत होते.', 'मला वाटत होते की मी खूप चिंताग्रस्त ऊर्जा वापरत आहे.', 'मी घाबरू शकेन अशा परिस्थितींबद्दल मला चिंता वाटत होती.', 'मला भविष्याबद्दल काहीही अपेक्षा नाही असे वाटत होते.', 'मला चिडचिड होत होती.', 'मला आराम करणे कठीण जात होते.', 'माझा मूड खिन्न आणि उदास होता.', 'मी करत असलेल्या कामात अडथळा आणणाऱ्या गोष्टी सहन करू शकत नव्हतो.', 'मला वाटत होते की मी घाबरण्याच्या स्थितीजवळ पोहोचलो आहे.', 'मला कोणत्याही गोष्टीबद्दल उत्साह वाटत नव्हता.', 'मला वाटत होते की मी व्यक्ती म्हणून मोलाचा नाही.', 'मला वाटत होते की मी खूप संवेदनशील/हळवा झालो आहे.', 'शारीरिक हालचाल नसतानाही माझ्या हृदयाचे ठोके जाणवत होते.', 'मला कोणतेही योग्य कारण नसताना भीती वाटत होती.', 'मला वाटत होते की जीवन निरर्थक आहे.']
WHOQOL_MARATHI = ['तुम्ही तुमच्या जीवनाच्या गुणवत्तेला किती प्रमाणात रेट कराल?', 'तुम्ही तुमच्या आरोग्याबद्दल किती समाधानी आहात?', 'वेदना तुम्हाला आवश्यक गोष्टी करण्यापासून किती प्रमाणात रोखतात?', 'दैनंदिन जीवनात कार्य करण्यासाठी तुम्हाला किती वैद्यकीय उपचारांची गरज आहे?', 'तुम्ही आयुष्याचा किती आनंद घेतला?', 'तुमच्या जीवनाला किती अर्थ आहे असे तुम्हाला वाटते?', 'तुम्हाला लक्ष केंद्रित करणे किती सोपे वाटते?', 'दैनंदिन जीवनात तुम्हाला किती सुरक्षित वाटते?', 'तुमचे भौतिक वातावरण किती आरोग्यदायी आहे?', 'दैनंदिन जीवनासाठी तुमच्याकडे पुरेशी ऊर्जा आहे का?', 'तुम्ही तुमच्या शरीराच्या स्वरूपाचा स्वीकार करू शकता का?', 'तुमच्या गरजा पूर्ण करण्यासाठी तुमच्याकडे पुरेसे पैसे आहेत का?', 'दैनंदिन जीवनासाठी आवश्यक माहिती तुम्हाला कितपत उपलब्ध आहे?', 'मनोरंजनासाठी तुम्हाला किती संधी मिळतात?', 'तुम्ही तुमच्या झोपेबद्दल किती समाधानी आहात?', 'दैनंदिन कामे करण्याच्या क्षमतेबद्दल तुम्ही किती समाधानी आहात?', 'काम करण्याच्या तुमच्या क्षमतेबद्दल तुम्ही किती समाधानी आहात?', 'तुम्ही स्वतःबद्दल किती समाधानी आहात?', 'तुमच्या वैयक्तिक नातेसंबंधांबद्दल तुम्ही किती समाधानी आहात?', 'तुमच्या लैंगिक जीवनाबद्दल तुम्ही किती समाधानी आहात?', 'तुमच्या मित्रांकडून मिळणाऱ्या मदतीबद्दल तुम्ही किती समाधानी आहात?', 'तुमच्या राहणीमानाबद्दल तुम्ही किती समाधानी आहात?', 'आरोग्यसेवा मिळण्याबद्दल तुम्ही किती समाधानी आहात?', 'वाहतुकीच्या सुविधांबद्दल तुम्ही किती समाधानी आहात?', 'तुम्हाला एका ठिकाणाहून दुसऱ्या ठिकाणी फिरणे किती सहज शक्य होते?', 'तुमच्या आजूबाजूच्या वातावरणाबद्दल तुम्ही किती समाधानी आहात?']

# --------------------------------------------------------------
# Session-state defaults
# --------------------------------------------------------------
_STATE_DEFAULTS = [
    ("verified", False), ("username", None), ("user_email", None),
    ("editing_profile", False), ("profile_doc", None), ("profile_user", None),
    ("recs", []), ("got_recs", False), ("pool", pd.DataFrame()),
    ("session_number", 1), ("session_finished", False),
    ("page", "Dashboard"),
    ("full_baseline_this_session", False),
    ("session_id", None),                                   # unique id of THIS login session (sessions collection)
    ("assessment_id", None),                                # unique id of the latest assessment (Get Recommendations click)
    ("wesad_context", None),
    ("admin_ok", False),
    ("login_flash", None),
    ("rr_saved", []), ("rr_features_latest", None),
    ("pulse_status", {}), ("pulse_port", "COM3"), ("pulse_baud", 115200),
]
for key, default in _STATE_DEFAULTS:
    if key not in st.session_state:
        st.session_state[key] = default


def _reset_session_state():
    """Clear everything that belongs to the signed-in participant (used on logout)."""
    for k, d in _STATE_DEFAULTS:
        if k in ("page", "admin_ok"):
            continue
        st.session_state[k] = d if not isinstance(d, (list, dict)) else type(d)()
    st.session_state["pool"] = pd.DataFrame()
    for k in list(st.session_state.keys()):
        if k.startswith(("fb_done_", "sess_fb_done_", "rate_", "phq4_", "pss4_", "login_")):
            del st.session_state[k]


# --------------------------------------------------------------
# Short-form state check-in for RETURNING users (8 items)
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
            else "moderate" if total <= 8 else "severe")


# --------------------------------------------------------------
# Data / model loading (cached, safe)
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_dataset():
    return ds.load_dataset()


@st.cache_resource(show_spinner=False)
def _load_models():
    return mdl.load_models()


df, is_demo_dataset, dataset_note = _load_dataset()
metadata, rnn_model, ncf_model, model_error = _load_models()


# --------------------------------------------------------------
# Secrets helper
# --------------------------------------------------------------
def _get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


# --------------------------------------------------------------
# MongoDB Atlas connection
# --------------------------------------------------------------
class MongoDBStore:
    def __init__(self, uri=None, db_name=None):
        uri = uri or _get_secret("MONGODB_URI")
        db_name = db_name or _get_secret("MONGODB_DATABASE", "musync")

        if not uri:
            raise RuntimeError(
                "MONGODB_URI is missing. Add MONGODB_URI and "
                "MONGODB_DATABASE in Streamlit Secrets."
            )

        self.client = MongoClient(
            uri,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=20000,
            connectTimeoutMS=20000,
            socketTimeoutMS=30000,
            retryWrites=True,
        )

        self.client.admin.command("ping")
        self.mongo_db = self.client[db_name]

        self.login_history = self.mongo_db["login_history"]
        self.profiles = self.mongo_db["user_profiles"]
        self.profile_history = self.mongo_db["profile_history"]
        self.physiological_measurements = self.mongo_db["physiological_measurements"]
        self.qtables = self.mongo_db["qtables"]
        self.recommendation_feedback = self.mongo_db["recommendation_feedback"]
        self.experiments = self.mongo_db["experiments"]
        self.bias_assessments = self.mongo_db["bias_assessments"]
        self.users = self.mongo_db["users"]
        self.state_checkins = self.mongo_db["state_checkins"]
        self.sessions = self.mongo_db["sessions"]
        self.assessments = self.mongo_db["assessments"]

        self.mode = "mongodb"
        self.index_warnings = []
        self._create_indexes()

    def _ix(self, coll, keys, **kw):
        try:
            coll.create_index(keys, **kw)
        except PyMongoError as e:
            self.index_warnings.append(f"{coll.name} {keys}: {e}")

    def _create_indexes(self):
        has_assess = {"assessment_id": {"$exists": True}}
        self._ix(self.users, "user", unique=True)
        self._ix(self.users, "email")
        self._ix(self.profiles, "user", unique=True)
        self._ix(self.profile_history, [("user", 1), ("timestamp", -1)])
        self._ix(self.sessions, "session_id", unique=True)
        self._ix(self.sessions, [("user", 1), ("session_number", 1)], unique=True)
        self._ix(self.login_history, [("user", 1), ("timestamp_utc", -1)])
        self._ix(self.login_history, "session_id")
        self._ix(self.assessments, "assessment_id", unique=True)
        self._ix(self.assessments, [("user", 1), ("session_id", 1)])
        self._ix(self.qtables, "user", unique=True)
        self._ix(self.recommendation_feedback, [("user", 1), ("timestamp", -1)])
        self._ix(self.recommendation_feedback, [("user", 1), ("assessment_id", 1), ("song_id", 1)],
                 unique=True, partialFilterExpression=has_assess, name="uniq_feedback_per_assessment_song")
        self._ix(self.experiments, [("user", 1), ("timestamp", -1)])
        self._ix(self.experiments, [("user", 1), ("assessment_id", 1)],
                 unique=True, partialFilterExpression=has_assess, name="uniq_session_feedback_per_assessment")
        self._ix(self.physiological_measurements, [("user", 1), ("timestamp", -1)])
        self._ix(self.physiological_measurements, "session_id")
        self._ix(self.state_checkins, [("user", 1), ("timestamp", -1)])
        self._ix(self.state_checkins, "assessment_id", unique=True, partialFilterExpression=has_assess,
                 name="uniq_checkin_per_assessment")
        self._ix(self.bias_assessments, [("user", 1), ("timestamp", -1)])


@st.cache_resource(show_spinner=False)
def _connect_mongodb(uri, db_name):
    return MongoDBStore(uri=uri, db_name=db_name)


_mongo_uri = _get_secret("MONGODB_URI")
_mongo_db_name = _get_secret("MONGODB_DATABASE", "musync")

try:
    db = _connect_mongodb(_mongo_uri, _mongo_db_name)
    mongodb_error = None
except Exception as e:
    db = None
    mongodb_error = str(e)


# --------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------
def _now_ist_str():
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")


def _utc_iso():
    return datetime.now(timezone.utc).isoformat()


def _normalize_username(raw):
    return re.sub(r"\s+", "_", raw.strip().lower())


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def spotify_link(song, artist):
    q = urllib.parse.quote_plus(f"{song} {artist}")
    return f"https://open.spotify.com/search/{q}"


# --------------------------------------------------------------
# Sign-in & Authentication
# --------------------------------------------------------------
def _complete_login(username, display_id):
    if not db:
        raise RuntimeError("MongoDB is not connected.")

    ist, utc = _now_ist_str(), _utc_iso()
    session_id = uuid.uuid4().hex

    with db.client.start_session() as mongo_session:
        with mongo_session.start_transaction():
            user_doc = db.users.find_one_and_update(
                {"user": username},
                {
                    "$set": {"last_login_ist": ist, "last_login_utc": utc, "name": display_id},
                    "$setOnInsert": {
                        "display_id": display_id,
                        "first_login_ist": ist, "first_login_utc": utc,
                        "created_at_utc": utc,
                    },
                    "$inc": {"login_count": 1},
                },
                upsert=True, return_document=ReturnDocument.AFTER, session=mongo_session,
            )
            if not user_doc:
                raise RuntimeError("MongoDB did not return the user document.")

            session_number = int(user_doc.get("login_count", 1))
            is_new = session_number == 1

            db.sessions.insert_one({
                "session_id": session_id, "user": username, "name": display_id,
                "session_number": session_number,
                "started_at_ist": ist, "started_at_utc": utc,
                "ended_at_ist": None, "ended_at_utc": None, "duration_min": None,
                "status": "active", "is_first_session": is_new,
            }, session=mongo_session)

            db.login_history.insert_one({
                "event": "login", "session_id": session_id, "user": username,
                "name": display_id, "session_number": session_number,
                "timestamp_ist": ist, "timestamp_utc": utc,
            }, session=mongo_session)

    st.session_state["verified"] = True
    st.session_state["username"] = username
    st.session_state["user_email"] = None
    st.session_state["session_number"] = session_number
    st.session_state["session_id"] = session_id
    st.session_state["full_baseline_this_session"] = False
    st.session_state["profile_user"] = None
    st.session_state["profile_doc"] = None
    st.session_state["login_flash"] = (
        "Name saved. Welcome!" if is_new else f"Welcome back - Session #{session_number}.")


def _login_with_name(raw_name):
    if not db:
        return False, "MongoDB is not connected."
    name = re.sub(r"\s+", " ", (raw_name or "").strip())
    if len(name) < NAME_MIN_LEN:
        return False, f"Please enter your name (at least {NAME_MIN_LEN} characters)."
    if len(name) > NAME_MAX_LEN:
        return False, f"Name must be at most {NAME_MAX_LEN} characters."
    username = _normalize_username(name)
    try:
        _complete_login(username, name)
    except Exception as e:
        return False, f"Name could not be saved in MongoDB ({type(e).__name__}: {e})."
    return True, "OK"


def _render_login_flow():
    flash = st.session_state.get("login_flash")
    if flash:
        st.success(flash)
        st.session_state["login_flash"] = None

    st.info("Welcome to MuSync 🎧 Type your **name** below and tap **Enter**.")
    with st.form("login_form", clear_on_submit=False):
        name = st.text_input("Your name", key="login_name_input", max_chars=NAME_MAX_LEN,
                             placeholder="e.g. Priya")
        submitted_login = st.form_submit_button("Enter ➜", type="primary", use_container_width=True)
    if submitted_login:
        ok, msg = _login_with_name(name)
        if ok:
            st.rerun()
        else:
            st.error(msg)


def _mark_checkout(reason):
    if not db:
        return False, "MongoDB is not connected."

    sid = st.session_state.get("session_id")
    if not sid:
        return False, "No active session ID is present in this browser session."

    now_utc = datetime.now(timezone.utc)
    ist = _now_ist_str()

    sdoc = db.sessions.find_one({"session_id": sid, "ended_at_utc": None})
    if not sdoc:
        return False, "The MongoDB session record was not found or was already closed."

    try:
        started = datetime.fromisoformat(str(sdoc["started_at_utc"]))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        duration = round((now_utc - started).total_seconds() / 60.0, 2)
    except Exception:
        duration = None

    try:
        with db.client.start_session() as mongo_session:
            with mongo_session.start_transaction():
                res = db.sessions.update_one(
                    {"session_id": sid, "ended_at_utc": None},
                    {"$set": {
                        "ended_at_ist": ist, "ended_at_utc": now_utc.isoformat(),
                        "duration_min": duration, "status": "closed", "end_reason": reason,
                    }},
                    session=mongo_session,
                )
                if res.modified_count != 1:
                    raise RuntimeError("MongoDB did not close the active session record.")

                db.login_history.insert_one({
                    "event": "logout", "session_id": sid, "user": sdoc["user"],
                    "name": sdoc.get("name"), "session_number": sdoc.get("session_number"),
                    "timestamp_ist": ist, "timestamp_utc": now_utc.isoformat(),
                    "session_duration_min": duration, "reason": reason,
                }, session=mongo_session)

                db.users.update_one(
                    {"user": sdoc["user"]},
                    {"$set": {"last_logout_ist": ist, "last_logout_utc": now_utc.isoformat()}},
                    session=mongo_session,
                )
        return True, "Logout saved to MongoDB."
    except Exception as e:
        return False, f"MongoDB could not save the logout ({type(e).__name__}: {e})."


def _user_feedback_df(name):
    try:
        rows = list(db.recommendation_feedback.find({"user": name}, {"_id": 0}).sort("timestamp", 1))
    except Exception:
        return None
    return pd.DataFrame(rows) if rows else None


def _user_session_feedback_df(name):
    try:
        rows = list(db.experiments.find({"user": name}, {"_id": 0}).sort("timestamp", 1))
    except Exception:
        return None
    return pd.DataFrame(rows) if rows else None


def _save_qtable(user, table, before, num_songs):
    try:
        table = np.asarray(table, dtype=float)
        before = np.asarray(before, dtype=float)
        meta = db.qtables.find_one({"user": user}, {"num_songs": 1, "_id": 0})
        if meta and meta.get("num_songs") == num_songs and table.shape == before.shape and table.shape[1] == num_songs:
            diff = table - before
            nz = np.argwhere(np.abs(diff) > 0)
            if len(nz) == 0:
                return
            inc = {f"qtable.{int(r)}.{int(c)}": float(diff[r, c]) for r, c in nz}
            db.qtables.update_one({"user": user}, {"$inc": inc, "$set": {"updated_at": _utc_iso()}})
        else:
            db.qtables.update_one({"user": user}, {"$set": {"qtable": table.tolist(), "num_songs": int(num_songs),
                                                            "updated_at": _utc_iso()}}, upsert=True)
    except Exception as e:
        st.warning(f"Q-table for '{user}' could not be saved: {e}")


# --------------------------------------------------------------
# WESAD research mode helpers
# --------------------------------------------------------------
WESAD_FEATURES_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "wesad_hrv_features.csv")
WESAD_REQUIRED_COLS = ["subject", "condition", "mean_hr_bpm", "rmssd_ms"]


@st.cache_data(show_spinner=False)
def _load_wesad_features():
    if not os.path.exists(WESAD_FEATURES_CSV):
        return None, "WESAD feature file not found (data/wesad_hrv_features.csv)."
    try:
        wdf = pd.read_csv(WESAD_FEATURES_CSV)
    except Exception as e:
        return None, f"Could not read WESAD feature file: {e}"
    missing = [c for c in WESAD_REQUIRED_COLS if c not in wdf.columns]
    if missing:
        return None, f"WESAD feature file is missing columns {missing}. Re-run extract_wesad_hrv.py."
    if wdf.empty:
        return None, "WESAD feature file is empty."
    return wdf, None


def _wesad_stress_index(rmssd_ms, wdf):
    lo = float(np.nanpercentile(wdf["rmssd_ms"], 5))
    hi = float(np.nanpercentile(wdf["rmssd_ms"], 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 50.0
    return float(np.clip(100.0 * (1.0 - (rmssd_ms - lo) / (hi - lo)), 0.0, 100.0))


# --------------------------------------------------------------
# RESEARCH EVIDENCE
# --------------------------------------------------------------
RAGA_EVIDENCE = {
    "bhairavi": {
        "raga": "Bhairavi",
        "pattern": re.compile(r"\bbhairavi\b|\bbhairvi\b", re.I),
        "citation": ("Chand K, Chandra S, Dutt V. Raga Bhairavi in virtual reality reduces stress-related "
                     "psychophysiological markers. Scientific Reports 14, 24816 (2024)."),
        "short_citation": "Chand, Chandra & Dutt (2024), Sci Rep 14:24816",
        "doi": "10.1038/s41598-024-74932-1",
        "design": ("Randomised controlled study, N = 44 healthy adults (18-35 y, mean age 24.4), IIT Mandi. "
                   "Intervention: 15 min/day for 6 consecutive days listening to raga Bhairavi as a 360-degree video "
                   "on a Meta Quest 2 headset with headphones (VR-raga, n = 22) vs. 15 min sitting without "
                   "intervention (control, n = 22)."),
        "stimulus": "Meditative Raag Bhairavi | Kaushiki Chakraborty | Dawn | Darbar VR360 (2019)",
        "stimulus_url": "https://www.youtube.com/watch?v=6_bfPzclL08",
        "tempo": "Not reported in the paper",
        "mood": ("Mood labels were not tested; outcomes were DASS-21 stress, anxiety and depression and HRV."),
        "instrumentation": "Not reported in the paper",
        "musical_characteristics": ("'Sampoorn' (complete) raga using all seven notes; traditionally associated with "
                                    "early-morning performance and often presented at the close of a concert."),
        "psychological": ("After 6 days all DASS-21 subscales fell within the VR-raga group."),
        "physiological": ("Short-term seated HRV (PPG, ear lobe): SDNN, SDHR, HTI, LF/HF, respiration rate, SD2/SD1, SampEn differed significantly."),
        "limitations": [
            "The effect of VR was not isolated from the effect of the music.",
            "No placebo/sham group; expectancy effects are possible.",
            "Only 6 days; DASS-21 covers the whole week, not individual days.",
            "Healthy 18-35-year-olds at one institute.",
            "One raga and one specific recorded performance were tested.",
        ],
    }
}

RESEARCH_SOURCE_TRACKS = [
    {"key": "miyan_ki_todi", "song_id": "research_src_miyan_ki_todi", "song": "Raag Miyan Ki Todi",
     "artist": "Nikhil Banerjee", "genre": "Indian Classical", "raga": "Miyan Ki Todi",
     "evidence_level": "project_source", "evidence_key": "",
     "source_url": "https://open.spotify.com/track/2wmy0bj9Lchz0cQnmriBR0",
     "source_name": "Fond Memories-Sitar Vol-1"},
    {"key": "rageshree", "song_id": "research_src_rageshree", "song": "Raag Rageshree",
     "artist": "Nikhil Banerjee", "genre": "Indian Classical", "raga": "Rageshree",
     "evidence_level": "project_source", "evidence_key": "",
     "source_url": "https://open.spotify.com/album/4CbM5IC1txrx40X0AYyPmP",
     "source_name": "Fond Memories-Sitar Vol-1"},
    {"key": "nat_bhairav", "song_id": "research_src_nat_bhairav", "song": "Raag Nat Bhairav",
     "artist": "Nikhil Banerjee", "genre": "Indian Classical", "raga": "Nat Bhairav",
     "evidence_level": "project_source", "evidence_key": "",
     "source_url": "https://open.spotify.com/track/1Y76LppGA9FsyAackq9uLy",
     "source_name": "Fond Memories-Sitar Vol-1"},
]
RESEARCH_PAPER_TRACKS = [
    {"key": "bhairavi_darbar", "song_id": "research_src_bhairavi_darbar", "song": "Meditative Raag Bhairavi",
     "artist": "Kaushiki Chakraborty", "genre": "Indian Classical", "raga": "Bhairavi",
     "evidence_level": "studied_stimulus", "evidence_key": "bhairavi",
     "source_url": RAGA_EVIDENCE["bhairavi"]["stimulus_url"],
     "source_name": "Stimulus used in " + RAGA_EVIDENCE["bhairavi"]["short_citation"]},
]
RESEARCH_CANDIDATES = RESEARCH_PAPER_TRACKS + RESEARCH_SOURCE_TRACKS

EVIDENCE_WEIGHT = {"studied_stimulus": 0.60, "raga_category": 0.30, "project_source": 0.20}
RESEARCH_FIT_FLOOR = 0.30
RESEARCH_FIT_RANGE = 0.45
MAX_SOURCE_ONLY_RESEARCH = 2
RESEARCH_META_COLS = {"research_source": False, "evidence_level": "", "evidence_key": "", "source_url": "",
                      "source_name": "", "raga": "", "evidence_bonus": 0.0}
SCORE_COLS = ["rnn_score", "ncf_score", "personal_q", "pref_bias", "physio_fit", "psy_bias"]


def _norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().casefold()


def _research_fit(stress, hr, dass_n, mood_state, genre_pref, era_pref):
    stress_n = float(np.clip(float(stress) / 100.0, 0, 1))
    hr_n = float(np.clip((float(hr) - 70.0) / 50.0, 0, 1))
    dass_n = float(np.clip(float(dass_n), 0, 1))
    mood_d = {"sad": 1.0, "angry": 1.0, "happy": 0.1, "calm": 0.2, "energetic": 0.2}.get(str(mood_state).lower(), 0.3)
    taste = 1.0 if "classical" in f"{genre_pref} {era_pref}".lower() else 0.0
    return 0.35 * stress_n + 0.25 * dass_n + 0.15 * hr_n + 0.15 * mood_d + 0.10 * taste


def _flagged_set(flagged):
    try:
        return {str(x) for x in flagged}
    except Exception:
        return set()


def _apply_research_evidence(pool, fit_inputs, flagged):
    pool = pool.copy().reset_index(drop=True)
    for c, d in RESEARCH_META_COLS.items():
        if c not in pool.columns:
            pool[c] = d
    pool["final_score"] = pd.to_numeric(pool["final_score"], errors="coerce")
    scores = pool["final_score"].dropna()
    spread = float(scores.max() - scores.min()) if len(scores) > 1 and scores.max() > scores.min() else 1.0
    median = float(scores.median()) if len(scores) else 0.0

    fit = _research_fit(**fit_inputs)
    strength = float(np.clip((fit - RESEARCH_FIT_FLOOR) / RESEARCH_FIT_RANGE, 0, 1))
    flagged = _flagged_set(flagged)

    song_n = pool["song"].map(_norm) if "song" in pool.columns else pd.Series([""] * len(pool))
    artist_n = pool["artist"].map(_norm) if "artist" in pool.columns else pd.Series([""] * len(pool))

    def tag(idx, level, key, raga, url, name):
        pool.at[idx, "research_source"] = True
        pool.at[idx, "evidence_level"] = level
        pool.at[idx, "evidence_key"] = key
        pool.at[idx, "raga"] = raga
        pool.at[idx, "source_url"] = url
        pool.at[idx, "source_name"] = name

    matched = set()
    for cand in RESEARCH_CANDIDATES:
        m = (song_n == _norm(cand["song"])) & (artist_n == _norm(cand["artist"]))
        for idx in pool.index[m]:
            tag(idx, cand["evidence_level"], cand["evidence_key"], cand["raga"], cand["source_url"], cand["source_name"])
            matched.add(cand["key"])

    for idx in pool.index:
        if pool.at[idx, "evidence_level"]:
            continue
        text = " ".join(str(pool.at[idx, c]) for c in ("song", "genre", "raga_name") if c in pool.columns)
        for ek, e in RAGA_EVIDENCE.items():
            if e["pattern"].search(text):
                tag(idx, "raga_category", ek, e["raga"], "", e["short_citation"])
                break

    for idx in pool.index[pool["evidence_level"] != ""]:
        if str(pool.at[idx, "song_id"]) in flagged:
            continue
        bonus = EVIDENCE_WEIGHT.get(pool.at[idx, "evidence_level"], 0.0) * strength * spread
        pool.at[idx, "evidence_bonus"] = bonus
        if pd.notna(pool.at[idx, "final_score"]):
            pool.at[idx, "final_score"] += bonus

    new_rows = []
    if strength > 0:
        cands = []
        for cand in RESEARCH_CANDIDATES:
            if cand["key"] in matched or str(cand["song_id"]) in flagged:
                continue
            cands.append((EVIDENCE_WEIGHT[cand["evidence_level"]] * strength * spread, cand))
        cands.sort(key=lambda x: x[0], reverse=True)
        for bonus, cand in cands[:MAX_SOURCE_ONLY_RESEARCH]:
            row = {c: (np.nan if pd.api.types.is_numeric_dtype(pool[c]) else "") for c in pool.columns}
            for c in SCORE_COLS:
                if c in pool.columns:
                    s = pd.to_numeric(pool[c], errors="coerce")
                    row[c] = float(s.median()) if s.notna().any() else 0.0
            row.update({"song_id": cand["song_id"], "song": cand["song"], "artist": cand["artist"],
                        "genre": cand["genre"], "research_source": True, "evidence_level": cand["evidence_level"],
                        "evidence_key": cand["evidence_key"], "raga": cand["raga"], "source_url": cand["source_url"],
                        "source_name": cand["source_name"], "evidence_bonus": bonus, "final_score": median + bonus})
            new_rows.append(row)
    if new_rows:
        pool = pd.concat([pool, pd.DataFrame(new_rows, columns=pool.columns)], ignore_index=True)
    pool["research_source"] = pool["research_source"].fillna(False).astype(bool)
    return pool, fit, strength


def _select_recommendations(pool, n=5):
    try:
        chosen = rec.select_recommendations(pool, n=n)
        if chosen is None or len(chosen) == 0:
            raise ValueError("empty selection")
        chosen = chosen.reset_index(drop=True)
    except Exception:
        chosen = pool.sort_values("final_score", ascending=False).head(n).reset_index(drop=True)

    top = pool.sort_values("final_score", ascending=False).head(n)
    chosen_ids = [str(x) for x in chosen["song_id"]]
    meta_flag = pool.set_index(pool["song_id"].astype(str))["research_source"].to_dict()
    for _, r in top.iterrows():
        if bool(r.get("research_source")) and str(r["song_id"]) not in chosen_ids:
            cand_idx = [i for i, sid in enumerate(chosen_ids) if not meta_flag.get(sid, False)]
            if cand_idx:
                drop = cand_idx[-1]
                chosen.iloc[drop] = r[chosen.columns].values if all(c in r.index for c in chosen.columns) else chosen.iloc[drop]
                chosen_ids[drop] = str(r["song_id"])
    return chosen


# --------------------------------------------------------------
# Per-song recommendation explanations
# --------------------------------------------------------------
def _research_explanation(row, ctx):
    song = str(row.get("song", "This track"))
    artist = str(row.get("artist", "the artist"))
    level = str(row.get("evidence_level", ""))
    ek = str(row.get("evidence_key", ""))
    e = RAGA_EVIDENCE.get(ek)
    stress = ctx.get("stress")
    state = (f"your reported stress ({stress}/100), psychological scores, heart rate and mood" if stress is not None
             else "your reported stress, psychological scores, heart rate and mood")
    if level == "studied_stimulus" and e:
        return (f"{song} by {artist} is the recording used in a published randomised study ({e['short_citation']}).")
    if level == "raga_category" and e:
        return (f"{song} by {artist} is a piece in raga {e['raga']}.")
    return (f"{song} by {artist} comes from the project's supplied Spotify research-source collection.")


def _song_explanation(song_row, ctx, is_research=False):
    song = str(song_row.get("song", "This song"))
    artist = str(song_row.get("artist", "the artist"))
    genre = str(song_row.get("genre", "this style of music"))
    mood = str(ctx.get("mood_state", "your current mood"))

    if is_research:
        return _research_explanation(song_row, ctx)

    return f"{song} by {artist} ({genre}) was selected to match your reported state ({mood})."


# --------------------------------------------------------------
# Navigation & Page Layout
# --------------------------------------------------------------
USER_PAGES = [
    ("Dashboard", "🏠 Home"),
    ("Profile", "📝 1 · My Profile"),
    ("Physiological Input", "💓 2 · How I Feel"),
    ("Music Preference & Recommendation", "🎵 3 · My Music"),
]
USER_PAGE_KEYS = [k for k, _ in USER_PAGES]
ADMIN_PAGES = ["Psychological Assessment", "Evaluation & Validation", "Bias & Risk of Bias",
               "Research Evidence", "Dataset / Research Mode", "Backend Monitor (Admin)"]
ADMIN_NONE = "👤 Participant view (back)"
st.session_state.setdefault("admin_nav", ADMIN_NONE)


def _go_user_page(key):
    st.session_state["page"] = key
    st.session_state["admin_nav"] = ADMIN_NONE


def _go_admin_page():
    sel = st.session_state.get("admin_nav", ADMIN_NONE)
    st.session_state["page"] = sel if sel in ADMIN_PAGES else "Dashboard"


def render_user_nav():
    st.markdown("<div class='mus-nav-title'>👇 Where would you like to go?</div>", unsafe_allow_html=True)
    cols = st.columns(len(USER_PAGES))
    for i, (key, label) in enumerate(USER_PAGES):
        with cols[i]:
            st.button(label, key=f"unav_{i}", use_container_width=True,
                      type="primary" if st.session_state["page"] == key else "secondary",
                      on_click=_go_user_page, args=(key,))


with st.sidebar:
    st.markdown("<span class='mus-admin-tag'>🔬 RESEARCHER / ADMIN ONLY</span>", unsafe_allow_html=True)
    st.markdown(f"### 🎧 {APP_NAME}")
    mode_badge("MongoDB Atlas connected" if db else "MongoDB connection failed", "research" if db else "warning")
    st.divider()
    st.radio("Researcher / Admin pages", [ADMIN_NONE] + ADMIN_PAGES, key="admin_nav", on_change=_go_admin_page)

page = st.session_state["page"]


def safety_banner():
    bg = COLORS['neutral']
    accent = COLORS['accent']
    html = (f"<div style='background:{bg};border-left:4px solid {accent};"
            f"padding:8px 14px;border-radius:6px;font-size:0.85em;margin-bottom:12px;'>"
            f"⚠ {saf.SAFETY_DISCLAIMER}</div>")
    st.markdown(html, unsafe_allow_html=True)


if db and st.session_state.verified and page in USER_PAGE_KEYS and page != "Dashboard":
    render_user_nav()

# ================================================================
# PAGE: Dashboard / Login
# ================================================================
if page == "Dashboard":
    st.title(f"🎧 {APP_NAME}")
    st.caption(APP_TAGLINE)
    safety_banner()

    card_open()
    st.subheader("👋 Hi! What should we call you?")
    if not db:
        st.error("Sign-in is disabled until MongoDB is connected.")
    elif not st.session_state.verified:
        _render_login_flow()
    else:
        st.success(f"😊 Hello **{st.session_state.username}**!")
        render_user_nav()
        if st.button("🚪 Logout"):
            ok, msg = _mark_checkout("logout")
            if ok:
                _reset_session_state()
                st.rerun()
    card_close()

# ================================================================
# PAGE: Profile
# ================================================================
elif page == "Profile":
    name = st.session_state.username
    st.title("📝 Your Profile")
    safety_banner()

    if st.session_state.get("profile_user") != name:
        st.session_state["profile_user"] = name
        st.session_state["profile_doc"] = db.profiles.find_one({"user": name}, {"_id": 0})
        st.session_state["editing_profile"] = st.session_state["profile_doc"] is None

    profile_doc = st.session_state["profile_doc"]
    if profile_doc and not st.session_state["editing_profile"]:
        card_open()
        st.write(f"**Age:** {profile_doc.get('age')}")
        st.write(f"**Preferred Genre:** {profile_doc.get('genre_pref')}")
        st.write(f"**Preferred Vibe:** {profile_doc.get('era_pref')}")
        if st.button("✏️ Update Profile"):
            st.session_state["editing_profile"] = True
            st.rerun()
        card_close()

    if st.session_state["editing_profile"]:
        defaults = profile_doc or {}
        card_open()
        with st.form("profile_form"):
            age = st.number_input("Age", min_value=0, max_value=100, value=int(defaults.get("age", 18)), step=1)
            genre_pref = st.selectbox("Preferred Genre", ["Bollywood", "Hindi Pop", "Ghazal", "Classical"])
            era_pref = st.selectbox("Preferred Vibe", ["60s songs", "90s songs", "Energetic songs", "calming songs", "Classical songs"])
            
            st.subheader("🧩 TIPI (Big Five)")
            saved_tipi = defaults.get("tipi", [4] * len(psy.TIPI_ALL))
            tipi = []
            for i, q in enumerate(psy.TIPI_ALL):
                label = f"{q}\n\nमराठी: {TIPI_MARATHI[i]}" if i < len(TIPI_MARATHI) else q
                tipi.append(st.slider(label, 1, 7, int(saved_tipi[i]) if i < len(saved_tipi) else 4, key=f"baseline_tipi_{i}"))

            st.subheader("💭 DASS-21 (baseline)")
            saved_dass = defaults.get("dass", [1] * len(psy.DASS_ALL))
            dass = []
            for i, q in enumerate(psy.DASS_ALL):
                label = f"{q}\n\nमराठी: {DASS_MARATHI[i]}" if i < len(DASS_MARATHI) else q
                dass.append(st.slider(label, 0, 3, int(saved_dass[i]) if i < len(saved_dass) else 1, key=f"baseline_dass_{i}"))

            st.subheader("🌍 WHOQOL-BREF")
            saved_whoqol = defaults.get("whoqol", [3] * len(psy.WHOQOL_ALL))
            whoqol = []
            for i, q in enumerate(psy.WHOQOL_ALL):
                label = f"{q}\n\nमराठी: {WHOQOL_MARATHI[i]}" if i < len(WHOQOL_MARATHI) else q
                whoqol.append(st.slider(label, 1, 5, int(saved_whoqol[i]) if i < len(saved_whoqol) else 3, key=f"baseline_whoqol_{i}"))

            submitted = st.form_submit_button("💾 Save Profile")
        if submitted:
            profile_data = {
                "name": st.session_state.get("username"), "age": int(age),
                "genre_pref": genre_pref, "era_pref": era_pref,
                "tipi": tipi, "dass": dass, "whoqol": whoqol,
                "updated_at_ist": _now_ist_str()
            }
            db.profiles.update_one({"user": name}, {"$set": profile_data}, upsert=True)
            st.session_state["profile_doc"] = profile_data
            st.session_state["editing_profile"] = False
            st.success("Profile saved.")
            st.rerun()
        card_close()

# ================================================================
# PAGE: Physiological Input
# ================================================================
elif page == "Physiological Input":
    st.title("💓 How are you feeling right now?")
    safety_banner()

    card_open()
    mood = st.selectbox("How is your mood?", ["Happy", "Sad", "Angry", "Calm", "Energetic"])
    stress = st.slider("How stressed do you feel? (0-100)", 0, 100, 40)
    hrv = st.slider("Heart rate (bpm)", 20, 200, 90)
    st.session_state["hrv_selfreport"] = hrv
    st.session_state["stress_selfreport"] = stress
    st.session_state["mood_selfreport"] = mood
    card_close()
    st.button("Next ➜ 🎵 Get my music", on_click=_go_user_page, args=("Music Preference & Recommendation",))

# ================================================================
# PAGE: Music Preference & Recommendation (WITH PULSE HARDWARE INTEGRATION)
# ================================================================
elif page == "Music Preference & Recommendation":
    st.title("🎵 Your Music & Therapeutic Pulse")
    safety_banner()

    profile_doc = st.session_state.get("profile_doc") or db.profiles.find_one({"user": st.session_state.username}, {"_id": 0})
    if not profile_doc:
        st.warning("Please complete **📝 1 · My Profile** first.")
        st.stop()

    name = st.session_state.username
    hrv = st.session_state.get("hrv_selfreport", 90)
    stress = st.session_state.get("stress_selfreport", 40)
    mood = st.session_state.get("mood_selfreport", "Calm")

    get_btn = st.button("🎧 Get Recommendations")

    if get_btn:
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                st.error("The music dataset is empty or could not be loaded. Please check modules/dataset.py and the dataset file.")
                st.stop()

            # Derive scores from the saved baseline questionnaires rather than passing
            # an incomplete context to modules/recommender.py.
            saved_tipi = profile_doc.get("tipi", []) or []
            saved_dass = profile_doc.get("dass", []) or []
            saved_whoqol = profile_doc.get("whoqol", []) or []

            def _valid_answers(values, expected, low, high):
                cleaned = []
                for value in list(values)[:expected]:
                    try:
                        number = float(value)
                        cleaned.append(number if np.isfinite(number) and low <= number <= high else None)
                    except (TypeError, ValueError):
                        cleaned.append(None)
                return cleaned

            def _mean_or(values, default):
                usable = [float(v) for v in values if v is not None]
                return float(np.mean(usable)) if usable else float(default)

            tipi_answers = _valid_answers(saved_tipi, 10, 1, 7)
            dass_answers = _valid_answers(saved_dass, 21, 0, 3)
            whoqol_answers = _valid_answers(saved_whoqol, 26, 1, 5)

            # TIPI items are 1-based: Extraversion Q1 + reverse Q6; Openness Q5 + reverse Q10.
            extraversion = (tipi_answers[0] + (8 - tipi_answers[5])) / 2 if len(tipi_answers) >= 6 and tipi_answers[0] is not None and tipi_answers[5] is not None else 4.0
            openness = (tipi_answers[4] + (8 - tipi_answers[9])) / 2 if len(tipi_answers) >= 10 and tipi_answers[4] is not None and tipi_answers[9] is not None else 4.0

            # DASS-21 standard subscales (each 7 items; multiply raw subscale by 2).
            # If the saved baseline is incomplete, use neutral zeros and disclose this in the assessment record.
            dep_idx = [2, 4, 9, 12, 15, 16, 20]
            anx_idx = [1, 3, 6, 8, 14, 18, 19]
            stress_idx = [0, 5, 7, 10, 11, 13, 17]
            dass_complete = len(dass_answers) == 21 and all(v is not None for v in dass_answers)
            depression = float(sum(dass_answers[i] for i in dep_idx) * 2) if dass_complete else 0.0
            anxiety = float(sum(dass_answers[i] for i in anx_idx) * 2) if dass_complete else 0.0
            dass_stress = float(sum(dass_answers[i] for i in stress_idx) * 2) if dass_complete else 0.0

            # WHOQOL-BREF domain scores use standard item groupings; reverse-score items 3 and 4.
            whoqol_complete = len(whoqol_answers) == 26 and all(v is not None for v in whoqol_answers)
            if whoqol_complete:
                q = list(whoqol_answers)
                q[2], q[3] = 6 - q[2], 6 - q[3]
                physical_qol = float(np.mean([q[i] for i in [2, 3, 9, 14, 15, 16, 17]]) * 20)
                social_qol = float(np.mean([q[i] for i in [19, 20, 21]]) * 20)
            else:
                physical_qol = social_qol = 60.0

            tipi_n = float(np.clip(_mean_or(tipi_answers, 4.0) / 7.0, 0.0, 1.0))
            whoql_n = float(np.clip(_mean_or(whoqol_answers, 3.0) / 5.0, 0.0, 1.0))
            dass_n = float(np.clip((depression + anxiety + dass_stress) / 126.0, 0.0, 1.0)) if dass_complete else 0.0
            mood_n = float({"Sad": 0.0, "Angry": 0.25, "Energetic": 0.5, "Calm": 0.75, "Happy": 1.0}.get(mood, 0.5))

            # The candidate-pool function takes depression and anxiety as arguments.
            pool = rec.build_candidate_pool(
                df, mood, hrv, stress,
                profile_doc.get("genre_pref", "Bollywood"),
                profile_doc.get("era_pref", "calming songs"),
                depression, anxiety, extraversion, physical_qol, social_qol
            )
            if pool is None or not isinstance(pool, pd.DataFrame) or pool.empty:
                st.warning("No candidate tracks were returned. Check the dataset columns and modules/recommender.py filters.")
                st.stop()

            score_context = {
                "mood_state": mood, "hrv": hrv, "stress": stress,
                "genre_pref": profile_doc.get("genre_pref", "Bollywood"),
                "era_pref": profile_doc.get("era_pref", "calming songs"),
                "extraversion": extraversion, "openness": openness,
                "depression": depression, "anxiety": anxiety,
                "physical_qol": physical_qol, "social_qol": social_qol,
                "tipi_n": tipi_n, "whoql_n": whoql_n, "dass_n": dass_n,
                "mood_n": mood_n, "user_name": name,
            }
            song_count = max(1, len(df))
            personal_q = np.zeros((100, song_count), dtype=float)
            global_q = np.zeros((100, song_count), dtype=float)
            if db:
                saved_q = db.qtables.find_one({"user": name}, {"qtable": 1, "num_songs": 1, "_id": 0})
                if saved_q and saved_q.get("num_songs") == song_count:
                    try:
                        loaded_q = np.asarray(saved_q.get("qtable"), dtype=float)
                        if loaded_q.shape == personal_q.shape:
                            personal_q = loaded_q
                    except (TypeError, ValueError):
                        pass

            # score_pool's RL lookup uses modulo/index arithmetic on song_id.
            # Preserve numeric IDs as-is; temporarily map non-numeric IDs to stable
            # numeric positions so string IDs cannot raise TypeError during scoring.
            original_song_ids = None
            if "song_id" in pool.columns and not pd.api.types.is_numeric_dtype(pool["song_id"]):
                original_song_ids = pool["song_id"].copy()
                unique_ids = list(dict.fromkeys(original_song_ids.astype(str).tolist()))
                numeric_id_map = {song_id: idx for idx, song_id in enumerate(unique_ids)}
                pool["song_id"] = original_song_ids.astype(str).map(numeric_id_map).astype(int)

            pool, _ = rec.score_pool(
                pool, score_context, personal_q, global_q,
                {"metadata": metadata, "rnn_model": rnn_model, "ncf_model": ncf_model},
                lambda: rec.FALLBACK_WEIGHTS, song_count
            )
            if original_song_ids is not None:
                pool["song_id"] = original_song_ids.values
            chosen = _select_recommendations(pool, n=5)
            if chosen is None or chosen.empty:
                st.warning("The recommender did not select any tracks. Please check the recommender configuration.")
                st.stop()

            assessment_id = uuid.uuid4().hex
            st.session_state["assessment_id"] = assessment_id
            st.session_state["got_recs"] = True
            st.session_state["recs"] = chosen.to_dict("records")
            st.session_state["pool"] = pool

            # Store the recommendation context/results in MongoDB only.
            if db:
                db.assessments.insert_one(_clean({
                    "assessment_id": assessment_id,
                    "user": name,
                    "session_id": st.session_state.get("session_id"),
                    "timestamp_ist": _now_ist_str(),
                    "timestamp_utc": _utc_iso(),
                    "mood_state": mood, "heart_rate_bpm": hrv, "self_reported_stress": stress,
                    "genre_pref": score_context["genre_pref"], "era_pref": score_context["era_pref"],
                    "depression_score": depression, "anxiety_score": anxiety,
                    "dass_stress_score": dass_stress, "dass_baseline_complete": dass_complete,
                    "whoqol_baseline_complete": whoqol_complete,
                    "recommendations": chosen.to_dict("records"),
                }))
        except Exception as exc:
            st.error(f"Recommendation generation failed ({type(exc).__name__}): {exc}")
            st.caption("Check that the deployed modules/recommender.py matches this main app and that required dataset/model files are present.")
            st.stop()

    if st.session_state.get("recs"):
        st.markdown("### 🎶 Recommended Tracks")
        st.warning(
            "Research prototype only. The displayed frequency is a software calculation, not a validated treatment setting. "
            "Do not connect electrodes or use this prototype on a person. Hardware output is restricted to disconnected bench testing."
        )
        bench_test_ack = st.checkbox(
            "I confirm the NodeMCU/ESP32 is disconnected from any person and this is a bench-only test.",
            key="bench_test_ack",
        )

        with st.expander("⚙️ Serial settings (NodeMCU / ESP32 — bench test only)"):
            hc1, hc2 = st.columns(2)
            port_in = hc1.text_input("Serial port", value=st.session_state["pulse_port"], key="pulse_port_input")
            baud_in = hc2.number_input("Baud rate", min_value=300, max_value=2_000_000,
                                       value=int(st.session_state["pulse_baud"]), step=100, key="pulse_baud_input")
            st.session_state["pulse_port"], st.session_state["pulse_baud"] = port_in.strip() or "COM3", int(baud_in)
            st.caption("pyserial detected." if pulse_hardware.serial_available
                       else "pyserial not installed - pulses run in safe simulation mode.")
        pulse_hardware.configure(st.session_state["pulse_port"], st.session_state["pulse_baud"])

        for i, s in enumerate(st.session_state["recs"]):
            card_open()
            col_info, col_pulse_ui = st.columns([1.2, 1])

            # Track BPM from dataset metadata, else a safe fallback
            try:
                track_bpm = float(s.get("tempo", s.get("bpm", 95.0)))
            except (TypeError, ValueError):
                track_bpm = float("nan")
            if not np.isfinite(track_bpm) or track_bpm <= 0:
                track_bpm = 110.0 if "energetic" in str(s.get("genre", "")).lower() else 75.0

            # Pure calculation only - no serial I/O on rerun
            freq_hz, period_ms = pulse_hardware.calculate_pulse(track_bpm, stress)
            track_key = str(s.get("song_id", i))

            with col_info:
                st.markdown(f"#### {i+1}. {s['song']} — {s['artist']}")
                st.caption(f"Genre: {s.get('genre', 'General')} | Estimated Tempo: {track_bpm:.0f} BPM")
                st.markdown(f"[🎧 Open on Spotify]({spotify_link(s['song'], s['artist'])})")

            with col_pulse_ui:
                st.markdown("**⚙️ Calculated demo signal**")
                m1, m2 = st.columns(2)
                m1.metric("Calculated Frequency", f"{freq_hz} Hz")
                m2.metric("Calculated Period", f"{period_ms} ms")
                st.caption("These values are not medically validated and must not be used to set stimulation for a person.")

                if st.button("Send demo frequency to board", key=f"btn_actuate_{track_key}_{i}"):
                    if not bench_test_ack:
                        st.warning("Confirm disconnected bench testing before sending any serial command.")
                    else:
                        f_sent, p_sent, mode, msg = pulse_hardware.send_pulse(track_bpm, stress)
                        st.session_state["pulse_status"][track_key] = {"mode": mode, "msg": msg}

                last = st.session_state["pulse_status"].get(track_key)
                if last is None:
                    st.caption("Status: not actuated yet (simulation unless hardware is reachable).")
                elif last["mode"] == "hardware":
                    st.success(last["msg"])
                else:
                    st.info(last["msg"])

            # Explicit participant feedback is written to MongoDB only.
            st.divider()
            fb_col1, fb_col2 = st.columns([1.2, 2])
            with fb_col1:
                rating = st.selectbox(
                    "Your feedback on this track",
                    ["Choose…", "Loved it", "Liked it", "Neutral", "Disliked it"],
                    key=f"track_rating_{st.session_state.get('assessment_id', 'na')}_{track_key}_{i}",
                )
            with fb_col2:
                comment = st.text_input(
                    "Optional comment",
                    key=f"track_comment_{st.session_state.get('assessment_id', 'na')}_{track_key}_{i}",
                    max_chars=500,
                )
            if st.button("Save feedback", key=f"save_track_feedback_{st.session_state.get('assessment_id', 'na')}_{track_key}_{i}"):
                if rating == "Choose…":
                    st.warning("Choose a rating before saving feedback.")
                elif not db:
                    st.error("MongoDB is unavailable; feedback was not saved.")
                else:
                    try:
                        aid = st.session_state.get("assessment_id")
                        if not aid:
                            raise RuntimeError("Missing assessment ID. Generate recommendations again first.")
                        feedback_doc = _clean({
                            "user": name,
                            "session_id": st.session_state.get("session_id"),
                            "assessment_id": aid,
                            "song_id": track_key,
                            "song": s.get("song", ""),
                            "artist": s.get("artist", ""),
                            "rating": rating,
                            "comment": comment.strip(),
                            "timestamp": _utc_iso(),
                            "timestamp_ist": _now_ist_str(),
                        })
                        db.recommendation_feedback.update_one(
                            {"user": name, "assessment_id": aid, "song_id": track_key},
                            {"$set": feedback_doc},
                            upsert=True,
                        )
                        st.success("Feedback saved to MongoDB.")
                    except Exception as exc:
                        st.error(f"Feedback could not be saved to MongoDB ({type(exc).__name__}): {exc}")

            card_close()

# ================================================================
# FALLBACK / OTHER PAGES
# ================================================================
else:
    st.title(f"📄 {page}")
    st.info("Section active.")
