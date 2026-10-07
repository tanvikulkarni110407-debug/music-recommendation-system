"""
MuSync - research prototype (Streamlit + MongoDB Atlas)

REQUIRED STREAMLIT SECRETS  (App -> Settings -> Secrets).  NOTHING secret lives in this file.
-----------------------------------------------------------------------------------------
MONGODB_URI        = "mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority"
MONGODB_DATABASE   = "musync"
ADMIN_PASSWORD     = "choose-a-strong-password"

# --- OTP authentication ---
OTP_PEPPER         = "long-random-string"        # secret key used to HMAC-hash OTPs (python -c "import secrets;print(secrets.token_hex(32))")
AUTHENTICATOR_EMAILS = ["a1@gmail.com", "a2@gmail.com", "a3@gmail.com", "a4@gmail.com", "a5@gmail.com"]
SMTP_HOST          = "smtp.gmail.com"            # optional, default smtp.gmail.com
SMTP_PORT          = 465                         # optional, 465 (SSL) or 587 (STARTTLS)
SMTP_USER          = "your.sender@gmail.com"     # the Gmail that SENDS the mails
SMTP_PASSWORD      = "gmail-app-password"        # Google "App password" (needs 2-step verification)

requirements.txt must contain (besides what you already have):
    pymongo[srv]  dnspython  certifi  openpyxl
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
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import certifi                                              # TLS fix for MongoDB Atlas
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo


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

st.set_page_config(page_title=APP_NAME, layout="wide")
inject_theme()

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
OTP_TTL_MIN = 5                      # OTP lifetime
OTP_MAX_ATTEMPTS = 5                 # wrong-code attempts per OTP
OTP_RESEND_COOLDOWN_S = 60           # per browser session
OTP_MAX_REQUESTS_PER_ID = 5          # per (User ID + Gmail) per 15 minutes
OTP_MAX_REQUESTS_GLOBAL = 40         # whole app per 15 minutes (anti-spam for the authenticators)
ENFORCE_UNIQUE_EMAIL = True          # one Gmail <-> one User ID (set False if you want one Gmail to hold several IDs)
USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{2,39}$")
GMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@gmail\.com$", re.I)
ANY_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

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
    # OTP flow - strictly per browser session, never global
    ("otp_request_id", None), ("otp_user", None), ("otp_display_id", None), ("otp_email", None),
    ("otp_client_token", None), ("otp_sent_at", None), ("otp_expires_ist", None), ("otp_flash", None),
    ("rr_saved", []), ("rr_features_latest", None),
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
        if k.startswith(("fb_done_", "sess_fb_done_", "rate_", "phq4_", "pss4_", "login_", "otp_code")):
            del st.session_state[k]


# --------------------------------------------------------------
# Short-form state check-in for RETURNING users (8 items)
#   PHQ-4 : Kroenke, Spitzer, Williams & Lowe (2009), Psychosomatics 50(6):613-621
#           (validated in the general population: Lowe et al., 2010, J Affect Disord 122:86-95)
#   PSS-4 : Cohen, Kamarck & Mermelstein (1983), J Health Soc Behav 24:385-396
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
# Secrets helper (Streamlit secrets first, then environment variables)
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
    """
    Collections (all linked by `user` = normalised Participant ID; session data also by `session_id`):
      users                      1 doc per participant  (identity, Gmail, login_count, counters, registration-email flag)
      user_profiles              1 doc per participant  (current profile; updated in place)
      profile_history            1 doc per profile save (permanent versions)
      sessions                   1 doc per login        (session_id, session_number, start/end/duration)
      login_history              1 doc per login EVENT and 1 doc per logout EVENT
      assessments                1 doc per "Get Recommendations" (inputs, scores, recommendations) -> session_id
      physiological_measurements self-report + uploaded-RR readings -> session_id / assessment_id
      state_checkins             short PHQ-4/PSS-4 check-ins -> session_id / assessment_id
      recommendation_feedback    1 doc per rated song -> session_id / assessment_id
      experiments                end-of-session feedback -> session_id / assessment_id
      qtables                    RL tables (per user + "global"), updated atomically with $inc
      bias_assessments           bias checklists
      otp_requests               hashed OTPs (TTL-expired automatically; the OTP itself is never stored)
    """

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
        self.otp_requests = self.mongo_db["otp_requests"]

        self.mode = "mongodb"
        self.index_warnings = []
        self._create_indexes()

    def _ix(self, coll, keys, **kw):
        """Each index is created on its own, so one failing index (e.g. old duplicate data) never blocks the others."""
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
        self._ix(self.otp_requests, "request_id", unique=True)
        self._ix(self.otp_requests, [("user", 1), ("email", 1), ("created_at", -1)])
        self._ix(self.otp_requests, "expires_at", expireAfterSeconds=3600)     # hashed OTP docs vanish 1 h after expiry


@st.cache_resource(show_spinner=False)
def _connect_mongodb(uri, db_name):
    # Cache is keyed by the actual URI/database so changing Streamlit Secrets
    # cannot leave the app attached to an old MongoDB target.
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
    """'Ratika', ' ratika ', 'RATIKA' and 'Ratika  Sharma' -> same stored identity every time."""
    return re.sub(r"\s+", "_", raw.strip().lower())


def _clean(o):
    """Make any object BSON-safe (numpy scalars/arrays, NaN, Timestamps -> plain python)."""
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


def _mask_email(e):
    try:
        name, dom = e.split("@")
        return name[0] + "***@" + dom
    except Exception:
        return "***"


# --------------------------------------------------------------
# E-mail (OTP to authenticators, one-time registration mail to participant)
# --------------------------------------------------------------
def _authenticator_emails():
    raw = _get_secret("AUTHENTICATOR_EMAILS")
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = re.split(r"[,\s;]+", raw)
    else:
        try:
            parts = [str(x) for x in list(raw)]
        except Exception:
            parts = []
    out = []
    for p in parts:
        p = p.strip().lower()
        if p and ANY_EMAIL_RE.match(p) and p not in out:
            out.append(p)
    return out


def _auth_config_missing():
    missing = []
    if not _get_secret("OTP_PEPPER"):
        missing.append("OTP_PEPPER")
    if not _get_secret("BREVO_API_KEY"):
        missing.append("BREVO_API_KEY")
    if not _get_secret("SENDER_EMAIL"):
        missing.append("SENDER_EMAIL")
    if not _authenticator_emails():
        missing.append("AUTHENTICATOR_EMAILS")
    return missing


def _brevo_send(subject, html_content, recipients):
    api_key = str(_get_secret("BREVO_API_KEY") or "").strip()
    sender_email = str(_get_secret("SENDER_EMAIL") or "").strip()
    if not api_key or not sender_email:
        raise RuntimeError("BREVO_API_KEY or SENDER_EMAIL is missing")

    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key["api-key"] = api_key

    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(configuration)
    )

    email_data = sib_api_v3_sdk.SendSmtpEmail(
        sender={"email": sender_email},
        to=[{"email": str(r).strip()} for r in recipients if str(r).strip()],
        subject=subject,
        html_content=html_content,
    )

    api_instance.send_transac_email(email_data)


def _send_otp_email(display_id, participant_email, otp, expires_ist):
    auths = _authenticator_emails()
    if not auths:
        raise RuntimeError("AUTHENTICATOR_EMAILS is empty")

    subject = f"MuSync login OTP for participant {display_id}"
    html_content = f"""
    <html>
    <body>
        <h2>MuSync Login OTP</h2>
        <p>A participant is trying to sign in to MuSync.</p>
        <p><b>Participant ID:</b> {display_id}</p>
        <p><b>Gmail:</b> {participant_email}</p>
        <h2>{otp}</h2>
        <p><b>Valid until:</b> {expires_ist} IST</p>
        <p>Valid for {OTP_TTL_MIN} minutes and {OTP_MAX_ATTEMPTS} attempts.</p>
        <p>Give this code to the participant only after you have confirmed their identity.</p>
        <p>If you did not expect this request, ignore this e-mail.</p>
    </body>
    </html>
    """
    _brevo_send(subject, html_content, auths)


def _send_registration_email(display_id, username, participant_email):
    subject = f"Your {APP_NAME} Participant ID"
    html_content = f"""
    <html>
    <body>
        <h2>Welcome to {APP_NAME}!</h2>
        <p>You are now registered.</p>
        <p><b>Your User ID / Participant ID:</b> {display_id}</p>
        <p><b>Registered Gmail:</b> {participant_email}</p>
        <p>Please use exactly this User ID and this Gmail every time you return.</p>
        <p>Do not share this e-mail.</p>
    </body>
    </html>
    """
    _brevo_send(subject, html_content, [participant_email])


# --------------------------------------------------------------
# OTP authentication  (hashed, expiring, attempt-limited, bound to User ID + Gmail + request + browser session)
# --------------------------------------------------------------
def _otp_hash(request_id, username, email, otp):
    pepper = str(_get_secret("OTP_PEPPER"))
    return hmac.new(pepper.encode(), f"{request_id}|{username}|{email}|{otp}".encode(), hashlib.sha256).hexdigest()


def _check_identity(username, email):
    """Same User ID must always come with the same Gmail. Returns (ok, message)."""
    doc = db.users.find_one({"user": username}, {"email": 1})
    if doc:
        stored = (doc.get("email") or "").lower()
        if stored and not stored.endswith("@local") and stored != email:
            return False, "This User ID is already registered with a different Gmail address."
        return True, ""
    if ENFORCE_UNIQUE_EMAIL and db.users.find_one({"email": email}, {"_id": 1}):
        return False, "This Gmail address is already registered with a different User ID."
    return True, ""


def _clear_otp_state():
    for k in ("otp_request_id", "otp_user", "otp_display_id", "otp_email", "otp_client_token", "otp_expires_ist"):
        st.session_state[k] = None
    for k in ("otp_code_input",):
        if k in st.session_state:
            del st.session_state[k]


def _request_otp(raw_uid, raw_email):
    """Create ONE hashed OTP request and e-mail the OTP to the authenticators. Returns (ok, message)."""
    if not db:
        return False, "MongoDB is not connected."
    raw_uid = (raw_uid or "").strip()
    email = (raw_email or "").strip().lower()
    if not USER_ID_RE.match(raw_uid):
        return False, "User ID must be 3-40 characters (letters, digits, space, _ . -)."
    if not GMAIL_RE.match(email):
        return False, "Please enter a valid Gmail address (name@gmail.com)."
    missing = _auth_config_missing()
    if missing:
        return False, f"Authentication is not configured on the server (missing secrets: {', '.join(missing)})."
    last = st.session_state.get("otp_sent_at")
    if last and time.time() - last < OTP_RESEND_COOLDOWN_S:
        return False, f"Please wait {int(OTP_RESEND_COOLDOWN_S - (time.time() - last))} s before requesting another OTP."

    username = _normalize_username(raw_uid)
    try:
        ok, msg = _check_identity(username, email)
        if not ok:
            return False, msg

        now = datetime.now(timezone.utc)
        window = now - timedelta(minutes=15)
        if db.otp_requests.count_documents({"user": username, "email": email, "created_at": {"$gt": window}}) >= OTP_MAX_REQUESTS_PER_ID:
            return False, "Too many OTP requests for this User ID. Please try again in a few minutes."
        if db.otp_requests.count_documents({"created_at": {"$gt": window}}) >= OTP_MAX_REQUESTS_GLOBAL:
            return False, "The system is receiving too many OTP requests. Please try again shortly."

        request_id = uuid.uuid4().hex
        client_token = secrets.token_urlsafe(24)               # lives only in THIS browser session
        otp = f"{secrets.randbelow(1_000_000):06d}"
        expires = now + timedelta(minutes=OTP_TTL_MIN)
        expires_ist = expires.astimezone(ZoneInfo("Asia/Kolkata")).strftime("%I:%M:%S %p")

        db.otp_requests.update_many({"user": username, "email": email, "status": "pending"},
                                    {"$set": {"status": "superseded"}})
        db.otp_requests.insert_one({
            "request_id": request_id,
            "user": username,
            "display_id": raw_uid,
            "email": email,
            "otp_hash": _otp_hash(request_id, username, email, otp),
            "session_token_hash": hashlib.sha256(client_token.encode()).hexdigest(),
            "created_at": now,
            "expires_at": expires,
            "attempts": 0,
            "max_attempts": OTP_MAX_ATTEMPTS,
            "status": "pending",
        })
    except Exception as e:
        return False, f"Could not create the OTP request ({type(e).__name__})."

    try:
        _send_otp_email(raw_uid, email, otp, expires_ist)
    except Exception as e:
        try:
            db.otp_requests.update_one({"request_id": request_id}, {"$set": {"status": "send_failed"}})
        except Exception:
            pass
        return False, f"The OTP e-mail could not be sent ({type(e).__name__}). Check the SMTP secrets."
    finally:
        otp = None                                             # never keep the plain OTP around

    st.session_state["otp_request_id"] = request_id
    st.session_state["otp_user"] = username
    st.session_state["otp_display_id"] = raw_uid
    st.session_state["otp_email"] = email
    st.session_state["otp_client_token"] = client_token
    st.session_state["otp_sent_at"] = time.time()
    st.session_state["otp_expires_ist"] = expires_ist
    st.session_state["otp_flash"] = "OTP sent to the study authenticators."
    return True, "OTP sent."


def _complete_login(username, display_id, email):
    """Persist a verified login as ONE atomic MongoDB transaction."""
    if not db:
        raise RuntimeError("MongoDB is not connected.")

    ist, utc = _now_ist_str(), _utc_iso()
    session_id = uuid.uuid4().hex

    # Atlas supports transactions. This guarantees that users + sessions +
    # login_history are either ALL written or NONE are written.
    with db.client.start_session() as mongo_session:
        with mongo_session.start_transaction():
            user_doc = db.users.find_one_and_update(
                {"user": username},
                {
                    "$set": {"last_login_ist": ist, "last_login_utc": utc, "email": email},
                    "$setOnInsert": {
                        "display_id": display_id,
                        "first_login_ist": ist, "first_login_utc": utc,
                        "registration_email_sent": False,
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
                "session_id": session_id, "user": username, "email": email,
                "session_number": session_number,
                "started_at_ist": ist, "started_at_utc": utc,
                "ended_at_ist": None, "ended_at_utc": None, "duration_min": None,
                "status": "active", "is_first_session": is_new,
            }, session=mongo_session)

            db.login_history.insert_one({
                "event": "login", "session_id": session_id, "user": username,
                "user_email": email, "session_number": session_number,
                "timestamp_ist": ist, "timestamp_utc": utc,
            }, session=mongo_session)

    # Registration mail is deliberately outside the DB transaction. An e-mail
    # failure must NEVER delete an otherwise valid login/session.
    reg_msg = None
    claim = db.users.find_one_and_update(
        {"user": username, "registration_email_sent": False},
        {"$set": {"registration_email_sent": "sending"}},
    )
    if claim:
        try:
            _send_registration_email(display_id, username, email)
            db.users.update_one({"user": username}, {"$set": {
                "registration_email_sent": True,
                "registration_email_sent_ist": _now_ist_str(),
            }})
            reg_msg = "Your Participant ID has been e-mailed to your Gmail."
        except Exception as e:
            db.users.update_one({"user": username}, {"$set": {"registration_email_sent": False}})
            reg_msg = f"Registered, but the ID e-mail could not be sent ({type(e).__name__})."

    st.session_state["verified"] = True
    st.session_state["username"] = username
    st.session_state["user_email"] = email
    st.session_state["session_number"] = session_number
    st.session_state["session_id"] = session_id
    st.session_state["full_baseline_this_session"] = False
    st.session_state["profile_user"] = None
    st.session_state["profile_doc"] = None
    st.session_state["otp_flash"] = (
        ("Registration complete. " if is_new else f"Welcome back - Session #{session_number}. ") + (reg_msg or ""))
    _clear_otp_state()


def _verify_otp(code):
    """Verify the OTP of THIS browser session's request only. Returns (ok, message)."""
    rid, username, email = (st.session_state.get(k) for k in ("otp_request_id", "otp_user", "otp_email"))
    token = st.session_state.get("otp_client_token")
    if not (rid and username and email and token):
        return False, "No active OTP request. Please request a new OTP."
    code = (code or "").strip()
    if not re.fullmatch(r"\d{6}", code):
        return False, "Enter the 6-digit code."

    now = datetime.now(timezone.utc)
    base = {"request_id": rid, "user": username, "email": email,
            "session_token_hash": hashlib.sha256(token.encode()).hexdigest()}
    try:
        # Atomically consume ONE attempt; fails if not pending, expired or attempts exhausted.
        req = db.otp_requests.find_one_and_update(
            {**base, "status": "pending", "attempts": {"$lt": OTP_MAX_ATTEMPTS}, "expires_at": {"$gt": now}},
            {"$inc": {"attempts": 1}}, return_document=ReturnDocument.AFTER)
        if req is None:
            d = db.otp_requests.find_one(base, {"status": 1, "attempts": 1, "expires_at": 1})
            if not d:
                return False, "This OTP request does not belong to this browser session. Request a new OTP."
            if d.get("status") == "verified":
                return False, "This OTP was already used."
            if d.get("expires_at") and d["expires_at"] < now.replace(tzinfo=None):
                return False, "The OTP has expired. Please request a new one."
            if d.get("attempts", 0) >= OTP_MAX_ATTEMPTS:
                return False, "Too many wrong attempts. Please request a new OTP."
            return False, "This OTP request is no longer active. Please request a new one."

        if not hmac.compare_digest(str(req["otp_hash"]), _otp_hash(rid, username, email, code)):
            left = OTP_MAX_ATTEMPTS - int(req.get("attempts", 0))
            if left <= 0:
                db.otp_requests.update_one({"request_id": rid, "status": "pending"}, {"$set": {"status": "locked"}})
                return False, "Incorrect OTP. No attempts left - please request a new OTP."
            return False, f"Incorrect OTP. {left} attempt(s) left."

        # Single-use: only the call that flips pending -> verified may log in.
        done = db.otp_requests.find_one_and_update({"request_id": rid, "status": "pending"},
                                                   {"$set": {"status": "verified", "verified_at": now}})
        if done is None:
            return False, "This OTP was already used. Please request a new one."
        try:
            _complete_login(username, req.get("display_id") or username, email)
        except Exception as e:
            db.otp_requests.update_one({"request_id": rid}, {"$set": {"status": "error"}})
            return False, f"Login could not be saved in MongoDB ({type(e).__name__}: {e}). Please request a new OTP."
        return True, "Verified."
    except PyMongoError as e:
        return False, f"Database error during verification ({type(e).__name__})."


def _render_login_flow():
    flash = st.session_state.get("otp_flash")
    if flash:
        st.success(flash)
        st.session_state["otp_flash"] = None

    if not st.session_state.get("otp_request_id"):
        st.info("Welcome to MuSync. Enter your **User ID (Participant ID)** and the **Gmail** you registered with. "
                "New participants are registered automatically after verification.")
        uid = st.text_input("User ID / Participant ID", key="login_uid_input", max_chars=40)
        gm = st.text_input("Gmail address", key="login_gmail_input", max_chars=100)
        if st.button("Send OTP", type="primary", key="send_otp_btn"):
            ok, msg = _request_otp(uid, gm)
            if ok:
                st.rerun()
            else:
                st.error(msg)
    else:
        st.info(f"A 6-digit OTP for **{st.session_state['otp_display_id']}** "
                f"({_mask_email(st.session_state['otp_email'])}) was sent to the study authenticators. "
                f"Ask an authenticator for the code. It is valid until {st.session_state['otp_expires_ist']} IST "
                f"and allows {OTP_MAX_ATTEMPTS} attempts.")
        code = st.text_input("Enter the 6-digit OTP", key="otp_code_input", max_chars=6, type="password")
        c1, c2, c3 = st.columns(3)
        if c1.button("Verify & Continue", type="primary", key="verify_otp_btn"):
            ok, msg = _verify_otp(code)
            if ok:
                st.rerun()
            else:
                st.error(msg)
        if c2.button("Resend OTP", key="resend_otp_btn"):
            ok, msg = _request_otp(st.session_state["otp_display_id"], st.session_state["otp_email"])
            if ok:
                st.rerun()
            else:
                st.error(msg)
        if c3.button("Change User ID / Gmail", key="change_id_btn"):
            _clear_otp_state()
            st.rerun()


# --------------------------------------------------------------
# Check-out (one logout event + closes the session row, only once per session)
# --------------------------------------------------------------
def _mark_checkout(reason):
    """Atomically close the active session and create exactly one logout event."""
    if not db:
        return False, "MongoDB is not connected."

    sid = st.session_state.get("session_id")
    if not sid:
        return False, "No active session ID is present in this browser session."

    now_utc = datetime.now(timezone.utc)
    ist = _now_ist_str()

    sdoc = db.sessions.find_one({"session_id": sid, "ended_at_utc": None})
    if not sdoc:
        # Already closed is not an error; there is simply nothing more to write.
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
                    "user_email": sdoc.get("email"), "session_number": sdoc.get("session_number"),
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


# --------------------------------------------------------------
# Data access helpers (MongoDB is the single source of truth)
# --------------------------------------------------------------
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
    """Concurrency-safe Q-table persistence: only the CHANGED cells are applied with $inc, so two users
    updating the shared 'global' table at the same time never overwrite each other."""
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
# WESAD research mode helpers (precomputed HRV features)
# Reference: Schmidt et al. (2018), ACM ICMI, pp. 400-408.
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
    """Heuristic 0-100 stress index: lower RMSSD -> higher index (5th-95th percentile normalisation).
    Research heuristic only (Task Force of ESC/NASPE, 1996) - not a clinical measure."""
    lo = float(np.nanpercentile(wdf["rmssd_ms"], 5))
    hi = float(np.nanpercentile(wdf["rmssd_ms"], 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 50.0
    return float(np.clip(100.0 * (1.0 - (rmssd_ms - lo) / (hi - lo)), 0.0, 100.0))


# --------------------------------------------------------------
# RESEARCH EVIDENCE
# --------------------------------------------------------------
# 1) Evidence that is actually stated in the supplied paper. Nothing here is invented:
#    fields the paper does not report are marked "Not reported in the paper".
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
        "mood": ("Mood labels were not tested; outcomes were DASS-21 stress, anxiety and depression and HRV. "
                 "(The authors describe raga as carrying emotional resonance/'Rasa' in general terms only.)"),
        "instrumentation": "Not reported in the paper",
        "musical_characteristics": ("'Sampoorn' (complete) raga using all seven notes; traditionally associated with "
                                    "early-morning performance and often presented at the close of a concert "
                                    "(as stated by the authors)."),
        "psychological": ("After 6 days all DASS-21 subscales fell within the VR-raga group (stress dz = 1.11, "
                          "anxiety dz = 1.04, depression dz = 1.16; all p < .001) with no significant change in the "
                          "control group; group x time interactions were significant for all three subscales, and "
                          "the post-pre change differed from control with large effects (dz = 1.04, 1.07, 0.95)."),
        "physiological": ("Short-term seated HRV (PPG, ear lobe): on day 1 only respiration rate differed between groups "
                          "(lower in the VR-raga group); on day 6 all seven selected HRV parameters (SDNN, SDHR, HTI, "
                          "LF/HF, respiration rate, SD2/SD1, SampEn) differed significantly between groups after "
                          "Benjamini-Hochberg correction. The authors interpret this as reduced physiological stress "
                          "and enhanced autonomic balance."),
        "limitations": [
            "The effect of VR was not isolated from the effect of the music (no VR-without-raga or music-without-VR arm).",
            "No placebo/sham group; expectancy effects are possible.",
            "Only 6 days; DASS-21 covers the whole week, not individual days.",
            "Healthy 18-35-year-olds (mean 24.4) at one institute; not clinical patients.",
            "One raga and one specific recorded performance were tested; audio-only listening was not tested.",
        ],
    }
}

# 2) Candidate tracks that can enter the SAME ranking pool as the dataset songs.
#    evidence_level:  studied_stimulus = the exact recording used in a published study
#                     raga_category    = a song of a raga that a paper studied (the SONG itself was not studied)
#                     project_source   = from the project's supplied Spotify source pool (no outcome claim is made)
RESEARCH_SOURCE_TRACKS = [
    {"key": "miyan_ki_todi", "song_id": "research_src_miyan_ki_todi", "song": "Raag Miyan Ki Todi",
     "artist": "Nikhil Banerjee", "genre": "Indian Classical", "raga": "Miyan Ki Todi",
     "evidence_level": "project_source", "evidence_key": "",
     "source_url": "https://open.spotify.com/track/2wmy0bj9Lchz0cQnmriBR0",
     "source_name": "Fond Memories-Sitar Vol-1 (provided Spotify source)"},
    {"key": "rageshree", "song_id": "research_src_rageshree", "song": "Raag Rageshree",
     "artist": "Nikhil Banerjee", "genre": "Indian Classical", "raga": "Rageshree",
     "evidence_level": "project_source", "evidence_key": "",
     "source_url": "https://open.spotify.com/album/4CbM5IC1txrx40X0AYyPmP",
     "source_name": "Fond Memories-Sitar Vol-1 (provided Spotify source)"},
    {"key": "nat_bhairav", "song_id": "research_src_nat_bhairav", "song": "Raag Nat Bhairav",
     "artist": "Nikhil Banerjee", "genre": "Indian Classical", "raga": "Nat Bhairav",
     "evidence_level": "project_source", "evidence_key": "",
     "source_url": "https://open.spotify.com/track/1Y76LppGA9FsyAackq9uLy",
     "source_name": "Fond Memories-Sitar Vol-1 (provided Spotify source)"},
]
RESEARCH_PAPER_TRACKS = [
    {"key": "bhairavi_darbar", "song_id": "research_src_bhairavi_darbar", "song": "Meditative Raag Bhairavi",
     "artist": "Kaushiki Chakraborty", "genre": "Indian Classical", "raga": "Bhairavi",
     "evidence_level": "studied_stimulus", "evidence_key": "bhairavi",
     "source_url": RAGA_EVIDENCE["bhairavi"]["stimulus_url"],
     "source_name": "Stimulus used in " + RAGA_EVIDENCE["bhairavi"]["short_citation"] + " (Darbar VR360 video)"},
]
RESEARCH_CANDIDATES = RESEARCH_PAPER_TRACKS + RESEARCH_SOURCE_TRACKS

# Exact Spotify sources supplied by the project owner (recorded as metadata only).
RESEARCH_SPOTIFY_SOURCES = [
    "https://open.spotify.com/album/4CbM5IC1txrx40X0AYyPmP",
    "https://open.spotify.com/playlist/0efes1si9D7BtI93izeQJ1",
    "https://open.spotify.com/playlist/1WZr6aA4096hUr9ssO2bcZ",
    "https://open.spotify.com/playlist/5Y4kPb6Q4Ftrui4e5cRsKb",
]

# Ranking heuristics (application design choices, NOT findings of the paper):
# a research track receives a score bonus only when the participant's current inputs resemble the population/outcomes
# the evidence is about (elevated stress / psychological distress / HR, distress mood, taste for classical music).
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
    """0..1 match between the participant's current state and the state the studies address."""
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
    """Put research tracks through the SAME scoring pool. Dataset songs of a studied raga are tagged; source-only tracks
    are appended with neutral model scores. A track is boosted only if the participant's state matches its evidence."""
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
    for cand in RESEARCH_CANDIDATES:                         # exact recording already in the dataset
        m = (song_n == _norm(cand["song"])) & (artist_n == _norm(cand["artist"]))
        for idx in pool.index[m]:
            tag(idx, cand["evidence_level"], cand["evidence_key"], cand["raga"], cand["source_url"], cand["source_name"])
            matched.add(cand["key"])

    for idx in pool.index:                                   # dataset songs that belong to a studied raga
        if pool.at[idx, "evidence_level"]:
            continue
        text = " ".join(str(pool.at[idx, c]) for c in ("song", "genre", "raga_name") if c in pool.columns)
        for ek, e in RAGA_EVIDENCE.items():
            if e["pattern"].search(text):
                tag(idx, "raga_category", ek, e["raga"], "", e["short_citation"])
                break

    for idx in pool.index[pool["evidence_level"] != ""]:     # bonus for tagged dataset rows
        if str(pool.at[idx, "song_id"]) in flagged:
            continue
        bonus = EVIDENCE_WEIGHT.get(pool.at[idx, "evidence_level"], 0.0) * strength * spread
        pool.at[idx, "evidence_bonus"] = bonus
        if pd.notna(pool.at[idx, "final_score"]):
            pool.at[idx, "final_score"] += bonus

    new_rows = []                                            # source-only candidates (not in the dataset)
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
    """Existing selector first; if it fails, fall back to top-n. Any research track that is genuinely in the top-n by
    final score is guaranteed to survive the diversity step (nothing is forced if it did not earn its place)."""
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
            # replace the lowest-ranked non-research item
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
        return (f"{song} by {artist} is the recording used in a published randomised study ({e['short_citation']}). "
                f"Healthy adults who listened to raga {e['raga']} for 15 minutes a day for 6 days (in VR with headphones) "
                f"reported lower DASS-21 stress, anxiety and depression scores than a no-music control and showed "
                f"different HRV patterns. It ranked well here because {state} resemble the outcomes that study addressed. "
                f"Caveats: audio-only listening was not tested, tempo and instrumentation are not reported in the paper, "
                f"and the study cannot promise the same effect for you.")
    if level == "raga_category" and e:
        return (f"{song} by {artist} is a piece in raga {e['raga']}. A published study ({e['short_citation']}) examined "
                f"raga {e['raga']} using one specific recording, not this song, so this is raga-level evidence only. "
                f"It ranked well because {state} fit the outcomes that study addressed alongside your other preferences.")
    return (f"{song} by {artist} comes from the project's supplied Spotify research-source collection. No outcome "
            f"evidence for this specific recording is claimed. It was ranked together with all other songs and chosen "
            f"because it fit {state} and your listening preferences.")


def _song_explanation(song_row, ctx, is_research=False):
    """Explain one recommendation in plain, user-friendly language."""
    song = str(song_row.get("song", "This song"))
    artist = str(song_row.get("artist", "the artist"))
    genre = str(song_row.get("genre", "this style of music"))
    mood = str(ctx.get("mood_state", "your current mood"))
    preferred_genre = str(ctx.get("genre_pref", "")).strip()
    preferred_vibe = str(ctx.get("era_pref", "")).strip()

    if is_research:
        return _research_explanation(song_row, ctx)

    reasons = []

    signal_labels = [
        ("pref_bias", "your music preferences"),
        ("physio_fit", "the state you reported in the physiological-input section"),
        ("psy_bias", "the information from your psychological profile"),
        ("rnn_score", "the song's suitability based on patterns in the music recommendations"),
        ("ncf_score", "similar listening patterns represented in the recommendation history"),
        ("personal_q", "your previous feedback on recommendations"),
    ]

    scored = []
    for col, label in signal_labels:
        try:
            value = float(song_row.get(col, np.nan))
            if np.isfinite(value):
                scored.append((value, label))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)

    if preferred_genre:
        if preferred_genre.casefold() in genre.casefold() or genre.casefold() in preferred_genre.casefold():
            reasons.append(f"it matches your preference for {preferred_genre}")
        else:
            reasons.append("it adds variety while staying within the broader listening style considered for you")

    mood_text = {
        "happy": "its overall musical character can complement a positive mood",
        "sad": "its overall musical character can provide a gentle listening experience",
        "angry": "its overall character can offer a more balanced listening direction",
        "calm": "its overall character fits a calm and relaxed listening experience",
        "energetic": "its overall character fits an energetic listening experience",
    }.get(mood.casefold(), "its musical character was considered in relation to your current mood")
    reasons.append(mood_text)

    if preferred_vibe and preferred_vibe.casefold() in (song + " " + genre).casefold():
        reasons.append(f"it also fits the {preferred_vibe} style you selected")

    used_labels = set()
    for _, label in scored:
        if label not in used_labels:
            reasons.append(f"the recommendation also gave weight to {label}")
            used_labels.add(label)
        if len(used_labels) >= 2:
            break

    pool = st.session_state.get("pool")
    try:
        final_score = float(song_row.get("final_score", np.nan))
    except Exception:
        final_score = np.nan

    if pool is not None and not pool.empty and np.isfinite(final_score) and "final_score" in pool.columns:
        scores = pd.to_numeric(pool["final_score"], errors="coerce").dropna().sort_values(ascending=False)
        if len(scores):
            rank = int((scores > final_score).sum()) + 1
            if rank == 1:
                reasons.append("it was the strongest overall match among the personalized options")
            elif rank == 2:
                reasons.append("it was one of the strongest matches in the final selection")
            elif rank == 3:
                reasons.append("it provides a strong alternative within the final selection")
            elif rank == 4:
                reasons.append("it was kept to add useful variety to the final selection")
            else:
                reasons.append("it was included to broaden the final selection while remaining relevant")

    unique = []
    for r in reasons:
        if r not in unique:
            unique.append(r)
    unique = unique[:4]

    if len(unique) == 1:
        reason_text = unique[0]
    elif len(unique) == 2:
        reason_text = unique[0] + " and " + unique[1]
    else:
        reason_text = ", ".join(unique[:-1]) + ", and " + unique[-1]

    return (
        f"{song} by {artist} was recommended because {reason_text}. "
        "The goal is to give you a relevant and varied listening choice based on the information you provided."
    )


# --------------------------------------------------------------
# Export helpers (admin) - MongoDB -> CSV / Excel
# --------------------------------------------------------------
def _docs_to_df(coll, drop=("_id",), query=None):
    rows = list(coll.find(query or {}, {f: 0 for f in drop}))
    if not rows:
        return pd.DataFrame()
    d = pd.json_normalize(rows, sep="_")

    def fix(v):
        if isinstance(v, (list, dict)):
            return json.dumps(v, default=str)[:32000]
        if isinstance(v, str):
            return v[:32000]
        return v
    return d.apply(lambda col: col.map(fix))


def _build_export_tables():
    t = {}
    t["participants"] = _docs_to_df(db.users)
    t["profiles"] = _docs_to_df(db.profiles)
    t["profile_history"] = _docs_to_df(db.profile_history)
    t["sessions"] = _docs_to_df(db.sessions)
    t["login_logout_history"] = _docs_to_df(db.login_history)
    t["assessments"] = _docs_to_df(db.assessments)
    t["physiological"] = _docs_to_df(db.physiological_measurements)
    t["short_checkins"] = _docs_to_df(db.state_checkins)
    t["song_feedback"] = _docs_to_df(db.recommendation_feedback)
    t["session_feedback"] = _docs_to_df(db.experiments)
    t["bias_assessments"] = _docs_to_df(db.bias_assessments)

    fb, asm, ses, usr = t["song_feedback"], t["assessments"], t["sessions"], t["participants"]
    if not fb.empty:
        m = fb.copy()
        if not asm.empty and "assessment_id" in m.columns:
            keep = [c for c in asm.columns if c != "recommendations" and (c == "assessment_id" or c not in m.columns)]
            m = m.merge(asm[keep], on="assessment_id", how="left")
        if not ses.empty and "session_id" in m.columns:
            keep = [c for c in ses.columns if c == "session_id" or c not in m.columns]
            m = m.merge(ses[keep], on="session_id", how="left")
        if not usr.empty and "user" in m.columns:
            keep = [c for c in ["user", "display_id", "email", "first_login_ist", "login_count"] if c in usr.columns]
            keep = [c for c in keep if c == "user" or c not in m.columns]
            m = m.merge(usr[keep], on="user", how="left")
        t["MASTER_flat_feedback"] = m
    return t


def _excel_bytes(tables):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in tables.items():
            (d if not d.empty else pd.DataFrame({"info": ["no data"]})).to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


# --------------------------------------------------------------
# Sidebar: navigation + status banners
# --------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 🎧 {APP_NAME}")
    st.caption(APP_TAGLINE)
    mode_badge("MongoDB Atlas connected" if db else "MongoDB connection failed", "research" if db else "warning")
    st.write("")
    mode_badge("DEMO DATASET" if is_demo_dataset else "Real dataset loaded",
               "demo" if is_demo_dataset else "research")
    if model_error:
        mode_badge("Content-based fallback (no trained models)", "warning")
    else:
        mode_badge("RNN + NCF models loaded", "research")
    if st.session_state.verified:
        st.caption(f"Signed in: **{st.session_state.username}** · Session #{st.session_state.session_number}")
    st.divider()

    PAGES = ["Dashboard", "Profile", "Psychological Assessment", "Physiological Input",
              "Music Preference & Recommendation", "Evaluation & Validation",
              "Bias & Risk of Bias", "Research Evidence", "Dataset / Research Mode",
              "Backend Monitor (Admin)"]
    st.session_state["page"] = st.radio("Navigate", PAGES,
                                         index=PAGES.index(st.session_state["page"]))

page = st.session_state["page"]


# --------------------------------------------------------------
# Shared safety disclaimer banner (shown on every page)
# --------------------------------------------------------------
def safety_banner():
    bg = COLORS['neutral']
    accent = COLORS['accent']
    html = (f"<div style='background:{bg};border-left:4px solid {accent};"
            f"padding:8px 14px;border-radius:6px;font-size:0.85em;margin-bottom:12px;'>"
            f"⚠ {saf.SAFETY_DISCLAIMER}</div>")
    st.markdown(html, unsafe_allow_html=True)


# ================================================================
# PAGE: Dashboard / Login
# ================================================================
if page == "Dashboard":
    st.title(f"🎧 {APP_NAME}")
    st.caption(APP_TAGLINE)
    safety_banner()

    if mongodb_error:
        st.error(
            "MongoDB is not connected. Check MONGODB_URI in Streamlit Secrets "
            "and restart the app."
        )
        st.code(mongodb_error)

    card_open()
    st.subheader("👤 Sign in (OTP verification)")

    if db:
        st.success(
            "MongoDB Atlas connected — application data will be stored in MongoDB."
        )
        if db.index_warnings:
            st.warning("Some MongoDB indexes could not be created (see Backend Monitor).")

    if not db:
        st.error("Sign-in is disabled until MongoDB is connected.")
    elif not st.session_state.verified:
        _render_login_flow()
    else:
        flash = st.session_state.get("otp_flash")
        if flash:
            st.success(flash)
            st.session_state["otp_flash"] = None
        st.success(f"Signed in as **{st.session_state.username}** — Session #{st.session_state.session_number}")
        try:
            _has_prof = db.profiles.find_one({"user": st.session_state.username}, {"_id": 1}) is not None
        except Exception:
            _has_prof = False
        if _has_prof:
            st.info(f"Welcome back — session #{st.session_state.session_number}. "
                    "You will only answer a short 8-question check-in before recommendations. "
                    "You can still update your profile on the **Profile** page.")
        else:
            st.info("First session — please complete the full questionnaires on the **Profile** page (one time only).")

        if st.button("🚪 Logout"):
            ok, msg = _mark_checkout("logout")
            if ok:
                _reset_session_state()
                st.success(msg)
                st.rerun()
            else:
                # Do NOT clear Streamlit session state when MongoDB failed.
                # The user stays signed in and can retry, so the logout event is not lost.
                st.error(f"Logout was NOT completed: {msg}")

    card_close()

    if st.session_state.verified:
        st.markdown(
            "Use the sidebar to continue: **Profile → "
            "Psychological Assessment → Physiological Input → "
            "Music Preference & Recommendation**."
        )

# ================================================================
# Everything below requires sign-in (except the admin page, which has its own password)
# ================================================================
elif not db:
    st.error("MongoDB connection is required. Configure MONGODB_URI in Streamlit Secrets.")
elif not st.session_state.verified and page != "Backend Monitor (Admin)":
    st.warning("Please sign in on the Dashboard page first.")

# ================================================================
# PAGE: Profile
# ================================================================
elif page == "Profile":
    name = st.session_state.username
    st.title("👤 Your Profile")
    safety_banner()

    if st.session_state.get("profile_user") != name:
        st.session_state["profile_user"] = name
        st.session_state["profile_doc"] = db.profiles.find_one({"user": name}, {"_id": 0})
        st.session_state["editing_profile"] = st.session_state["profile_doc"] is None

    profile_doc = st.session_state["profile_doc"]
    has_profile = profile_doc is not None

    if has_profile and not st.session_state["editing_profile"]:
        card_open()
        st.write(f"**Age:** {profile_doc.get('age')}")
        st.write(f"**Preferred Genre:** {profile_doc.get('genre_pref')}")
        st.write(f"**Preferred Vibe/Era:** {profile_doc.get('era_pref')}")
        st.caption("Your full questionnaires (TIPI, DASS-21, WHOQOL-BREF) are saved. "
                   "In future sessions you will only answer a short 8-question check-in.")
        if st.button("✏️ Update Profile"):
            st.session_state["editing_profile"] = True
            st.rerun()
        card_close()

    if st.session_state["editing_profile"]:
        defaults = profile_doc or {}
        card_open()
        st.subheader("Complete / Update Your Profile")
        st.caption("Collected once and reused on every future login. Updating it keeps all earlier versions in the history.")
        with st.form("profile_form"):
            age = st.number_input("Age", min_value=0, max_value=100, value=int(defaults.get("age", 18)), step=1)
            genre_options = ["Bollywood", "Hindi Pop", "Ghazal", "Classical"]
            era_options = ["60s songs", "90s songs", "Energetic songs", "calming songs", "Classical songs"]
            c1, c2 = st.columns(2)
            with c1:
                genre_pref = st.selectbox("Preferred Genre", genre_options,
                                           index=genre_options.index(defaults["genre_pref"]) if defaults.get("genre_pref") in genre_options else 0)
            with c2:
                era_pref = st.selectbox("Preferred Vibe", era_options,
                                         index=era_options.index(defaults["era_pref"]) if defaults.get("era_pref") in era_options else 0)

            st.subheader("🧩 TIPI (Big Five)")
            st.caption(psy.INSTRUMENT_METADATA["TIPI"]["scoring"])
            saved_tipi = defaults.get("tipi", [4] * len(psy.TIPI_ALL))
            tipi = [st.slider(q, 1, 7, int(saved_tipi[i]) if i < len(saved_tipi) else 4) for i, q in enumerate(psy.TIPI_ALL)]

            st.subheader("💭 DASS-21 (baseline)")
            saved_dass = defaults.get("dass", [1] * len(psy.DASS_ALL))
            dass = [st.slider(q, 0, 3, int(saved_dass[i]) if i < len(saved_dass) else 1) for i, q in enumerate(psy.DASS_ALL)]

            st.subheader("🌍 WHOQOL-BREF")
            saved_whoqol = defaults.get("whoqol", [3] * len(psy.WHOQOL_ALL))
            whoqol = [st.slider(q, 1, 5, int(saved_whoqol[i]) if i < len(saved_whoqol) else 3) for i, q in enumerate(psy.WHOQOL_ALL)]

            submitted = st.form_submit_button("💾 Save Profile")
        if submitted:
            if age < 18:
                st.error("Age must be 18 or above to use this system.")
                st.stop()
            ist_text = _now_ist_str()
            profile_data = {
                "email": st.session_state.user_email, "age": int(age),
                "genre_pref": genre_pref, "era_pref": era_pref,
                "tipi": [int(x) for x in tipi], "dass": [int(x) for x in dass], "whoqol": [int(x) for x in whoqol],
                "updated_at_ist": ist_text, "last_session_id": st.session_state.get("session_id"),
            }
            try:
                db.profiles.update_one(
                    {"user": name},
                    {"$set": profile_data, "$setOnInsert": {"created_at_ist": ist_text,
                                                            "created_session_id": st.session_state.get("session_id")}},
                    upsert=True)                                  # unique on "user" -> one profile per participant, never a duplicate
                db.profile_history.insert_one({**profile_data, "user": name, "timestamp": _utc_iso(),
                                               "session_id": st.session_state.get("session_id"),
                                               "session_number": st.session_state.get("session_number")})
                db.users.update_one({"user": name}, {"$set": {"baseline_completed_ist": ist_text}})
            except Exception as e:
                st.error(f"Profile could not be saved to MongoDB: {e}")
                st.stop()
            st.session_state["profile_doc"] = {**profile_data, "user": name}
            st.session_state["editing_profile"] = False
            st.session_state["full_baseline_this_session"] = True    # full questionnaires done -> no check-in this session
            st.success("Profile saved.")
            st.rerun()
        card_close()
    elif not has_profile:
        st.info("No profile yet — fill in the form above.")

# ================================================================
# PAGE: Psychological Assessment (documentation + score display)
# ================================================================
elif page == "Psychological Assessment":
    st.title("🧠 Psychological Assessment")
    safety_banner()
    st.info(psy.DISCLAIMER)

    for key, meta in psy.INSTRUMENT_METADATA.items():
        card_open()
        st.markdown(f"#### {meta['name']}")
        st.write(f"**Purpose:** {meta['purpose']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Items", meta["items"])
        c2.write(f"**Source:** {meta['source']}")
        c3.write(f"**Scoring:** {meta['scoring']}")
        with st.expander("Validation evidence & limitations"):
            st.write(f"**Validation:** {meta['validation']}")
            st.write(f"**Limitations:** {meta['limitations']}")
        card_close()

    card_open()
    st.markdown("#### Returning-user short check-in (PHQ-4 + PSS-4, 8 items)")
    st.write("**Purpose:** track current anxiety, depressed mood and perceived stress in sessions after the first one, "
             "without repeating the full 21+10+26-item baseline.")
    st.write("**Sources:** Kroenke et al. (2009) Psychosomatics 50(6):613-621 (PHQ-4); Lowe et al. (2010) J Affect Disord "
             "122(1-2):86-95 (general-population validation); Cohen, Kamarck & Mermelstein (1983) J Health Soc Behav "
             "24:385-396 (PSS).")
    st.write("**Limitations:** PHQ-4 and PSS-4 use 2-week / 1-month recall windows; PSS-4 has modest reliability; scores are "
             "linearly rescaled to the DASS-21 0-42 range for the recommender (approximation, not a validated conversion).")
    card_close()

    profile_doc = st.session_state.get("profile_doc")
    if not profile_doc and db:
        profile_doc = db.profiles.find_one({"user": st.session_state.username}, {"_id": 0})
    if profile_doc:
        card_open()
        st.markdown("#### Your saved baseline scores")
        tipi_scores = psy.score_tipi(profile_doc["tipi"])
        dass_scores = psy.score_dass21(profile_doc["dass"])
        whoqol_scores = psy.score_whoqol(profile_doc["whoqol"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Big Five (TIPI)**")
            st.json({k: round(v, 2) for k, v in tipi_scores.items()})
        with c2:
            st.write("**DASS-21**")
            for sub in ["depression", "anxiety", "stress"]:
                band = psy.dass_severity_band(sub, dass_scores[sub])
                st.write(f"{sub.title()}: {dass_scores[sub]} ({band})")
            st.caption("Bands are the published DASS-21 severity labels — screening categories, not diagnoses.")
        with c3:
            st.write("**WHOQOL-BREF (0-100)**")
            st.json({k: round(v, 1) for k, v in whoqol_scores.items() if k != "psych_mean_1to5"})
        card_close()
    else:
        st.warning("Complete your Profile first to see computed scores.")

# ================================================================
# PAGE: Physiological Input
# ================================================================
elif page == "Physiological Input":
    st.title("⌚ Physiological Input")
    safety_banner()

    tab1, tab2, tab3 = st.tabs(["Self-report (used for recommendations)", "Measured HRV (upload RR-intervals)", "WESAD Research Mode"])

    with tab1:
        card_open()
        mode_badge("SELF-REPORT — not a physiological measurement", "warning")
        st.caption("These sliders feed the recommendation engine below, exactly as in the original app. "
                   "They represent perceived state, not a sensor reading. They are saved with your session "
                   "when you press 'Get Recommendations'.")
        hrv = st.slider("Perceived HR (bpm)", 20, 200, 90, key="hr_slider")
        stress = st.slider("Perceived Stress Level", 0, 100, 40, key="stress_slider")
        mood = st.selectbox("Current Mood", ["Happy", "Sad", "Angry", "Calm", "Energetic"], key="mood_select")
        st.session_state["hrv_selfreport"] = hrv
        st.session_state["stress_selfreport"] = stress
        st.session_state["mood_selfreport"] = mood
        card_close()

    with tab2:
        card_open()
        mode_badge("MEASURED — real HRV computation", "research")
        st.caption("Upload a CSV with one column of RR intervals in milliseconds "
                   "(e.g. exported from a chest-strap or PPG app) to compute real, "
                   f"literature-defined HRV features. Reference: {physio.HRV_REFERENCE}")
        with st.expander("What each feature means"):
            for k, m in physio.HRV_METADATA.items():
                st.write(f"**{m['name']}** ({m['unit']}) — {m['meaning']} *Requires:* {m['requires']}")
        uploaded = st.file_uploader("RR-interval CSV (single column, ms)", type=["csv"])
        if uploaded is not None:
            try:
                raw = uploaded.getvalue()
                fhash = hashlib.sha256(raw).hexdigest()
                rr_df = pd.read_csv(io.BytesIO(raw), header=None)
                rr_values = pd.to_numeric(rr_df.iloc[:, 0], errors="coerce").dropna().values
                features, note = physio.compute_hrv_features(rr_values)
                if features is None:
                    st.error(note)
                else:
                    st.success("HRV features computed from your uploaded data:")
                    st.json(features)
                    if note:
                        st.caption(f"⚠ {note}")
                    save_key = f"{st.session_state.get('session_id')}:{fhash}"
                    if save_key not in st.session_state["rr_saved"]:      # a Streamlit rerun must not re-insert the same file
                        mid = uuid.uuid4().hex
                        db.physiological_measurements.insert_one(_clean({
                            "measurement_id": mid, "user": st.session_state.username,
                            "session_id": st.session_state.get("session_id"),
                            "session_number": st.session_state.get("session_number"),
                            "features": features, "file_sha256": fhash, "n_rr": int(len(rr_values)),
                            "timestamp": _utc_iso(), "source": "uploaded_rr",
                        }))
                        st.session_state["rr_saved"].append(save_key)
                        st.session_state["rr_features_latest"] = {"measurement_id": mid, "features": _clean(features)}
                        st.caption("Saved to your session record.")
            except Exception as e:
                st.error(f"Could not parse/save file: {e}")
        card_close()

    with tab3:
        card_open()
        mode_badge("RESEARCH DATASET MODE", "research")
        st.caption("WESAD (Schmidt et al., 2018, ACM ICMI): chest ECG sampled at 700 Hz -> R-peaks -> RR intervals -> HRV "
                   "features per condition (baseline / stress / amusement / meditation). WESAD contains no music; it supplies "
                   "physiological context only.")

        wdf, wesad_err = _load_wesad_features()
        if wdf is None:
            st.warning(f"{wesad_err} Falling back to self-report physiological input (the app keeps working normally). "
                       "To enable this mode run `extract_wesad_hrv.py` on the computer that holds WESAD and commit "
                       "`data/wesad_hrv_features.csv`.")
        else:
            st.success(f"Precomputed WESAD HRV features loaded: {wdf['subject'].nunique()} subjects, {len(wdf)} segments.")
            with st.expander("View WESAD HRV feature table"):
                st.dataframe(wdf, use_container_width=True)
            num_cols = [c for c in wdf.select_dtypes("number").columns if c != "duration_s"]
            with st.expander("Mean HRV features by condition"):
                st.dataframe(wdf.groupby("condition")[num_cols].mean().round(2))

            subj = st.selectbox("WESAD subject", sorted(wdf["subject"].astype(str).unique()), key="wesad_subj_sel")
            conds = wdf[wdf["subject"].astype(str) == subj]["condition"].astype(str).tolist()
            cond = st.selectbox("Condition segment", conds, key="wesad_cond_sel")
            seg = wdf[(wdf["subject"].astype(str) == subj) & (wdf["condition"].astype(str) == cond)].iloc[0]
            try:
                w_hr = float(np.clip(float(seg["mean_hr_bpm"]), 20, 200))
                w_rmssd = float(seg["rmssd_ms"])
                w_stress = _wesad_stress_index(w_rmssd, wdf)
                m1, m2, m3 = st.columns(3)
                m1.metric("Mean HR (bpm)", f"{w_hr:.0f}")
                m2.metric("RMSSD (ms)", f"{w_rmssd:.1f}")
                m3.metric("HRV-derived stress index (0-100)", f"{w_stress:.0f}")
                st.caption("Stress index = heuristic: lower RMSSD (lower vagally-mediated HRV; Task Force of ESC/NASPE, 1996) "
                           "maps to a higher index, normalised to the 5th-95th percentile of RMSSD in this WESAD table. "
                           "It is a research heuristic, not a clinical measure.")
                if st.button("✅ Use this WESAD context for recommendations"):
                    st.session_state["wesad_context"] = {
                        "subject": subj, "condition": cond,
                        "hr": int(round(w_hr)), "stress": int(round(w_stress)), "rmssd": w_rmssd,
                    }
                    st.success("Saved. On the recommendation page you can now choose 'WESAD research mode' as the "
                               "physiological context source. Self-report remains the default.")
            except Exception as e:
                st.error(f"Could not derive physiological context from this WESAD segment: {e}")

        st.markdown("---")
        st.markdown("**Raw WESAD (local machine only)**")
        try:
            subjects = physio.list_available_wesad_subjects()
        except Exception as e:
            subjects = []
            st.caption(f"Raw WESAD folder could not be scanned: {e}")
        if not subjects:
            st.warning(
                "WESAD not found locally. WESAD (Schmidt et al., 2018) requires registration with "
                "the original authors and cannot be auto-downloaded here.\n\n"
                f"Place downloaded subject folders at: `{physio.WESAD_DIR}/S<id>/S<id>.pkl` "
                "(official per-subject pickle format) to enable this mode.")
        else:
            chosen = st.selectbox("Available WESAD subjects", subjects)
            if st.button("Load subject & extract HRV from chest ECG"):
                data, err = physio.load_wesad_subject(chosen)
                if err:
                    st.error(err)
                else:
                    try:
                        ecg = data["signal"]["chest"]["ECG"]
                        rr = physio.wesad_ecg_to_rr(ecg, fs=700)
                        if rr is None:
                            st.error("R-peak detection failed on this signal.")
                        else:
                            features, note = physio.compute_hrv_features(rr)
                            st.json(features)
                            if note:
                                st.caption(f"⚠ {note}")
                    except Exception as e:
                        st.error(f"Could not process WESAD signal: {e}")
        card_close()

# ================================================================
# PAGE: Music Preference & Recommendation
# ================================================================
elif page == "Music Preference & Recommendation":
    st.title("🎵 Music Preference & Recommendation")
    safety_banner()

    profile_doc = st.session_state.get("profile_doc")
    if not profile_doc:
        profile_doc = db.profiles.find_one({"user": st.session_state.username}, {"_id": 0})
        st.session_state["profile_doc"] = profile_doc
    if not profile_doc:
        st.warning("Complete your Profile first.")
        st.stop()

    name = st.session_state.username
    age, genre_pref, era_pref = profile_doc["age"], profile_doc["genre_pref"], profile_doc["era_pref"]
    tipi, whoqol = profile_doc["tipi"], profile_doc["whoqol"]
    dass_baseline = profile_doc.get("dass", [1] * len(psy.DASS_ALL))

    hrv = st.session_state.get("hrv_selfreport", 90)
    stress = st.session_state.get("stress_selfreport", 40)
    mood = st.session_state.get("mood_selfreport", "Calm")
    if "hrv_selfreport" not in st.session_state:
        st.info("Set your self-reported mood/HR/stress on the **Physiological Input** page first "
                 "(defaults are being used for now).")

    # Physiological context source: self-report (default) or WESAD research mode
    physio_source = "self-report"
    wctx = st.session_state.get("wesad_context")
    if wctx:
        card_open()
        src_choice = st.radio(
            "Physiological context source",
            ["Self-report (default)",
             f"WESAD research mode ({wctx['subject']}, {wctx['condition']}: HR {wctx['hr']} bpm, stress index {wctx['stress']})"],
            key="physio_source_choice")
        if src_choice.startswith("WESAD"):
            hrv, stress, physio_source = wctx["hr"], wctx["stress"], "wesad"
            st.caption("Research mode: HR / stress come from a WESAD participant segment, not from you.")
        card_close()

    # First session -> full questionnaires were already answered on the Profile page (no check-in).
    # Later sessions -> only the 8-item short check-in below (no full questionnaires).
    first_session = st.session_state.get("full_baseline_this_session", False)
    dass = dass_baseline.copy()
    short_used = False
    short_scores = None
    phq4, pss4 = [], []

    if first_session:
        dass_scores = psy.score_dass21(dass)
    else:
        short_used = True
        card_open()
        st.subheader("💭 Quick Check-in (8 questions)")
        st.caption(SHORT_FORM_NOTE)
        st.markdown("**Over the last 2 weeks, how often have you been bothered by...**")
        phq4 = [st.radio(q, list(PHQ4_OPTIONS), format_func=PHQ4_OPTIONS.get, horizontal=True, key=f"phq4_{i}")
                for i, q in enumerate(PHQ4_ITEMS)]
        st.markdown("**In the last month, how often have you...**")
        pss4 = [st.radio(q, list(PSS4_OPTIONS), format_func=PSS4_OPTIONS.get, horizontal=True, key=f"pss4_{i}")
                for i, q in enumerate(PSS4_ITEMS)]
        card_close()
        short_scores = score_short_checkin(phq4, pss4)
        dass_scores = {k: short_scores[k] for k in ("depression", "anxiety", "stress")}

    tipi_scores = psy.score_tipi(tipi)
    whoqol_scores = psy.score_whoqol(whoqol)
    dass_mood = psy.get_dass_mood(dass_scores["depression"], dass_scores["stress"], dass_scores["anxiety"])
    mood_state = mood if (mood == dass_mood or random.random() < 0.7) else dass_mood

    tipi_mean = np.mean(tipi)
    ctx = {
        "mood_state": mood_state, "hrv": hrv, "stress": stress,
        "genre_pref": genre_pref, "era_pref": era_pref,
        "extraversion": tipi_scores["extraversion"], "openness": tipi_scores["openness"],
        "depression": dass_scores["depression"], "anxiety": dass_scores["anxiety"],
        "physical_qol": whoqol_scores["physical"], "social_qol": whoqol_scores["social"],
        "tipi_n": (tipi_mean - 1) / 6.0, "whoql_n": (whoqol_scores["psych_mean_1to5"] - 1) / 4.0,
        "dass_n": (np.mean([dass_scores["depression"], dass_scores["anxiety"], dass_scores["stress"]]) / 42.0
                   if short_used else np.mean(dass) / 3.0),
        "mood_n": rec.MOOD_MAP[mood_state] / 4.0,
        "user_name": name,
    }

    num_songs = len(df)
    user_qdoc = db.qtables.find_one({"user": name})
    global_qdoc = db.qtables.find_one({"user": "global"})
    personal_q = np.array(user_qdoc["qtable"]) if user_qdoc else np.zeros((100, num_songs))
    global_q = np.array(global_qdoc["qtable"]) if global_qdoc else np.zeros((100, num_songs))
    if personal_q.shape[1] != num_songs:
        newq = np.zeros((100, num_songs)); n = min(personal_q.shape[1], num_songs); newq[:, :n] = personal_q[:, :n]; personal_q = newq
    if global_q.shape[1] != num_songs:
        newq = np.zeros((100, num_songs)); n = min(global_q.shape[1], num_songs); newq[:, :n] = global_q[:, :n]; global_q = newq

    def weights_getter():
        fdf = _user_feedback_df(name)                              # permanent history from MongoDB (all sessions)
        needed = ["rnn_score", "ncf_score", "personal_q", "pref_bias", "physio_fit", "psy_bias", "rating"]
        if fdf is None or len(fdf) < 20 or not all(c in fdf.columns for c in needed):
            return rec.FALLBACK_WEIGHTS
        try:
            from sklearn.linear_model import Ridge
            X, y = fdf[needed[:-1]].values, fdf["rating"].values
            m = Ridge(alpha=1.0).fit(X, y)
            clipped = np.clip(m.coef_, 0.01, None)
            return clipped / clipped.sum()
        except Exception:
            return rec.FALLBACK_WEIGHTS

    get_btn = st.button("🎧 Get Recommendations", disabled=age < 18)

    if get_btn:
        assessment_id = uuid.uuid4().hex
        st.session_state["got_recs"] = True
        st.session_state["feedback_count"] = 0
        st.session_state["physio_source_used"] = physio_source
        st.session_state["assessment_id"] = assessment_id
        sid = st.session_state.get("session_id")
        snum = st.session_state.get("session_number")
        ts = _utc_iso()

        pool = rec.build_candidate_pool(df, mood_state, hrv, stress, genre_pref, era_pref,
                                         ctx["depression"], ctx["anxiety"], ctx["extraversion"],
                                         ctx["physical_qol"], ctx["social_qol"])
        pool, mode_note = rec.score_pool(pool, ctx, personal_q, global_q,
                                          {"metadata": metadata, "rnn_model": rnn_model, "ncf_model": ncf_model},
                                          weights_getter, num_songs)

        confidence, spread = saf.assess_confidence(pool["final_score"])
        flagged = saf.get_flagged_songs(db.recommendation_feedback, name)
        pool, safety_note = saf.apply_safety_layer(pool, confidence, flagged, rec.neutral_safe_pool)

        # Research-evidence layer: research tracks compete in the SAME ranking. They are boosted only when the
        # participant's HR / stress / psychological scores / mood resemble what the evidence addresses.
        pool, research_fit, research_strength = _apply_research_evidence(
            pool,
            {"stress": stress, "hr": hrv, "dass_n": ctx["dass_n"], "mood_state": mood_state,
             "genre_pref": genre_pref, "era_pref": era_pref},
            flagged)
        chosen = _select_recommendations(pool, n=5)

        st.session_state["pool"] = pool
        meta_by_id = {str(r.get("song_id")): r for r in pool.to_dict("records")}
        rec_records = []
        for _, r in chosen.iterrows():
            m = meta_by_id.get(str(r["song_id"]), r.to_dict())
            sid_val = m.get("song_id")
            sid_val = sid_val.item() if isinstance(sid_val, np.generic) else sid_val
            rec_records.append({
                "song_id": sid_val, "song": m.get("song"), "artist": m.get("artist"), "genre": m.get("genre"),
                "research_source": bool(m.get("research_source", False)),
                "evidence_level": m.get("evidence_level", "") or "",
                "evidence_key": m.get("evidence_key", "") or "",
                "raga": m.get("raga", "") or "",
                "source_url": m.get("source_url", "") or "",
                "source_name": m.get("source_name", "") or "",
                "source_only": str(sid_val).startswith("research_src_"),   # not a dataset column -> no Q-table index
            })
        st.session_state["recs"] = rec_records
        n_res = sum(1 for r in rec_records if r["research_source"])
        st.session_state["mode_note"] = (
            f"{mode_note}\n\nResearch-evidence layer: research tracks were ranked in the same pool as every other song. "
            f"A track receives a bonus only if your stress, psychological scores, heart rate and mood fit the evidence "
            f"(fit {research_fit:.2f}, strength {research_strength:.2f}). "
            + (f"{n_res} of the 5 selected songs carry research evidence." if n_res else
               "None of the 5 selected songs needed a research boost this time.")
        )
        st.session_state["safety_note"] = safety_note
        st.session_state["confidence"] = confidence
        st.session_state["mood_state_used"] = mood_state

        # ---------- persistence: everything linked to User ID + session_id + assessment_id ----------
        try:
            u = db.users.find_one_and_update({"user": name}, {"$inc": {"assessment_counter": 1}},
                                             return_document=ReturnDocument.AFTER)
            anum = int((u or {}).get("assessment_counter", 1))
            if short_used:
                try:
                    db.state_checkins.insert_one({
                        "user": name, "session_id": sid, "session_number": snum,
                        "assessment_id": assessment_id, "assessment_number": anum,
                        "phq4": [int(v) for v in phq4], "pss4": [int(v) for v in pss4],
                        "phq4_total": short_scores["phq4_total"], "pss4_total": short_scores["pss4_total"],
                        "phq4_band": phq4_band(short_scores["phq4_total"]),
                        "scores_dass_scale": {k: float(short_scores[k]) for k in ("depression", "anxiety", "stress")},
                        "timestamp": ts,
                    })
                except DuplicateKeyError:
                    pass
            rr_latest = st.session_state.get("rr_features_latest")
            db.physiological_measurements.insert_one(_clean({
                "measurement_id": uuid.uuid4().hex, "user": name, "session_id": sid, "session_number": snum,
                "assessment_id": assessment_id, "source": "self_report" if physio_source == "self-report" else "wesad_context",
                "hr_bpm": hrv, "stress_0_100": stress, "mood_selected": mood,
                "wesad_context": wctx if physio_source == "wesad" else None, "timestamp": ts,
            }))
            db.assessments.insert_one(_clean({
                "assessment_id": assessment_id, "assessment_number": anum,
                "user": name, "session_id": sid, "session_number": snum, "timestamp": ts, "timestamp_ist": _now_ist_str(),
                "physiological": {"hr_bpm": hrv, "stress_0_100": stress, "mood_selected": mood,
                                  "mood_state_used": mood_state, "source": physio_source,
                                  "measured_hrv": rr_latest},
                "questionnaire": {"type": "short_checkin" if short_used else "full_baseline",
                                  "phq4": phq4 if short_used else None, "pss4": pss4 if short_used else None},
                "psychological_scores": {"dass_scale": dass_scores, "tipi": tipi_scores, "whoqol": whoqol_scores},
                "research_layer": {"fit": research_fit, "strength": research_strength},
                "confidence": confidence, "safety_note": safety_note,
                "recommendations": rec_records,
            }))
        except Exception as e:
            st.error(f"This assessment could not be fully saved to MongoDB: {e}")

    if st.session_state["recs"]:
        card_open()
        st.markdown("### 🧠 Explanation")
        mode_badge(st.session_state.get("confidence", "high").upper() + " CONFIDENCE",
                    "research" if st.session_state.get("confidence") == "high" else "warning")
        st.write(st.session_state.get("mode_note", ""))
        if st.session_state.get("physio_source_used") == "wesad":
            st.caption("Physiological context for this playlist: WESAD research mode (not the user's own measurement).")
        if st.session_state.get("safety_note"):
            st.warning(st.session_state["safety_note"])
        ev_entry = rec.explain_evidence_for(st.session_state.get("mood_state_used", mood_state))
        st.markdown(f"**Evidence-based direction applied:** {ev_entry['audio_characteristics']}")
        st.caption(f"Claim level: {ev_entry['claim_level']}")
        st.caption("See the Research Evidence page for full citations.")
        card_close()

        aid = st.session_state.get("assessment_id") or "na"
        for i, s in enumerate(st.session_state["recs"]):
            card_open()
            st.markdown(f"**{i+1}. {s['song']} — {s['artist']}** ({s['genre']})")
            level = s.get("evidence_level", "")
            if level == "studied_stimulus":
                st.success("🔬 Evidence-supported: this recording was the stimulus of a published randomised study.")
            elif level == "raga_category":
                st.info("🎼 Raga-level evidence: this raga was studied (the song itself was not).")
            elif s.get("research_source"):
                st.info("📚 From the project's research-source pool (no recording-specific outcome evidence claimed).")
            with st.expander("💡 Why was this song recommended?", expanded=True):
                row_match = st.session_state["pool"]
                row_match = row_match[row_match["song_id"].astype(str) == str(s["song_id"])] if not row_match.empty else row_match
                row = row_match.iloc[0].to_dict() if len(row_match) else dict(s)
                row.update({k: s.get(k) for k in ("research_source", "evidence_level", "evidence_key", "raga", "source_url")})
                st.write(_song_explanation(row, {
                    "mood_state": st.session_state.get("mood_state_used", mood_state),
                    "stress": stress, "genre_pref": genre_pref, "era_pref": era_pref
                }, is_research=bool(s.get("research_source", False))))
                if s.get("research_source"):
                    st.caption("Research basis: see the Research Evidence page. Evidence describes the studied music/raga "
                               "context; it does not mean a song is universally therapeutic.")
            rating = st.radio("Rate this song (1=dislike, 5=like)", [1, 2, 3, 4, 5], horizontal=True,
                               key=f"rate_{aid}_{i}_{s['song_id']}")
            url = s.get("source_url") if (s.get("research_source") and s.get("source_url")) else spotify_link(s["song"], s["artist"])
            st.markdown(f"[🎧 Open recording]({url})")
            if s.get("research_source") and s.get("source_name"):
                st.caption(f"Source: {s.get('source_name')}")
            flag_key = f"fb_done_{aid}_{i}_{s['song_id']}"
            if flag_key not in st.session_state:
                st.session_state[flag_key] = False
            if st.button(f"Submit Feedback for Song {i+1}", key=f"fb_{aid}_{i}_{s['song_id']}", disabled=st.session_state[flag_key]):
                st.session_state[flag_key] = True
                song_action = s["song_id"]
                reward = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}[rating]
                pool_all = st.session_state["pool"]
                pool_row = pool_all[pool_all["song_id"].astype(str) == str(song_action)]
                def get(c):
                    try:
                        v = float(pool_row[c].values[0]) if len(pool_row) and c in pool_row.columns else 0.0
                        return v if np.isfinite(v) else 0.0
                    except Exception:
                        return 0.0
                entry = {
                    "user": name, "session_id": st.session_state.get("session_id"),
                    "session_number": st.session_state["session_number"],
                    "assessment_id": st.session_state.get("assessment_id"),
                    "song_id": song_action, "song": s["song"], "artist": s["artist"],
                    "rating": rating, "hrv": hrv, "stress": stress,
                    "mood_state": st.session_state.get("mood_state_used", mood_state),
                    "rnn_score": get("rnn_score"), "ncf_score": get("ncf_score"),
                    "personal_q": get("personal_q"), "pref_bias": get("pref_bias"),
                    "physio_fit": get("physio_fit"), "psy_bias": get("psy_bias"),
                    "physio_source": st.session_state.get("physio_source_used", "self-report"),
                    "research_source": bool(s.get("research_source")),
                    "evidence_level": s.get("evidence_level", ""),
                    "timestamp": _utc_iso(),
                }
                try:
                    db.recommendation_feedback.insert_one(_clean(entry))
                except DuplicateKeyError:
                    st.info("Feedback for this song was already recorded.")
                    st.stop()

                # Source-only research tracks are logged for feedback but have no Q-table column.
                # Dataset-backed songs keep the original RL/Q-table learning behaviour.
                if not s.get("source_only"):
                    cur_state = rec.get_user_state(st.session_state.get("mood_state_used", mood_state), stress, ctx["depression"])
                    before_p, before_g = personal_q.copy(), global_q.copy()
                    rec.update_q(personal_q, cur_state, song_action, reward, cur_state)
                    rec.update_q(global_q, cur_state, song_action, reward, cur_state)
                    _save_qtable(name, personal_q, before_p, num_songs)
                    _save_qtable("global", global_q, before_g, num_songs)

                if saf.is_adverse_rating(rating, st.session_state.get("mood_state_used", mood_state), stress):
                    st.error("This recommendation is flagged as a possible adverse response and will be "
                              "deprioritized in your future sessions.")
                st.success("Feedback recorded.")
            card_close()

        if st.button("✅ Finish Listening Session"):
            st.session_state.session_finished = True

        if st.session_state.session_finished:
            card_open()
            st.subheader("⭐ Overall System Feedback")
            comfort = st.slider("Comfort using the system (1-10)", 1, 10, 5)
            satisfaction = st.slider("Satisfaction with recommendations (1-10)", 1, 10, 5)
            mood_alignment = st.slider("How well songs matched your mood (1-10)", 1, 10, 5)
            experience = st.slider("Overall experience (1-10)", 1, 10, 5)
            continue_use = st.radio("Continue using this system?", ["Yes", "No"])
            sess_key = f"sess_fb_done_{aid}"
            if st.button("Submit Session Feedback", disabled=st.session_state.get(sess_key, False)):
                st.session_state[sess_key] = True
                entry = {"user": name, "session_id": st.session_state.get("session_id"),
                         "session_number": st.session_state["session_number"],
                         "assessment_id": st.session_state.get("assessment_id"),
                         "comfort": comfort, "satisfaction": satisfaction,
                         "mood_alignment": mood_alignment, "experience": experience,
                         "continue": continue_use, "mood_state": mood_state,
                         "stress": stress, "hrv": hrv, "timestamp": _utc_iso()}
                try:
                    db.experiments.insert_one(_clean(entry))
                    db.sessions.update_one({"session_id": st.session_state.get("session_id")},
                                           {"$set": {"session_feedback_submitted_ist": _now_ist_str()}})
                    st.success("Thank you — recorded.")
                except DuplicateKeyError:
                    st.info("Session feedback for this assessment was already recorded.")
                st.session_state.session_finished = False
            card_close()

# ================================================================
# PAGE: Evaluation & Validation
# ================================================================
elif page == "Evaluation & Validation":
    st.title("📊 Model & System Validation")
    safety_banner()
    name = st.session_state.username
    fdf = _user_feedback_df(name)
    sdf = _user_session_feedback_df(name)

    card_open()
    st.markdown("#### A. Input data quality")
    if fdf is not None:
        report = val.data_quality_report(fdf, ["song_id", "rating"])
        st.json(report)
    else:
        st.info("No feedback data yet — submit some song ratings to populate this section.")
    card_close()

    card_open()
    st.markdown("#### B. Model score ↔ rating correlation")
    corr, reason = val.score_rating_correlation(fdf)
    if corr is None:
        st.warning(reason)
    else:
        st.json({k: (round(v, 3) if v is not None else "undefined (no variance)") for k, v in corr.items()})
        st.caption("Pearson correlation between each fusion component and the user's actual 1-5 rating.")
    card_close()

    card_open()
    st.markdown("#### C. Precision@K")
    prec, reason = val.precision_at_k(fdf)
    if prec is None:
        st.warning(reason)
    else:
        st.json(prec)
    card_close()

    card_open()
    st.markdown("#### D. User satisfaction (end-of-session survey)")
    summ, reason = val.satisfaction_summary(sdf)
    if summ is None:
        st.warning(reason)
    else:
        st.json(summ)
    card_close()

    card_open()
    st.markdown("#### E. Physiological cross-device validation")
    st.warning("Validation dataset required / not available: this deployment has no paired "
               "reference-device vs. app measurements to compute MAE/RMSE/agreement statistics. "
               "No numbers are fabricated here.")
    card_close()

# ================================================================
# PAGE: Bias & Risk of Bias
# ================================================================
elif page == "Bias & Risk of Bias":
    st.title("⚖️ Bias & Risk of Bias")
    safety_banner()
    st.caption("Methodology adapted from PROBAST domain structure (Wolff et al., 2019, Annals of "
               "Internal Medicine) plus project-specific bias sources. This is a structured "
               "checklist you fill in — not an auto-computed score.")

    if "bias_assessment" not in st.session_state:
        st.session_state["bias_assessment"] = bias_mod.blank_assessment()

    for key, domain in bias_mod.BIAS_DOMAINS.items():
        card_open()
        st.markdown(f"#### {domain['title']}")
        for p in domain["prompts"]:
            st.caption(f"• {p}")
        risk = st.selectbox("Risk level", bias_mod.RISK_LEVELS,
                             index=bias_mod.RISK_LEVELS.index(st.session_state["bias_assessment"][key]["risk"]),
                             key=f"risk_{key}")
        justification = st.text_area("Justification", st.session_state["bias_assessment"][key]["justification"], key=f"just_{key}")
        mitigation = st.text_area("Mitigation applied / planned", st.session_state["bias_assessment"][key]["mitigation"], key=f"mit_{key}")
        st.session_state["bias_assessment"][key] = {"risk": risk, "justification": justification, "mitigation": mitigation}
        card_close()

    if st.button("💾 Save Bias Assessment"):
        db.bias_assessments.insert_one(_clean({
            "user": st.session_state.username, "session_id": st.session_state.get("session_id"),
            "assessment": st.session_state["bias_assessment"], "timestamp": _utc_iso(),
        }))
        st.success("Saved.")

    card_open()
    st.markdown("#### Summary")
    st.json(bias_mod.summarize(st.session_state["bias_assessment"]))
    card_close()

# ================================================================
# PAGE: Research Evidence
# ================================================================
elif page == "Research Evidence":
    st.title("📚 Research Evidence")
    safety_banner()
    st.info(ev.DISCLAIMER)

    st.markdown("#### Characteristic → Evidence mapping")
    for target, entry in ev.CHARACTERISTIC_EVIDENCE_MAP.items():
        card_open()
        st.markdown(f"**{target}**")
        st.write(f"Audio characteristics: {entry['audio_characteristics']}")
        st.write(f"Claim level: {entry['claim_level']}")
        for ref_key in entry["supported_by"]:
            r = ev.REFERENCES[ref_key]
            with st.expander(r["citation"]):
                st.write(f"DOI: {r['doi']}")
                st.write(f"Studied: {r['studied']}")
                st.write(f"Finding: {r['finding']}")
                st.write(f"Evidence type: {r['evidence_type']}")
        card_close()

    st.markdown("#### Raga-level evidence (from the supplied paper)")
    st.caption("Only facts stated in the paper are listed. Fields the paper does not report are marked as such. "
               "The evidence concerns raga Bhairavi as performed in one specific recording; it is not evidence about any other song.")
    for ek, e in RAGA_EVIDENCE.items():
        card_open()
        st.markdown(f"**Raga {e['raga']}** — {e['short_citation']}")
        st.write(f"Citation: {e['citation']} DOI: {e['doi']}")
        st.write(f"**Design:** {e['design']}")
        st.write(f"**Stimulus recording:** {e['stimulus']} — [{e['stimulus_url']}]({e['stimulus_url']})")
        st.write(f"**Tempo:** {e['tempo']}")
        st.write(f"**Mood:** {e['mood']}")
        st.write(f"**Instrumentation:** {e['instrumentation']}")
        st.write(f"**Musical characteristics:** {e['musical_characteristics']}")
        st.write(f"**Psychological associations:** {e['psychological']}")
        st.write(f"**Physiological associations:** {e['physiological']}")
        with st.expander("Limitations stated/implied by the study"):
            for lim in e["limitations"]:
                st.write(f"• {lim}")
        card_close()

    st.markdown("#### How research tracks enter the recommender")
    st.write("Research tracks are scored in the same candidate pool as all other songs (RNN / NCF / Q-table / personalisation). "
             "A research track receives a ranking bonus only when the participant's stress, psychological scores, heart rate "
             "and mood resemble the outcomes the evidence addresses; otherwise it is not forced into the list. "
             "The bonus size and fit formula are design choices of this application, not findings of the papers.")
    st.dataframe(pd.DataFrame([{
        "Track": c["song"], "Artist": c["artist"], "Raga": c["raga"],
        "Evidence level": {"studied_stimulus": "Recording used in a published study",
                           "raga_category": "Raga studied (song itself not studied)",
                           "project_source": "Project Spotify source (no outcome claim)"}[c["evidence_level"]],
        "Link": c["source_url"]} for c in RESEARCH_CANDIDATES]), use_container_width=True)

    st.markdown("#### Full reference list")
    for key, r in ev.REFERENCES.items():
        st.markdown(f"- {r['citation']} (DOI: {r['doi']})")
    st.markdown(f"- {RAGA_EVIDENCE['bhairavi']['citation']} (DOI: {RAGA_EVIDENCE['bhairavi']['doi']})")

# ================================================================
# PAGE: Dataset / Research Mode
# ================================================================
elif page == "Dataset / Research Mode":
    st.title("🗂 Dataset / Research Mode")
    safety_banner()

    card_open()
    st.markdown("#### Music catalog")
    mode_badge("DEMO" if is_demo_dataset else "REAL DATA", "demo" if is_demo_dataset else "research")
    st.write(dataset_note)
    card_close()

    card_open()
    st.markdown("#### Trained recommendation models")
    if model_error:
        mode_badge("NOT LOADED", "warning")
        st.write(model_error)
        st.caption("Recommendations currently run in content-based fallback mode.")
    else:
        mode_badge("LOADED", "research")
        st.write("RNN and NCF trained weights loaded successfully.")
    card_close()

    card_open()
    st.markdown("#### WESAD (physiological research dataset)")
    try:
        subjects = physio.list_available_wesad_subjects()
    except Exception:
        subjects = []
    _wdf, _werr = _load_wesad_features()
    if _wdf is not None:
        mode_badge(f"PRECOMPUTED HRV FEATURES: {_wdf['subject'].nunique()} SUBJECTS", "research")
    if subjects:
        mode_badge(f"{len(subjects)} RAW SUBJECT(S) AVAILABLE", "research")
    elif _wdf is None:
        mode_badge("NOT AVAILABLE", "warning")
        st.write(f"Place downloaded subject folders at `{physio.WESAD_DIR}/S<id>/S<id>.pkl`. "
                 "WESAD requires registration at the official source (Schmidt et al., 2018).")
    card_close()

    card_open()
    st.markdown("#### Known dataset distinctions (do not merge blindly)")
    st.markdown("""
- **DEAM** — music emotion/audio-feature labels only. No physiological data.
- **WESAD** — physiological stress dataset (ECG/EDA/EMG/resp/temp/ACC). No music stimuli.
- **PMEmo** — music + emotion annotation + EDA (not HRV), song-level.
- **DEAP** — music-video stimuli + physiological signals; not directly comparable to WESAD's protocol.
""")
    card_close()

# ================================================================
# PAGE: Backend Monitor (Admin) — monitoring + CSV/Excel export
# ================================================================
elif page == "Backend Monitor (Admin)":
    st.title("🛠 Backend Monitor (Admin)")
    admin_pw = _get_secret("ADMIN_PASSWORD")
    if not admin_pw:
        st.info("Set ADMIN_PASSWORD in Streamlit Secrets to unlock this page.")
    elif not st.session_state["admin_ok"]:
        pw = st.text_input("Admin password", type="password")
        if st.button("Unlock"):
            if hmac.compare_digest(str(pw), str(admin_pw)):
                st.session_state["admin_ok"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    else:
        if st.button("🔒 Lock"):
            st.session_state["admin_ok"] = False
            st.rerun()

        def _table(coll, sort_field, limit=100,
                   drop=("_id", "qtable", "tipi", "dass", "whoqol", "otp_hash", "session_token_hash", "recommendations")):
            proj = {f: 0 for f in drop}
            rows = list(coll.find({}, proj).sort(sort_field, -1).limit(limit))
            return pd.DataFrame(rows) if rows else pd.DataFrame()

        card_open()
        st.markdown("#### Collection sizes")
        names = ["users", "user_profiles", "profile_history", "sessions", "login_history", "assessments",
                 "state_checkins", "recommendation_feedback", "experiments", "physiological_measurements",
                 "bias_assessments", "qtables", "otp_requests"]
        st.dataframe(pd.DataFrame({"collection": names,
                                   "documents": [db.mongo_db[n].count_documents({}) for n in names]}),
                     use_container_width=True)
        if db.index_warnings:
            st.warning("Index warnings (usually old duplicate data blocking a unique index):")
            for w in db.index_warnings:
                st.code(w)
        card_close()

        card_open()
        st.markdown("#### Duplicate check")
        dups = list(db.users.aggregate([{"$group": {"_id": "$user", "n": {"$sum": 1}}}, {"$match": {"n": {"$gt": 1}}}]))
        pdups = list(db.profiles.aggregate([{"$group": {"_id": "$user", "n": {"$sum": 1}}}, {"$match": {"n": {"$gt": 1}}}]))
        sdups = list(db.sessions.aggregate([{"$group": {"_id": {"u": "$user", "n": "$session_number"}, "c": {"$sum": 1}}},
                                            {"$match": {"c": {"$gt": 1}}}]))
        if not dups and not pdups and not sdups:
            st.success("No duplicates: one document per person in `users` and `user_profiles`, one per session number.")
        else:
            st.error(f"Duplicates found — users: {dups}, profiles: {pdups}, sessions: {sdups}")
        card_close()

        card_open()
        st.markdown("#### Users (one row per person — login count, first/last login)")
        udf = _table(db.users, "last_login_utc")
        st.dataframe(udf, use_container_width=True) if not udf.empty else st.info("No users yet.")
        card_close()

        card_open()
        st.markdown("#### Sessions (one row per login — start, end, duration)")
        sdf_ = _table(db.sessions, "started_at_utc")
        st.dataframe(sdf_, use_container_width=True) if not sdf_.empty else st.info("No sessions yet.")
        st.caption("ended_at_* stays empty if the person closed the tab without pressing Logout.")
        card_close()

        card_open()
        st.markdown("#### Login / logout events (newest first)")
        ldf = _table(db.login_history, "timestamp_utc")
        st.dataframe(ldf, use_container_width=True) if not ldf.empty else st.info("No events yet.")
        card_close()

        card_open()
        st.markdown("#### Assessments (one row per 'Get Recommendations')")
        adf = _table(db.assessments, "timestamp")
        st.dataframe(adf, use_container_width=True) if not adf.empty else st.info("None yet.")
        card_close()

        card_open()
        st.markdown("#### Short check-ins (returning users)")
        cdf = _table(db.state_checkins, "timestamp")
        st.dataframe(cdf, use_container_width=True) if not cdf.empty else st.info("None yet.")
        card_close()

        card_open()
        st.markdown("#### Song feedback (newest first)")
        fbdf = _table(db.recommendation_feedback, "timestamp")
        st.dataframe(fbdf, use_container_width=True) if not fbdf.empty else st.info("None yet.")
        card_close()

        card_open()
        st.markdown("#### OTP requests (status only — OTPs/hashes are never shown)")
        odf = _table(db.otp_requests, "created_at", limit=50)
        st.dataframe(odf, use_container_width=True) if not odf.empty else st.info("None yet.")
        card_close()

        card_open()
        st.markdown("#### Export to CSV / Excel")
        st.caption("Participants, profiles, sessions, login/logout history, assessments, physiological readings, "
                   "check-ins, feedback — all linked by `user`, `session_id` and `assessment_id`. "
                   "`MASTER_flat_feedback` has one row per rated song joined with its assessment, session and participant.")
        if st.button("📦 Prepare export"):
            try:
                st.session_state["export_tables"] = _build_export_tables()
            except Exception as e:
                st.error(f"Export failed: {e}")
        tables = st.session_state.get("export_tables")
        if tables:
            try:
                st.download_button("⬇ Download everything (Excel .xlsx)", _excel_bytes(tables),
                                   file_name="musync_export.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning(f"Excel export unavailable ({type(e).__name__}); add `openpyxl` to requirements.txt. CSV below still works.")
            for tname, tdf in tables.items():
                st.download_button(f"⬇ {tname}.csv ({len(tdf)} rows)", tdf.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{tname}.csv", mime="text/csv", key=f"dl_{tname}")
        card_close()
