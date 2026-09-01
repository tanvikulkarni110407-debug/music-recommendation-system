"""
config.py
---------
Central configuration: secrets loading, constants, and a graceful,
fail-safe MongoDB connector. Every other module should import `get_db()`
from here rather than opening its own connection.

Design decision (documented per project requirement #37 - "existing
code must remain workable"): if MongoDB is unreachable or credentials
are missing, the app must NOT crash. It falls back to a local on-disk
JSON store (see modules/local_store.py) and shows a visible banner
telling the user they are in "local fallback" persistence mode.
"""

import os
import streamlit as st

APP_NAME = "MuSync"
APP_TAGLINE = "AI-Assisted, Evidence-Based Personalized Music Recommendation (Research Prototype)"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
WESAD_DIR = os.path.join(DATA_DIR, "wesad")
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Music_dataset2.csv")
QTABLE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "QTables")
MODEL_DIR = os.path.dirname(os.path.dirname(__file__))

os.makedirs(WESAD_DIR, exist_ok=True)
os.makedirs(QTABLE_DIR, exist_ok=True)


def _secret(name, default=None):
    """Read a secret from st.secrets if present, else from env vars, else default.
    Never raises - missing secrets degrade functionality, they don't crash the app."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


MONGO_URI = _secret("MONGO_URI")
# NOTE: Gemini, Brevo (OTP email) and related secrets were removed.
# Login is plain email-based (no verification code) - see app.py.
# Explainability is handled by modules/explainability.py (real SHAP),
# not by an LLM narrative - so no external AI API key is required at all.


@st.cache_resource(show_spinner=False)
def get_mongo_client():
    """Return a live MongoClient, or None if unavailable. Never raises."""
    if not MONGO_URI:
        return None
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client
    except Exception:
        return None


def get_db():
    """Return the musync database, or None if Mongo is unavailable.
    Callers must handle the None case (use modules.local_store as a fallback)."""
    client = get_mongo_client()
    if client is None:
        return None
    return client["musync"]


def storage_mode():
    """'mongodb' or 'local' - used to render the persistence-mode banner."""
    return "mongodb" if get_db() is not None else "local"
