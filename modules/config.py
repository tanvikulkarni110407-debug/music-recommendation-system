"""
config.py
---------
Central configuration for MuSync.

- Loads configuration from Streamlit Secrets or environment variables.
- Connects to MongoDB Atlas using MONGO_URI.
- Uses the "musync" database.
- Provides get_db() and storage_mode() for the rest of the application.
"""

import os
import streamlit as st


# ================================================================
# APPLICATION SETTINGS
# ================================================================

APP_NAME = "MuSync"

APP_TAGLINE = (
    "AI-Assisted, Evidence-Based Personalized Music "
    "Recommendation (Research Prototype)"
)


# ================================================================
# PROJECT PATHS
# ================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
WESAD_DIR = os.path.join(DATA_DIR, "wesad")

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Music_dataset2.csv"
)

QTABLE_DIR = os.path.join(
    BASE_DIR,
    "QTables"
)

MODEL_DIR = BASE_DIR


# Create required directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WESAD_DIR, exist_ok=True)
os.makedirs(QTABLE_DIR, exist_ok=True)


# ================================================================
# SECRET LOADER
# ================================================================

def _secret(name, default=None):
    """
    Read a value from Streamlit Secrets first,
    then environment variables.

    Returns the default value if the secret is unavailable.
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.environ.get(name, default)


# ================================================================
# MONGODB CONFIGURATION
# ================================================================

# IMPORTANT:
# Your Streamlit Secret must be named exactly:
#
# MONGO_URI = "mongodb+srv://..."
#
MONGO_URI = _secret("MONGO_URI", "")

# Database name
MONGODB_DATABASE = _secret(
    "MONGODB_DATABASE",
    "musync"
)


# ================================================================
# MONGODB CONNECTION
# ================================================================

@st.cache_resource(show_spinner=False)
def get_mongo_client():
    """
    Create and test the MongoDB connection.

    Returns:
        MongoClient if connection succeeds.
        None if MongoDB is unavailable.
    """

    if not MONGO_URI:
        return None

    try:
        from pymongo import MongoClient

        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000
        )

        # Test connection
        client.admin.command("ping")

        return client

    except Exception:
        return None


# ================================================================
# DATABASE ACCESS
# ================================================================

def get_db():
    """
    Return the MuSync MongoDB database.

    Returns:
        MongoDB database object if available.
        None if MongoDB cannot be reached.
    """

    client = get_mongo_client()

    if client is None:
        return None

    return client[MONGODB_DATABASE]


# ================================================================
# STORAGE MODE
# ================================================================

def storage_mode():
    """
    Return the currently active persistence mode.

    Returns:
        "mongodb" when MongoDB is available.
        "local" otherwise.
    """

    return (
        "mongodb"
        if get_db() is not None
        else "local"
    )
