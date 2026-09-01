"""
db.py
-----
Unified data-access layer. Every other module calls the functions here
(e.g. collections().feedback) instead of touching MongoDB or the local
store directly. Automatically falls back to modules.local_store when
MongoDB is not configured/reachable, and exposes storage_mode() so the
UI can show a clear banner about which persistence mode is active.

Collections (per project schema requirement):
  users, profiles, psychological_assessments, physiological_measurements,
  music_preferences, music_catalog, recommendations,
  recommendation_feedback, model_predictions, validation_results,
  bias_assessments, experiments, login_history, qtables
"""

from .config import get_db, storage_mode
from .local_store import LocalDB

_local_db = LocalDB()

COLLECTION_NAMES = [
    "users", "profiles", "psychological_assessments",
    "physiological_measurements", "music_preferences", "music_catalog",
    "recommendations", "recommendation_feedback", "model_predictions",
    "validation_results", "bias_assessments", "experiments",
    "login_history", "qtables",
]


class Collections:
    """Attribute-style access, e.g. collections().feedback"""
    def __init__(self):
        db = get_db()
        self._db = db if db is not None else _local_db
        self.mode = storage_mode()

    def __getattr__(self, name):
        return self._db[name]


def collections():
    return Collections()
