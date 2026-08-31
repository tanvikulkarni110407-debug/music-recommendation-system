import os

APP_NAME = "MuSync"
APP_TAGLINE = "AI-Assisted, Evidence-Based Personalized Music Recommendation (Research Prototype)"

QTABLE_DIR = os.path.join("QTables")

BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
HOST_EMAILS = os.getenv("HOST_EMAILS", "")