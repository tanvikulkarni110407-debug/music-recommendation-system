import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import os
import json
import urllib.parse
import requests
from pymongo import MongoClient
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException


client = MongoClient(st.secrets["MONGO_URI"])

SMTP_LOGIN = st.secrets["SMTP_LOGIN"]
SMTP_PASSWORD = st.secrets["SMTP_PASSWORD"]
SENDER_EMAIL = st.secrets["SENDER_EMAIL"]
BREVO_API_KEY = st.secrets["BREVO_API_KEY"]

# GenAI key is OPTIONAL on purpose — the app must run fully without it.
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    GEMINI_API_KEY = ""

HOST_EMAILS = [
    "aryadagare@gmail.com",
    "ratika.ind@gmail.com",
    "sahilkhopkar15@gmail.com",
    "tanvikulkarni110407@gmail.com",
    "yogesh.c@fcrit.ac.in"
]

db = client["music_recommendation"]
feedback_collection = db["feedback"]
qtable_collection = db["qtables"]

st.set_page_config(page_title="MRS", layout="wide")

st.markdown("""
<style>
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, #0f3d3e, #145c5f);
}
header[data-testid="stHeader"] * {
    color: #e6fffa !important;
}
.stApp {
    background: linear-gradient(135deg, #0f3d3e, #1f7a6d, #2fa4a9);
    color: #e6fffa;
}
div[data-testid="stAppViewContainer"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# ==================================================================
# Load Dataset
# ==================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Music_dataset2.csv")
    df.columns = df.columns.str.strip().str.lower()

    df.rename(columns={
        "track name": "song",
        "track": "song",
        "artist name(s)": "artist",
        "artist name": "artist",
        "artists": "artist",
        "artist": "artist",
        "genres": "genre",
        "release date": "release_date",
        "release_date": "release_date"
    }, inplace=True)

    if "artist" not in df.columns:
        df["artist"] = "Unknown Artist"

    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["year"] = df["release_date"].dt.year
    else:
        df["year"] = np.nan

    if "genre" not in df.columns:
        df["genre"] = "Unknown"
    required_cols = [c for c in ["song", "artist"] if c in df.columns]
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ["valence", "energy", "tempo"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5)
            if col != "tempo":
                df[col] = df[col].clip(0.0, 1.0)

    return df

df = load_data()

# ==================================================================
# Mood-HRV-Stress Alignment (audio-feature based, bias-free)
# ==================================================================
MOOD_AUDIO_TARGETS = {
    "Sad":       {"valence": (0.0, 0.35), "energy": (0.0, 0.40)},
    "Calm":      {"valence": (0.3, 0.60), "energy": (0.0, 0.35)},
    "Energetic": {"valence": (0.5, 1.00), "energy": (0.7, 1.00)},
    "Angry":     {"valence": (0.0, 0.40), "energy": (0.6, 1.00)},
    "Happy":     {"valence": (0.6, 1.00), "energy": (0.4, 0.80)},
}

def mood_physiology_fit(row, mood_state, hrv, stress):
    score = 0.0
    target = MOOD_AUDIO_TARGETS.get(mood_state, {})
    for feature, (low, high) in target.items():
        val = row.get(feature, None)
        if val is not None and low <= val <= high:
            score += 1.0

    energy = row.get("energy", None)
    if energy is not None:
        if hrv > 80 and energy < 0.4:
            score += 0.4
        elif hrv < 50 and energy > 0.6:
            score += 0.4

    if stress > 60:
        if energy is not None and energy < 0.4:
            score += 0.3
        if energy is not None and energy > 0.7:
            score -= 0.3

    return score

# ==================================================================
# Load Training Metadata
# ==================================================================
try:
    metadata = torch.load("metadata2.pth", map_location="cpu")
    genre_mapping = metadata["genre_mapping"]
    vibe_mapping = metadata["vibe_mapping"]
except Exception:
    st.error("Metadata loading failed. Please retrain models.")
    st.stop()

NUM_SONGS_TRAINED = metadata["num_songs"]
NUM_GENRES_TRAINED = metadata["num_genres"]
NUM_VIBES_TRAINED = metadata["num_vibes"]

if "genre" not in df.columns:
    df["genre"] = "Unknown"
df["genre"] = df["genre"].fillna("Unknown")

if "vibe" not in df.columns:
    df["vibe"] = "Neutral"
df["vibe"] = df["vibe"].fillna("Neutral")

num_songs = len(df)

# ==================================================================
# RECOMMENDATION ENGINE — Model definitions (RNN, NCF)
# No GenAI dependency anywhere in this section, by design.
# ==================================================================
class ContextRNN(nn.Module):
    def __init__(self, num_songs, num_genres, num_vibes, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.song_emb = nn.Embedding(num_songs, embed_dim)
        self.genre_emb = nn.Embedding(num_genres, 8)
        self.vibe_emb = nn.Embedding(num_vibes, 8)
        self.context_fc = nn.Linear(6, 16)
        self.lstm = nn.LSTM(160, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.attention = nn.Linear(64, 1)
        self.fc = nn.Linear(64, num_songs)

    def forward(self, seq, genre, vibe, ctx):
        s = self.song_emb(seq)
        g = self.genre_emb(genre).unsqueeze(1).repeat(1, seq.size(1), 1)
        v = self.vibe_emb(vibe).unsqueeze(1).repeat(1, seq.size(1), 1)
        c = self.context_fc(ctx).unsqueeze(1).repeat(1, seq.size(1), 1)
        x = torch.cat([s, g, v, c], dim=2)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)


class ContextNCF(nn.Module):
    def __init__(self, num_users, num_songs, num_genres, num_vibes):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 16)
        self.song_emb = nn.Embedding(num_songs, 32)
        self.genre_emb = nn.Embedding(num_genres, 8)
        self.vibe_emb = nn.Embedding(num_vibes, 8)
        self.mood_emb = nn.Embedding(6, 4)

        total_input = 16 + 32 + 8 + 8 + 4 + 1 + 1 + 1 + 1 + 1 + 1

        self.fc1 = nn.Linear(total_input, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, u, s, g, v, m, st_, h, t, d, w, pop):
        x = torch.cat([
            self.user_emb(u),
            self.song_emb(s),
            self.genre_emb(g),
            self.vibe_emb(v),
            self.mood_emb(m),
            st_, h, t, d, w, pop
        ], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


df["genre_id"] = (
    df["genre"].map(genre_mapping).fillna(0).astype(int).clip(0, NUM_GENRES_TRAINED - 1)
)
df["vibe_id"] = (
    df["vibe"].map(vibe_mapping).fillna(0).astype(int).clip(0, NUM_VIBES_TRAINED - 1)
)

assert df["genre_id"].max() < NUM_GENRES_TRAINED
assert df["vibe_id"].max() < NUM_VIBES_TRAINED

NUM_USERS_TRAINED = metadata["num_users"]


@st.cache_resource
def load_models():
    rnn = ContextRNN(
        num_songs=NUM_SONGS_TRAINED,
        num_genres=NUM_GENRES_TRAINED,
        num_vibes=NUM_VIBES_TRAINED
    )
    ncf = ContextNCF(
        num_users=NUM_USERS_TRAINED,
        num_songs=NUM_SONGS_TRAINED,
        num_genres=NUM_GENRES_TRAINED,
        num_vibes=NUM_VIBES_TRAINED
    )
    rnn.load_state_dict(torch.load("rnn_model_trained2.pth", map_location="cpu"), strict=True)
    ncf.load_state_dict(torch.load("ncf_model_trained2.pth", map_location="cpu"), strict=True)
    rnn.eval()
    ncf.eval()
    return rnn, ncf

rnn_model, ncf_model = load_models()

df["song_id"] = df.index.astype(int)
num_songs = len(df)

# ==================================================================
# RL Agent
# ==================================================================
class RLAgent:
    def __init__(self, n_actions):
        self.alpha = 0.2
        self.gamma = 0.9
        self.q_table = np.zeros((100, n_actions))

def update_q(q, s, a, r, ns, alpha=0.1, gamma=0.9):
    q[s, a] += alpha * (r + gamma * np.max(q[ns]) - q[s, a])

# ==================================================================
# Spotify metadata link (Client Credentials scope only — see spec doc)
# ==================================================================
def spotify_link(song, artist):
    song = song if pd.notna(song) else ""
    artist = artist if pd.notna(artist) else ""
    q = urllib.parse.quote_plus(f"{song} {artist}")
    return f"https://open.spotify.com/search/{q}"

def send_otp(email, otp):
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key["api-key"] = BREVO_API_KEY
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))

    email_data = sib_api_v3_sdk.SendSmtpEmail(
        sender={"email": SENDER_EMAIL},
        to=[{"email": host} for host in HOST_EMAILS],
        subject="Your OTP for Music Recommendation System",
        html_content=f"""
        <h2>Your OTP is: {otp}</h2>
        <p>This OTP is valid for 5 minutes.</p>
        """
    )
    try:
        api_instance.send_transac_email(email_data)
        return True
    except ApiException as e:
        st.error(f"Email Error: {e}")
        return False


def stable_user_hash(name: str, mod: int) -> int:
    """Deterministic replacement for Python's randomized hash()."""
    import hashlib
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest, 16) % mod


# ==================================================================
# MAIN PAGE — User Inputs
# ==================================================================
st.title("🎧 Music Recommendation System")
if "verified" not in st.session_state:
    st.session_state.verified = False

st.header("👤 User Details")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    email = st.text_input("Email")

    if "otp" not in st.session_state:
        st.session_state.otp = None
    if "verified" not in st.session_state:
        st.session_state.verified = False

    if st.button("Send OTP"):
        if email:
            otp = str(random.randint(100000, 999999))
            st.session_state.otp = otp
            if send_otp(email, otp):
                st.success("OTP sent successfully!")
        else:
            st.warning("Enter your email first.")

    entered_otp = st.text_input("Enter OTP")

    if st.button("Verify OTP"):
        if entered_otp == st.session_state.otp:
            st.session_state.verified = True
            st.success("Email verified!")
        else:
            st.error("Invalid OTP")

    if st.session_state.verified:
        name = email.split("@")[0]
    else:
        name = "Guest"

    if not st.session_state.verified:
        st.info("📧 Please verify your email first.")
        st.stop()

if "age_touched" not in st.session_state:
    st.session_state.age_touched = False

def mark_age_touched():
    st.session_state.age_touched = True

with col2:
    age = st.number_input(
        "Age", min_value=0, max_value=100, value=0, step=1,
        key="age_input", on_change=mark_age_touched
    )

age_valid = True
if st.session_state.age_touched and age < 18:
    age_valid = False
    st.markdown(
        "<span style='color:#ff4b4b;'>⚠ Age must be 18 or above to receive recommendations.</span>",
        unsafe_allow_html=True
    )

with col3:
    mood = st.selectbox("Current Mood", ["Happy", "Sad", "Angry", "Calm", "Energetic"])

st.subheader("🎶 Music Preferences")
colp1, colp2 = st.columns(2)
with colp1:
    genre_pref = st.selectbox("Preferred Genre", ["Bollywood", "Hindi Pop", "Ghazal"])
with colp2:
    era_pref = st.selectbox(
        "Preferred Vibe",
        ["60s songs", "90s songs", "Energetic songs", "calming songs", "Classical songs"]
    )

st.header("⌚ Smartwatch Data")
col3, col4 = st.columns(2)
with col3:
    hrv = st.slider("HR (bpm)", 20, 200, 90)
with col4:
    stress = st.slider("Stress Level", 0, 100, 40)

hrv_n = (hrv - 20) / 180
stress_n = stress / 100

st.header("🧠 Psychological Inputs")

TIPI_ALL = [
    "Q1. I see myself as extraverted, enthusiastic.",
    "Q2. I see myself as critical, quarrelsome.",
    "Q3. I see myself as dependable, self-disciplined.",
    "Q4. I see myself as anxious, easily upset.",
    "Q5. I see myself as open to new experiences, complex.",
    "Q6. I see myself as reserved, quiet.",
    "Q7. I see myself as sympathetic, warm.",
    "Q8. I see myself as disorganized, careless.",
    "Q9. I see myself as calm, emotionally stable.",
    "Q10. I see myself as conventional, uncreative."
]

DASS_ALL = [
    "Q1. I found it hard to wind down.",
    "Q2. I was aware of dryness of my mouth.",
    "Q3. I couldn’t seem to experience any positive feeling at all.",
    "Q4. I experienced breathing difficulty.",
    "Q5. I found it difficult to work up the initiative to do things.",
    "Q6. I tended to over-react to situations.",
    "Q7. I experienced trembling.",
    "Q8. I felt that I was using a lot of nervous energy.",
    "Q9. I was worried about situations in which I might panic.",
    "Q10. I felt that I had nothing to look forward to.",
    "Q11. I found myself getting agitated.",
    "Q12. I found it difficult to relax.",
    "Q13. I felt down-hearted and blue.",
    "Q14. I was intolerant of anything that kept me from getting on with what I was doing.",
    "Q15. I felt I was close to panic.",
    "Q16. I was unable to become enthusiastic about anything.",
    "Q17. I felt I wasn’t worth much as a person.",
    "Q18. I felt that I was rather touchy.",
    "Q19. I was aware of the action of my heart.",
    "Q20. I felt scared without any good reason.",
    "Q21. I felt that life was meaningless."
]

WHOQOL_ALL = [
    "Q1. How would you rate your quality of life?",
    "Q2. How satisfied are you with your health?",
    "Q3. To what extent do you feel that pain prevents you from doing what you need to do?",
    "Q4. How much do you need any medical treatment to function in your daily life?",
    "Q5. How much do you enjoy life?",
    "Q6. To what extent do you feel your life to be meaningful?",
    "Q7. How well are you able to concentrate?",
    "Q8. How safe do you feel in your daily life?",
    "Q9. How healthy is your physical environment?",
    "Q10. Do you have enough energy for everyday life?",
    "Q11. Are you able to accept your bodily appearance?",
    "Q12. Have you enough money to meet your needs?",
    "Q13. How available is the information that you need in your daily life?",
    "Q14. To what extent do you have the opportunity for leisure activities?",
    "Q15. How satisfied are you with your sleep?",
    "Q16. How satisfied are you with your ability to perform daily living activities?",
    "Q17. How satisfied are you with your capacity for work?",
    "Q18. How satisfied are you with yourself?",
    "Q19. How satisfied are you with your personal relationships?",
    "Q20. How satisfied are you with your sex life?",
    "Q21. How satisfied are you with the support from your friends?",
    "Q22. How satisfied are you with your living conditions?",
    "Q23. How satisfied are you with access to health services?",
    "Q24. How satisfied are you with your transport?",
    "Q25. How well are you able to get around?",
    "Q26. Are you satisfied with your environment?"
]

st.subheader("🧩 TIPI (Big Five)")
st.caption("1 = Disagree strongly | 7 = Agree strongly")
tipi = [st.slider(q, 1, 7, 4) for q in TIPI_ALL]

st.subheader("💭 DASS-21")
st.caption("0 = Did not apply | 3 = Applied very much")
dass = [st.slider(q, 0, 3, 1) for q in DASS_ALL]

st.subheader("🌍 WHOQOL-BREF")
st.caption("1 = Very poor | 5 = Very good")
whoqol = [st.slider(q, 1, 5, 3) for q in WHOQOL_ALL]

# ==================================================================
# Score calculation (TIPI / DASS-21 / WHOQOL-BREF)
# ==================================================================
def rev_tipi(x):
    return 8 - x

tipi_scored = tipi.copy()
for idx in [1, 3, 5, 7, 9]:
    tipi_scored[idx] = rev_tipi(tipi_scored[idx])

extraversion = (tipi_scored[0] + tipi_scored[5]) / 2
agreeableness = (tipi_scored[1] + tipi_scored[6]) / 2
conscientiousness = (tipi_scored[2] + tipi_scored[7]) / 2
emotional_stability = (tipi_scored[3] + tipi_scored[8]) / 2
openness = (tipi_scored[4] + tipi_scored[9]) / 2

dep_items = [2, 4, 9, 12, 15, 16, 20]
anx_items = [1, 3, 6, 8, 14, 18, 19]
str_items = [0, 5, 7, 10, 11, 13, 17]

depression = sum(dass[i] for i in dep_items) * 2
anxiety = sum(dass[i] for i in anx_items) * 2
stress_s = sum(dass[i] for i in str_items) * 2
dass_total = depression + anxiety + stress_s

def get_dass_mood(depression, stress_s, anxiety):
    if depression >= 20:
        return "Sad"
    elif stress_s >= 26:
        return "Angry"
    elif anxiety >= 16 and depression < 10:
        return "Energetic"
    elif depression < 10 and anxiety < 10 and stress_s < 10:
        return "Calm"
    else:
        return "Happy"

def final_mood(user_mood, dass_mood, weight_user=0.7):
    if user_mood == dass_mood:
        return user_mood
    return user_mood if random.random() < weight_user else dass_mood

dass_mood = get_dass_mood(depression, stress_s, anxiety)
mood_state = final_mood(mood, dass_mood, weight_user=0.7)

def rev_whoqol(x):
    return 6 - x

whoqol_scored = whoqol.copy()
for idx in [2, 3, 25]:
    whoqol_scored[idx] = rev_whoqol(whoqol_scored[idx])

physical_raw = sum(whoqol_scored[i] for i in [2, 3, 9, 14, 15, 16, 17])
psych_raw = sum(whoqol_scored[i] for i in [4, 5, 6, 10, 18, 25])
social_raw = sum(whoqol_scored[i] for i in [19, 20, 21])
env_raw = sum(whoqol_scored[i] for i in [7, 8, 11, 12, 13, 22, 23, 24])

physical_mean = physical_raw / 7
psych_mean = psych_raw / 6
social_mean = social_raw / 3
env_mean = env_raw / 8

physical_qol = (physical_mean - 4) * (100 / 16)
psych_qol = (psych_mean - 4) * (100 / 16)
social_qol = (social_mean - 4) * (100 / 16)
env_qol = (env_mean - 4) * (100 / 16)

# ==================================================================
# RL Setup
# ==================================================================
q_dir = "QTables"
os.makedirs(q_dir, exist_ok=True)

user_doc = qtable_collection.find_one({"user": name.lower()})
global_doc = qtable_collection.find_one({"user": "global"})

personal_q = np.array(user_doc["qtable"]) if user_doc else np.zeros((100, num_songs))
global_q = np.array(global_doc["qtable"]) if global_doc else np.zeros((100, num_songs))

if os.path.exists(feedback_file := os.path.join(q_dir, f"{name.lower()}_feedback.csv")):
    _existing = pd.read_csv(feedback_file)
    if "session_number" in _existing.columns:
        st.session_state["session_number"] = int(_existing["session_number"].max()) + 1
    else:
        st.session_state["session_number"] = 3
else:
    st.session_state["session_number"] = 1

if personal_q.shape[1] != num_songs:
    new_q = np.zeros((100, num_songs))
    min_cols = min(personal_q.shape[1], num_songs)
    new_q[:, :min_cols] = personal_q[:, :min_cols]
    personal_q = new_q

if global_q.shape[1] != num_songs:
    new_q = np.zeros((100, num_songs))
    min_cols = min(global_q.shape[1], num_songs)
    new_q[:, :min_cols] = global_q[:, :min_cols]
    global_q = new_q

if "recs" not in st.session_state:
    st.session_state["recs"] = []
if "got_recs" not in st.session_state:
    st.session_state["got_recs"] = False
if "feedback_count" not in st.session_state:
    st.session_state["feedback_count"] = 0
if "pool" not in st.session_state:
    st.session_state["pool"] = pd.DataFrame()
if "session_number" not in st.session_state:
    st.session_state["session_number"] = 1
if "genai_output" not in st.session_state:
    st.session_state["genai_output"] = None
if "genai_source" not in st.session_state:
    st.session_state["genai_source"] = None

def psychology_bias(row, mood_state, extraversion, openness,
                     depression, psych_qol, physical_qol, social_qol):
    score = 0.0
    genre = str(row.get("genre", "")).lower()
    mood_factor = 0.3 if mood_state in ["Sad", "Angry"] else 0.1

    if any(g in genre for g in ["slow", "soft", "ghazal", "classical"]):
        score += mood_factor * (depression / 42)

    personality_strength = (extraversion + openness) / 14
    if any(g in genre for g in ["dance", "pop", "bollywood"]):
        score += 0.2 * personality_strength

    if physical_qol < 40:
        score += 0.1 * (1 - physical_qol / 100)
    if social_qol < 40:
        score += 0.05 * (1 - social_qol / 100)

    return score

def get_user_state(mood_state, stress, depression):
    state = 0
    if mood_state == "Sad":
        state += 10
    elif mood_state == "Angry":
        state += 20
    elif mood_state == "Energetic":
        state += 30
    elif mood_state == "Calm":
        state += 40
    else:
        state += 50

    if stress > 70:
        state += 10
    elif stress > 40:
        state += 5

    if depression > 20:
        state += 5

    return min(state, 99)

FALLBACK_WEIGHTS = np.array([0.40, 0.28, 0.18, 0.09, 0.03, 0.02])

def get_final_weights(feedback_file, fallback_weights):
    MIN_SAMPLES = 20
    if not os.path.exists(feedback_file):
        return fallback_weights
    feedback_df = pd.read_csv(feedback_file)
    required_cols = ["rnn_score", "ncf_score", "personal_q",
                      "pref_bias", "physio_fit", "psy_bias", "rating"]
    if not all(c in feedback_df.columns for c in required_cols):
        return fallback_weights
    if len(feedback_df) < MIN_SAMPLES:
        return fallback_weights
    try:
        from sklearn.linear_model import Ridge
        X = feedback_df[["rnn_score", "ncf_score", "personal_q",
                          "pref_bias", "physio_fit", "psy_bias"]].values
        y = feedback_df["rating"].values
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        clipped = np.clip(model.coef_, 0.01, None)
        return clipped / clipped.sum()
    except Exception:
        return fallback_weights

MOOD_MAP = {"Sad": 0, "Angry": 1, "Energetic": 2, "Calm": 3, "Happy": 4}
mood_id = MOOD_MAP[mood_state]
stress_n = stress / 100.0
hrv_n = (hrv - 20) / 180.0


# ==================================================================
# EXPLAINABLE AI (XAI) — rule-based, deterministic, auditable
# Reads ONLY the scores the recommendation engine already computed.
# No external API call lives anywhere in this section.
# ==================================================================
FACTOR_PHRASES = {
    "rnn_score": "it fits the natural flow of songs you tend to listen to in sequence",
    "ncf_score": "it matches patterns from your overall listening profile",
    "personal_q": "you've responded positively to similar songs in your past sessions",
    "pref_bias": None,   # filled in dynamically with genre/vibe preference
    "physio_fit": None,  # filled in dynamically with HR/stress
    "psy_bias": None,    # filled in dynamically with psych profile
}

def rule_based_song_explanation(row, weights, mood_state, hrv, stress,
                                 depression, anxiety, genre_pref, era_pref):
    """
    Deterministic, template-based explanation.
    Ranks the same score columns the engine used for final_score,
    picks the top contributing factors, and turns them into a sentence.
    No LLM involved — this must remain fully auditable.
    """
    factor_cols = ["rnn_score", "ncf_score", "personal_q", "pref_bias", "physio_fit", "psy_bias"]
    contributions = {c: float(row.get(c, 0.0)) * w for c, w in zip(factor_cols, weights[:6])}
    ranked = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    top_factors = [f for f, v in ranked if v > 0][:3]

    reasons = []
    for f in top_factors:
        if f == "physio_fit":
            reasons.append(
                f"your heart rate ({hrv} bpm) and stress level ({stress}/100) suggest this "
                f"song's energy matches how your body feels right now"
            )
        elif f == "psy_bias":
            if depression >= 15:
                reasons.append("your assessment results indicate you may benefit from calmer, uplifting music")
            elif anxiety >= 12:
                reasons.append("your assessment results suggest lower-intensity music may help right now")
            else:
                reasons.append("your personality and wellbeing profile align well with this style")
        elif f == "pref_bias":
            reasons.append(f"it matches your stated preference for {genre_pref} / {era_pref}")
        else:
            reasons.append(FACTOR_PHRASES[f])

    if not reasons:
        reasons.append(f"it broadly matches your current mood ({mood_state}) and stated preferences")

    if len(reasons) == 1:
        body = reasons[0]
    else:
        body = ", ".join(reasons[:-1]) + " and " + reasons[-1]

    return f"Recommended because {body}."


# ==================================================================
# GENERATIVE AI — Gemini (optional), with deterministic fallback
# Called ONLY after the recommendation engine + XAI have finished.
# Never blocks or alters the recommendation list itself.
# ==================================================================
def _call_gemini(prompt: str, api_key: str, timeout: int = 8) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 300}
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

def _parse_gemini_sections(text: str):
    sections = {"wellness_summary": "", "recommendation_summary": "", "listening_advice": ""}
    key_map = {
        "WELLNESS_SUMMARY": "wellness_summary",
        "RECOMMENDATION_SUMMARY": "recommendation_summary",
        "LISTENING_ADVICE": "listening_advice",
    }
    current = None
    for line in text.splitlines():
        line = line.strip()
        matched = False
        for tag, key in key_map.items():
            if line.upper().startswith(tag):
                current = key
                sections[key] = line.split(":", 1)[-1].strip()
                matched = True
                break
        if not matched and current and line:
            sections[current] += " " + line

    if not any(sections.values()):
        return None
    return sections

def template_wellness_output(mood_state, stress, hrv, depression, anxiety, top_recs):
    """Deterministic fallback used whenever Gemini is unavailable, slow, or errors."""
    if depression >= 20:
        wellness = ("Your responses suggest you might be feeling low today. It's okay to have "
                    "days like this — small steps, like listening to music that feels comforting, "
                    "can help.")
    elif stress >= 60:
        wellness = (f"Your stress indicators are on the higher side today (stress level {stress}/100). "
                    "Taking a short break with calming music may help you reset.")
    elif mood_state == "Energetic":
        wellness = ("You're bringing good energy today. This is a great time to channel that "
                    "into music that keeps you moving.")
    else:
        wellness = "Your overall profile looks balanced today. Music can be a great companion right now."

    song_names = ", ".join([r["song"] for r in top_recs[:3]]) if top_recs else "your recommendations"
    rec_summary = (
        f"Based on your mood, physiological readings, and past feedback, we've picked songs like "
        f"{song_names} that should resonate with how you're feeling."
    )

    if stress >= 60 or anxiety >= 16:
        advice = "Try listening somewhere quiet, and take a few slow breaths between songs if you can."
    elif mood_state == "Energetic":
        advice = "This could be a great soundtrack for a walk, workout, or getting through your to-do list."
    else:
        advice = "Feel free to let these play in the background while you go about your day."

    return {
        "wellness_summary": wellness,
        "recommendation_summary": rec_summary,
        "listening_advice": advice,
    }

def generate_genai_layer(mood_state, stress, hrv, depression, anxiety, top_recs):
    """
    Writes prose ABOUT an already-finalized recommendation list + XAI
    explanations. Never re-ranks or re-selects songs. Returns
    (output_dict, source) where source is 'gemini' or 'template'.
    """
    fallback = template_wellness_output(mood_state, stress, hrv, depression, anxiety, top_recs)

    if not GEMINI_API_KEY:
        return fallback, "template"

    try:
        song_lines = "\n".join(
            f"- {r['song']} by {r['artist']}: {r.get('explanation', '')}" for r in top_recs[:3]
        )
        prompt = f"""
You are a supportive wellness assistant inside a music recommendation app.
The song list below was already generated by a separate machine-learning
system (RNN + NCF + Reinforcement Learning). Do not change, question, or
re-rank the recommendations — only write supportive, plain-language prose
about them. No medical claims, no diagnosis.

Context:
- Mood: {mood_state}
- Stress level: {stress}/100
- Heart rate: {hrv} bpm
- Depression indicator (DASS-21): {depression}
- Anxiety indicator (DASS-21): {anxiety}

Top recommended songs and why the system picked them:
{song_lines}

Respond with exactly these three labeled sections, each 1-2 short sentences:
WELLNESS_SUMMARY: <text>
RECOMMENDATION_SUMMARY: <text>
LISTENING_ADVICE: <text>
"""
        text = _call_gemini(prompt, GEMINI_API_KEY, timeout=8)
        parsed = _parse_gemini_sections(text)
        if parsed:
            return parsed, "gemini"
        return fallback, "template"
    except Exception:
        return fallback, "template"


# ==================================================================
# Recommendations (Single Click) — RECOMMENDATION ENGINE STAGE
# ==================================================================
st.header("🎵 Recommendations")
get_recs_btn = st.button("🎧 Get Recommendations", disabled=not age_valid)

if get_recs_btn:
    if not st.session_state.verified:
        st.error("⚠ Please verify your email first.")
        st.stop()

if get_recs_btn and not st.session_state["got_recs"]:
    st.session_state["got_recs"] = True

    pool = df.copy()

    def safe_filter(base_pool, condition, min_size=5):
        filtered = base_pool[condition]
        return filtered if len(filtered) >= min_size else base_pool

    if era_pref == "Classical songs":
        classical_pool = pool[pool["genre"].str.contains(
            "raga|classical|hindustani|carnatic|traditional music|contemporary classical|chamber music",
            case=False, na=False
        )]
        if len(classical_pool) >= 1:
            pool = classical_pool
    else:
        if era_pref == "60s songs":
            era_pool = pool[(pool["year"] >= 1960) & (pool["year"] <= 1969)]
            if len(era_pool) >= 20:
                pool = era_pool
        elif era_pref == "90s songs":
            era_pool = pool[(pool["year"] >= 1990) & (pool["year"] <= 1999)]
            if len(era_pool) >= 20:
                pool = era_pool

        genre_pool = pool[pool["genre"].str.contains(genre_pref.lower(), case=False, na=False)]
        if len(genre_pool) >= 10:
            pool = genre_pool

        if era_pref == "Energetic songs":
            vibe_pool = pool[(pool["energy"] > 0.60) & (pool["valence"] > 0.45)]
            if len(vibe_pool) >= 10:
                pool = vibe_pool
        elif era_pref == "calming songs":
            vibe_pool = pool[(pool["energy"] < 0.50) & (pool["valence"] > 0.30)]
            if len(vibe_pool) >= 10:
                pool = vibe_pool

    if mood_state == "Sad":
        pool = safe_filter(pool, (pool["energy"] < 0.50) & (pool["valence"] < 0.55))
    elif mood_state == "Happy":
        pool = safe_filter(pool, (pool["valence"] > 0.50) & (pool["energy"] > 0.45))
    elif mood_state == "Angry":
        pool = safe_filter(pool, (pool["energy"] > 0.50) & (pool["valence"] < 0.60))
    elif mood_state == "Calm":
        pool = safe_filter(pool, (pool["energy"] < 0.55) & (pool["valence"] > 0.30))
    elif mood_state == "Energetic":
        pool = safe_filter(pool, (pool["energy"] > 0.55) & (pool["valence"] > 0.45))

    if hrv > 100:
        pool = safe_filter(pool, pool["energy"] < 0.55)
    elif hrv < 50:
        pool = safe_filter(pool, pool["energy"] > 0.35)

    if stress > 70:
        pool = safe_filter(pool, (pool["energy"] < 0.55) & (pool["valence"] > 0.25))
    elif stress < 30:
        pool = safe_filter(pool, pool["energy"] > 0.35)

    if depression >= 20:
        pool = safe_filter(pool, pool["valence"] > 0.30)
    if anxiety >= 16:
        pool = safe_filter(pool, pool["energy"] < 0.65)
    if extraversion > 5:
        pool = safe_filter(pool, pool["energy"] > 0.40)
    if physical_qol < 25:
        pool = safe_filter(pool, pool["energy"] < 0.60)
    if social_qol < 25:
        pool = safe_filter(pool, pool["valence"] > 0.35)

    clean_pool = pool[~pool["song"].str.contains(
        "trap mix|remix|mashup|lo-fi mix", case=False, na=False
    )]
    if len(clean_pool) >= 5:
        pool = clean_pool

    def preference_bias(row, genre_pref, era_pref):
        score = 0.0
        g = str(row.get("genre", "")).lower()
        y = row.get("year", None)

        if genre_pref.lower() in g:
            score += 0.3
        if era_pref == "Energetic songs" and any(k in g for k in ["dance", "upbeat", "pop"]):
            score += 0.2
        if era_pref == "calming songs" and any(k in g for k in ["soft", "slow", "instrumental"]):
            score += 0.2
        if era_pref == "Classical songs" and any(
            k in g for k in ["raga", "classical", "hindustani", "carnatic",
                              "traditional music", "contemporary classical", "chamber music"]
        ):
            score += 0.2
        if y is not None:
            if era_pref == "60s songs" and 1960 <= y <= 1969:
                score += 0.25
            elif era_pref == "90s songs" and 1990 <= y <= 1999:
                score += 0.25

        return score

    pool = pool.copy()
    pool["physio_fit"] = pool.apply(
        lambda row: mood_physiology_fit(row, mood_state, hrv, stress), axis=1
    )

    state = get_user_state(mood_state, stress, depression)

    pool["personal_q"] = pool["song_id"].apply(lambda a: personal_q[state, a])
    pool["global_q"] = pool["song_id"].apply(lambda a: global_q[state, a])

    pool["psy_bias"] = pool.apply(
        lambda row: psychology_bias(
            row, mood_state, extraversion, openness,
            depression, psych_qol, physical_qol, social_qol
        ), axis=1
    )

    SEQ_LEN = 10
    shared_seq = torch.tensor(
        [random.sample(range(NUM_SONGS_TRAINED), min(SEQ_LEN, NUM_SONGS_TRAINED))],
        dtype=torch.long
    )

    tipi_mean = np.mean(tipi)
    tipi_n = (tipi_mean - 1) / 6.0
    whoql_n = (psych_mean - 1) / 4.0
    dass_mean = np.mean(dass)
    dass_n = dass_mean / 3.0
    mood_n = MOOD_MAP[mood_state] / 4.0

    context = torch.tensor(
        [[mood_n, stress_n, hrv_n, tipi_n, whoql_n, dass_n]],
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = rnn_model(shared_seq, torch.tensor([0]), torch.tensor([0]), context)
        probs = torch.softmax(logits, dim=1).squeeze()

    pool["rnn_score"] = pool["song_id"].apply(lambda i: probs[i % NUM_SONGS_TRAINED].item())

    user_hash = stable_user_hash(name.lower(), NUM_USERS_TRAINED)
    user_id = torch.tensor([user_hash], dtype=torch.long)

    def ncf_score(song_id):
        genre_id = int(df.loc[song_id, "genre_id"])
        vibe_id = int(df.loc[song_id, "vibe_id"])
        genre_id = max(0, min(genre_id, NUM_GENRES_TRAINED - 1))
        vibe_id = max(0, min(vibe_id, NUM_VIBES_TRAINED - 1))
        song_id_safe = song_id % NUM_SONGS_TRAINED

        song_id_t = torch.tensor([song_id_safe], dtype=torch.long)
        genre_id_t = torch.tensor([genre_id], dtype=torch.long)
        vibe_id_t = torch.tensor([vibe_id], dtype=torch.long)
        mood_t_t = torch.tensor([MOOD_MAP[mood_state]], dtype=torch.long)
        stress_t_t = torch.tensor([[stress_n]], dtype=torch.float32)
        hrv_t_t = torch.tensor([[hrv_n]], dtype=torch.float32)
        tipi_t = torch.tensor([[tipi_n]], dtype=torch.float32)
        dass_t = torch.tensor([[dass_n]], dtype=torch.float32)
        whoql_t = torch.tensor([[whoql_n]], dtype=torch.float32)
        pop_t = torch.tensor([[0.5]], dtype=torch.float32)

        with torch.no_grad():
            score = ncf_model(user_id, song_id_t, genre_id_t, vibe_id_t, mood_t_t,
                               stress_t_t, hrv_t_t, tipi_t, dass_t, whoql_t, pop_t)
        return score.item()

    pool["ncf_score"] = pool["song_id"].apply(ncf_score)
    pool["pref_bias"] = pool.apply(lambda r: preference_bias(r, genre_pref, era_pref), axis=1)

    weights = get_final_weights(
        os.path.join(q_dir, f"{name.lower()}_feedback.csv"), FALLBACK_WEIGHTS
    )

    def safe_normalize(col):
        min_v = col.min()
        max_v = col.max()
        if max_v - min_v < 1e-6:
            return np.zeros_like(col) + 0.5
        return (col - min_v) / (max_v - min_v)

    genre_counts = pool["genre"].value_counts()
    pool["diversity_penalty"] = pool["genre"].map(lambda g: np.log1p(genre_counts[g])) * 0.01

    global_feedback_file = os.path.join(q_dir, "global_session_feedback.csv")
    if os.path.exists(global_feedback_file):
        global_feedback_df = pd.read_csv(global_feedback_file)
        if "song_id" in global_feedback_df.columns:
            song_counts = np.bincount(global_feedback_df["song_id"].astype(int), minlength=num_songs)
        else:
            song_counts = np.zeros(num_songs)
    else:
        song_counts = np.zeros(num_songs)

    exploration_bonus = 0.1 / (1 + song_counts)
    pool["exploration_bonus"] = pool["song_id"].map(lambda i: exploration_bonus[i])

    cols = ["personal_q", "global_q", "psy_bias", "physio_fit",
            "pref_bias", "rnn_score", "ncf_score", "exploration_bonus"]
    for c in cols:
        pool[c] = safe_normalize(pool[c])

    pool["final_score"] = (
        weights[0] * pool["rnn_score"] +
        weights[1] * pool["ncf_score"] +
        weights[2] * pool["personal_q"] +
        weights[3] * pool["pref_bias"] +
        weights[4] * pool["physio_fit"] +
        weights[5] * pool["psy_bias"] +
        0.05 * pool["exploration_bonus"]
    ) - pool["diversity_penalty"]

    pool_sorted = pool.sort_values("final_score", ascending=False)
    top_candidates = pool_sorted.head(15)

    epsilon = 0.15
    n_available = len(top_candidates)
    n_pick = min(5, n_available)

    if random.random() < epsilon:
        chosen = top_candidates.sample(n_pick, replace=False)
    else:
        chosen = top_candidates.head(n_pick)

    if len(chosen) < 5:
        remaining = pool[~pool["song_id"].isin(chosen["song_id"])]
        extras = remaining.nlargest(5 - len(chosen), "final_score")
        chosen = pd.concat([chosen, extras], ignore_index=True)

    # ---- END OF RECOMMENDATION ENGINE. Everything above this line is
    #      RNN + NCF + RL + rule-based filters only. No GenAI call has
    #      happened yet. ----------------------------------------------

    st.session_state["pool"] = pool

    recs_list = chosen[["song_id", "song", "artist", "genre"]].to_dict("records")

    # ---- XAI STAGE: rule-based explanation per song, using the
    #      scores the engine already produced. Still no GenAI call. ---
    weights_list = list(weights)
    for rec in recs_list:
        row = pool[pool["song_id"] == rec["song_id"]].iloc[0]
        rec["explanation"] = rule_based_song_explanation(
            row, weights_list, mood_state, hrv, stress,
            depression, anxiety, genre_pref, era_pref
        )

    st.session_state["recs"] = recs_list

    # ---- GENAI STAGE: writes prose ABOUT the finished list. Runs once
    #      per recommendation batch; failure here never affects the
    #      recommendations or explanations above. -----------------------
    genai_output, genai_source = generate_genai_layer(
        mood_state, stress, hrv, depression, anxiety, recs_list
    )
    st.session_state["genai_output"] = genai_output
    st.session_state["genai_source"] = genai_source

if get_recs_btn:
    st.session_state["feedback_count"] = 0

# ==================================================================
# Show Recommendations
# ==================================================================
if "recs" in st.session_state and st.session_state["recs"]:

    # ---- Wellness summary (GenAI or template fallback) ----
    if st.session_state.get("genai_output"):
        genai_output = st.session_state["genai_output"]
        source_label = "AI-generated" if st.session_state["genai_source"] == "gemini" else "template-based"
        st.subheader("🧠 Wellness Insight")
        st.write(genai_output.get("wellness_summary", ""))
        st.write(genai_output.get("recommendation_summary", ""))
        st.write(genai_output.get("listening_advice", ""))
        st.caption(
            f"This summary is {source_label} and is written *after* the recommendations "
            "below were finalized by the RNN + NCF + RL engine — it never influences which "
            "songs were chosen."
        )
        st.markdown("---")

    if "spotify_touched" not in st.session_state:
        st.session_state.spotify_touched = {}
    if "pending_feedback_song" not in st.session_state:
        st.session_state.pending_feedback_song = None
    if "lock_warning_song" not in st.session_state:
        st.session_state.lock_warning_song = None
    if "song_touched" not in st.session_state:
        st.session_state.song_touched = {}

    for i, s in enumerate(st.session_state["recs"]):

        if i not in st.session_state.spotify_touched:
            st.session_state.spotify_touched[i] = False

        st.subheader(f"{i+1}. {s['song']} – {s['artist']}")

        # ---- Rule-based XAI explanation, shown per song ----
        if s.get("explanation"):
            st.caption(f"🔍 {s['explanation']}")

        flag_key = f"fb_done_{i}_{s['song_id']}"
        if flag_key not in st.session_state:
            st.session_state[flag_key] = False

        rating = st.radio(
            "Rate this song (1 = Strongly dislike, 5 = Strongly like)",
            [1, 2, 3, 4, 5], horizontal=True, key=f"rate_{i}_{s['song_id']}"
        )

        feedback_btn = st.button(
            f"Submit Feedback for Song {i+1}",
            key=f"fb_{i}_{s['song_id']}",
            disabled=st.session_state[flag_key]
        )

        if feedback_btn:
            st.session_state[flag_key] = True
            if st.session_state.pending_feedback_song == i:
                st.session_state.pending_feedback_song = None
                st.session_state.lock_warning_song = None

        def mark_song_touched(idx):
            st.session_state.song_touched[idx] = True

        spotify_url = spotify_link(s["song"], s["artist"])
        spotify_key = f"spotify_{i}_{s['song_id']}"

        if st.button(f"🎧 Open Song {i+1} in Spotify", key=spotify_key):
            allow_open = True
            locked = st.session_state.pending_feedback_song

            if locked is not None and locked != i:
                locked_song = st.session_state["recs"][locked]
                locked_flag = f"fb_done_{locked}_{locked_song['song_id']}"
                if not st.session_state.get(locked_flag, False):
                    st.session_state.lock_warning_song = locked
                    allow_open = False

            if allow_open:
                st.session_state.pending_feedback_song = i
                st.session_state.lock_warning_song = None
                st.markdown(
                    f'<a href="{spotify_url}" target="_blank">🎧 Click here to open Song {i+1} in Spotify</a>',
                    unsafe_allow_html=True
                )

        if st.session_state.lock_warning_song == i:
            warn_song = st.session_state["recs"][i]
            warn_flag = f"fb_done_{i}_{warn_song['song_id']}"
            if not st.session_state.get(warn_flag, False):
                st.markdown(
                    "<span style='color:#ff4b4b;'>⚠ Submit feedback before opening another song</span>",
                    unsafe_allow_html=True
                )

        if feedback_btn:
            song_action = s["song_id"]
            reward_map = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
            reward = reward_map.get(rating, 0)

            feedback_file = os.path.join(q_dir, f"{name.lower()}_feedback.csv")

            pool_ref = st.session_state["pool"]
            def _val(col):
                m = pool_ref.loc[pool_ref["song_id"] == song_action, col]
                return float(m.values[0]) if len(m) > 0 else 0.0

            new_entry = {
                "song_id": song_action,
                "song": s["song"],
                "artist": s["artist"],
                "rating": rating,
                "hrv": hrv,
                "stress": stress,
                "session_number": st.session_state["session_number"],
                "extraversion": extraversion,
                "agreeableness": agreeableness,
                "conscientiousness": conscientiousness,
                "emotional_stability": emotional_stability,
                "openness": openness,
                "depression": depression,
                "anxiety": anxiety,
                "stress_score": stress_s,
                "physical_qol": physical_qol,
                "psych_qol": psych_qol,
                "social_qol": social_qol,
                "env_qol": env_qol,
                "age": age,
                "mood_id": mood_id,
                "explanation": s.get("explanation", ""),
                "rnn_score": _val("rnn_score"),
                "ncf_score": _val("ncf_score"),
                "personal_q": _val("personal_q"),
                "pref_bias": _val("pref_bias"),
                "physio_fit": _val("physio_fit"),
                "psy_bias": _val("psy_bias"),
            }

            if os.path.exists(feedback_file):
                feedback_df = pd.read_csv(feedback_file)
            else:
                feedback_df = pd.DataFrame(columns=new_entry.keys())

            feedback_collection.insert_one(new_entry)
            feedback_df = pd.concat([feedback_df, pd.DataFrame([new_entry])], ignore_index=True)
            feedback_df.to_csv(feedback_file, index=False)

            current_state = get_user_state(mood_state, stress, depression)
            update_q(personal_q, current_state, song_action, reward, current_state)
            update_q(global_q, current_state, song_action, reward, current_state)

            qtable_collection.update_one(
                {"user": name.lower()}, {"$set": {"qtable": personal_q.tolist()}}, upsert=True
            )
            qtable_collection.update_one(
                {"user": "global"}, {"$set": {"qtable": global_q.tolist()}}, upsert=True
            )

            st.session_state[flag_key] = True
            st.session_state["feedback_count"] += 1
            st.success("Feedback recorded")

# ==================================================================
# Footer / Session-end feedback
# ==================================================================
st.markdown("---")

if "session_finished" not in st.session_state:
    st.session_state.session_finished = False

if st.session_state["recs"]:
    if st.button("✅ Finish Listening Session"):
        st.session_state.session_finished = True

if st.session_state.session_finished:
    st.header("⭐ Overall System Feedback")

    comfort = st.slider(
        "Q1. How comfortable did you feel using the system? (1 = Very uncomfortable, 10 = Extremely comfortable)",
        1, 10, 5
    )
    satisfaction = st.slider(
        "Q2. How satisfied are you with the recommendations? (1 = Very dissatisfied, 10 = Extremely satisfied)",
        1, 10, 5
    )
    mood_alignment = st.slider(
        "Q3. How well did songs match your mood? (1 = Not at all, 10 = Perfectly matched)",
        1, 10, 5
    )
    experience = st.slider("Q4. Rate your overall experience? (1 = Poor, 10 = Excellent)", 1, 10, 5)
    continue_use = st.radio("Q5. Would you like to continue using this system?", ["Yes", "No"])

    overall_btn = st.button("Submit the Feedback")

    if overall_btn:
        entry = {
            "user": name.lower(),
            "comfort": comfort,
            "satisfaction": satisfaction,
            "mood_alignment": mood_alignment,
            "experience": experience,
            "continue": continue_use,
            "avg_song_rating": np.mean([
                st.session_state.get(f"rate_{i}_{s['song_id']}", 3)
                for i, s in enumerate(st.session_state["recs"])
            ]),
            "mood_state": mood_state,
            "stress": stress,
            "hrv": hrv,
            "genai_source": st.session_state.get("genai_source", "template"),
        }

        personal_file = os.path.join(q_dir, f"{name.lower()}_session_feedback.csv")
        if os.path.exists(personal_file):
            df_personal = pd.read_csv(personal_file)
        else:
            df_personal = pd.DataFrame(columns=entry.keys())
        df_personal = pd.concat([df_personal, pd.DataFrame([entry])], ignore_index=True)
        df_personal.to_csv(personal_file, index=False)

        global_file = os.path.join(q_dir, "global_session_feedback.csv")
        if os.path.exists(global_file):
            df_global = pd.read_csv(global_file)
        else:
            df_global = pd.DataFrame(columns=entry.keys())
        df_global = pd.concat([df_global, pd.DataFrame([entry])], ignore_index=True)
        df_global.to_csv(global_file, index=False)

        st.success("Thank you for participating!")
        st.session_state.session_finished = False
