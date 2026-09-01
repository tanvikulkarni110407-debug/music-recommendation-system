

import os
import random
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


from modules.config import APP_NAME, APP_TAGLINE, QTABLE_DIR
from modules.theme import inject_theme, mode_badge, card_open, card_close, COLORS
from pymongo import MongoClient
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
]:
    if key not in st.session_state:
        st.session_state[key] = default


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
# MongoDB Atlas connection
# --------------------------------------------------------------
# Streamlit Secrets:
# MONGODB_URI = "mongodb+srv://<username>:<password>@<cluster>/..."
# MONGODB_DATABASE = "musync"
# --------------------------------------------------------------
def _get_secret(name, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


class MongoDBStore:
    def __init__(self):
        uri = _get_secret("MONGODB_URI")
        db_name = _get_secret("MONGODB_DATABASE", "musync")

        if not uri:
            raise RuntimeError(
                "MONGODB_URI is missing. Add MONGODB_URI and "
                "MONGODB_DATABASE in Streamlit Secrets."
            )

        self.client = MongoClient(
            uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
            retryWrites=True,
        )
        self.client.admin.command("ping")
        self.mongo_db = self.client[db_name]

        # Collections used by the app.
        self.login_history = self.mongo_db["login_history"]
        self.profiles = self.mongo_db["user_profiles"]
        self.physiological_measurements = self.mongo_db["physiological_measurements"]
        self.qtables = self.mongo_db["qtables"]
        self.recommendation_feedback = self.mongo_db["recommendation_feedback"]
        self.experiments = self.mongo_db["experiments"]
        self.bias_assessments = self.mongo_db["bias_assessments"]
        self.mode = "mongodb"

        # Useful indexes.
        try:
            self.login_history.create_index("user_email")
            self.profiles.create_index("user", unique=True)
            self.qtables.create_index("user", unique=True)
            self.recommendation_feedback.create_index(
                [("user", 1), ("timestamp", -1)]
            )
            self.experiments.create_index([("user", 1), ("timestamp", -1)])
            self.physiological_measurements.create_index(
                [("user", 1), ("timestamp", -1)]
            )
            self.bias_assessments.create_index([("user", 1), ("timestamp", -1)])
        except PyMongoError:
            pass


@st.cache_resource(show_spinner=False)
def _connect_mongodb():
    return MongoDBStore()


try:
    db = _connect_mongodb()
    mongodb_error = None
except Exception as e:
    db = None
    mongodb_error = str(e)


def spotify_link(song, artist):
    q = urllib.parse.quote_plus(f"{song} {artist}")
    return f"https://open.spotify.com/search/{q}"


# --------------------------------------------------------------
# Research-source recommendation anchors
# --------------------------------------------------------------
# At least ONE recommendation per generated set is guaranteed to come
# from the Spotify sources supplied for this project.  The first source
# is an album whose tracks can be resolved reliably without requiring a
# Spotify API key.
RESEARCH_SOURCE_TRACKS = [
    {
        "song": "Raag Miyan Ki Todi",
        "artist": "Nikhil Banerjee",
        "genre": "Indian Classical",
        "source_url": "https://open.spotify.com/track/2wmy0bj9Lchz0cQnmriBR0",
        "source_name": "Fond Memories-Sitar Vol-1 (provided Spotify source)",
    },
    {
        "song": "Raag Rageshree",
        "artist": "Nikhil Banerjee",
        "genre": "Indian Classical",
        "source_url": "https://open.spotify.com/album/4CbM5IC1txrx40X0AYyPmP",
        "source_name": "Fond Memories-Sitar Vol-1 (provided Spotify source)",
    },
    {
        "song": "Raag Nat Bhairav",
        "artist": "Nikhil Banerjee",
        "genre": "Indian Classical",
        "source_url": "https://open.spotify.com/track/1Y76LppGA9FsyAackq9uLy",
        "source_name": "Fond Memories-Sitar Vol-1 (provided Spotify source)",
    },
]

# These are the exact Spotify sources supplied by the project owner.
# The app records them as research-source metadata; it does not claim
# that every track in these playlists is clinically/scientifically proven.
RESEARCH_SPOTIFY_SOURCES = [
    "https://open.spotify.com/album/4CbM5IC1txrx40X0AYyPmP",
    "https://open.spotify.com/playlist/0efes1si9D7BtI93izeQJ1",
    "https://open.spotify.com/playlist/1WZr6aA4096hUr9ssO2bcZ",
    "https://open.spotify.com/playlist/5Y4kPb6Q4Ftrui4e5cRsKb",
]


def _add_required_research_song(pool, df):
    """Guarantee one supplied research-source track in the final pool.

    Prefer a matching song already present in the project's music dataset.
    If the dataset does not contain one, append a source-only anchor row.
    Source-only rows are logged normally but are excluded from Q-table updates
    because they do not have a dataset column/index.
    """
    anchor = RESEARCH_SOURCE_TRACKS[0]

    # Prefer an exact match already present in the loaded dataset/pool.
    mask = (
        pool["song"].astype(str).str.strip().str.casefold().eq(anchor["song"].casefold())
        & pool["artist"].astype(str).str.strip().str.casefold().eq(anchor["artist"].casefold())
    ) if not pool.empty and "song" in pool.columns and "artist" in pool.columns else pd.Series(dtype=bool)

    if len(mask) and mask.any():
        idx = pool.index[mask][0]
        pool.loc[idx, "research_source"] = True
        pool.loc[idx, "source_url"] = anchor["source_url"]
        pool.loc[idx, "source_name"] = anchor["source_name"]
        return pool, anchor, False

    # Dataset does not contain the anchor: append a source-only row.
    row = {c: np.nan for c in pool.columns}
    row.update({
        "song_id": "research_anchor_raag_miyan_ki_todi",
        "song": anchor["song"],
        "artist": anchor["artist"],
        "genre": anchor["genre"],
        "research_source": True,
        "source_url": anchor["source_url"],
        "source_name": anchor["source_name"],
    })

    numeric_score_cols = [
        "rnn_score", "ncf_score", "personal_q", "pref_bias",
        "physio_fit", "psy_bias", "final_score"
    ]
    for col in numeric_score_cols:
        if col in pool.columns:
            row[col] = float(pool[col].max()) if pool[col].notna().any() else 0.0
    if "final_score" in pool.columns:
        max_score = pd.to_numeric(pool["final_score"], errors="coerce").max()
        row["final_score"] = (float(max_score) + 1.0) if pd.notna(max_score) else 1.0

    pool = pd.concat([pool, pd.DataFrame([row])], ignore_index=True)
    return pool, anchor, True


def _choose_with_research_anchor(pool, n=5):
    """Select n recommendations while forcing one supplied research track."""
    if pool.empty:
        return pool.head(0)

    research_mask = pool.get("research_source", pd.Series(False, index=pool.index)).fillna(False).astype(bool)
    research_rows = pool[research_mask]
    normal_pool = pool[~research_mask]

    if research_rows.empty:
        return rec.select_recommendations(pool, n=n)

    anchor_row = research_rows.sort_values("final_score", ascending=False).head(1)
    remaining = max(0, n - 1)
    if remaining:
        normal_rows = rec.select_recommendations(normal_pool, n=remaining) if not normal_pool.empty else normal_pool.head(0)
        return pd.concat([anchor_row, normal_rows], ignore_index=True)
    return anchor_row.reset_index(drop=True)



# --------------------------------------------------------------
# Per-song recommendation explanations
# --------------------------------------------------------------
def _song_explanation(song_row, ctx, is_research=False):
    """Explain one recommendation in plain, user-friendly language."""
    song = str(song_row.get("song", "This song"))
    artist = str(song_row.get("artist", "the artist"))
    genre = str(song_row.get("genre", "this style of music"))
    mood = str(ctx.get("mood_state", "your current mood"))
    preferred_genre = str(ctx.get("genre_pref", "")).strip()
    preferred_vibe = str(ctx.get("era_pref", "")).strip()

    if is_research:
        return (
            f"{song} by {artist} is the research-supported choice in this 5-song playlist. "
            "It was included from the research-source music selected for this project, "
            "so you receive one evidence-informed option alongside the personalized songs. "
            "It is also presented in the context of your current listening preferences. "
            "This does not mean the individual song is a medical treatment or will have the same effect for everyone."
        )

    reasons = []

    # Natural interpretation of the actual recommendation signals.
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
            reasons.append(f"it adds variety while staying within the broader listening style considered for you")

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

    # Add one or two song-specific strongest signals so different songs can
    # receive meaningfully different explanations.
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

    # De-duplicate and cap the explanation so it remains readable.
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
    st.divider()

    PAGES = ["Dashboard", "Profile", "Psychological Assessment", "Physiological Input",
              "Music Preference & Recommendation", "Evaluation & Validation",
              "Bias & Risk of Bias", "Research Evidence", "Dataset / Research Mode"]
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
            "MongoDB is not connected. Check MONGO_URI in Streamlit Secrets "
            "and restart the app."
        )
        st.code(mongodb_error)

    card_open()
    st.subheader("👤 Sign in")

    if db:
        st.success(
            "MongoDB Atlas connected — application data will be stored in MongoDB."
        )

    if not st.session_state.verified:
        st.info(
            "Welcome to MuSync. Enter your name or participant ID to continue."
        )

        name_input = st.text_input(
            "Enter your name or participant ID"
        )

        email_input = st.text_input(
            "Email (optional, used only as an identifier)"
        )

        if st.button("Continue", type="primary"):
            if name_input.strip():
                st.session_state.verified = True
                st.session_state.username = (
                    name_input.strip().lower().replace(" ", "_")
                )
                st.session_state.user_email = (
                    email_input.strip()
                    if email_input.strip()
                    else f"{st.session_state.username}@local"
                )

                ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))

                try:
                    if db:
                        db.login_history.insert_one({
                            "user_email": st.session_state.user_email,
                            "username": st.session_state.username,
                            "login_time_ist": ist_now.strftime(
                                "%Y-%m-%d %I:%M:%S %p"
                            ),
                        })
                except Exception:
                    pass

                st.success("Signed in successfully!")
                st.rerun()
            else:
                st.warning("Please enter your name or participant ID.")
    else:
        st.success(
            f"Signed in as **{st.session_state.username}**"
        )

        if st.button("🚪 Logout"):
            for k in [
                "verified", "username", "user_email",
                "profile_doc", "profile_user"
            ]:
                st.session_state[k] = (
                    None if k != "verified" else False
                )
            st.rerun()

    card_close()

    if st.session_state.verified:
        st.markdown(
            "Use the sidebar to continue: **Profile → "
            "Psychological Assessment → Physiological Input → "
            "Music Preference & Recommendation**."
        )

# ================================================================
# Everything below requires sign-in
# ================================================================
elif not db:
    st.error("MongoDB connection is required. Configure MONGO_URI in Streamlit Secrets.")
elif not st.session_state.verified:
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
        st.session_state["profile_doc"] = db.profiles.find_one({"user": name})
        st.session_state["editing_profile"] = st.session_state["profile_doc"] is None

    profile_doc = st.session_state["profile_doc"]
    has_profile = profile_doc is not None

    if has_profile and not st.session_state["editing_profile"]:
        card_open()
        st.write(f"**Age:** {profile_doc.get('age')}")
        st.write(f"**Preferred Genre:** {profile_doc.get('genre_pref')}")
        st.write(f"**Preferred Vibe/Era:** {profile_doc.get('era_pref')}")
        st.caption("TIPI, DASS-21 baseline, and WHOQOL-BREF answers are saved and reused automatically.")
        if st.button("✏️ Update Profile"):
            st.session_state["editing_profile"] = True
            st.rerun()
        card_close()

    if st.session_state["editing_profile"]:
        defaults = profile_doc or {}
        card_open()
        st.subheader("Complete / Update Your Profile")
        st.caption("Collected once and reused on every future login.")
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
            ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
            profile_data = {
                "user": name, "email": st.session_state.user_email, "age": int(age),
                "genre_pref": genre_pref, "era_pref": era_pref,
                "tipi": tipi, "dass": dass, "whoqol": whoqol,
                "updated_at_ist": ist_now.strftime("%Y-%m-%d %I:%M:%S %p"),
            }
            db.profiles.update_one({"user": name}, {"$set": profile_data}, upsert=True)
            st.session_state["profile_doc"] = profile_data
            st.session_state["editing_profile"] = False
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

    profile_doc = st.session_state.get("profile_doc")
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
                   "They represent perceived state, not a sensor reading.")
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
                rr_df = pd.read_csv(uploaded, header=None)
                rr_values = pd.to_numeric(rr_df.iloc[:, 0], errors="coerce").dropna().values
                features, note = physio.compute_hrv_features(rr_values)
                if features is None:
                    st.error(note)
                else:
                    st.success("HRV features computed from your uploaded data:")
                    st.json(features)
                    if note:
                        st.caption(f"⚠ {note}")
                    db.physiological_measurements.insert_one({
                        "user": st.session_state.username, "features": features,
                        "timestamp": datetime.now(timezone.utc).isoformat(), "source": "uploaded_rr",
                    })
            except Exception as e:
                st.error(f"Could not parse file: {e}")
        card_close()

    with tab3:
        card_open()
        mode_badge("RESEARCH DATASET MODE", "research")
        subjects = physio.list_available_wesad_subjects()
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

    card_open()
    st.subheader("💭 Quick Mood Check-in (DASS-21, 10 items)")
    dass = dass_baseline.copy()
    for idx in psy.DASS_DYNAMIC_INDICES:
        dass[idx] = st.slider(psy.DASS_ALL[idx], 0, 3, int(dass_baseline[idx]) if idx < len(dass_baseline) else 1,
                               key=f"dass_dyn_{idx}")
    card_close()

    tipi_scores = psy.score_tipi(tipi)
    dass_scores = psy.score_dass21(dass)
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
        "dass_n": np.mean(dass) / 3.0, "mood_n": rec.MOOD_MAP[mood_state] / 4.0,
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

    feedback_file = os.path.join(QTABLE_DIR, f"{name}_feedback.csv")

    def weights_getter():
        if not os.path.exists(feedback_file):
            return rec.FALLBACK_WEIGHTS
        fdf = pd.read_csv(feedback_file)
        needed = ["rnn_score", "ncf_score", "personal_q", "pref_bias", "physio_fit", "psy_bias", "rating"]
        if len(fdf) < 20 or not all(c in fdf.columns for c in needed):
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
        st.session_state["got_recs"] = True
        st.session_state["feedback_count"] = 0
        pool = rec.build_candidate_pool(df, mood_state, hrv, stress, genre_pref, era_pref,
                                         ctx["depression"], ctx["anxiety"], ctx["extraversion"],
                                         ctx["physical_qol"], ctx["social_qol"])
        pool, mode_note = rec.score_pool(pool, ctx, personal_q, global_q,
                                          {"metadata": metadata, "rnn_model": rnn_model, "ncf_model": ncf_model},
                                          weights_getter, num_songs)

        confidence, spread = saf.assess_confidence(pool["final_score"])
        flagged = saf.get_flagged_songs(db.recommendation_feedback, name)
        pool, safety_note = saf.apply_safety_layer(pool, confidence, flagged, rec.neutral_safe_pool)

        # Guarantee that at least one recommendation is from the supplied
        # Spotify research sources.  This is enforced after the safety layer
        # so the requirement cannot disappear during normal ranking/filtering.
        pool, research_anchor, anchor_was_added = _add_required_research_song(pool, df)
        chosen = _choose_with_research_anchor(pool, n=5)

        st.session_state["pool"] = pool
        rec_records = chosen[["song_id", "song", "artist", "genre"]].to_dict("records")
        for rec_record in rec_records:
            if rec_record["song"] == research_anchor["song"] and rec_record["artist"] == research_anchor["artist"]:
                rec_record["research_source"] = True
                rec_record["source_url"] = research_anchor["source_url"]
                rec_record["source_name"] = research_anchor["source_name"]
            else:
                rec_record["research_source"] = False
        st.session_state["recs"] = rec_records
        st.session_state["mode_note"] = (
            f"{mode_note}\n\nResearch-source constraint applied: at least one recommendation "
            f"is guaranteed from the supplied Spotify sources — {research_anchor['song']} "
            f"by {research_anchor['artist']}."
        )
        st.session_state["safety_note"] = safety_note
        st.session_state["confidence"] = confidence
        st.session_state["mood_state_used"] = mood_state

    if st.session_state["recs"]:
        card_open()
        st.markdown("### 🧠 Explanation")
        mode_badge(st.session_state.get("confidence", "high").upper() + " CONFIDENCE",
                    "research" if st.session_state.get("confidence") == "high" else "warning")
        st.write(st.session_state.get("mode_note", ""))
        if st.session_state.get("safety_note"):
            st.warning(st.session_state["safety_note"])
        ev_entry = rec.explain_evidence_for(st.session_state.get("mood_state_used", mood_state))
        st.markdown(f"**Evidence-based direction applied:** {ev_entry['audio_characteristics']}")
        st.caption(f"Claim level: {ev_entry['claim_level']}")
        st.caption("See the Research Evidence page for full citations.")
        card_close()

        for i, s in enumerate(st.session_state["recs"]):
            card_open()
            st.markdown(f"**{i+1}. {s['song']} — {s['artist']}** ({s['genre']})")
            if s.get("research_source"):
                st.success("🔬 Evidence-supported Raga/music option — selected from the project's research-source pool.")
            with st.expander("💡 Why was this song recommended?", expanded=True):
                # Pull the corresponding scored row when available.
                row_match = st.session_state["pool"]
                row_match = row_match[row_match["song_id"] == s["song_id"]] if not row_match.empty else row_match
                row = row_match.iloc[0].to_dict() if len(row_match) else s
                st.write(_song_explanation(row, {
                    "mood_state": st.session_state.get("mood_state_used", mood_state),
                    "stress": stress, "genre_pref": genre_pref, "era_pref": era_pref
                }, is_research=s.get("research_source", False)))
                if s.get("research_source"):
                    st.caption("Research basis: see the Research Evidence page and the cited papers used for the project. The evidence supports the studied music/Raga intervention context; it does not mean the song is universally therapeutic.")
            rating = st.radio("Rate this song (1=dislike, 5=like)", [1, 2, 3, 4, 5], horizontal=True,
                               key=f"rate_{i}_{s['song_id']}")
            url = s.get("source_url") if s.get("research_source") else spotify_link(s["song"], s["artist"])
            st.markdown(f"[🎧 Open in Spotify]({url})")
            if s.get("research_source"):
                st.caption(f"Source: {s.get('source_name', 'Supplied Spotify source')}")
            flag_key = f"fb_done_{i}_{s['song_id']}"
            if flag_key not in st.session_state:
                st.session_state[flag_key] = False
            if st.button(f"Submit Feedback for Song {i+1}", key=f"fb_{i}_{s['song_id']}", disabled=st.session_state[flag_key]):
                st.session_state[flag_key] = True
                song_action = s["song_id"]
                reward = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}[rating]
                pool_row = st.session_state["pool"][st.session_state["pool"]["song_id"] == song_action]
                get = lambda c: float(pool_row[c].values[0]) if len(pool_row) and c in pool_row.columns else 0.0
                entry = {
                    "user": name, "song_id": song_action, "song": s["song"], "artist": s["artist"],
                    "rating": rating, "hrv": hrv, "stress": stress,
                    "session_number": st.session_state["session_number"],
                    "mood_state": st.session_state.get("mood_state_used", mood_state),
                    "rnn_score": get("rnn_score"), "ncf_score": get("ncf_score"),
                    "personal_q": get("personal_q"), "pref_bias": get("pref_bias"),
                    "physio_fit": get("physio_fit"), "psy_bias": get("psy_bias"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                db.recommendation_feedback.insert_one(entry)
                fdf = pd.read_csv(feedback_file) if os.path.exists(feedback_file) else pd.DataFrame(columns=entry.keys())
                fdf = pd.concat([fdf, pd.DataFrame([entry])], ignore_index=True)
                fdf.to_csv(feedback_file, index=False)

                # Source-only research anchors are not columns in the trained
                # dataset, so they are logged for feedback but are not used as
                # Q-table indices. Dataset-backed recommendations keep the
                # original RL/Q-table learning behaviour unchanged.
                if not s.get("research_source"):
                    cur_state = rec.get_user_state(st.session_state.get("mood_state_used", mood_state), stress, ctx["depression"])
                    rec.update_q(personal_q, cur_state, song_action, reward, cur_state)
                    rec.update_q(global_q, cur_state, song_action, reward, cur_state)
                    db.qtables.update_one({"user": name}, {"$set": {"qtable": personal_q.tolist()}}, upsert=True)
                    db.qtables.update_one({"user": "global"}, {"$set": {"qtable": global_q.tolist()}}, upsert=True)

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
            if st.button("Submit Session Feedback"):
                entry = {"user": name, "comfort": comfort, "satisfaction": satisfaction,
                         "mood_alignment": mood_alignment, "experience": experience,
                         "continue": continue_use, "mood_state": mood_state,
                         "stress": stress, "hrv": hrv,
                         "timestamp": datetime.now(timezone.utc).isoformat()}
                db.experiments.insert_one(entry)
                personal_file = os.path.join(QTABLE_DIR, f"{name}_session_feedback.csv")
                pdf_ = pd.read_csv(personal_file) if os.path.exists(personal_file) else pd.DataFrame(columns=entry.keys())
                pdf_ = pd.concat([pdf_, pd.DataFrame([entry])], ignore_index=True)
                pdf_.to_csv(personal_file, index=False)
                st.success("Thank you — recorded.")
                st.session_state.session_finished = False
            card_close()

# ================================================================
# PAGE: Evaluation & Validation
# ================================================================
elif page == "Evaluation & Validation":
    st.title("📊 Model & System Validation")
    safety_banner()
    name = st.session_state.username
    feedback_file = os.path.join(QTABLE_DIR, f"{name}_feedback.csv")
    session_file = os.path.join(QTABLE_DIR, f"{name}_session_feedback.csv")
    fdf = pd.read_csv(feedback_file) if os.path.exists(feedback_file) else None
    sdf = pd.read_csv(session_file) if os.path.exists(session_file) else None

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
        db.bias_assessments.insert_one({
            "user": st.session_state.username, "assessment": st.session_state["bias_assessment"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
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

    st.markdown("#### Full reference list")
    for key, r in ev.REFERENCES.items():
        st.markdown(f"- {r['citation']} (DOI: {r['doi']})")

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
    subjects = physio.list_available_wesad_subjects()
    if subjects:
        mode_badge(f"{len(subjects)} SUBJECT(S) AVAILABLE", "research")
    else:
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