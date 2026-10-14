"""
recommender.py
---------------
The core fusion engine. Preserves the seniors' filtering + scoring
pipeline (mood/HRV/stress audio-feature filters, preference bias,
psychology bias, RL Q-tables, RNN/NCF neural scores) and adds:
  - graceful fallback to content-based-only scoring if the trained
    RNN/NCF weights aren't present (see modules/models.py)
  - the safety layer (confidence gating + adverse-history exclusion)
  - an evidence tag per recommendation explaining which characteristic-
    level research (modules/evidence.py) motivated the direction of
    the filter that was applied
"""

import random
import numpy as np
import pandas as pd
import torch

from .safety import assess_confidence, apply_safety_layer, get_flagged_songs
from .evidence import CHARACTERISTIC_EVIDENCE_MAP

MOOD_AUDIO_TARGETS = {
    "Sad":       {"valence": (0.0, 0.35), "energy": (0.0, 0.40)},
    "Calm":      {"valence": (0.3, 0.60), "energy": (0.0, 0.35)},
    "Energetic": {"valence": (0.5, 1.00), "energy": (0.7, 1.00)},
    "Angry":     {"valence": (0.0, 0.40), "energy": (0.6, 1.00)},
    "Happy":     {"valence": (0.6, 1.00), "energy": (0.4, 0.80)},
}
MOOD_MAP = {"Sad": 0, "Angry": 1, "Energetic": 2, "Calm": 3, "Happy": 4}
FALLBACK_WEIGHTS = np.array([0.40, 0.28, 0.18, 0.09, 0.03, 0.02])


def mood_physiology_fit(row, mood_state, hrv, stress):
    score = 0.0
    target = MOOD_AUDIO_TARGETS.get(mood_state, {})
    for feature, (low, high) in target.items():
        val = row.get(feature)
        if val is not None and low <= val <= high:
            score += 1.0
    energy = row.get("energy")
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


def preference_bias(row, genre_pref, era_pref):
    score = 0.0
    g = str(row.get("genre", "")).lower()
    y = row.get("year")
    if genre_pref.lower() in g:
        score += 0.3
    if era_pref == "Energetic songs" and any(k in g for k in ["dance", "upbeat", "pop"]):
        score += 0.2
    if era_pref == "calming songs" and any(k in g for k in ["soft", "slow", "instrumental"]):
        score += 0.2
    if era_pref == "Classical songs" and any(k in g for k in ["raga", "classical", "hindustani", "carnatic"]):
        score += 0.2
    if y is not None and not pd.isna(y):
        if era_pref == "60s songs" and 1960 <= y <= 1969:
            score += 0.25
        elif era_pref == "90s songs" and 1990 <= y <= 1999:
            score += 0.25
    return score


def psychology_bias(row, mood_state, extraversion, openness, depression, physical_qol, social_qol):
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
    state = {"Sad": 10, "Angry": 20, "Energetic": 30, "Calm": 40, "Happy": 50}.get(mood_state, 50)
    if stress > 70:
        state += 10
    elif stress > 40:
        state += 5
    if depression > 20:
        state += 5
    return min(state, 99)


def safe_filter(base_pool, condition, min_size=5):
    filtered = base_pool[condition]
    return filtered if len(filtered) >= min_size else base_pool


def neutral_safe_pool(df):
    """Used only when the safety layer detects low model confidence -
    a moderate valence / moderate-low energy pool, i.e. a conservative
    default rather than trusting a noisy top-ranked pick."""
    return df[(df["valence"] > 0.4) & (df["valence"] < 0.75) & (df["energy"] < 0.6)]


def safe_normalize(col):
    min_v, max_v = col.min(), col.max()
    if max_v - min_v < 1e-6:
        return np.zeros_like(col) + 0.5
    return (col - min_v) / (max_v - min_v)


def build_candidate_pool(df, mood_state, hrv, stress, genre_pref, era_pref,
                          depression, anxiety, extraversion, physical_qol, social_qol):
    pool = df.copy()

    if era_pref == "Classical songs":
        classical_pool = pool[pool["genre"].str.contains(
            "raga|classical|hindustani|carnatic|traditional", case=False, na=False)]
        if len(classical_pool) >= 1:
            pool = classical_pool
    else:
        if era_pref == "60s songs":
            era_pool = pool[(pool["year"] >= 1960) & (pool["year"] <= 1969)]
            if len(era_pool) >= 5:
                pool = era_pool
        elif era_pref == "90s songs":
            era_pool = pool[(pool["year"] >= 1990) & (pool["year"] <= 1999)]
            if len(era_pool) >= 5:
                pool = era_pool
        genre_pool = pool[pool["genre"].str.contains(genre_pref.lower(), case=False, na=False)]
        if len(genre_pool) >= 3:
            pool = genre_pool
        if era_pref == "Energetic songs":
            vibe_pool = pool[(pool["energy"] > 0.60) & (pool["valence"] > 0.45)]
            if len(vibe_pool) >= 3:
                pool = vibe_pool
        elif era_pref == "calming songs":
            vibe_pool = pool[(pool["energy"] < 0.50) & (pool["valence"] > 0.30)]
            if len(vibe_pool) >= 3:
                pool = vibe_pool

    if mood_state == "Sad":
        pool = safe_filter(pool, (pool["energy"] < 0.50) & (pool["valence"] < 0.55), min_size=3)
    elif mood_state == "Happy":
        pool = safe_filter(pool, (pool["valence"] > 0.50) & (pool["energy"] > 0.45), min_size=3)
    elif mood_state == "Angry":
        pool = safe_filter(pool, (pool["energy"] > 0.50) & (pool["valence"] < 0.60), min_size=3)
    elif mood_state == "Calm":
        pool = safe_filter(pool, (pool["energy"] < 0.55) & (pool["valence"] > 0.30), min_size=3)
    elif mood_state == "Energetic":
        pool = safe_filter(pool, (pool["energy"] > 0.55) & (pool["valence"] > 0.45), min_size=3)

    if hrv > 100:
        pool = safe_filter(pool, pool["energy"] < 0.55, min_size=3)
    elif hrv < 50:
        pool = safe_filter(pool, pool["energy"] > 0.35, min_size=3)
    if stress > 70:
        pool = safe_filter(pool, (pool["energy"] < 0.55) & (pool["valence"] > 0.25), min_size=3)
    if depression >= 20:
        pool = safe_filter(pool, pool["valence"] > 0.30, min_size=3)
    if anxiety >= 16:
        pool = safe_filter(pool, pool["energy"] < 0.65, min_size=3)
    if extraversion > 5:
        pool = safe_filter(pool, pool["energy"] > 0.40, min_size=3)
    if physical_qol < 25:
        pool = safe_filter(pool, pool["energy"] < 0.60, min_size=3)
    if social_qol < 25:
        pool = safe_filter(pool, pool["valence"] > 0.35, min_size=3)

    return pool.copy()


def score_pool(pool, ctx, personal_q, global_q, rnn_ncf_state, weights_getter, num_songs):
    """
    ctx: dict with mood_state, hrv, stress, genre_pref, era_pref,
         extraversion, openness, depression, anxiety, physical_qol,
         social_qol, tipi_n, whoql_n, dass_n, mood_n, user_name
    rnn_ncf_state: dict with metadata, rnn_model, ncf_model (any may be None)
    Returns scored pool (DataFrame) and a note on which scoring mode ran.
    """
    mood_state, hrv, stress = ctx["mood_state"], ctx["hrv"], ctx["stress"]
    state = get_user_state(mood_state, stress, ctx["depression"])

    pool["physio_fit"] = pool.apply(lambda r: mood_physiology_fit(r, mood_state, hrv, stress), axis=1)
    pool["pref_bias"] = pool.apply(lambda r: preference_bias(r, ctx["genre_pref"], ctx["era_pref"]), axis=1)
    pool["psy_bias"] = pool.apply(lambda r: psychology_bias(
        r, mood_state, ctx["extraversion"], ctx["openness"], ctx["depression"],
        ctx["physical_qol"], ctx["social_qol"]), axis=1)
    pool["personal_q"] = pool["song_id"].apply(lambda a: personal_q[state, a % personal_q.shape[1]])
    pool["global_q"] = pool["song_id"].apply(lambda a: global_q[state, a % global_q.shape[1]])

    metadata, rnn_model, ncf_model = rnn_ncf_state["metadata"], rnn_ncf_state["rnn_model"], rnn_ncf_state["ncf_model"]

    if rnn_model is not None and ncf_model is not None:
        mode_note = "Full fusion mode: RNN + NCF + RL + preference + physiology + psychology."
        num_songs_trained = metadata["num_songs"]
        num_genres_trained = metadata["num_genres"]
        num_vibes_trained = metadata["num_vibes"]
        num_users_trained = metadata["num_users"]

        SEQ_LEN = 10
        shared_seq = torch.tensor(
            [random.sample(range(num_songs_trained), min(SEQ_LEN, num_songs_trained))], dtype=torch.long)
        context = torch.tensor([[ctx["mood_n"], stress / 100.0, (hrv - 20) / 180.0,
                                  ctx["tipi_n"], ctx["whoql_n"], ctx["dass_n"]]], dtype=torch.float32)
        with torch.no_grad():
            logits = rnn_model(shared_seq, torch.tensor([0]), torch.tensor([0]), context)
            probs = torch.softmax(logits, dim=1).squeeze()
        pool["rnn_score"] = pool["song_id"].apply(lambda i: probs[i % num_songs_trained].item())

        user_hash = hash(ctx["user_name"]) % num_users_trained
        genre_map = metadata.get("genre_mapping", {})
        vibe_map = metadata.get("vibe_mapping", {})

        def ncf_score(row):
            genre_id = max(0, min(int(genre_map.get(row.get("genre", "Unknown"), 0)), num_genres_trained - 1))
            vibe_id = max(0, min(int(vibe_map.get(row.get("vibe", "Neutral"), 0)), num_vibes_trained - 1))
            song_id_safe = int(row["song_id"]) % num_songs_trained
            with torch.no_grad():
                score = ncf_model(
                    torch.tensor([user_hash]), torch.tensor([song_id_safe]),
                    torch.tensor([genre_id]), torch.tensor([vibe_id]),
                    torch.tensor([MOOD_MAP[mood_state]]),
                    torch.tensor([[stress / 100.0]], dtype=torch.float32),
                    torch.tensor([[(hrv - 20) / 180.0]], dtype=torch.float32),
                    torch.tensor([[ctx["tipi_n"]]], dtype=torch.float32),
                    torch.tensor([[ctx["dass_n"]]], dtype=torch.float32),
                    torch.tensor([[ctx["whoql_n"]]], dtype=torch.float32),
                    torch.tensor([[0.5]], dtype=torch.float32),
                )
            return score.item()
        pool["ncf_score"] = pool.apply(ncf_score, axis=1)
    else:
        mode_note = ("CONTENT-BASED FALLBACK MODE: trained RNN/NCF weights not found - "
                      "recommendations use preference/physiology/psychology/RL scores only, "
                      "not the neural models. See Dataset/Research Mode page for setup.")
        pool["rnn_score"] = 0.5
        pool["ncf_score"] = 0.5

    weights = weights_getter()
    genre_counts = pool["genre"].value_counts()
    pool["diversity_penalty"] = pool["genre"].map(lambda g: np.log1p(genre_counts.get(g, 1))) * 0.01
    pool["exploration_bonus"] = 0.05  # simplified: exploration handled by epsilon sampling below

    cols = ["personal_q", "global_q", "psy_bias", "physio_fit", "pref_bias", "rnn_score", "ncf_score"]
    for c in cols:
        pool[c] = safe_normalize(pool[c])

    pool["final_score"] = (
        weights[0] * pool["rnn_score"] + weights[1] * pool["ncf_score"] +
        weights[2] * pool["personal_q"] + weights[3] * pool["pref_bias"] +
        weights[4] * pool["physio_fit"] + weights[5] * pool["psy_bias"] +
        0.05 * pool["exploration_bonus"] - pool["diversity_penalty"]
    )
    return pool, mode_note


def select_recommendations(pool, n=5, epsilon=0.15):
    pool_sorted = pool.sort_values("final_score", ascending=False)
    top_candidates = pool_sorted.head(15)
    n_pick = min(n, len(top_candidates))
    if n_pick == 0:
        return pool_sorted.head(0)
    if random.random() < epsilon:
        chosen = top_candidates.sample(n_pick, replace=False)
    else:
        chosen = top_candidates.head(n_pick)
    if len(chosen) < n:
        remaining = pool[~pool["song_id"].isin(chosen["song_id"])]
        extras = remaining.nlargest(n - len(chosen), "final_score")
        chosen = pd.concat([chosen, extras], ignore_index=True)
    return chosen


def update_q(q, s, a, r, ns, alpha=0.1, gamma=0.9):
    a = a % q.shape[1]
    ns_col = min(ns, q.shape[0] - 1)
    q[s, a] += alpha * (r + gamma * np.max(q[ns_col]) - q[s, a])


def explain_evidence_for(mood_state):
    """Returns the evidence-map entry most relevant to this mood target,
    for the Explainable-AI panel."""
    if mood_state in ("Sad", "Calm"):
        return CHARACTERISTIC_EVIDENCE_MAP["Calm / low-arousal target"]
    if mood_state in ("Energetic", "Angry"):
        return CHARACTERISTIC_EVIDENCE_MAP["High-arousal / energetic target"]
    return CHARACTERISTIC_EVIDENCE_MAP["Positive valence target"]
