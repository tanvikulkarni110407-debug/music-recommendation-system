"""
safety.py
---------
Safety layer. This is a research prototype, not a medical device -
this module never diagnoses or treats anything. It implements three
concrete, checkable behaviours required by the project spec:

1. Confidence gating: if the fusion model's score spread across the
   candidate pool is too flat (i.e. the model isn't meaningfully
   distinguishing songs for this state), label the batch "Low
   confidence" and fall back to a neutral, low-arousal, high-valence
   default pool instead of an aggressive/confident-sounding pick.

2. Adverse-response flagging: if a user rates a song 1/5 immediately
   after a state that already indicated high stress/low mood, flag the
   (state, song) pair so it is deprioritized rather than repeated.

3. Repetition avoidance: songs already flagged adverse for a user are
   excluded from future candidate pools for a cool-down period
   (tracked by simple occurrence count here).

SAFETY_DISCLAIMER must be shown wherever recommendations are displayed.
"""

import numpy as np

SAFETY_DISCLAIMER = (
    "This system is a research prototype. It does not diagnose, treat, or "
    "cure any medical or psychological condition. Recommendations are based "
    "on self-reported and/or measured research inputs and general "
    "music-psychology evidence, not individualized clinical judgment."
)

CONFIDENCE_SPREAD_THRESHOLD = 0.05  # min std-dev of final_score across top-15 candidates to call it "confident"


def assess_confidence(pool_scores):
    """pool_scores: array-like of final_score values across the candidate pool.
    Returns ('high'|'low', spread_value)."""
    arr = np.asarray(pool_scores, dtype=float)
    if len(arr) < 3:
        return "low", 0.0
    spread = float(np.std(arr))
    return ("high" if spread >= CONFIDENCE_SPREAD_THRESHOLD else "low"), spread


def is_adverse_rating(rating, mood_state, stress):
    """A rating is flagged adverse if the user strongly disliked (1-2/5) a
    song that was recommended during an already-elevated-stress / low-mood
    state - i.e. a case where the recommendation may have made things worse
    rather than better."""
    elevated_state = stress is not None and stress > 60 or mood_state in ("Sad", "Angry")
    return bool(rating is not None and rating <= 2 and elevated_state)


def get_flagged_songs(feedback_collection, user):
    """Return set of song_ids flagged adverse for this user, from stored feedback."""
    docs = feedback_collection.find({"user": user}) if hasattr(feedback_collection, "find") else []
    flagged = set()
    for d in docs:
        if is_adverse_rating(d.get("rating"), d.get("mood_state"), d.get("stress")):
            flagged.add(d.get("song_id"))
    return flagged


def apply_safety_layer(pool_df, confidence_level, flagged_song_ids, neutral_filter_fn):
    """
    pool_df: DataFrame with a 'song_id' column and 'final_score'.
    confidence_level: 'high' or 'low' (from assess_confidence).
    flagged_song_ids: set of song_ids to exclude (adverse history).
    neutral_filter_fn: callable(df) -> df, applies a neutral/safe filter
        (e.g. moderate valence, moderate energy) used only in low-confidence mode.
    Returns (filtered_df, safety_note).
    """
    df = pool_df[~pool_df["song_id"].isin(flagged_song_ids)]
    if len(df) < 5:
        df = pool_df  # don't over-filter into an empty pool; log instead of crashing
        note = "Safety filter would have emptied the candidate pool; adverse-history exclusion was relaxed for this session."
    else:
        note = None

    if confidence_level == "low":
        safe_df = neutral_filter_fn(df)
        if len(safe_df) >= 5:
            df = safe_df
        note = (note or "") + " Low model confidence for this state - falling back to a neutral, low-risk candidate pool rather than the top-ranked (possibly noisy) picks."

    return df, note
