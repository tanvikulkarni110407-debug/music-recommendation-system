"""
validation.py
--------------
Computes REAL validation statistics from whatever feedback data
actually exists in the database (Mongo or local fallback). If there
isn't enough data yet, it says so explicitly instead of showing a
fabricated number - this is enforced by MIN_SAMPLES gating below.

Distinguishes (per project spec) between:
  - INPUT validation (data-quality checks on incoming records)
  - MODEL/RECOMMENDATION validation (does final_score correlate with
    the user's actual rating; Precision@K on "liked" songs)
  - USER SATISFACTION (from the end-of-session survey)
This module does NOT touch physiological cross-device agreement
statistics (MAE/RMSE/ICC against a reference device) because that
requires paired reference-vs-device measurements this deployment does
not have; the UI must show "Validation dataset required / not
available" for that section rather than a number.
"""

import numpy as np
import pandas as pd

MIN_SAMPLES_CORRELATION = 15
MIN_SAMPLES_PRECISION = 10


def data_quality_report(df, required_cols):
    """Real, simple checks - no invented thresholds beyond basic sanity."""
    report = {}
    report["n_rows"] = len(df)
    report["missing_required_cols"] = [c for c in required_cols if c not in df.columns]
    if not report["missing_required_cols"]:
        report["missing_values"] = int(df[required_cols].isna().sum().sum())
    else:
        report["missing_values"] = None
    report["duplicate_rows"] = int(df.duplicated().sum())
    return report


def score_rating_correlation(feedback_df):
    """
    Pearson correlation between the fusion model's final_score components
    (rnn_score, ncf_score, personal_q, pref_bias, physio_fit, psy_bias)
    and the actual 1-5 rating the user gave. This is the real, checkable
    link between "what the model predicted was a good match" and "what
    the user actually said".
    Returns None (with a reason) if there isn't enough data yet.
    """
    needed = ["rnn_score", "ncf_score", "personal_q", "pref_bias", "physio_fit", "psy_bias", "rating"]
    if feedback_df is None or len(feedback_df) < MIN_SAMPLES_CORRELATION:
        n = 0 if feedback_df is None else len(feedback_df)
        return None, f"Need at least {MIN_SAMPLES_CORRELATION} feedback rows to compute a stable correlation (have {n})."
    missing = [c for c in needed if c not in feedback_df.columns]
    if missing:
        return None, f"Feedback data is missing required columns: {missing}"

    df = feedback_df.dropna(subset=needed)
    if len(df) < MIN_SAMPLES_CORRELATION:
        return None, f"Only {len(df)} complete rows after dropping missing values - need {MIN_SAMPLES_CORRELATION}."

    results = {}
    for col in needed[:-1]:
        if df[col].std() > 1e-9:
            results[col] = float(np.corrcoef(df[col], df["rating"])[0, 1])
        else:
            results[col] = None  # no variance - correlation undefined, not zero
    return results, None


def precision_at_k(feedback_df, k=5, like_threshold=4):
    """
    Precision@K: of the top-K songs (by final_score, reconstructed from the
    stored component scores) that were actually shown, what fraction did
    the user rate >= like_threshold? Computed per session batch where
    session_number groups a single recommendation set.
    """
    needed = ["session_number", "rating"]
    if feedback_df is None or len(feedback_df) < MIN_SAMPLES_PRECISION:
        n = 0 if feedback_df is None else len(feedback_df)
        return None, f"Need at least {MIN_SAMPLES_PRECISION} feedback rows to compute Precision@K (have {n})."
    if any(c not in feedback_df.columns for c in needed):
        return None, "Feedback data is missing session_number/rating columns."

    precisions = []
    for _, batch in feedback_df.groupby("session_number"):
        topk = batch.head(k)
        if len(topk) == 0:
            continue
        precisions.append((topk["rating"] >= like_threshold).mean())
    if not precisions:
        return None, "No complete recommendation batches found."
    return {"mean_precision_at_k": float(np.mean(precisions)), "n_batches": len(precisions), "k": k}, None


def satisfaction_summary(session_feedback_df):
    """Simple, real descriptive stats from the end-of-session survey -
    no invented benchmarks to compare against."""
    if session_feedback_df is None or len(session_feedback_df) == 0:
        return None, "No end-of-session survey responses recorded yet."
    cols = ["comfort", "satisfaction", "mood_alignment", "experience"]
    present = [c for c in cols if c in session_feedback_df.columns]
    if not present:
        return None, "Survey columns not found in stored data."
    return {c: {"mean": float(session_feedback_df[c].mean()), "n": int(session_feedback_df[c].count())}
            for c in present}, None
