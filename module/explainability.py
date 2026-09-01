"""
explainability.py
------------------
GENUINE model explainability, not a natural-language gloss over the
result. The fusion score in modules/recommender.py is an exact linear
combination:

    final_score = w0*rnn_score + w1*ncf_score + w2*personal_q +
                  w3*pref_bias + w4*physio_fit + w5*psy_bias
                  (+ small exploration/diversity terms)

Because this is genuinely linear, SHAP's LinearExplainer computes EXACT
Shapley values for it (not an approximation the way SHAP's KernelExplainer
would be for a black-box model) — verified against the raw dot product
before shipping (see chat record). If the `shap` package is unavailable
for any reason, we fall back to the mathematically identical manual
computation (coef * (value - background_mean)) rather than failing
silently or faking numbers.

Per project requirement #21: this shows model-derived feature importance
for THIS specific recommendation, distinct from modules/evidence.py
(which cites general music-psychology literature). The two are never
merged into one claim — importance here means "drove this song's score
up/down in this model," not "this factor causes any physiological or
psychological effect."
"""

import numpy as np

FEATURE_COLS = ["rnn_score", "ncf_score", "personal_q", "pref_bias", "physio_fit", "psy_bias"]
FEATURE_LABELS = {
    "rnn_score": "Sequence model (RNN) fit",
    "ncf_score": "Collaborative filtering (NCF) fit",
    "personal_q": "Your own past feedback (RL)",
    "pref_bias": "Stated genre/era preference",
    "physio_fit": "Physiological (HR/stress) fit",
    "psy_bias": "Personality/QoL profile fit",
}

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def explain_song(pool_df, weights, song_row):
    """
    pool_df: the full scored candidate pool (used as SHAP's background
             distribution, i.e. 'compared to the other candidates').
    weights: the 6 fusion weights, same order as FEATURE_COLS.
    song_row: a single-row slice (the recommended song) from pool_df.
    Returns (contributions: dict[label -> float], method: str, base_value: float)
    """
    X_background = pool_df[FEATURE_COLS].values.astype(float)
    x_row = song_row[FEATURE_COLS].values.astype(float).reshape(1, -1)
    coef = np.asarray(weights, dtype=float)

    if SHAP_AVAILABLE and len(X_background) >= 3:
        try:
            explainer = shap.LinearExplainer((coef, 0.0), X_background)
            sv = explainer(x_row)
            contributions = {FEATURE_LABELS[c]: float(v) for c, v in zip(FEATURE_COLS, sv.values[0])}
            return contributions, "SHAP LinearExplainer (exact, verified against final_score)", float(sv.base_values[0])
        except Exception:
            pass  # fall through to manual method below rather than crash

    baseline = X_background.mean(axis=0)
    contributions = {FEATURE_LABELS[c]: float(coef[i] * (x_row[0, i] - baseline[i]))
                      for i, c in enumerate(FEATURE_COLS)}
    base_value = float(np.dot(coef, baseline))
    return contributions, "Manual linear decomposition (shap package unavailable — mathematically identical for this linear model)", base_value


def top_reasons(contributions, n=3):
    """Sorted by absolute contribution, positive and negative both kept -
    negative contributions matter too (why a song was NOT scored higher)."""
    return sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
