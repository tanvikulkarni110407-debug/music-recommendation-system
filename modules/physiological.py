"""
physiological.py
-----------------
Physiological-input layer. Deliberately keeps two paths SEPARATE and
clearly labeled, per project safety requirements:

1. SELF-REPORT path (current-mood sliders for perceived HR / stress) -
   what the original app used. This is NOT a physiological measurement;
   it's a subjective input, and the UI must label it as such.

2. MEASURED path - real HRV time-domain features computed from an
   uploaded RR-interval (IBI) series, e.g. exported from a chest strap,
   PPG app, or the WESAD dataset. This is only computed when a real
   RR-interval series is provided; it is never fabricated.

HRV feature definitions (standard time-domain metrics; see:
  Shaffer, F., & Ginsberg, J. P. (2017). "An Overview of Heart Rate
  Variability Metrics and Norms." Frontiers in Public Health, 5, 258.
  https://doi.org/10.3389/fpubh.2017.00258
for definitions, required signal, and interpretation caveats.)

WESAD (Schmidt et al., 2018, ICMI) loader:
  WESAD is NOT auto-downloadable here (requires registration with the
  original authors: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/).
  This module provides the correct loader for its pickle-per-subject
  format IF you place the files under data/wesad/S<id>/S<id>.pkl. If
  the files are absent, the loader returns None and the UI clearly
  shows "WESAD not found - Research Mode disabled for this feature".
"""

import os
import numpy as np
import pandas as pd
from .config import WESAD_DIR

HRV_METADATA = {
    "mean_rr": {
        "name": "Mean RR interval", "unit": "ms",
        "meaning": "Average time between successive heartbeats.",
        "requires": "A clean RR-interval (IBI) series.",
    },
    "sdnn": {
        "name": "SDNN", "unit": "ms",
        "meaning": "Standard deviation of RR intervals; reflects overall (short + long term) heart-rate variability.",
        "requires": "At least ~5 minutes of continuous RR intervals for standard interpretation (shorter windows are reported but less comparable to norms).",
    },
    "rmssd": {
        "name": "RMSSD", "unit": "ms",
        "meaning": "Root mean square of successive RR-interval differences; primarily reflects parasympathetic (vagal) activity.",
        "requires": "RR-interval series; robust even over shorter windows (~1-5 min) relative to SDNN.",
    },
    "pnn50": {
        "name": "pNN50", "unit": "%",
        "meaning": "Percentage of successive RR-interval differences greater than 50ms; related to parasympathetic activity.",
        "requires": "RR-interval series.",
    },
    "heart_rate": {
        "name": "Mean Heart Rate", "unit": "bpm",
        "meaning": "60000 / mean_rr(ms).",
        "requires": "RR-interval series.",
    },
}
HRV_REFERENCE = "Shaffer & Ginsberg (2017), Frontiers in Public Health, 5:258. doi:10.3389/fpubh.2017.00258"


def compute_hrv_features(rr_intervals_ms):
    """
    rr_intervals_ms: array-like of RR intervals in milliseconds.
    Returns a dict of real, computed HRV features, or None if the
    series is too short/invalid to compute anything meaningful.
    NEVER fabricates values - insufficient data returns None with a reason.
    """
    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr = rr[(rr > 250) & (rr < 2500)]  # physiologically plausible RR range (24-240 bpm)
    if len(rr) < 10:
        return None, "Fewer than 10 valid RR intervals after cleaning - cannot compute stable HRV features."

    diffs = np.diff(rr)
    mean_rr = float(np.mean(rr))
    sdnn = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnn50 = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100)
    heart_rate = float(60000.0 / mean_rr)

    quality_note = None
    if len(rr) < 300:
        quality_note = (f"Only {len(rr)} beats available (~{len(rr)*mean_rr/60000:.1f} min of data). "
                         "SDNN in particular is less comparable to population norms below ~5 minutes; "
                         "treat as a within-subject relative measure, not an absolute clinical value.")

    return {
        "mean_rr": round(mean_rr, 1),
        "sdnn": round(sdnn, 1),
        "rmssd": round(rmssd, 1),
        "pnn50": round(pnn50, 1),
        "heart_rate": round(heart_rate, 1),
        "n_beats": int(len(rr)),
    }, quality_note


def wesad_subject_available(subject_id):
    path = os.path.join(WESAD_DIR, f"S{subject_id}", f"S{subject_id}.pkl")
    return os.path.exists(path)


def list_available_wesad_subjects():
    if not os.path.isdir(WESAD_DIR):
        return []
    found = []
    for entry in sorted(os.listdir(WESAD_DIR)):
        if entry.startswith("S") and wesad_subject_available(entry[1:]):
            found.append(entry[1:])
    return found


def load_wesad_subject(subject_id):
    """
    Loads one WESAD subject's pickle file (official format: a dict with
    'signal' -> {'chest':{...}, 'wrist':{...}} and 'label' arrays at 700Hz
    for chest, per Schmidt et al. 2018 documentation).
    Returns the raw dict, or (None, reason) if the file is missing.
    Requires the `pandas`+`numpy` stack only; does not require any
    WESAD-specific library.
    """
    path = os.path.join(WESAD_DIR, f"S{subject_id}", f"S{subject_id}.pkl")
    if not os.path.exists(path):
        return None, (f"WESAD file not found at {path}. Download WESAD from the official source "
                       "(registration required) and place it at this path to enable Research Mode "
                       "for this subject.")
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data, None


def wesad_ecg_to_rr(ecg_signal, fs=700):
    """
    Very simple Pan-Tompkins-style R-peak detector for WESAD's chest ECG
    (700 Hz), sufficient for demonstrating the raw-signal -> RR-interval
    -> HRV-feature pipeline described in the project architecture.
    For a submitted research analysis, a validated peak detector
    (e.g. NeuroKit2 or BioSPPy) should be used instead of this minimal
    implementation - documented here as a known limitation.
    """
    from scipy.signal import butter, filtfilt, find_peaks
    ecg = np.asarray(ecg_signal, dtype=float).flatten()
    b, a = butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    filtered = filtfilt(b, a, ecg)
    squared = filtered ** 2
    peaks, _ = find_peaks(squared, distance=int(0.3 * fs), height=np.percentile(squared, 90))
    if len(peaks) < 2:
        return None
    rr_ms = np.diff(peaks) / fs * 1000.0
    return rr_ms
