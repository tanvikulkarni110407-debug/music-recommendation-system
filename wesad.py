"""
Run this ON THE LAPTOP THAT HOLDS WESAD (not on Streamlit Cloud).

  python extract_wesad_hrv.py --wesad_dir "C:\\path\\to\\WESAD"

Pipeline (Schmidt et al., 2018, ACM ICMI, "Introducing WESAD"):
  chest ECG (RespiBAN, 700 Hz) -> R-peaks -> RR intervals -> HRV features
  per labelled condition: 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation.

Output: data/wesad_hrv_features.csv  (a few KB - this is what you commit to GitHub).
Columns used by the app: subject, condition, duration_s, mean_hr_bpm, rmssd_ms, sdnn_ms, pnn50_pct
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd

FS = 700
LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
MIN_SECONDS = 60          # skip very short segments
RR_MIN_MS, RR_MAX_MS = 300.0, 2000.0

try:                       # use your project's own detector when available
    from modules import physiological as physio
except Exception:
    physio = None


def contiguous_runs(mask):
    """Return (start, end) index pairs of consecutive True values."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def rr_fallback(ecg, fs):
    """Simple band-pass + squared-signal R-peak detector (used only if physio fails)."""
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    sq = filtfilt(b, a, ecg) ** 2
    peaks, _ = find_peaks(sq, distance=int(0.3 * fs), height=0.3 * np.percentile(sq, 99))
    return np.diff(peaks) / fs * 1000.0


def get_rr_ms(segment):
    rr = None
    if physio is not None:
        try:
            rr = physio.wesad_ecg_to_rr(segment, fs=FS)
        except Exception:
            rr = None
    if rr is None or len(rr) < 10:
        rr = rr_fallback(segment, FS)
    rr = np.asarray(rr, dtype=float)
    if rr.size and np.nanmedian(rr) < 10:          # seconds -> milliseconds
        rr = rr * 1000.0
    return rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]


def hrv_from_rr(rr):
    d = np.diff(rr)
    return {
        "mean_hr_bpm": float(60000.0 / np.mean(rr)),
        "sdnn_ms": float(np.std(rr, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(d ** 2))),
        "pnn50_pct": float(100.0 * np.mean(np.abs(d) > 50.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_dir", required=True, help="folder that contains S2, S3, ... S17")
    ap.add_argument("--out", default=os.path.join("data", "wesad_hrv_features.csv"))
    args = ap.parse_args()

    rows = []
    for sdir in sorted(os.listdir(args.wesad_dir)):
        pkl = os.path.join(args.wesad_dir, sdir, f"{sdir}.pkl")
        if not os.path.exists(pkl):
            continue
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            ecg = np.asarray(data["signal"]["chest"]["ECG"], dtype=float).ravel()
            lab = np.asarray(data["label"]).ravel()
        except Exception as e:
            print(f"[skip] {sdir}: cannot read ({e})")
            continue
        n = min(len(ecg), len(lab))
        ecg, lab = ecg[:n], lab[:n]

        for code, cname in LABELS.items():
            rr_all, dur = [], 0.0
            for s, e in contiguous_runs(lab == code):
                if (e - s) < FS * 20:                  # ignore runs shorter than 20 s
                    continue
                rr = get_rr_ms(ecg[s:e])
                if rr.size >= 10:
                    rr_all.append(rr)
                    dur += (e - s) / FS
            if dur < MIN_SECONDS or not rr_all:
                print(f"[skip] {sdir} {cname}: not enough data")
                continue
            rr_cat = np.concatenate(rr_all)
            row = {"subject": sdir, "condition": cname, "duration_s": round(dur, 1), "n_rr": int(rr_cat.size)}
            row.update({k: round(v, 3) for k, v in hrv_from_rr(rr_cat).items()})
            rows.append(row)
            print(f"[ok]   {sdir} {cname}: HR {row['mean_hr_bpm']:.0f} bpm, RMSSD {row['rmssd_ms']:.1f} ms")

    if not rows:
        raise SystemExit("No features extracted - check --wesad_dir (expects S2/S2.pkl, S3/S3.pkl, ...).")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\nSaved {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()