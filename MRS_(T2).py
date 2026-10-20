"""
MuSync - research prototype (Streamlit + MongoDB Atlas)

REQUIRED STREAMLIT SECRETS  (App -> Settings -> Secrets).  NOTHING secret lives in this file.
-----------------------------------------------------------------------------------------
MONGODB_URI        = "mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority"
MONGODB_DATABASE   = "musync"
ADMIN_PASSWORD     = "choose-a-strong-password"

requirements.txt must contain (besides what you already have):
    pymongo[srv]  dnspython  certifi  openpyxl  pyserial
    scikit-learn  scipy  matplotlib        (SoundSync pulse/graph engine)
    librosa                                (optional - only to decode mp3/flac/ogg uploads)
"""
import os
import re
import io
import json
import time
import uuid
import hmac
import ssl
import hashlib
import secrets
import random
import urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
import certifi                                              # TLS fix for MongoDB Atlas
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Safe import for serial communication with NodeMCU/ESP32
try:
    import serial
except ImportError:
    serial = None


from modules.config import APP_NAME, APP_TAGLINE, QTABLE_DIR
from modules.theme import inject_theme, mode_badge, card_open, card_close, COLORS
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError, DuplicateKeyError
from modules import psychology as psy
from modules import physiological as physio
from modules import dataset as ds
from modules import models as mdl
from modules import recommender as rec
from modules import safety as saf
from modules import bias as bias_mod
from modules import validation as val
from modules import evidence as ev

st.set_page_config(page_title=APP_NAME, layout="wide", initial_sidebar_state="collapsed")
inject_theme()

# --------------------------------------------------------------
# NodeMCU / ESP32 Therapeutic Pulse Generator
# --------------------------------------------------------------
class NodeMCUPulseGenerator:
    """Maps track tempo + the automatically calculated stress score to a pulse frequency (20-150 Hz) and
    optionally streams it to a NodeMCU / ESP32 over a serial port.

    * calculate_*  : pure maths, no I/O - safe to call on every Streamlit rerun.
    * send_pulse   : performs serial I/O - call ONLY from an explicit user action.
    Falls back to simulation if pyserial is missing or the board is unreachable.
    """
    MIN_FREQ_HZ = 20
    MAX_FREQ_HZ = 150
    MIN_BPM = 60.0
    MAX_BPM = 180.0
    HIGH_STRESS_THRESHOLD = 60      # stress > 60/100 -> grounding scale
    HIGH_STRESS_SCALE = 0.7

    def __init__(self, port="COM3", baudrate=115200):
        self.port = port
        self.baudrate = int(baudrate)
        self.max_freq = self.MAX_FREQ_HZ
        self.min_freq = self.MIN_FREQ_HZ

    @property
    def serial_available(self):
        return serial is not None

    def configure(self, port=None, baudrate=None):
        if port:
            self.port = str(port).strip()
        if baudrate:
            self.baudrate = int(baudrate)

    @staticmethod
    def _to_float(value, default):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float(default)
        return v if np.isfinite(v) else float(default)

    def calculate_safe_frequency(self, bpm, stress_level):
        """BPM 60-180 -> 20-150 Hz (linear); stress > 60 -> x0.7; result clamped to [20, 150]."""
        bpm_val = self._to_float(bpm, 90.0)
        stress_val = self._to_float(stress_level, 40.0)      # note: 0 is a valid stress value
        bpm_val = min(max(bpm_val, self.MIN_BPM), self.MAX_BPM)
        span = self.max_freq - self.min_freq
        raw = self.min_freq + ((bpm_val - self.MIN_BPM) / (self.MAX_BPM - self.MIN_BPM)) * span
        if stress_val > self.HIGH_STRESS_THRESHOLD:
            raw *= self.HIGH_STRESS_SCALE
        return int(max(self.min_freq, min(self.max_freq, round(raw))))

    @staticmethod
    def period_ms(freq_hz):
        """Period (ms) = 1000 / Frequency (Hz)."""
        return round(1000.0 / freq_hz, 2) if freq_hz and freq_hz > 0 else 0.0

    def calculate_pulse(self, bpm, stress_level):
        """Return (freq_hz, period_ms) without touching any hardware."""
        freq = self.calculate_safe_frequency(bpm, stress_level)
        return freq, self.period_ms(freq)

    def send_pulse(self, bpm, stress_level):
        """Compute and transmit f"{freq}\\n". Returns (freq, period_ms, mode, status_text)
        where mode is 'hardware' or 'simulation'."""
        freq, period = self.calculate_pulse(bpm, stress_level)
        if serial is None:
            return freq, period, "simulation", (
                f"🧪 Safe simulation mode: {freq} Hz computed, nothing sent (pyserial not installed).")
        try:
            with serial.Serial(self.port, self.baudrate, timeout=1) as ser:
                time.sleep(0.1)
                ser.write(f"{freq}\n".encode("utf-8"))
                ser.flush()
            return freq, period, "hardware", (
                f"✅ Dispatched to NodeMCU/ESP32 on {self.port} @ {self.baudrate} baud: {freq} Hz.")
        except Exception as exc:                              # port busy / missing / permission
            return freq, period, "simulation", (
                f"🧪 Safe simulation mode: {freq} Hz computed (hardware unreachable on {self.port}: {exc}).")


pulse_hardware = NodeMCUPulseGenerator(port="COM3", baudrate=115200)

# --------------------------------------------------------------
# SoundSync music-feature-driven electrical-stimulation ("pulse") research engine
# Ported from the SoundSync notebook code (Soundsync_Code.docx):
#   dataset generator -> RF/GB models -> safety layers L1-L6 -> cascade -> 12 graphs + 3-zoom viewer
# Software signal generation only. Nothing here touches hardware.
# --------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from scipy import signal as sp_signal
    from scipy.io import wavfile as sp_wavfile
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    ES_AVAILABLE, ES_IMPORT_ERROR = True, None
except Exception as _es_exc:                                  # missing scipy / sklearn / matplotlib
    ES_AVAILABLE, ES_IMPORT_ERROR = False, str(_es_exc)

import tempfile

ES_SR = 44100            # sample rate of the generated signals
ES_CHUNK = 2048          # analysis chunk (~46 ms)
ES_BASE_SECONDS = 30     # ML runs on this base clip only, then it is cascaded (looped) to full length
ES_FEATURE_NAMES = ["RMS", "SpectralCentroid", "LowRatio", "MidRatio", "HighRatio",
                    "DominantFreq", "MaxAmp", "EnergyRatio", "ConsecOnes", "CumCharge"]


class MLMusicToSafeStimulation:
    """Audio-feature-driven FM and stimulation-like research signals with conservative software guards."""
    SAFETY_MAX_AMP = 0.10
    SAFETY_MIN_AMP = 0.01
    SAFETY_MAX_FREQ = 150.0   # strict medical ceiling (Hz) - matches NodeMCUPulseGenerator.MAX_FREQ_HZ
    SAFETY_MIN_FREQ = 10.0
    SAFETY_MAX_CONSEC = 5
    SAFETY_CHARGE_LIM = 0.045
    BIN_THRESHOLD = 0.55
    DESAT_CHUNKS = 4

    def __init__(self, models, sample_rate=ES_SR, fm_f0=80, fm_f1=150, fm_max_amp=0.5, chunk_size=ES_CHUNK):
        self.m = models
        self.sr = sample_rate
        self.chunk = chunk_size
        self.fm_f0, self.fm_f1, self.fm_max_amp = fm_f0, fm_f1, fm_max_amp

    def _chunk_features(self, chunk):
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        max_amp = float(np.max(np.abs(chunk)) + 1e-9)
        if len(chunk) > 10:
            mag = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1 / self.sr)
            total = np.sum(mag) + 1e-9
            sc = float(np.sum(freqs * mag) / total)
            dom = float(freqs[np.argmax(mag)])
            low = float(np.sum(mag[freqs <= 250]) / total)
            mid = float(np.sum(mag[(freqs > 250) & (freqs <= 2000)]) / total)
            high = float(np.sum(mag[freqs > 2000]) / total)
        else:
            sc = dom = 500.0
            low = mid = high = 1 / 3
        return np.array([rms, sc, low, mid, high, dom, max_amp, mid + high, 0.0, 0.0])

    def _safety_guard(self, ml_bin, consec, charge, amp):
        if consec >= self.SAFETY_MAX_CONSEC:                      # L3 burst guard
            return 0, 0, charge
        if charge > self.SAFETY_CHARGE_LIM:                       # L4 charge guard
            return 0, 0, 0.0
        if ml_bin == 1:
            return 1, consec + 1, charge + amp * (self.chunk / self.sr)
        return 0, 0, charge

    def process_segment(self, audio):
        """Derive a changing, audio-driven profile from each chunk.

        This transparent rule-based mapping intentionally avoids the previous
        randomly trained surrogate models. It is a software research prototype,
        not a clinically validated or human-use stimulation controller.
        """
        audio = np.asarray(audio, dtype=np.float64).reshape(-1)
        if audio.size == 0:
            raise ValueError("No audio samples were supplied for pulse analysis.")

        starts = list(range(0, len(audio), self.chunk))
        feats = np.asarray([
            self._chunk_features(audio[i:i + self.chunk]) for i in starts
        ], dtype=np.float64)

        rms = feats[:, 0]
        centroid = feats[:, 1]
        low, mid, high = feats[:, 2], feats[:, 3], feats[:, 4]
        max_amp = feats[:, 6]

        def robust_norm(values):
            values = np.asarray(values, dtype=float)
            lo, hi = np.percentile(values, [10, 90]) if len(values) > 1 else (values[0], values[0])
            if not np.isfinite(hi - lo) or hi - lo < 1e-9:
                return np.zeros_like(values)
            return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

        energy_n = robust_norm(rms)
        brightness_n = robust_norm(centroid)
        spectral_balance = np.clip((high - low + 1.0) / 2.0, 0.0, 1.0)

        freq_span = self.SAFETY_MAX_FREQ - self.SAFETY_MIN_FREQ
        candidate_freq = self.SAFETY_MIN_FREQ + freq_span * (
            0.55 * energy_n + 0.30 * brightness_n + 0.15 * spectral_balance
        )
        candidate_amp = self.SAFETY_MIN_AMP + (
            self.SAFETY_MAX_AMP - self.SAFETY_MIN_AMP
        ) * (0.65 * energy_n + 0.35 * robust_norm(max_amp))

        rms_floor = max(1e-4, float(np.percentile(rms, 20)) * 0.5)
        active_candidates = (rms > rms_floor) & (rms > 1e-5)

        bins, consec, charge = [], 0, 0.0
        freqs, amps = [], []
        chunk_seconds = self.chunk / self.sr
        for i, active in enumerate(active_candidates):
            amp = float(np.clip(candidate_amp[i], self.SAFETY_MIN_AMP, self.SAFETY_MAX_AMP))
            freq = float(np.clip(candidate_freq[i], self.SAFETY_MIN_FREQ, self.SAFETY_MAX_FREQ))
            if active:
                final_bin, consec_new, charge_new = self._safety_guard(1, consec, charge, amp)
                if final_bin:
                    consec, charge = consec_new, charge_new
                else:
                    consec, charge = 0, charge_new
            else:
                final_bin, consec, charge = 0, 0, max(0.0, charge - amp * chunk_seconds)
            bins.append(int(final_bin))
            freqs.append(freq if final_bin else self.SAFETY_MIN_FREQ)
            amps.append(amp if final_bin else 0.0)

        bins = np.asarray(bins, dtype=int)
        freq_arr = np.asarray(freqs, dtype=float)
        if freq_arr.size and float(np.max(freq_arr)) > self.SAFETY_MAX_FREQ + 1e-9:
            raise ValueError(f"Safety stop: planned pulse frequency exceeds {self.SAFETY_MAX_FREQ:.0f} Hz.")
        amp_arr = np.asarray(amps, dtype=float)
        fm_freq = np.minimum(np.where(bins == 1, self.fm_f1, self.fm_f0), self.SAFETY_MAX_FREQ).astype(float)
        fm_amp = np.clip(rms * self.fm_max_amp * 3, 0.0, self.fm_max_amp)

        def synth(freq_c, amp_c):
            f = np.repeat(np.asarray(freq_c), self.chunk)[:len(audio)]
            a = np.repeat(np.asarray(amp_c), self.chunk)[:len(audio)]
            return (a * np.sin(np.cumsum(2 * np.pi * f / self.sr))).astype(np.float32)

        fm_out = synth(fm_freq, fm_amp)
        stim_out = synth(freq_arr, amp_arr)
        params = {
            "frequencies": freq_arr.tolist(),
            "amplitudes": amp_arr.tolist(),
            "timestamps": [i / self.sr for i in starts],
            "binary_sequence": bins.tolist(),
        }
        return fm_out, bins.copy(), stim_out, params, bins

    def global_features(self, audio, bpm):
        audio = np.asarray(audio, dtype=np.float64)
        ft = {"rms_energy": float(np.sqrt(np.mean(audio ** 2))),
              "max_amplitude": float(np.max(np.abs(audio))), "bpm": float(bpm)}
        fw, psd = sp_signal.welch(audio, fs=self.sr, nperseg=1024)
        ft["psd_frequencies"], ft["psd_power"] = fw.tolist(), psd.tolist()
        ft["psd_peak_frequency"] = float(fw[np.argmax(psd)])
        trapz = getattr(np, "trapezoid", None) or np.trapz
        ft["psd_total_power"] = float(trapz(psd, fw))
        ffreqs, _, Z = sp_signal.stft(audio, fs=self.sr, nperseg=2048)
        mag = np.abs(Z)
        ft["spectral_centroid_mean"] = float(np.mean(np.sum(ffreqs[:, None] * mag, axis=0) / (np.sum(mag, axis=0) + 1e-9)))
        bands = {"sub_bass": (0, 60), "bass": (60, 250), "low_mid": (250, 500),
                 "mid": (500, 2000), "high_mid": (2000, 4000), "treble": (4000, 20000)}
        for b, (lo, hi) in bands.items():
            mask = (ffreqs >= lo) & (ffreqs <= hi)
            ft[f"{b}_energy"] = float(np.mean(mag[mask, :])) if mask.any() else 0.0
        tot = sum(ft[f"{b}_energy"] for b in bands) + 1e-9
        ft["low_freq_ratio"] = (ft["sub_bass_energy"] + ft["bass_energy"]) / tot
        ft["mid_freq_ratio"] = (ft["low_mid_energy"] + ft["mid_energy"]) / tot
        ft["high_freq_ratio"] = (ft["high_mid_energy"] + ft["treble_energy"]) / tot
        return ft


@st.cache_resource(show_spinner=False)
def _load_es_models():
    rs = np.random.RandomState(42)
    n = 8000
    rms = rs.uniform(0.001, 0.5, n)
    centroid = rs.uniform(50, 8000, n)
    low = rs.uniform(0.0, 1.0, n)
    mid = rs.uniform(0.0, 1.0 - low)
    high = 1.0 - low - mid
    dom = rs.uniform(20, 6000, n)
    max_amp = rms * rs.uniform(1.2, 4.0, n)
    energy_ratio = mid + high
    consec = rs.randint(0, 8, n)
    charge = rs.uniform(0.0, 0.06, n)

    y_freq = np.clip(10 + (centroid / 8000) * 40 + rms * 6, 10, 50)
    y_amp = np.clip(rms * 0.4, 0.01, 0.1)
    y_amp = np.clip(np.where(centroid > 4000, y_amp * 0.7, y_amp), 0.01, 0.1)
    base_bin = ((energy_ratio > 0.5) & (rms > 0.05)).astype(int)
    force_zero = ((consec >= 5) | (charge > 0.045)).astype(int)
    y_bin = base_bin * (1 - force_zero)
    noise = rs.rand(n) < 0.05
    y_bin[noise] = 1 - y_bin[noise]
    X = np.column_stack([rms, centroid, low, mid, high, dom, max_amp, energy_ratio, consec, charge])

    X_tr, X_te, yf_tr, yf_te, ya_tr, ya_te, yb_tr, yb_te = train_test_split(
        X, y_freq, y_amp, y_bin, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    freq_m = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_split=5, n_jobs=-1, random_state=42).fit(Xtr, yf_tr)
    amp_m = RandomForestRegressor(n_estimators=80, max_depth=8, min_samples_split=5, n_jobs=-1, random_state=42).fit(Xtr, ya_tr)
    bin_m = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42).fit(Xtr, yb_tr)
    freq_m.n_jobs = amp_m.n_jobs = 1
    return {"freq": freq_m, "amp": amp_m, "bin": bin_m, "scaler": scaler}


def _synth_proxy_audio(bpm, genre_text, seed_text, seconds, sr=ES_SR):
    rng = np.random.default_rng(int(hashlib.md5(str(seed_text).encode()).hexdigest()[:8], 16))
    n = int(seconds * sr)
    t = np.arange(n) / sr
    soft = any(k in str(genre_text).lower() for k in ("classical", "ghazal", "calm", "raga", "sitar"))
    root = float(rng.choice([98.0, 110.0, 123.5, 130.8, 146.8]))
    sig = 0.5 * np.sin(2 * np.pi * root * t) + 0.3 * np.sin(2 * np.pi * root * 1.5 * t) \
        + 0.2 * np.sin(2 * np.pi * root * 2 * t)
    sig *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 6.28))
    beat = 60.0 / float(bpm)
    scale = root * np.array([2, 2.25, 2.5, 3, 3.375, 4, 4.5])
    nb = int(seconds / beat) + 1
    perc = np.zeros(n)
    bar_gain = rng.uniform(0.15, 1.0, nb // 4 + 2)
    for b in range(nb):
        s0 = int(b * beat * sr)
        L = min(int(0.45 * sr), n - s0)
        if L <= 0:
            break
        tt = np.arange(L) / sr
        f = float(rng.choice(scale)) * (1.0 if b % 4 else 0.5)
        accent = 1.0 if b % 4 == 0 else 0.6
        perc[s0:s0 + L] += accent * bar_gain[b // 4] * np.exp(-tt / 0.12) * np.sin(2 * np.pi * f * tt)
    sig = 0.5 * sig + 1.0 * perc + 0.003 * rng.standard_normal(n)
    target_rms = 0.035 if soft else 0.12
    sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9) * target_rms
    return np.clip(sig, -1, 1).astype(np.float32)


def _decode_uploaded_audio(raw, filename, seconds=None, sr=ES_SR):
    """Decode to mono float32 @ ES_SR. seconds=None -> decode the FULL file (pulse length == song length)."""
    ext = os.path.splitext(filename or "")[1].lower() or ".wav"
    try:
        import librosa
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            y, _ = librosa.load(path, sr=sr, mono=True, duration=seconds)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        return y.astype(np.float32) if len(y) > sr else None
    except Exception:
        pass
    try:
        rate, data = sp_wavfile.read(io.BytesIO(raw))
        data = data.astype(np.float64)
        if data.ndim == 2:
            data = data.mean(axis=1)
        peak = np.max(np.abs(data)) or 1.0
        data = data / (32768.0 if peak > 1.5 else 1.0)
        if rate != sr:
            from math import gcd
            g = gcd(int(rate), sr)
            data = sp_signal.resample_poly(data, sr // g, int(rate) // g)
        if seconds:
            data = data[: int(seconds * sr)]
        return data.astype(np.float32) if len(data) > sr else None
    except Exception:
        return None


def _get_song_duration_sec(song_record):
    """Safely extract known song duration in seconds from available metadata without inventing values."""
    for key in ("duration", "duration_sec", "duration_seconds", "length", "track_duration"):
        val = song_record.get(key)
        if val is not None:
            try:
                d = float(val)
                if np.isfinite(d) and d > 0:
                    if d > 1000:  # handle milliseconds if applicable
                        d = d / 1000.0
                    return d
            except (TypeError, ValueError):
                continue
    return None


def _run_es_pipeline(track_key, track_bpm, genre_text, uploaded, dur_sec, strict_upload=False):
    conv = MLMusicToSafeStimulation({})
    dur_sec = max(1.0, float(dur_sec or 30.0))
    base_audio, source = None, "synthetic proxy (metadata-driven simulation)"

    if uploaded is not None:
        base_audio = _decode_uploaded_audio(uploaded[0], uploaded[1], None)   # full length of the upload
        if base_audio is not None:
            source = f"uploaded audio, full length ({len(base_audio) / ES_SR:.1f} s): {uploaded[1]}"
        elif strict_upload:
            raise ValueError("Could not read the uploaded file as audio (or it is shorter than 1 second). "
                             "Please upload a valid WAV file.")

    if base_audio is None:
        base_audio = _synth_proxy_audio(track_bpm, genre_text, track_key, dur_sec)

    fm_out, fm_bin, stim_out, stim_params, stim_bin = conv.process_segment(base_audio)
    actual_duration = len(base_audio) / ES_SR
    features = conv.global_features(base_audio, track_bpm)
    return {
        "base_audio": base_audio,
        "fm_out": fm_out,
        "stim_out": stim_out,
        "fm_bin": fm_bin,
        "stim_bin": stim_bin,
        "stim_params": stim_params,
        "features": features,
        "dur_sec": float(actual_duration),
        "base_sec": float(actual_duration),
        "source": source,
    }


def _fig_png(fig, dpi=80):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def _ds(x, max_pts=150_000):
    step = max(1, len(x) // max_pts)
    return x[::step]


def _render_analysis_png(res, music_name):
    SR, dur = ES_SR, res["dur_sec"]
    audio, fm_sig, stim_sig = res["base_audio"], res["fm_out"], res["stim_out"]
    sp, ft = res["stim_params"], res["features"]
    fm_bin, stim_bin = np.asarray(res["fm_bin"]), np.asarray(res["stim_bin"])
    bin_time = np.array(sp["timestamps"])
    chunk_dur = ES_CHUNK / SR

    audio_p = _ds(audio)
    t_audio = np.linspace(0, len(audio) / SR, len(audio_p))
    fm_p, stim_p = _ds(fm_sig), _ds(stim_sig)
    t_full = np.linspace(0, dur, len(fm_p))

    fig = Figure(figsize=(22, 15))
    fig.suptitle(f"Experimental Software Simulation Profile — {music_name}  ({dur/60:.1f} min)\n"
                 f"Not validated for use on a person  ·  Source: {res['source']}", fontsize=13, fontweight="bold")

    ax = fig.add_subplot(3, 4, 1)
    ax.plot(t_audio, audio_p, "b-", alpha=0.6, lw=0.4)
    ax.set_title("Base Audio Clip"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(True)

    ax = fig.add_subplot(3, 4, 2)
    ax.plot(t_full, fm_p, "r-", alpha=0.8, lw=0.4)
    ax.set_title(f"FM Signal"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(True)

    ax = fig.add_subplot(3, 4, 3)
    ax.plot(t_full, stim_p, "g-", alpha=0.8, lw=0.4)
    ax.axhline(0.1, color="r", ls="--", lw=1.2, label="Safety ±0.1"); ax.axhline(-0.1, color="r", ls="--", lw=1.2)
    ax.set_title(f"ES Signal"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7); ax.grid(True)

    ax = fig.add_subplot(3, 4, 4)
    ax.semilogy(ft["psd_frequencies"], np.maximum(ft["psd_power"], 1e-20), "b-", lw=1.2, alpha=0.7)
    ax.axvline(ft["psd_peak_frequency"], color="r", ls="--", alpha=0.8, label=f"Peak {ft['psd_peak_frequency']:.1f} Hz")
    ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("PSD"); ax.set_title("Power Spectral Density"); ax.legend(fontsize=7); ax.grid(True)

    ax = fig.add_subplot(3, 4, 5)
    ax.step(bin_time, fm_bin, "purple", where="post", lw=1.2)
    ax.fill_between(bin_time, fm_bin, step="post", alpha=0.25, color="purple")
    ax.set_title("FM Binary Sequence"); ax.set_ylabel("Binary"); ax.set_yticks([0, 1]); ax.grid(True)

    ax = fig.add_subplot(3, 4, 6)
    ax.step(bin_time, stim_bin, "orange", where="post", lw=1.2)
    ax.fill_between(bin_time, stim_bin, step="post", alpha=0.25, color="orange")
    ax.set_title("ES Binary Sequence"); ax.set_ylabel("Binary"); ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1]); ax.grid(True)

    ax = fig.add_subplot(3, 4, 7)
    n = min(len(fm_bin), len(stim_bin))
    ax.step(bin_time[:n], fm_bin[:n], "purple", where="post", lw=1.2, label="FM", alpha=0.7)
    ax.step(bin_time[:n], stim_bin[:n], "orange", where="post", lw=1.2, label="ES", alpha=0.7)
    ax.set_title("FM vs ES Binary Comparison"); ax.set_ylabel("Binary"); ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1]); ax.legend(fontsize=7); ax.grid(True)

    ax = fig.add_subplot(3, 4, 8)
    ax.plot(bin_time, sp["frequencies"], "darkorange", lw=1.2)
    ax.axhline(10, color="r", ls="--", alpha=0.5, label="Min 10 Hz"); ax.axhline(MLMusicToSafeStimulation.SAFETY_MAX_FREQ, color="r", ls="--", alpha=0.5, label=f"Max {MLMusicToSafeStimulation.SAFETY_MAX_FREQ:.0f} Hz")
    ax.set_title("Pulse Frequency Over Time"); ax.set_ylabel("Freq (Hz)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.grid(True)

    ax = fig.add_subplot(3, 4, 9)
    ax.bar(bin_time, stim_bin, width=chunk_dur * 0.8,
           color=["#FF6B6B" if b == 1 else "#4ECDC4" for b in stim_bin], alpha=0.85)
    ax.set_title("Pulse Map"); ax.set_ylabel("Binary"); ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1]); ax.grid(True, axis="y")

    ax = fig.add_subplot(3, 4, 10)
    cum = np.cumsum(np.array(sp["amplitudes"]) * stim_bin * chunk_dur)
    ax.plot(bin_time, cum, "teal", lw=1.5)
    ax.axhline(0.045, color="red", ls="--", lw=2, label="Safety limit")
    ax.set_title("Cumulative Charge Accumulation"); ax.set_ylabel("Charge"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.grid(True)

    ax = fig.add_subplot(3, 4, 11)
    ax.plot(bin_time, sp["amplitudes"], "#9D4EDD", lw=1.2, label="Amplitude")
    ax.axhline(0.1, color="r", ls="--", lw=1.5, label="Max 0.1"); ax.axhline(0.01, color="b", ls="--", lw=1.0, label="Min 0.01")
    ax.fill_between(bin_time, sp["amplitudes"], alpha=0.25, color="#9D4EDD")
    ax.set_title("Amplitude Over Time"); ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.grid(True)

    ax = fig.add_subplot(3, 4, 12)
    bands = ["sub_bass", "bass", "low_mid", "mid", "high_mid", "treble"]
    energies = [ft.get(f"{b}_energy", 0) for b in bands]
    bars = ax.bar(bands, energies, color=["#FF6B6B", "#FF9F1C", "#4ECDC4", "#9D4EDD", "#4361EE", "#06D6A0"],
                  edgecolor="black", alpha=0.85)
    for bar, e in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{e:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Spectral Band Energy"); ax.set_ylabel("Mean Energy"); ax.set_xlabel("Band")
    ax.tick_params(axis="x", rotation=30); ax.grid(True, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _fig_png(fig)


def _render_viewer_png(res):
    SR, dur, stim = ES_SR, res["dur_sec"], res["stim_out"]
    fig = Figure(figsize=(16, 11))
    fig.suptitle(f"Experimental Software Simulation — Pulse Viewer  ({dur/60:.2f} min total) — Not validated for use on a person", fontsize=14, fontweight="bold")
    ax = fig.add_subplot(3, 1, 1)
    z1 = min(int(SR * 0.5), len(stim)); t1 = np.linspace(0, z1 / SR * 1000, z1); s1 = stim[:z1]
    ax.plot(t1, s1, "mediumseagreen", lw=1.2)
    ax.fill_between(t1, s1, 0, where=(s1 > 0), alpha=0.4, color="mediumseagreen")
    ax.fill_between(t1, s1, 0, where=(s1 < 0), alpha=0.4, color="tomato")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("First 500 ms — individual cycles")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(3, 1, 2)
    z2 = min(int(SR * 10), len(stim)); t2 = np.linspace(0, z2 / SR, z2)
    ax.plot(t2, stim[:z2], "steelblue", lw=0.7)
    ax.axhline(0.1, color="red", ls="--", lw=1.2, label="Safety limit ±0.1"); ax.axhline(-0.1, color="red", ls="--", lw=1.2)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("First 10 seconds")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(3, 1, 3)
    s3 = _ds(stim, 200_000); t3 = np.linspace(0, dur / 60, len(s3))
    ax.plot(t3, s3, "#9D4EDD", lw=0.4, alpha=0.7); ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Full duration overview — {dur/60:.2f} min")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _fig_png(fig, dpi=90)


def _render_hero_png(res, music_name):
    """Clean, presentation-style chart: pulse frequency over the whole song + pulse strength underneath."""
    sp, dur = res["stim_params"], res["dur_sec"]
    limit = MLMusicToSafeStimulation.SAFETY_MAX_FREQ
    t = np.asarray(sp["timestamps"], dtype=float) / 60.0
    on = np.asarray(sp["binary_sequence"]) == 1
    f = np.where(on, np.asarray(sp["frequencies"], dtype=float), 0.0)
    a = np.asarray(sp["amplitudes"], dtype=float)
    ink, grid = "#334155", "#E2E8F0"

    fig = Figure(figsize=(12, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.15], hspace=0.14)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax.fill_between(t, f, step="post", color="#7C3AED", alpha=0.22)
    ax.step(t, f, where="post", color="#6D28D9", lw=1.6)
    ax.axhline(limit, color="#EF4444", ls="--", lw=1.4)
    ax.text(0.995, limit, f" {limit:.0f} Hz safety limit", color="#EF4444", fontsize=9, fontweight="bold",
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_ylim(0, limit * 1.12); ax.set_ylabel("Pulse frequency (Hz)", color=ink)
    ax.set_title(f"Pulse for “{music_name}”  ·  {dur/60:.2f} min", loc="left", fontsize=12.5, fontweight="bold", color=ink)
    ax2.fill_between(t, a, step="post", color="#4361EE", alpha=0.25)
    ax2.step(t, a, where="post", color="#3B5BDB", lw=1.2)
    ax2.set_ylim(0, 0.115); ax2.set_ylabel("Strength", color=ink); ax2.set_xlabel("Time (minutes)", color=ink)
    ax.tick_params(labelbottom=False)
    for axx in (ax, ax2):
        axx.set_xlim(0, max(t.max() if len(t) else 1.0, dur / 60.0))
        axx.grid(True, color=grid, lw=0.8); axx.set_axisbelow(True)
        axx.tick_params(colors=ink, labelsize=9)
        for sp_name in ("top", "right"):
            axx.spines[sp_name].set_visible(False)
        for sp_name in ("left", "bottom"):
            axx.spines[sp_name].set_color(grid)
    return _fig_png(fig, dpi=110)


def _pulse_stats(res):
    sp = res["stim_params"]
    desat_ms = MLMusicToSafeStimulation.DESAT_CHUNKS * ES_CHUNK / ES_SR * 1000
    return {"duration_sec": res["dur_sec"], "avg_freq_hz": float(np.mean(sp["frequencies"])),
            "avg_amp": float(np.mean(sp["amplitudes"])),
            "max_freq_hz": float(np.max(sp["frequencies"])), "pct_active": float(np.mean(sp["binary_sequence"]) * 100),
            "desat_ms": float(desat_ms), "bpm": float(res["features"]["bpm"]),
            "psd_peak_hz": float(res["features"]["psd_peak_frequency"])}


def _wav_bytes(sig):
    buf = io.BytesIO()
    sp_wavfile.write(buf, ES_SR, (np.clip(sig, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


def _audible_preview_wav(res, seconds=20):
    sr = ES_SR
    n = min(int(seconds * sr), len(res["stim_out"]))
    nch = (n + ES_CHUNK - 1) // ES_CHUNK
    sp = res["stim_params"]
    on = np.repeat(np.asarray(sp["binary_sequence"][:nch], dtype=float), ES_CHUNK)[:n]
    fq = np.repeat(np.asarray(sp["frequencies"][:nch], dtype=float), ES_CHUNK)[:n]
    am = np.repeat(np.asarray(sp["amplitudes"][:nch], dtype=float), ES_CHUNK)[:n]
    k = int(0.005 * sr)
    on = np.convolve(on, np.ones(k) / k, mode="same")
    t = np.arange(n) / sr
    carrier = np.sin(2 * np.pi * 440.0 * t)
    trem = 0.5 + 0.5 * np.sin(np.cumsum(2 * np.pi * fq / sr))
    level = 0.25 + 0.55 * np.clip(am / 0.1, 0, 1)
    return _wav_bytes((0.8 * on * level * carrier * (0.4 + 0.6 * trem)).astype(np.float32))


def _build_pulse_package(res, music_name):
    stats = _pulse_stats(res)
    meta = {"music_name": music_name, "source": res["source"], "full_duration_sec": res["dur_sec"],
            "sample_rate": ES_SR, "signal_type": "Experimental Software Simulation Waveform",
            "safety_standard": "Experimental bench prototype - NOT VALIDATED FOR USE ON A PERSON",
            "freq_range_Hz": [MLMusicToSafeStimulation.SAFETY_MIN_FREQ, MLMusicToSafeStimulation.SAFETY_MAX_FREQ], "amplitude_range": [0.01, 0.1],
            "avg_frequency_Hz": stats["avg_freq_hz"], "avg_amplitude": stats["avg_amp"],
            "pct_active": stats["pct_active"]}
    return {"stats": stats, "hero_png": _render_hero_png(res, music_name), "viewer_png": _render_viewer_png(res),
            "preview_wav": _audible_preview_wav(res), "stim_wav": _wav_bytes(res["stim_out"]), "fm_wav": _wav_bytes(res["fm_out"]),
            "json": json.dumps(meta, indent=2).encode("utf-8")}

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
NAME_MIN_LEN = 2
NAME_MAX_LEN = 40

TIPI_MARATHI = ['मी स्वतःला मोकळा-मिळणारा, उत्साही मानतो.', 'मी स्वतःला टीकात्मक, वाद घालणारा मानतो.', 'मी स्वतःला विश्वासार्ह, स्वअनुशासित मानतो.', 'मी स्वतःला चिंताग्रस्त, सहज अस्वस्थ होणारा मानतो.', 'मी स्वतःला नवीन अनुभवांसाठी खुले, गुंतागुंतीचा मानतो.', 'मी स्वतःला राखीव, शांत मानतो.', 'मी स्वतःला सहकार्यशील, उबदार स्वभावाचा मानतो.', 'मी स्वतःला असंघटित, निष्काळजी मानतो.', 'मी स्वतःला शांत, भावनिकदृष्ट्या स्थिर मानतो.', 'मी स्वतःला पारंपरिक, कल्पनाशक्ती कमी असलेला मानतो.']
DASS_MARATHI = ['मला शांत होणे कठीण वाटत होते.', 'माझ्या तोंडात कोरडेपणा जाणवत होता.', 'मला कोणताही सकारात्मक अनुभव घेता येत नव्हता.', 'मला श्वास घेणे कठीण वाटत होते.', 'काहीतरी करण्याची प्रेरणा मिळवणे कठीण वाटत होते.', 'मी परिस्थितींवर अतिप्रतिक्रिया देत होतो.', 'माझे हात थरथरत होते.', 'मला वाटत होते की मी खूप चिंताग्रस्त ऊर्जा वापरत आहे.', 'मी घाबरू शकेन अशा परिस्थितींबद्दल मला चिंता वाटत होती.', 'मला भविष्याबद्दल काहीही अपेक्षा नाही असे वाटत होते.', 'मला चिडचिड होत होती.', 'मला आराम करणे कठीण जात होते.', 'माझा मूड खिन्न आणि उदास होता.', 'मी करत असलेल्या कामात अडथळा आणणाऱ्या गोष्टी सहन करू शकत नव्हतो.', 'मला वाटत होते की मी घाबरण्याच्या स्थितीजवळ पोहोचलो आहे.', 'मला कोणत्याही गोष्टीबद्दल उत्साह वाटत नव्हता.', 'मला वाटत होते की मी व्यक्ती म्हणून मोलाचा नाही.', 'मला वाटत होते की मी खूप संवेदनशील/हळवा झालो आहे.', 'शारीरिक हालचाल नसतानाही माझ्या हृदयाचे ठोके जाणवत होते.', 'मला कोणतेही योग्य कारण नसताना भीती वाटत होती.', 'मला वाटत होते की जीवन निरर्थक आहे.']
WHOQOL_MARATHI = ['तुम्ही तुमच्या जीवनाच्या गुणवत्तेला किती प्रमाणात रेट कराल?', 'तुम्ही तुमच्या आरोग्याबद्दल किती समाधानी आहात?', 'वेदना तुम्हाला आवश्यक गोष्टी करण्यापासून किती प्रमाणात रोखतात?', 'दैनंदिन जीवनात कार्य करण्यासाठी तुम्हाला किती वैद्यकीय उपचारांची गरज आहे?', 'तुम्ही आयुष्याचा किती आनंद घेतला?', 'तुमच्या जीवनाला किती अर्थ आहे असे तुम्हाला वाटते?', 'तुम्हाला लक्ष केंद्रित करणे किती सोपे वाटते?', 'दैनंदिन जीवनात तुम्हाला किती सुरक्षित वाटते?', 'तुमचे भौतिक वातावरण किती आरोग्यदायी आहे?', 'दैनंदिन जीवनासाठी तुमच्याकडे पुरेशी ऊर्जा आहे का?', 'तुम्ही तुमच्या शरीराच्या स्वरूपाचा स्वीकार करू शकता का?', 'तुमच्या गरजा पूर्ण करण्यासाठी तुमच्याकडे पुरेसे पैसे आहेत का?', 'दैनंदिन जीवनासाठी आवश्यक माहिती तुम्हाला कितपत उपलब्ध आहे?', 'मनोरंजनासाठी तुम्हाला किती संधी मिळतात?', 'तुम्ही तुमच्या झोपेबद्दल किती समाधानी आहात?', 'दैनंदिन कामे करण्याच्या क्षमतेबद्दल तुम्ही किती समाधानी आहात?', 'काम करण्याच्या तुमच्या क्षमतेबद्दल तुम्ही किती समाधानी आहात?', 'तुम्ही स्वतःबद्दल किती समाधानी आहात?', 'तुमच्या वैयक्तिक नातेसंबंधांबद्दल तुम्ही किती समाधानी आहात?', 'तुमच्या लैंगिक जीवनाबद्दल तुम्ही किती समाधानी आहात?', 'तुमच्या मित्रांकडून मिळणाऱ्या मदतीबद्दल तुम्ही किती समाधानी आहात?', 'तुम्ही तुमच्या राहणीमानाबद्दल तुम्ही किती समाधानी आहात?', 'आरोग्यसेवा मिळण्याबद्दल तुम्ही किती समाधानी आहात?', 'वाहतुकीच्या सुविधांबद्दल तुम्ही किती समाधानी आहात?', 'तुम्हाला एका ठिकाणाहून दुसऱ्या ठिकाणी फिरणे किती सहज शक्य होते?', 'तुमच्या आजूबाजूच्या वातावरणाबद्दल तुम्ही किती समाधानी आहात?']

_STATE_DEFAULTS = [
    ("verified", False), ("username", None), ("user_email", None),
    ("editing_profile", False), ("profile_doc", None), ("profile_user", None),
    ("recs", []), ("got_recs", False), ("pool", pd.DataFrame()),
    ("session_number", 1), ("session_finished", False),
    ("page", "Dashboard"),
    ("full_baseline_this_session", False),
    ("session_id", None),
    ("assessment_id", None),
    ("wesad_context", None),
    ("admin_ok", False),
    ("login_flash", None),
    ("rr_saved", []), ("rr_features_latest", None),
    ("pulse_status", {}), ("pulse_port", "COM3"), ("pulse_baud", 115200),
    ("es_analysis", {}), ("pulse_results", {}), ("auto_pulse_results", {}),
    ("song_wavs", {}), ("upload_pulse_results", {}),
    ("stress_score", None), ("stress_source", None), ("stress_detail", None),
    ("checkin_done", False),
]
for key, default in _STATE_DEFAULTS:
    if key not in st.session_state:
        st.session_state[key] = default


def _reset_session_state():
    for k, d in _STATE_DEFAULTS:
        if k in ("page", "admin_ok"):
            continue
        st.session_state[k] = d if not isinstance(d, (list, dict)) else type(d)()
    st.session_state["pool"] = pd.DataFrame()
    for k in list(st.session_state.keys()):
        if k.startswith(("fb_done_", "sess_fb_done_", "rate_", "phq4_", "pss4_", "login_")):
            del st.session_state[k]


PHQ4_ITEMS = [
    "Feeling nervous, anxious or on edge",
    "Not being able to stop or control worrying",
    "Feeling down, depressed or hopeless",
    "Little interest or pleasure in doing things",
]
PHQ4_OPTIONS = {0: "Not at all", 1: "Several days",
                2: "More than half the days", 3: "Nearly every day"}
PSS4_ITEMS = [
    "felt that you were unable to control the important things in your life?",
    "felt confident about your ability to handle your personal problems?",
    "felt that things were going your way?",
    "felt difficulties were piling up so high that you could not overcome them?",
]
PSS4_REVERSED = {1, 2}
PSS4_OPTIONS = {0: "Never", 1: "Almost never", 2: "Sometimes",
                3: "Fairly often", 4: "Very often"}
SHORT_FORM_NOTE = (
    "Short check-in: PHQ-4 (Kroenke et al., 2009) + PSS-4 (Cohen et al., 1983)."
)


def score_short_checkin(phq4, pss4):
    anxiety_raw = phq4[0] + phq4[1]
    depression_raw = phq4[2] + phq4[3]
    pss_total = sum((4 - v) if i in PSS4_REVERSED else v for i, v in enumerate(pss4))
    return {
        "anxiety": anxiety_raw * 7.0,
        "depression": depression_raw * 7.0,
        "stress": pss_total * (42.0 / 16.0),
        "phq4_total": int(sum(phq4)),
        "pss4_total": int(pss_total),
    }


def phq4_band(total):
    return ("normal" if total <= 2 else "mild" if total <= 5
            else "moderate" if total <= 8 else "severe")


@st.cache_data(show_spinner=False)
def _load_dataset():
    return ds.load_dataset()


@st.cache_resource(show_spinner=False)
def _load_models():
    return mdl.load_models()


df, is_demo_dataset, dataset_note = _load_dataset()
metadata, rnn_model, ncf_model, model_error = _load_models()


def _get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


class MongoDBStore:
    def __init__(self, uri=None, db_name=None):
        uri = uri or _get_secret("MONGODB_URI")
        db_name = db_name or _get_secret("MONGODB_DATABASE", "musync")

        if not uri:
            raise RuntimeError(
                "MONGODB_URI is missing. Add MONGODB_URI and "
                "MONGODB_DATABASE in Streamlit Secrets."
            )

        self.client = MongoClient(
            uri,
            tls=True,
            tlsCAFile=certifi.where(),
            serverSelectionTimeoutMS=20000,
            connectTimeoutMS=20000,
            socketTimeoutMS=30000,
            retryWrites=True,
        )

        self.client.admin.command("ping")
        self.mongo_db = self.client[db_name]

        self.login_history = self.mongo_db["login_history"]
        self.profiles = self.mongo_db["user_profiles"]
        self.profile_history = self.mongo_db["profile_history"]
        self.physiological_measurements = self.mongo_db["physiological_measurements"]
        self.qtables = self.mongo_db["qtables"]
        self.recommendation_feedback = self.mongo_db["recommendation_feedback"]
        self.experiments = self.mongo_db["experiments"]
        self.bias_assessments = self.mongo_db["bias_assessments"]
        self.users = self.mongo_db["users"]
        self.state_checkins = self.mongo_db["state_checkins"]
        self.sessions = self.mongo_db["sessions"]
        self.assessments = self.mongo_db["assessments"]

        self.mode = "mongodb"
        self.index_warnings = []
        self._create_indexes()

    def _ix(self, coll, keys, **kw):
        try:
            coll.create_index(keys, **kw)
        except PyMongoError as e:
            self.index_warnings.append(f"{coll.name} {keys}: {e}")

    def _create_indexes(self):
        has_assess = {"assessment_id": {"$exists": True}}
        self._ix(self.users, "user", unique=True)
        self._ix(self.users, "email")
        self._ix(self.profiles, "user", unique=True)
        self._ix(self.profile_history, [("user", 1), ("timestamp", -1)])
        self._ix(self.sessions, "session_id", unique=True)
        self._ix(self.sessions, [("user", 1), ("session_number", 1)], unique=True)
        self._ix(self.login_history, [("user", 1), ("timestamp_utc", -1)])
        self._ix(self.login_history, "session_id")
        self._ix(self.assessments, "assessment_id", unique=True)
        self._ix(self.assessments, [("user", 1), ("session_id", 1)])
        self._ix(self.qtables, "user", unique=True)
        self._ix(self.recommendation_feedback, [("user", 1), ("timestamp", -1)])
        self._ix(self.recommendation_feedback, [("user", 1), ("assessment_id", 1), ("song_id", 1)],
                 unique=True, partialFilterExpression=has_assess, name="uniq_feedback_per_assessment_song")
        self._ix(self.experiments, [("user", 1), ("timestamp", -1)])
        self._ix(self.experiments, [("user", 1), ("assessment_id", 1)],
                 unique=True, partialFilterExpression=has_assess, name="uniq_session_feedback_per_assessment")
        self._ix(self.physiological_measurements, [("user", 1), ("timestamp", -1)])
        self._ix(self.physiological_measurements, "session_id")
        self._ix(self.state_checkins, [("user", 1), ("timestamp", -1)])
        self._ix(self.state_checkins, "assessment_id", unique=True, partialFilterExpression=has_assess,
                 name="uniq_checkin_per_assessment")
        self._ix(self.bias_assessments, [("user", 1), ("timestamp", -1)])


@st.cache_resource(show_spinner=False)
def _connect_mongodb(uri, db_name):
    return MongoDBStore(uri=uri, db_name=db_name)


_mongo_uri = _get_secret("MONGODB_URI")
_mongo_db_name = _get_secret("MONGODB_DATABASE", "musync")

try:
    db = _connect_mongodb(_mongo_uri, _mongo_db_name)
    mongodb_error = None
except Exception as e:
    db = None
    mongodb_error = str(e)


def _now_ist_str():
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")


def _utc_iso():
    return datetime.now(timezone.utc).isoformat()


def _normalize_username(raw):
    return re.sub(r"\s+", "_", raw.strip().lower())


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    return o


def _resolve_recommended_audio(song_record):
    candidates = []
    for key in ("audio_file", "audio_path", "audio_filename", "local_audio_path"):
        value = song_record.get(key)
        if value is not None and str(value).strip():
            candidates.append(str(value).strip())

    audio_roots = (
        os.path.join("data", "audio"),
        os.path.join("audio"),
        os.path.join("assets", "audio"),
    )
    song_id = str(song_record.get("song_id", "")).strip()
    song = str(song_record.get("song", "")).strip()
    artist = str(song_record.get("artist", "")).strip()

    def slug(value):
        value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
        return value[:120]

    names = []
    if song_id:
        names.extend([f"{slug(song_id)}{ext}" for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")])
    if song:
        for stem in (slug(song), slug(f"{song} - {artist}"), slug(f"{artist} - {song}")):
            names.extend([f"{stem}{ext}" for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a")])

    for root in audio_roots:
        candidates.extend(os.path.join(root, name) for name in names)

    for candidate in candidates:
        path = os.path.expanduser(candidate)
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if os.path.isfile(path):
            try:
                with open(path, "rb") as audio_f:
                    return audio_f.read(), os.path.basename(path)
            except OSError:
                continue
    return None


def spotify_link(song, artist):
    q = urllib.parse.quote_plus(f"{song} {artist}")
    return f"https://open.spotify.com/search/{q}"


STRESS_W_DASS = 0.60
STRESS_W_NEURO = 0.20
STRESS_W_QOL = 0.20
STRESS_CHECKIN_W_PSS = 0.70
STRESS_CHECKIN_W_PHQ = 0.30
STRESS_UPDATE_ALPHA = 0.70
STRESS_DEFAULT = 40.0
DASS_STRESS_IDX = [0, 5, 7, 10, 11, 13, 17]
STRESS_NOTE = (
    "The stress score is calculated automatically (heuristic research composite, not a validated "
    "clinical measure)."
)


def _valid_answers(values, expected, low, high):
    cleaned = []
    for value in list(values)[:expected]:
        try:
            number = float(value)
            cleaned.append(number if np.isfinite(number) and low <= number <= high else None)
        except (TypeError, ValueError):
            cleaned.append(None)
    return cleaned


def _mean_or(values, default):
    usable = [float(v) for v in values if v is not None]
    return float(np.mean(usable)) if usable else float(default)


def _stress_band(score):
    s = float(score)
    return "Low" if s < 35 else "Moderate" if s <= NodeMCUPulseGenerator.HIGH_STRESS_THRESHOLD else "High"


def compute_baseline_stress(profile_doc):
    profile_doc = profile_doc or {}
    tipi = _valid_answers(profile_doc.get("tipi", []) or [], 10, 1, 7)
    dass = _valid_answers(profile_doc.get("dass", []) or [], 21, 0, 3)
    whoqol = _valid_answers(profile_doc.get("whoqol", []) or [], 26, 1, 5)

    parts = {}
    if len(dass) == 21 and all(v is not None for v in dass):
        parts["dass_stress"] = ((sum(dass[i] for i in DASS_STRESS_IDX) * 2) / 42.0, STRESS_W_DASS)
    if len(tipi) >= 9 and tipi[3] is not None and tipi[8] is not None:
        neuroticism = (tipi[3] + (8 - tipi[8])) / 2.0
        parts["neuroticism"] = ((neuroticism - 1.0) / 6.0, STRESS_W_NEURO)
    if len(whoqol) == 26 and all(v is not None for v in whoqol):
        q = list(whoqol)
        q[2], q[3] = 6 - q[2], 6 - q[3]
        parts["low_quality_of_life"] = (1.0 - (float(np.mean(q)) - 1.0) / 4.0, STRESS_W_QOL)

    if not parts:
        return {"score": STRESS_DEFAULT, "components": {}, "complete": False}
    total_w = sum(w for _, w in parts.values())
    score = 100.0 * sum(v * w for v, w in parts.values()) / total_w
    return {"score": float(np.clip(score, 0.0, 100.0)),
            "components": {k: round(100.0 * v, 1) for k, (v, _) in parts.items()},
            "complete": len(parts) == 3}


def compute_checkin_stress(phq4, pss4):
    res = score_short_checkin(phq4, pss4)
    pss_n = res["pss4_total"] / 16.0
    phq_n = res["phq4_total"] / 12.0
    score = 100.0 * (STRESS_CHECKIN_W_PSS * pss_n + STRESS_CHECKIN_W_PHQ * phq_n)
    return float(np.clip(score, 0.0, 100.0)), res


def update_stress(previous, checkin_score):
    try:
        prev = float(previous)
    except (TypeError, ValueError):
        prev = float(checkin_score)
    if not np.isfinite(prev):
        prev = float(checkin_score)
    return float(np.clip(STRESS_UPDATE_ALPHA * float(checkin_score) + (1.0 - STRESS_UPDATE_ALPHA) * prev, 0.0, 100.0))


def _previous_stress(profile_doc):
    try:
        v = float((profile_doc or {}).get("stress_score"))
        if np.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return compute_baseline_stress(profile_doc)["score"]


def _needs_checkin():
    return (int(st.session_state.get("session_number", 1)) > 1
            and not st.session_state.get("full_baseline_this_session")
            and not st.session_state.get("checkin_done"))


def _resolve_session_stress(profile_doc):
    cur = st.session_state.get("stress_score")
    if cur is not None:
        return float(cur), (st.session_state.get("stress_source") or "baseline")
    if _needs_checkin():
        return None, None
    base = compute_baseline_stress(profile_doc)
    st.session_state["stress_score"] = float(base["score"])
    st.session_state["stress_source"] = "baseline"
    st.session_state["stress_detail"] = {"previous": None, "updated": float(base["score"]),
                                         "components": base["components"]}
    return float(base["score"]), "baseline"


def _persist_stress(user, score, source, extra=None):
    fields = {"stress_score": round(float(score), 2), "stress_source": source,
              "stress_updated_at_ist": _now_ist_str(), "stress_updated_at_utc": _utc_iso()}
    if db:
        try:
            db.profiles.update_one({"user": user}, {"$set": fields})
            db.profile_history.insert_one(_clean({
                "user": user, "type": "stress_update",
                "timestamp": fields["stress_updated_at_utc"], "timestamp_ist": fields["stress_updated_at_ist"],
                "session_id": st.session_state.get("session_id"),
                "session_number": st.session_state.get("session_number"),
                "stress_score": fields["stress_score"], "source": source, **(extra or {}),
            }))
        except Exception as exc:
            st.warning(f"Stress score could not be saved to MongoDB: {exc}")
    return fields


def _apply_checkin(user, profile_doc, phq4, pss4):
    checkin_score, res = compute_checkin_stress(phq4, pss4)
    previous = _previous_stress(profile_doc)
    updated = update_stress(previous, checkin_score)

    st.session_state["stress_score"] = float(updated)
    st.session_state["stress_source"] = "checkin"
    st.session_state["stress_detail"] = {"previous": float(previous), "checkin": float(checkin_score),
                                         "updated": float(updated)}
    st.session_state["checkin_done"] = True
    if isinstance(st.session_state.get("profile_doc"), dict):
        st.session_state["profile_doc"]["stress_score"] = round(float(updated), 2)

    if db:
        try:
            db.state_checkins.insert_one(_clean({
                "user": user, "session_id": st.session_state.get("session_id"),
                "session_number": st.session_state.get("session_number"),
                "timestamp": _utc_iso(), "timestamp_ist": _now_ist_str(),
                "phq4": list(phq4), "pss4": list(pss4),
                "phq4_total": res["phq4_total"], "pss4_total": res["pss4_total"],
                "phq4_band": phq4_band(res["phq4_total"]),
                "anxiety_dass_scale": res["anxiety"], "depression_dass_scale": res["depression"],
                "stress_dass_scale": res["stress"],
                "checkin_stress_100": checkin_score, "previous_stress_100": previous,
                "updated_stress_100": updated, "note": SHORT_FORM_NOTE,
            }))
        except Exception as exc:
            st.warning(f"Check-in could not be saved to MongoDB: {exc}")
    _persist_stress(user, updated, "checkin", {"checkin_stress_100": checkin_score, "previous_stress_100": previous})
    return updated


def _complete_login(username, display_id):
    if not db:
        raise RuntimeError("MongoDB is not connected.")

    ist, utc = _now_ist_str(), _utc_iso()
    session_id = uuid.uuid4().hex

    with db.client.start_session() as mongo_session:
        with mongo_session.start_transaction():
            user_doc = db.users.find_one_and_update(
                {"user": username},
                {
                    "$set": {"last_login_ist": ist, "last_login_utc": utc, "name": display_id},
                    "$setOnInsert": {
                        "display_id": display_id,
                        "first_login_ist": ist, "first_login_utc": utc,
                        "created_at_utc": utc,
                    },
                    "$inc": {"login_count": 1},
                },
                upsert=True, return_document=ReturnDocument.AFTER, session=mongo_session,
            )
            if not user_doc:
                raise RuntimeError("MongoDB did not return the user document.")

            session_number = int(user_doc.get("login_count", 1))
            is_new = session_number == 1

            db.sessions.insert_one({
                "session_id": session_id, "user": username, "name": display_id,
                "session_number": session_number,
                "started_at_ist": ist, "started_at_utc": utc,
                "ended_at_ist": None, "ended_at_utc": None, "duration_min": None,
                "status": "active", "is_first_session": is_new,
            }, session=mongo_session)

            db.login_history.insert_one({
                "event": "login", "session_id": session_id, "user": username,
                "name": display_id, "session_number": session_number,
                "timestamp_ist": ist, "timestamp_utc": utc,
            }, session=mongo_session)

    st.session_state["verified"] = True
    st.session_state["username"] = username
    st.session_state["user_email"] = None
    st.session_state["session_number"] = session_number
    st.session_state["session_id"] = session_id
    st.session_state["full_baseline_this_session"] = False
    st.session_state["stress_score"] = None
    st.session_state["stress_source"] = None
    st.session_state["stress_detail"] = None
    st.session_state["checkin_done"] = False
    st.session_state["profile_user"] = None
    st.session_state["profile_doc"] = None
    st.session_state["login_flash"] = (
        "Name saved. Welcome!" if is_new else f"Welcome back - Session #{session_number}.")


def _login_with_name(raw_name):
    if not db:
        return False, "MongoDB is not connected."
    name = re.sub(r"\s+", " ", (raw_name or "").strip())
    if len(name) < NAME_MIN_LEN:
        return False, f"Please enter your name (at least {NAME_MIN_LEN} characters)."
    if len(name) > NAME_MAX_LEN:
        return False, f"Name must be at most {NAME_MAX_LEN} characters."
    username = _normalize_username(name)
    try:
        _complete_login(username, name)
    except Exception as e:
        return False, f"Name could not be saved in MongoDB ({e})."
    return True, "OK"


def _render_login_flow():
    flash = st.session_state.get("login_flash")
    if flash:
        st.success(flash)
        st.session_state["login_flash"] = None

    st.info("Welcome to MuSync 🎧 Type your **name** below and tap **Enter**.")
    with st.form("login_form", clear_on_submit=False):
        name = st.text_input("Your name", key="login_name_input", max_chars=NAME_MAX_LEN,
                             placeholder="e.g. Priya")
        submitted_login = st.form_submit_button("Enter ➜", type="primary", use_container_width=True)
    if submitted_login:
        ok, msg = _login_with_name(name)
        if ok:
            st.rerun()
        else:
            st.error(msg)


def _mark_checkout(reason):
    if not db:
        return False, "MongoDB is not connected."

    sid = st.session_state.get("session_id")
    if not sid:
        return False, "No active session ID is present."

    now_utc = datetime.now(timezone.utc)
    ist = _now_ist_str()

    sdoc = db.sessions.find_one({"session_id": sid, "ended_at_utc": None})
    if not sdoc:
        return False, "The session record was not found or already closed."

    try:
        started = datetime.fromisoformat(str(sdoc["started_at_utc"]))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        duration = round((now_utc - started).total_seconds() / 60.0, 2)
    except Exception:
        duration = None

    try:
        with db.client.start_session() as mongo_session:
            with mongo_session.start_transaction():
                db.sessions.update_one(
                    {"session_id": sid, "ended_at_utc": None},
                    {"$set": {
                        "ended_at_ist": ist, "ended_at_utc": now_utc.isoformat(),
                        "duration_min": duration, "status": "closed", "end_reason": reason,
                    }},
                    session=mongo_session,
                )
                db.login_history.insert_one({
                    "event": "logout", "session_id": sid, "user": sdoc["user"],
                    "name": sdoc.get("name"), "session_number": sdoc.get("session_number"),
                    "timestamp_ist": ist, "timestamp_utc": now_utc.isoformat(),
                    "session_duration_min": duration, "reason": reason,
                }, session=mongo_session)
        return True, "Logout saved."
    except Exception as e:
        return False, f"Logout could not be saved ({e})."


RAGA_EVIDENCE = {
    "bhairavi": {
        "raga": "Bhairavi",
        "pattern": re.compile(r"\bbhairavi\b|\bbhairvi\b", re.I),
        "citation": ("Chand K, Chandra S, Dutt V. Raga Bhairavi in virtual reality reduces stress-related "
                     "psychophysiological markers. Scientific Reports 14, 24816 (2024)."),
        "short_citation": "Chand, Chandra & Dutt (2024), Sci Rep 14:24816",
        "stimulus_url": "https://www.youtube.com/watch?v=6_bfPzclL08",
    }
}

RESEARCH_PAPER_TRACKS = [
    {"key": "bhairavi_darbar", "song_id": "research_src_bhairavi_darbar", "song": "Meditative Raag Bhairavi",
     "artist": "Kaushiki Chakraborty", "genre": "Indian Classical", "raga": "Bhairavi",
     "evidence_level": "studied_stimulus", "evidence_key": "bhairavi",
     "source_url": RAGA_EVIDENCE["bhairavi"]["stimulus_url"],
     "source_name": "Stimulus used in research"},
]
RESEARCH_CANDIDATES = RESEARCH_PAPER_TRACKS
EVIDENCE_WEIGHT = {"studied_stimulus": 0.60, "raga_category": 0.30, "project_source": 0.20}
RESEARCH_FIT_FLOOR = 0.30
RESEARCH_FIT_RANGE = 0.45
MAX_SOURCE_ONLY_RESEARCH = 2
RESEARCH_META_COLS = {"research_source": False, "evidence_level": "", "evidence_key": "", "source_url": "",
                      "source_name": "", "raga": "", "evidence_bonus": 0.0}
SCORE_COLS = ["rnn_score", "ncf_score", "personal_q", "pref_bias", "physio_fit", "psy_bias"]


def _norm(x):
    return re.sub(r"\s+", " ", str(x)).strip().casefold()


def _research_fit(stress, hr, dass_n, mood_state, genre_pref, era_pref):
    stress_n = float(np.clip(float(stress) / 100.0, 0, 1))
    hr_n = float(np.clip((float(hr) - 70.0) / 50.0, 0, 1))
    dass_n = float(np.clip(float(dass_n), 0, 1))
    mood_d = {"sad": 1.0, "angry": 1.0, "happy": 0.1, "calm": 0.2, "energetic": 0.2}.get(str(mood_state).lower(), 0.3)
    taste = 1.0 if "classical" in f"{genre_pref} {era_pref}".lower() else 0.0
    return 0.35 * stress_n + 0.25 * dass_n + 0.15 * hr_n + 0.15 * mood_d + 0.10 * taste


def _flagged_set(flagged):
    try:
        return {str(x) for x in flagged}
    except Exception:
        return set()


def _apply_research_evidence(pool, fit_inputs, flagged):
    pool = pool.copy().reset_index(drop=True)
    for c, d in RESEARCH_META_COLS.items():
        if c not in pool.columns:
            pool[c] = d
    pool["final_score"] = pd.to_numeric(pool["final_score"], errors="coerce")
    scores = pool["final_score"].dropna()
    spread = float(scores.max() - scores.min()) if len(scores) > 1 and scores.max() > scores.min() else 1.0
    median = float(scores.median()) if len(scores) else 0.0

    fit = _research_fit(**fit_inputs)
    strength = float(np.clip((fit - RESEARCH_FIT_FLOOR) / RESEARCH_FIT_RANGE, 0, 1))
    flagged = _flagged_set(flagged)

    song_n = pool["song"].map(_norm) if "song" in pool.columns else pd.Series([""] * len(pool))
    artist_n = pool["artist"].map(_norm) if "artist" in pool.columns else pd.Series([""] * len(pool))

    def tag(idx, level, key, raga, url, name):
        pool.at[idx, "research_source"] = True
        pool.at[idx, "evidence_level"] = level
        pool.at[idx, "evidence_key"] = key
        pool.at[idx, "raga"] = raga
        pool.at[idx, "source_url"] = url
        pool.at[idx, "source_name"] = name

    for cand in RESEARCH_CANDIDATES:
        m = (song_n == _norm(cand["song"])) & (artist_n == _norm(cand["artist"]))
        for idx in pool.index[m]:
            tag(idx, cand["evidence_level"], cand["evidence_key"], cand["raga"], cand["source_url"], cand["source_name"])

    for idx in pool.index[pool["evidence_level"] != ""]:
        if str(pool.at[idx, "song_id"]) in flagged:
            continue
        bonus = EVIDENCE_WEIGHT.get(pool.at[idx, "evidence_level"], 0.0) * strength * spread
        pool.at[idx, "evidence_bonus"] = bonus
        if pd.notna(pool.at[idx, "final_score"]):
            pool.at[idx, "final_score"] += bonus

    pool["research_source"] = pool["research_source"].fillna(False).astype(bool)
    return pool, fit, strength


def _select_recommendations(pool, n=5):
    pool = pool.copy()
    for _c, _d in RESEARCH_META_COLS.items():
        if _c not in pool.columns:
            pool[_c] = _d
    pool["research_source"] = pool["research_source"].fillna(False).astype(bool)
    try:
        chosen = rec.select_recommendations(pool, n=n)
        if chosen is None or len(chosen) == 0:
            raise ValueError("empty selection")
        chosen = chosen.reset_index(drop=True)
    except Exception:
        chosen = pool.sort_values("final_score", ascending=False).head(n).reset_index(drop=True)
    return chosen


USER_PAGES = [
    ("Dashboard", "🏠 Home"),
    ("Profile", "📝 1 · My Profile"),
    ("Physiological Input", "💓 2 · How I Feel"),
    ("Music Preference & Recommendation", "🎵 3 · My Music"),
]
USER_PAGE_KEYS = [k for k, _ in USER_PAGES]
ADMIN_PAGES = ["Psychological Assessment", "Evaluation & Validation", "Bias & Risk of Bias",
               "Research Evidence", "Dataset / Research Mode", "Backend Monitor (Admin)"]
ADMIN_NONE = "👤 Participant view (back)"
st.session_state.setdefault("admin_nav", ADMIN_NONE)


def _go_user_page(key):
    st.session_state["page"] = key
    st.session_state["admin_nav"] = ADMIN_NONE


def _go_admin_page():
    sel = st.session_state.get("admin_nav", ADMIN_NONE)
    st.session_state["page"] = sel if sel in ADMIN_PAGES else "Dashboard"


def render_user_nav():
    st.markdown("<div class='mus-nav-title'>👇 Where would you like to go?</div>", unsafe_allow_html=True)
    cols = st.columns(len(USER_PAGES))
    for i, (key, label) in enumerate(USER_PAGES):
        with cols[i]:
            st.button(label, key=f"unav_{i}", use_container_width=True,
                      type="primary" if st.session_state["page"] == key else "secondary",
                      on_click=_go_user_page, args=(key,))


with st.sidebar:
    st.markdown("<span class='mus-admin-tag'>🔬 RESEARCHER / ADMIN ONLY</span>", unsafe_allow_html=True)
    st.markdown(f"### 🎧 {APP_NAME}")
    mode_badge("MongoDB Atlas connected" if db else "MongoDB connection failed", "research" if db else "warning")
    st.divider()
    st.radio("Researcher / Admin pages", [ADMIN_NONE] + ADMIN_PAGES, key="admin_nav", on_change=_go_admin_page)

page = st.session_state["page"]


def safety_banner():
    bg = COLORS['neutral']
    accent = COLORS['accent']
    html = (f"<div style='background:{bg};border-left:4px solid {accent};"
            f"padding:8px 14px;border-radius:6px;font-size:0.85em;margin-bottom:12px;'>"
            f"⚠ {saf.SAFETY_DISCLAIMER}</div>")
    st.markdown(html, unsafe_allow_html=True)


if db and st.session_state.verified and page in USER_PAGE_KEYS and page != "Dashboard":
    render_user_nav()

# ================================================================
# PAGE: Dashboard / Login
# ================================================================
if page == "Dashboard":
    st.title(f"🎧 {APP_NAME}")
    st.caption(APP_TAGLINE)
    safety_banner()

    card_open()
    st.subheader("👋 Hi! What should we call you?")
    if not db:
        st.error("Sign-in is disabled until MongoDB is connected.")
    elif not st.session_state.verified:
        _render_login_flow()
    else:
        st.success(f"😊 Hello **{st.session_state.username}**!")
        render_user_nav()
        if st.button("🚪 Logout"):
            ok, msg = _mark_checkout("logout")
            if ok:
                _reset_session_state()
                st.rerun()
    card_close()

# ================================================================
# PAGE: Profile
# ================================================================
elif page == "Profile":
    name = st.session_state.username
    st.title("📝 Your Profile")
    safety_banner()

    if st.session_state.get("profile_user") != name:
        st.session_state["profile_user"] = name
        st.session_state["profile_doc"] = db.profiles.find_one({"user": name}, {"_id": 0})
        st.session_state["editing_profile"] = st.session_state["profile_doc"] is None

    profile_doc = st.session_state["profile_doc"]
    if profile_doc and not st.session_state["editing_profile"]:
        card_open()
        st.write(f"**Age:** {profile_doc.get('age')}")
        st.write(f"**Preferred Genre:** {profile_doc.get('genre_pref')}")
        st.write(f"**Preferred Vibe:** {profile_doc.get('era_pref')}")
        if st.button("✏️ Update Profile"):
            st.session_state["editing_profile"] = True
            st.rerun()
        card_close()

    if st.session_state["editing_profile"]:
        defaults = profile_doc or {}
        card_open()
        with st.form("profile_form"):
            age = st.number_input("Age", min_value=0, max_value=100, value=int(defaults.get("age", 18)), step=1)
            genre_pref = st.selectbox("Preferred Genre", ["Bollywood", "Hindi Pop", "Ghazal", "Classical"])
            era_pref = st.selectbox("Preferred Vibe", ["60s songs", "90s songs", "Energetic songs", "calming songs", "Classical songs"])
            
            st.subheader("🧩 TIPI (Big Five)")
            saved_tipi = defaults.get("tipi", [4] * len(psy.TIPI_ALL))
            tipi = []
            for i, q in enumerate(psy.TIPI_ALL):
                label = f"{q}\n\nमराठी: {TIPI_MARATHI[i]}" if i < len(TIPI_MARATHI) else q
                tipi.append(st.slider(label, 1, 7, int(saved_tipi[i]) if i < len(saved_tipi) else 4, key=f"baseline_tipi_{i}"))

            st.subheader("💭 DASS-21 (baseline)")
            saved_dass = defaults.get("dass", [1] * len(psy.DASS_ALL))
            dass = []
            for i, q in enumerate(psy.DASS_ALL):
                label = f"{q}\n\nमराठी: {DASS_MARATHI[i]}" if i < len(DASS_MARATHI) else q
                dass.append(st.slider(label, 0, 3, int(saved_dass[i]) if i < len(saved_dass) else 1, key=f"baseline_dass_{i}"))

            st.subheader("🌍 WHOQOL-BREF")
            saved_whoqol = defaults.get("whoqol", [3] * len(psy.WHOQOL_ALL))
            whoqol = []
            for i, q in enumerate(psy.WHOQOL_ALL):
                label = f"{q}\n\nमराठी: {WHOQOL_MARATHI[i]}" if i < len(WHOQOL_MARATHI) else q
                whoqol.append(st.slider(label, 1, 5, int(saved_whoqol[i]) if i < len(saved_whoqol) else 3, key=f"baseline_whoqol_{i}"))

            submitted = st.form_submit_button("💾 Save Profile")
        if submitted:
            prev_doc = profile_doc or {}
            answers_changed = (prev_doc.get("tipi") != tipi or prev_doc.get("dass") != dass
                               or prev_doc.get("whoqol") != whoqol)
            profile_data = {
                "name": st.session_state.get("username"), "age": int(age),
                "genre_pref": genre_pref, "era_pref": era_pref,
                "tipi": tipi, "dass": dass, "whoqol": whoqol,
                "updated_at_ist": _now_ist_str()
            }
            db.profiles.update_one({"user": name}, {"$set": profile_data}, upsert=True)
            merged_doc = {**prev_doc, **profile_data}
            if answers_changed:
                base = compute_baseline_stress(profile_data)
                merged_doc.update(_persist_stress(
                    name, base["score"], "baseline",
                    {"components": base["components"], "baseline_complete": base["complete"]}))
                st.session_state["stress_score"] = float(base["score"])
                st.session_state["stress_source"] = "baseline"
                st.session_state["stress_detail"] = {"previous": None, "updated": float(base["score"]),
                                                     "components": base["components"]}
                st.session_state["full_baseline_this_session"] = True
            st.session_state["profile_doc"] = merged_doc
            st.session_state["editing_profile"] = False
            st.success("Profile saved.")
            st.rerun()
        card_close()

# ================================================================
# PAGE: Physiological Input
# ================================================================
elif page == "Physiological Input":
    st.title("💓 How are you feeling right now?")
    safety_banner()

    name = st.session_state.username
    profile_doc = st.session_state.get("profile_doc") or db.profiles.find_one({"user": name}, {"_id": 0})
    if not profile_doc:
        st.warning("Please complete **📝 1 · My Profile** first.")
        st.button("Go to 📝 My Profile ➜", on_click=_go_user_page, args=("Profile",))
        st.stop()

    card_open()
    mood = st.selectbox("How is your mood?", ["Happy", "Sad", "Angry", "Calm", "Energetic"])
    hrv = st.slider("Heart rate (bpm)", 20, 200, 90)
    st.session_state["hrv_selfreport"] = hrv
    st.session_state["mood_selfreport"] = mood

    st.markdown("#### 🧠 Your stress score (calculated automatically)")
    if _needs_checkin():
        st.info("Welcome back! Answer these 8 short questions and your stress score will be updated automatically.")
        with st.form("checkin_form"):
            st.caption(SHORT_FORM_NOTE)
            phq4 = []
            for i, item in enumerate(PHQ4_ITEMS):
                phq4.append(st.radio(item, options=list(PHQ4_OPTIONS.keys()),
                                     format_func=lambda k: PHQ4_OPTIONS[k],
                                     index=None, horizontal=True, key=f"phq4_{i}"))
            pss4 = []
            for i, item in enumerate(PSS4_ITEMS):
                pss4.append(st.radio(item, options=list(PSS4_OPTIONS.keys()),
                                     format_func=lambda k: PSS4_OPTIONS[k],
                                     index=None, horizontal=True, key=f"pss4_{i}"))
            checkin_submitted = st.form_submit_button("✅ Save check-in & update my stress score")
        if checkin_submitted:
            if any(v is None for v in phq4 + pss4):
                st.error("Please answer all 8 questions.")
            else:
                _apply_checkin(name, profile_doc, phq4, pss4)
                st.rerun()
    else:
        stress_val, stress_src = _resolve_session_stress(profile_doc)
        detail = st.session_state.get("stress_detail") or {}
        delta = None
        if stress_src == "checkin" and detail.get("previous") is not None:
            delta = f"{detail['updated'] - detail['previous']:+.0f} vs last session"
        st.metric("Stress score", f"{stress_val:.0f} / 100", delta=delta, delta_color="inverse")
        st.write(f"**Level:** {_stress_band(stress_val)}")
    card_close()
    st.button("Next ➜ 🎵 Get my music", on_click=_go_user_page, args=("Music Preference & Recommendation",))

# ================================================================
# PAGE: Music Preference & Recommendation
# ================================================================
elif page == "Music Preference & Recommendation":
    st.title("🎵 Your Music & Experimental Pulse Profiles")
    safety_banner()

    profile_doc = st.session_state.get("profile_doc") or db.profiles.find_one({"user": st.session_state.username}, {"_id": 0})
    if not profile_doc:
        st.warning("Please complete **📝 1 · My Profile** first.")
        st.stop()

    name = st.session_state.username
    hrv = st.session_state.get("hrv_selfreport", 90)
    mood = st.session_state.get("mood_selfreport", "Calm")

    if _needs_checkin():
        st.warning("Please complete the short check-in on **💓 2 · How I Feel** first.")
        st.button("Go to check-in ➜", on_click=_go_user_page, args=("Physiological Input",))
        st.stop()
    stress_float, stress_src = _resolve_session_stress(profile_doc)
    stress = int(round(stress_float))
    st.caption(f"Stress score used for recommendations: **{stress}/100** ({_stress_band(stress)})")

    get_btn = st.button("🎧 Get Recommendations")

    if get_btn:
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                st.error("The music dataset is empty or could not be loaded.")
                st.stop()

            saved_tipi = profile_doc.get("tipi", []) or []
            saved_dass = profile_doc.get("dass", []) or []
            saved_whoqol = profile_doc.get("whoqol", []) or []

            tipi_answers = _valid_answers(saved_tipi, 10, 1, 7)
            dass_answers = _valid_answers(saved_dass, 21, 0, 3)
            whoqol_answers = _valid_answers(saved_whoqol, 26, 1, 5)

            extraversion = (tipi_answers[0] + (8 - tipi_answers[5])) / 2 if len(tipi_answers) >= 6 and tipi_answers[0] is not None and tipi_answers[5] is not None else 4.0
            openness = (tipi_answers[4] + (8 - tipi_answers[9])) / 2 if len(tipi_answers) >= 10 and tipi_answers[4] is not None and tipi_answers[9] is not None else 4.0

            dep_idx = [2, 4, 9, 12, 15, 16, 20]
            anx_idx = [1, 3, 6, 8, 14, 18, 19]
            stress_idx = [0, 5, 7, 10, 11, 13, 17]
            dass_complete = len(dass_answers) == 21 and all(v is not None for v in dass_answers)
            depression = float(sum(dass_answers[i] for i in dep_idx) * 2) if dass_complete else 0.0
            anxiety = float(sum(dass_answers[i] for i in anx_idx) * 2) if dass_complete else 0.0
            dass_stress = float(sum(dass_answers[i] for i in stress_idx) * 2) if dass_complete else 0.0

            whoqol_complete = len(whoqol_answers) == 26 and all(v is not None for v in whoqol_answers)
            if whoqol_complete:
                q = list(whoqol_answers)
                q[2], q[3] = 6 - q[2], 6 - q[3]
                physical_qol = float(np.mean([q[i] for i in [2, 3, 9, 14, 15, 16, 17]]) * 20)
                social_qol = float(np.mean([q[i] for i in [19, 20, 21]]) * 20)
            else:
                physical_qol = social_qol = 60.0

            tipi_n = float(np.clip(_mean_or(tipi_answers, 4.0) / 7.0, 0.0, 1.0))
            whoql_n = float(np.clip(_mean_or(whoqol_answers, 3.0) / 5.0, 0.0, 1.0))
            dass_n = float(np.clip((depression + anxiety + dass_stress) / 126.0, 0.0, 1.0)) if dass_complete else 0.0
            mood_n = float({"Sad": 0.0, "Angry": 0.25, "Energetic": 0.5, "Calm": 0.75, "Happy": 1.0}.get(mood, 0.5))

            pool = rec.build_candidate_pool(
                df, mood, hrv, stress,
                profile_doc.get("genre_pref", "Bollywood"),
                profile_doc.get("era_pref", "calming songs"),
                depression, anxiety, extraversion, physical_qol, social_qol
            )
            if pool is None or not isinstance(pool, pd.DataFrame) or pool.empty:
                st.warning("No candidate tracks were returned.")
                st.stop()

            score_context = {
                "mood_state": mood, "hrv": hrv, "stress": stress,
                "genre_pref": profile_doc.get("genre_pref", "Bollywood"),
                "era_pref": profile_doc.get("era_pref", "calming songs"),
                "extraversion": extraversion, "openness": openness,
                "depression": depression, "anxiety": anxiety,
                "physical_qol": physical_qol, "social_qol": social_qol,
                "tipi_n": tipi_n, "whoql_n": whoql_n, "dass_n": dass_n,
                "mood_n": mood_n, "user_name": name,
            }
            song_count = max(1, len(df))
            personal_q = np.zeros((100, song_count), dtype=float)
            global_q = np.zeros((100, song_count), dtype=float)

            original_song_ids = None
            if "song_id" in pool.columns and not pd.api.types.is_numeric_dtype(pool["song_id"]):
                original_song_ids = pool["song_id"].copy()
                unique_ids = list(dict.fromkeys(original_song_ids.astype(str).tolist()))
                numeric_id_map = {song_id: idx for idx, song_id in enumerate(unique_ids)}
                pool["song_id"] = original_song_ids.astype(str).map(numeric_id_map).astype(int)

            pool, _ = rec.score_pool(
                pool, score_context, personal_q, global_q,
                {"metadata": metadata, "rnn_model": rnn_model, "ncf_model": ncf_model},
                lambda: rec.FALLBACK_WEIGHTS, song_count
            )
            if original_song_ids is not None:
                pool["song_id"] = original_song_ids.values

            try:
                pool, _, _ = _apply_research_evidence(
                    pool,
                    {"stress": stress, "hr": hrv, "dass_n": dass_n, "mood_state": mood,
                     "genre_pref": score_context["genre_pref"], "era_pref": score_context["era_pref"]},
                    set(),
                )
            except Exception:
                pass

            chosen = _select_recommendations(pool, n=5)
            if chosen is None or chosen.empty:
                st.warning("No tracks selected.")
                st.stop()

            assessment_id = uuid.uuid4().hex
            st.session_state["assessment_id"] = assessment_id
            st.session_state["got_recs"] = True
            st.session_state["recs"] = chosen.to_dict("records")
            st.session_state["pool"] = pool
            st.session_state["es_analysis"] = {}
            st.session_state["pulse_results"] = {}
            st.session_state["auto_pulse_results"] = {}

            st.session_state["song_wavs"] = {}
            st.session_state["upload_pulse_results"] = {}

        except Exception as exc:
            st.error(f"Recommendation generation failed ({exc})")
            st.stop()

    if st.session_state.get("recs"):
        import html as _html
        st.markdown("""
<style>
.ms-head{display:flex;align-items:center;gap:12px;margin-bottom:4px}
.ms-rank{flex:0 0 38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  color:#fff;font-weight:800;background:linear-gradient(135deg,#7C3AED,#4361EE);box-shadow:0 4px 12px rgba(99,102,241,.35)}
.ms-title{font-size:1.28rem;font-weight:800;line-height:1.25}
.ms-artist{opacity:.72;font-size:.95rem}
.ms-chip{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.78rem;font-weight:600;margin:6px 6px 0 0;
  background:rgba(124,58,237,.10);color:#5B21B6;border:1px solid rgba(124,58,237,.25)}
.ms-steps{display:flex;gap:8px;margin:2px 0 12px}
.ms-step{flex:1;padding:9px 8px;border-radius:12px;font-size:.8rem;font-weight:700;text-align:center;
  border:1.5px solid rgba(128,128,128,.28);opacity:.55}
.ms-step.active{opacity:1;border-color:#7C3AED;background:rgba(124,58,237,.10);color:#5B21B6}
.ms-step.done{opacity:1;border-color:#10B981;background:rgba(16,185,129,.12);color:#047857}
.ms-tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:8px 0 14px}
.ms-tile{padding:14px 16px;border-radius:16px;background:rgba(255,255,255,.75);border:1px solid rgba(124,58,237,.18);
  box-shadow:0 2px 10px rgba(80,70,160,.08)}
.ms-tile .v{font-size:1.7rem;font-weight:800;color:#4C1D95;line-height:1.1}
.ms-tile .l{font-size:.78rem;font-weight:600;opacity:.7;margin-top:2px}
.ms-ok{padding:10px 14px;border-radius:12px;background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.35);
  color:#065F46;font-weight:600;margin-bottom:10px}
@media (max-width:700px){.ms-tiles{grid-template-columns:repeat(2,1fr)}}
</style>""", unsafe_allow_html=True)

        st.markdown("### 🎶 Your Recommended Tracks")
        st.caption("Follow the three steps on each track: **① create the WAV → ② upload it → ③ see its pulse**.")
        st.info("⚠️ Experimental software simulation — not validated for use on a person. For bench testing only.")
        LIMIT_HZ = MLMusicToSafeStimulation.SAFETY_MAX_FREQ

        def _tile(value, label):
            return f'<div class="ms-tile"><div class="v">{value}</div><div class="l">{label}</div></div>'

        for k_ in ("song_wavs", "upload_pulse_results"):
            if k_ not in st.session_state:
                st.session_state[k_] = {}

        for i, s in enumerate(st.session_state["recs"]):
            card_open()
            try:
                track_bpm = float(s.get("tempo", s.get("bpm", 95.0)))
            except (TypeError, ValueError):
                track_bpm = 110.0 if "energetic" in str(s.get("genre", "")).lower() else 75.0

            freq_hz, period_ms = pulse_hardware.calculate_pulse(track_bpm, stress)
            track_key = str(s.get("song_id", i))
            ukey = f"{st.session_state.get('assessment_id', 'na')}_{track_key}_{i}"
            song_title = f"{s.get('song', 'Track')} — {s.get('artist', '')}"
            safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(s.get('song', 'track')))[:40]
            known_dur = _get_song_duration_sec(s)

            # ---------- header ----------
            genres = [g.strip().title() for g in str(s.get("genre", "General")).split(",") if g.strip()][:3] or ["General"]
            chips = "".join(f'<span class="ms-chip">{_html.escape(g)}</span>' for g in genres)
            chips += f'<span class="ms-chip">🥁 {track_bpm:.0f} BPM</span>'
            if known_dur is not None:
                chips += f'<span class="ms-chip">⏱️ {int(known_dur // 60)}:{int(known_dur % 60):02d}</span>'
            st.markdown(
                f'<div class="ms-head"><div class="ms-rank">{i+1}</div><div>'
                f'<div class="ms-title">{_html.escape(str(s.get("song", "")))}</div>'
                f'<div class="ms-artist">{_html.escape(str(s.get("artist", "")))}</div></div></div>{chips}',
                unsafe_allow_html=True)

            col_info, col_work = st.columns([1, 1.5], gap="large")
            pkg_show = None

            with col_info:
                st.markdown("**Reference pulse for this track**")
                st.markdown(f'<div class="ms-tiles" style="grid-template-columns:repeat(2,1fr)">'
                            f'{_tile(f"{freq_hz} Hz", "Frequency")}{_tile(f"{period_ms} ms", "Period")}</div>',
                            unsafe_allow_html=True)
                st.caption(f"Hard safety limit: {LIMIT_HZ:.0f} Hz")
                st.markdown(f"[🎧 Open on Spotify]({spotify_link(s['song'], s['artist'])})")

            with col_work:
                if not ES_AVAILABLE:
                    st.error("Scientific libraries (scikit-learn/scipy/matplotlib) are missing.")
                    card_close()
                    continue

                wav_bytes_ = st.session_state["song_wavs"].get(ukey)
                up = st.file_uploader("Upload the song's WAV", type=["wav"], key=f"up_{ukey}",
                                      label_visibility="collapsed")
                pkg = st.session_state["upload_pulse_results"].get(ukey)
                done1 = wav_bytes_ is not None or up is not None
                done2 = up is not None
                done3 = bool(pkg and not pkg.get("error") and up is not None and pkg.get("file_sig") == (up.name, up.size))
                states = ["done" if done1 else "active",
                          "done" if done2 else ("active" if done1 else ""),
                          "done" if done3 else ("active" if done2 else "")]
                labels = ["① Get WAV (upload yours or create one)", "② Upload WAV", "③ See pulse"]
                st.markdown('<div class="ms-steps">' + "".join(
                    f'<div class="ms-step {c}">{("✓ " if c == "done" else "")}{l}</div>'
                    for c, l in zip(states, labels)) + '</div>', unsafe_allow_html=True)

                # Optional: stand-in WAV generator (only needed when you do not have the song's real WAV)
                if up is None:
                    if wav_bytes_ is None:
                        st.caption("Have the song's WAV? Upload it above. No WAV? Create a stand-in below "
                                   "(synthetic audio matched to this song's tempo, genre and length — not the real song).")
                        if known_dur is not None:
                            make_dur = known_dur
                            st.caption(f"The stand-in WAV will match this song's length ({known_dur / 60:.1f} min).")
                        else:
                            make_dur_min = st.number_input("Song length (minutes) — not in the dataset, please enter it",
                                                           min_value=0.5, max_value=20.0, value=3.0, step=0.5,
                                                           format="%.1f", key=f"dur_min_in_{ukey}")
                            make_dur = float(make_dur_min) * 60.0     # pipeline works in seconds internally
                        if st.button("🎼 Create stand-in WAV", key=f"mk_wav_{ukey}",
                                     use_container_width=True):
                            with st.spinner("Creating your WAV…"):
                                try:
                                    audio_ = _synth_proxy_audio(track_bpm, str(s.get("genre", "")), track_key, float(make_dur))
                                    st.session_state["song_wavs"][ukey] = _wav_bytes(audio_)
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"WAV creation failed: {exc}")
                    else:
                        c_a, c_b = st.columns([1.3, 1])
                        c_a.success("Stand-in WAV ready — download it, then upload it above 👆")
                        c_b.download_button("💾 Download WAV", wav_bytes_, file_name=f"{safe_name}.wav",
                                            mime="audio/wav", key=f"dl_song_{ukey}", use_container_width=True)
                else:
                    sig = (up.name, up.size)
                    if not pkg or pkg.get("file_sig") != sig:
                        with st.spinner("Listening to the song and building its pulse…"):
                            try:
                                res = _run_es_pipeline(track_key, track_bpm, str(s.get("genre", "")),
                                                       (up.getvalue(), up.name), known_dur, strict_upload=True)
                                pkg = _build_pulse_package(res, song_title)
                                pkg["known_duration"] = known_dur
                            except Exception as exc:
                                pkg = {"error": str(exc)}
                            pkg["file_sig"] = sig
                            st.session_state["upload_pulse_results"][ukey] = pkg
                        st.rerun()
                    pkg_show = pkg

            # ---------- result (full width) ----------
            if pkg_show is not None:
                st.markdown("---")
                if pkg_show.get("error"):
                    st.error(f"Pulse generation error: {pkg_show['error']}")
                else:
                    stt = pkg_show["stats"]
                    tab_o, tab_d, tab_f = st.tabs(["📊 Pulse overview", "🔬 Detailed view", "💾 Files (optional)"])
                    with tab_o:
                        st.markdown(
                            '<div class="ms-tiles">'
                            + _tile(f"{stt['duration_sec'] / 60:.1f} min", "Pulse length")
                            + _tile(f"{stt['avg_freq_hz']:.0f} Hz", "Average frequency")
                            + _tile(f"{stt['max_freq_hz']:.0f} Hz", "Peak frequency")
                            + _tile(f"{stt['pct_active']:.0f}%", "Time pulse is on")
                            + '</div>', unsafe_allow_html=True)
                        if stt["max_freq_hz"] <= LIMIT_HZ:
                            st.markdown(f'<div class="ms-ok">✅ Safe: peak {stt["max_freq_hz"]:.0f} Hz is within the '
                                        f'{LIMIT_HZ:.0f} Hz limit — pulse is exactly as long as your song.</div>',
                                        unsafe_allow_html=True)
                        st.progress(min(1.0, stt["max_freq_hz"] / LIMIT_HZ),
                                    text=f"Peak frequency: {stt['max_freq_hz']:.0f} of {LIMIT_HZ:.0f} Hz allowed")
                        kd = pkg_show.get("known_duration")
                        if kd is not None and abs(stt["duration_sec"] - kd) > 2.0:
                            st.warning(f"Your file is {stt['duration_sec'] / 60:.1f} min but the song's listed length is "
                                       f"{kd / 60:.1f} min — the pulse follows your file.")
                        st.image(pkg_show["hero_png"], use_container_width=True)
                        st.caption("🔊 Listen to the pulse rhythm (first 20 s). The real pulse is below hearing range, "
                                   "so this is an audible stand-in.")
                        st.audio(pkg_show["preview_wav"], format="audio/wav")
                    with tab_d:
                        st.caption("Zoom levels: single cycles (0.5 s) → first 10 s → whole song.")
                        st.image(pkg_show["viewer_png"], use_container_width=True)
                    with tab_f:
                        st.caption("Not needed to view the pulse — only if you want the files.")
                        d1, d2 = st.columns(2)
                        d1.download_button("💾 Pulse WAV", pkg_show["stim_wav"],
                                           file_name=f"{safe_name}_pulse_max{int(LIMIT_HZ)}Hz.wav",
                                           mime="audio/wav", key=f"dl_pulse_{ukey}", use_container_width=True)
                        d2.download_button("📄 Metadata (JSON)", pkg_show["json"],
                                           file_name=f"{safe_name}_pulse.json",
                                           mime="application/json", key=f"dl_json_{ukey}",
                                           use_container_width=True)

            card_close()

# ================================================================
# FALLBACK / OTHER PAGES
# ================================================================
else:
    st.title(f"📄 {page}")
    st.info("Section active.")