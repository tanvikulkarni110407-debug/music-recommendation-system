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
    """
    MIN_FREQ_HZ = 20
    MAX_FREQ_HZ = 150  # Strict medical limit upper bound
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
        """BPM 60-180 -> 20-150 Hz (linear); stress > 60 -> x0.7; result strictly clamped to [20, 150] Hz."""
        bpm_val = self._to_float(bpm, 90.0)
        stress_val = self._to_float(stress_level, 40.0)
        bpm_val = min(max(bpm_val, self.MIN_BPM), self.MAX_BPM)
        span = self.max_freq - self.min_freq
        raw = self.min_freq + ((bpm_val - self.MIN_BPM) / (self.MAX_BPM - self.MIN_BPM)) * span
        if stress_val > self.HIGH_STRESS_THRESHOLD:
            raw *= self.HIGH_STRESS_SCALE
        return int(max(self.MIN_FREQ_HZ, min(self.MAX_FREQ_HZ, round(raw))))

    @staticmethod
    def period_ms(freq_hz):
        """Period (ms) = 1000 / Frequency (Hz)."""
        return round(1000.0 / freq_hz, 2) if freq_hz and freq_hz > 0 else 0.0

    def calculate_pulse(self, bpm, stress_level):
        freq = self.calculate_safe_frequency(bpm, stress_level)
        return freq, self.period_ms(freq)

    def send_pulse(self, bpm, stress_level):
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
        except Exception as exc:
            return freq, period, "simulation", (
                f"🧪 Safe simulation mode: {freq} Hz computed (hardware unreachable on {self.port}: {exc}).")


pulse_hardware = NodeMCUPulseGenerator(port="COM3", baudrate=115200)

# --------------------------------------------------------------
# SoundSync music-feature-driven electrical-stimulation ("pulse") research engine
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
except Exception as _es_exc:
    ES_AVAILABLE, ES_IMPORT_ERROR = False, str(_es_exc)

import tempfile

ES_SR = 44100            # sample rate of the generated signals
ES_CHUNK = 2048          # analysis chunk (~46 ms)
ES_FEATURE_NAMES = ["RMS", "SpectralCentroid", "LowRatio", "MidRatio", "HighRatio",
                    "DominantFreq", "MaxAmp", "EnergyRatio", "ConsecOnes", "CumCharge"]


class MLMusicToSafeStimulation:
    """Audio-feature-driven FM and stimulation-like research signals with conservative software guards."""
    SAFETY_MAX_AMP = 0.10
    SAFETY_MIN_AMP = 0.01
    SAFETY_MAX_FREQ = 150.0   # Strict medical limit: 150 Hz MAX
    SAFETY_MIN_FREQ = 20.0    # Strict medical limit: 20 Hz MIN
    SAFETY_MAX_CONSEC = 5
    SAFETY_CHARGE_LIM = 0.045
    BIN_THRESHOLD = 0.55
    DESAT_CHUNKS = 4

    def __init__(self, models, sample_rate=ES_SR, fm_f0=80, fm_f1=150, fm_max_amp=0.5, chunk_size=ES_CHUNK):
        self.m = models
        self.sr = sample_rate
        self.chunk = chunk_size
        self.fm_f0 = min(fm_f0, 150.0)
        self.fm_f1 = min(fm_f1, 150.0)
        self.fm_max_amp = fm_max_amp

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
        if consec >= self.SAFETY_MAX_CONSEC:
            return 0, 0, charge
        if charge > self.SAFETY_CHARGE_LIM:
            return 0, 0, 0.0
        if ml_bin == 1:
            return 1, consec + 1, charge + amp * (self.chunk / self.sr)
        return 0, 0, charge

    def process_segment(self, audio):
        """Derive pulse dynamically from the full audio signal across its entire length."""
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
        amp_arr = np.asarray(amps, dtype=float)
        fm_freq = np.where(bins == 1, self.fm_f1, self.fm_f0).astype(float)
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


def _synth_proxy_audio(bpm, genre_text, seed_text, seconds, sr=ES_SR):
    """Synthesize full-length audio matching exact track duration and genre parameters."""
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
        perc[s0:s0 + L] += accent * bar_gain[min(b // 4, len(bar_gain)-1)] * np.exp(-tt / 0.12) * np.sin(2 * np.pi * f * tt)
    sig = 0.5 * sig + 1.0 * perc + 0.003 * rng.standard_normal(n)
    target_rms = 0.035 if soft else 0.12
    sig = sig / (np.sqrt(np.mean(sig ** 2)) + 1e-9) * target_rms
    return np.clip(sig, -1, 1).astype(np.float32)


def _decode_uploaded_audio(raw, filename, seconds=None, sr=ES_SR):
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
        return y.astype(np.float32) if len(y) > 0 else None
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
        if seconds is not None:
            data = data[: int(seconds * sr)]
        return data.astype(np.float32) if len(data) > 0 else None
    except Exception:
        return None


def _get_song_duration_sec(song_record):
    for key in ("duration", "duration_sec", "duration_seconds", "length", "track_duration"):
        val = song_record.get(key)
        if val is not None:
            try:
                d = float(val)
                if np.isfinite(d) and d > 0:
                    if d > 1000:
                        d = d / 1000.0
                    return d
            except (TypeError, ValueError):
                continue
    return None


def _run_es_pipeline(track_key, track_bpm, genre_text, uploaded, dur_sec):
    conv = MLMusicToSafeStimulation({})
    dur_sec = max(1.0, float(dur_sec))
    base_audio, source = None, f"Full Song Audio ({dur_sec:.1f} s)"

    if uploaded is not None:
        file_bytes, filename = uploaded if isinstance(uploaded, tuple) else (uploaded, "uploaded.wav")
        base_audio = _decode_uploaded_audio(file_bytes, filename, seconds=dur_sec)
        if base_audio is not None:
            source = f"Uploaded audio track ({len(base_audio)/ES_SR:.1f} s): {filename}"

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


# --------------------------------------------------------------
# FULL 12 GRAPH PANELS FUNCTION
# --------------------------------------------------------------
def _render_analysis_png(res, music_name):
    """Render complete 12-panel analysis graph set for the song pulse."""
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
    fig.suptitle(f"Experimental Software Simulation Profile — {music_name} ({dur/60:.2f} min full length)\n"
                 f"Strict Medical Limit: 20–150 Hz  ·  Source: {res['source']}", fontsize=13, fontweight="bold")

    # Panel 1: Base Audio Clip
    ax = fig.add_subplot(3, 4, 1)
    ax.plot(t_audio, audio_p, "b-", alpha=0.6, lw=0.4)
    ax.set_title("1. Base Audio Clip Waveform"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(True)

    # Panel 2: FM Signal
    ax = fig.add_subplot(3, 4, 2)
    ax.plot(t_full, fm_p, "r-", alpha=0.8, lw=0.4)
    ax.set_title("2. FM Carrier Signal"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.grid(True)

    # Panel 3: ES Medical Pulse Signal
    ax = fig.add_subplot(3, 4, 3)
    ax.plot(t_full, stim_p, "g-", alpha=0.8, lw=0.4)
    ax.axhline(0.1, color="r", ls="--", lw=1.2, label="Safety Limit ±0.1"); ax.axhline(-0.1, color="r", ls="--", lw=1.2)
    ax.set_title("3. Medical ES Pulse Signal"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7); ax.grid(True)

    # Panel 4: Power Spectral Density
    ax = fig.add_subplot(3, 4, 4)
    ax.semilogy(ft["psd_frequencies"], np.maximum(ft["psd_power"], 1e-20), "b-", lw=1.2, alpha=0.7)
    ax.axvline(ft["psd_peak_frequency"], color="r", ls="--", alpha=0.8, label=f"Peak {ft['psd_peak_frequency']:.1f} Hz")
    ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("PSD"); ax.set_title("4. Power Spectral Density"); ax.legend(fontsize=7); ax.grid(True)

    # Panel 5: FM Binary Sequence
    ax = fig.add_subplot(3, 4, 5)
    ax.step(bin_time, fm_bin, "purple", where="post", lw=1.2)
    ax.fill_between(bin_time, fm_bin, step="post", alpha=0.25, color="purple")
    ax.set_title("5. FM Binary Activation"); ax.set_ylabel("Binary State"); ax.set_yticks([0, 1]); ax.grid(True)

    # Panel 6: ES Binary Sequence
    ax = fig.add_subplot(3, 4, 6)
    ax.step(bin_time, stim_bin, "orange", where="post", lw=1.2)
    ax.fill_between(bin_time, stim_bin, step="post", alpha=0.25, color="orange")
    ax.set_title("6. ES Pulse Activation"); ax.set_ylabel("Binary State"); ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1]); ax.grid(True)

    # Panel 7: FM vs ES Binary Comparison
    ax = fig.add_subplot(3, 4, 7)
    n = min(len(fm_bin), len(stim_bin))
    ax.step(bin_time[:n], fm_bin[:n], "purple", where="post", lw=1.2, label="FM", alpha=0.7)
    ax.step(bin_time[:n], stim_bin[:n], "orange", where="post", lw=1.2, label="ES", alpha=0.7)
    ax.set_title("7. FM vs ES Binary Comparison"); ax.set_ylabel("Binary State"); ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1]); ax.legend(fontsize=7); ax.grid(True)

    # Panel 8: Pulse Frequency Over Time
    ax = fig.add_subplot(3, 4, 8)
    ax.plot(bin_time, sp["frequencies"], "darkorange", lw=1.2)
    ax.axhline(20, color="r", ls="--", alpha=0.5, label="Min 20 Hz"); ax.axhline(150, color="r", ls="--", alpha=0.5, label="Max 150 Hz")
    ax.set_title("8. Pulse Frequency Over Time (20-150 Hz)"); ax.set_ylabel("Freq (Hz)"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.grid(True)

    # Panel 9: Pulse Map
    ax = fig.add_subplot(3, 4, 9)
    ax.bar(bin_time, stim_bin, width=chunk_dur * 0.8,
           color=["#FF6B6B" if b == 1 else "#4ECDC4" for b in stim_bin], alpha=0.85)
    ax.set_title("9. Detailed Pulse Burst Map"); ax.set_ylabel("Binary State"); ax.set_xlabel("Time (s)")
    ax.set_yticks([0, 1]); ax.grid(True, axis="y")

    # Panel 10: Cumulative Charge Accumulation
    ax = fig.add_subplot(3, 4, 10)
    cum = np.cumsum(np.array(sp["amplitudes"]) * stim_bin * chunk_dur)
    ax.plot(bin_time, cum, "teal", lw=1.5)
    ax.axhline(0.045, color="red", ls="--", lw=2, label="Safety limit (0.045)")
    ax.set_title("10. Cumulative Charge Accumulation"); ax.set_ylabel("Charge"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.grid(True)

    # Panel 11: Amplitude Over Time
    ax = fig.add_subplot(3, 4, 11)
    ax.plot(bin_time, sp["amplitudes"], "#9D4EDD", lw=1.2, label="Amplitude")
    ax.axhline(0.1, color="r", ls="--", lw=1.5, label="Max 0.1"); ax.axhline(0.01, color="b", ls="--", lw=1.0, label="Min 0.01")
    ax.fill_between(bin_time, sp["amplitudes"], alpha=0.25, color="#9D4EDD")
    ax.set_title("11. Pulse Amplitude Envelope"); ax.set_ylabel("Amplitude"); ax.set_xlabel("Time (s)")
    ax.legend(fontsize=7); ax.grid(True)

    # Panel 12: Spectral Band Energy
    ax = fig.add_subplot(3, 4, 12)
    bands = ["sub_bass", "bass", "low_mid", "mid", "high_mid", "treble"]
    energies = [ft.get(f"{b}_energy", 0) for b in bands]
    bars = ax.bar(bands, energies, color=["#FF6B6B", "#FF9F1C", "#4ECDC4", "#9D4EDD", "#4361EE", "#06D6A0"],
                  edgecolor="black", alpha=0.85)
    for bar, e in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{e:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("12. Spectral Band Energy"); ax.set_ylabel("Mean Energy"); ax.set_xlabel("Band")
    ax.tick_params(axis="x", rotation=30); ax.grid(True, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _fig_png(fig)


def _render_viewer_png(res):
    SR, dur, stim = ES_SR, res["dur_sec"], res["stim_out"]
    fig = Figure(figsize=(16, 11))
    fig.suptitle(f"Experimental Pulse Viewer ({dur/60:.2f} min total full length) — Strict Medical Limit (20–150 Hz)", fontsize=14, fontweight="bold")
    ax = fig.add_subplot(3, 1, 1)
    z1 = min(int(SR * 0.5), len(stim)); t1 = np.linspace(0, z1 / SR * 1000, z1); s1 = stim[:z1]
    ax.plot(t1, s1, "mediumseagreen", lw=1.2)
    ax.fill_between(t1, s1, 0, where=(s1 > 0), alpha=0.4, color="mediumseagreen")
    ax.fill_between(t1, s1, 0, where=(s1 < 0), alpha=0.4, color="tomato")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("First 500 ms — Individual Pulse Cycles")
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(3, 1, 2)
    z2 = min(int(SR * 10), len(stim)); t2 = np.linspace(0, z2 / SR, z2)
    ax.plot(t2, stim[:z2], "steelblue", lw=0.7)
    ax.axhline(0.1, color="red", ls="--", lw=1.2, label="Safety Limit ±0.1"); ax.axhline(-0.1, color="red", ls="--", lw=1.2)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("First 10 Seconds Segment")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(3, 1, 3)
    s3 = _ds(stim, 200_000); t3 = np.linspace(0, dur / 60, len(s3))
    ax.plot(t3, s3, "#9D4EDD", lw=0.4, alpha=0.7); ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Full Duration Song Overview — {dur/60:.2f} min ({dur:.1f} seconds)")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _fig_png(fig, dpi=90)


def _pulse_stats(res):
    sp = res["stim_params"]
    desat_ms = MLMusicToSafeStimulation.DESAT_CHUNKS * ES_CHUNK / ES_SR * 1000
    return {"duration_sec": res["dur_sec"], "avg_freq_hz": float(np.mean(sp["frequencies"])),
            "avg_amp": float(np.mean(sp["amplitudes"])), "pct_active": float(np.mean(sp["binary_sequence"]) * 100),
            "desat_ms": float(desat_ms), "bpm": float(res["features"]["bpm"]),
            "psd_peak_hz": float(res["features"]["psd_peak_frequency"])}


def _wav_bytes(sig):
    buf = io.BytesIO()
    sp_wavfile.write(buf, ES_SR, (np.clip(sig, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()


def _build_pulse_package(res, music_name):
    stats = _pulse_stats(res)
    meta = {"music_name": music_name, "source": res["source"], "full_duration_sec": res["dur_sec"],
            "sample_rate": ES_SR, "signal_type": "Medical Limit Pulse Waveform",
            "safety_standard": "Strict Medical Limit Enforced: 20-150 Hz",
            "freq_range_Hz": [20, 150], "amplitude_range": [0.01, 0.1],
            "avg_frequency_Hz": stats["avg_freq_hz"], "avg_amplitude": stats["avg_amp"],
            "pct_active": stats["pct_active"]}
    return {
        "stats": stats,
        "analysis_png": _render_analysis_png(res, music_name),  # Full 12 Graphs
        "viewer_png": _render_viewer_png(res),                # 3-Zoom Pulse Viewer
        "stim_wav": _wav_bytes(res["stim_out"]),
        "fm_wav": _wav_bytes(res["fm_out"]),
        "audio_wav": _wav_bytes(res["base_audio"]),
        "json": json.dumps(meta, indent=2).encode("utf-8")
    }

# --------------------------------------------------------------
# App Core Setup
# --------------------------------------------------------------
_STATE_DEFAULTS = [
    ("verified", False), ("username", None), ("user_email", None),
    ("editing_profile", False), ("profile_doc", None), ("profile_user", None),
    ("recs", []), ("got_recs", False), ("pool", pd.DataFrame()),
    ("session_number", 1), ("session_finished", False),
    ("page", "Dashboard"),
    ("full_baseline_this_session", False),
    ("session_id", None),
    ("assessment_id", None),
    ("admin_ok", False),
    ("login_flash", None),
    ("auto_pulse_results", {}), ("uploaded_pulse_result", None),
    ("stress_score", None), ("stress_source", None), ("stress_detail", None),
    ("checkin_done", False),
]
for key, default in _STATE_DEFAULTS:
    if key not in st.session_state:
        st.session_state[key] = default


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
            raise RuntimeError("MONGODB_URI missing.")
        self.client = MongoClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=20000)
        self.client.admin.command("ping")
        self.mongo_db = self.client[db_name]
        self.profiles = self.mongo_db["user_profiles"]
        self.users = self.mongo_db["users"]
        self.sessions = self.mongo_db["sessions"]


@st.cache_resource(show_spinner=False)
def _connect_mongodb(uri, db_name):
    return MongoDBStore(uri=uri, db_name=db_name)


try:
    db = _connect_mongodb(_get_secret("MONGODB_URI"), _get_secret("MONGODB_DATABASE", "musync"))
except Exception as e:
    db = None


@st.cache_data(show_spinner=False)
def _load_dataset():
    return ds.load_dataset()


@st.cache_resource(show_spinner=False)
def _load_models():
    return mdl.load_models()


df, is_demo_dataset, dataset_note = _load_dataset()
metadata, rnn_model, ncf_model, model_error = _load_models()


def spotify_link(song, artist):
    return f"https://open.spotify.com/search/{urllib.parse.quote_plus(f'{song} {artist}')}"


def _go_user_page(key):
    st.session_state["page"] = key


def safety_banner():
    # Fixed f-string backslash syntax error
    bg_color = COLORS['neutral']
    accent_color = COLORS['accent']
    st.markdown(
        f"<div style='background:{bg_color};border-left:4px solid {accent_color};"
        f"padding:8px 14px;border-radius:6px;font-size:0.85em;margin-bottom:12px;'>"
        f"⚠ {saf.SAFETY_DISCLAIMER}</div>",
        unsafe_allow_html=True
    )


# ================================================================
# PAGE: Music Preference & Recommendation
# ================================================================
if st.session_state["page"] == "Music Preference & Recommendation":
    st.title("🎵 Recommendation & Full-Length Song Pulse Generator")
    safety_banner()

    profile_doc = st.session_state.get("profile_doc") or (db.profiles.find_one({"user": st.session_state.username}, {"_id": 0}) if db else {})
    name = st.session_state.username
    hrv = st.session_state.get("hrv_selfreport", 90)
    mood = st.session_state.get("mood_selfreport", "Calm")
    stress = int(round(st.session_state.get("stress_score", 40)))

    st.caption(f"User: **{name}** | Current Stress Level: **{stress}/100**")
    get_btn = st.button("🎧 Get Recommendations", type="primary")

    if get_btn:
        try:
            pool = rec.build_candidate_pool(
                df, mood, hrv, stress,
                profile_doc.get("genre_pref", "Bollywood"),
                profile_doc.get("era_pref", "calming songs"),
                0, 0, 4.0, 60.0, 60.0
            )
            chosen = rec.select_recommendations(pool, n=5).reset_index(drop=True)
            st.session_state["assessment_id"] = uuid.uuid4().hex
            st.session_state["recs"] = chosen.to_dict("records")
            st.session_state["auto_pulse_results"] = {}
            st.session_state["uploaded_pulse_result"] = None
        except Exception as exc:
            st.error(f"Error fetching recommendations: {exc}")

    if st.session_state.get("recs"):
        st.markdown("### 🎶 Recommended Tracks & Full-Length Pulse Signals")
        st.info("ℹ️ **Strict Medical Safety Limits Active (20 Hz - 150 Hz)**. Each generated pulse strictly matches the full duration of its corresponding song.")

        for i, s in enumerate(st.session_state["recs"]):
            card_open()
            col_info, col_pulse_ui = st.columns([1.2, 1])

            track_bpm = float(s.get("tempo", s.get("bpm", 95.0))) if "tempo" in s or "bpm" in s else 90.0
            freq_hz, period_ms = pulse_hardware.calculate_pulse(track_bpm, stress)
            track_key = str(s.get("song_id", i))
            ukey = f"{st.session_state.get('assessment_id', 'na')}_{track_key}_{i}"
            song_title = f"{s.get('song', 'Track')} — {s.get('artist', '')}"
            known_dur = _get_song_duration_sec(s) or 180.0  # Default to full 3.0 min (180s) if unlisted

            with col_info:
                st.markdown(f"#### {i+1}. {s['song']} — {s['artist']}")
                st.caption(f"Genre: {s.get('genre', 'General')} | Tempo: {track_bpm:.0f} BPM")
                st.caption(f"⏱️ Full Song Duration: **{known_dur:.1f} seconds ({known_dur/60:.2f} min)**")
                st.markdown(f"[🎧 Open on Spotify]({spotify_link(s['song'], s['artist'])})")

            with col_pulse_ui:
                st.markdown("**⚙️ Medical Pulse Generation**")
                m1, m2 = st.columns(2)
                m1.metric("Calculated Pulse Freq", f"{freq_hz} Hz", help="Safe Limit: 20-150 Hz")
                m2.metric("Computed Period", f"{period_ms} ms")

                create_btn = st.button(f"🎵 Generate Full Song & Matching Pulse ({known_dur/60:.1f} min)", key=f"create_wav_{ukey}")

                if create_btn and ES_AVAILABLE:
                    with st.spinner(f"Generating full {known_dur/60:.2f} min matching pulse for {s['song']}..."):
                        try:
                            res = _run_es_pipeline(track_key, track_bpm, str(s.get("genre", "")), None, known_dur)
                            pkg = _build_pulse_package(res, song_title)
                            st.session_state["auto_pulse_results"][ukey] = pkg
                        except Exception as exc:
                            st.session_state["auto_pulse_results"][ukey] = {"error": str(exc)}

                pkg = st.session_state["auto_pulse_results"].get(ukey)
                if pkg and not pkg.get("error"):
                    st.success(f"✅ Generated! Pulse matching full duration of {known_dur/60:.2f} min created.")
                    safe_name = re.sub(r'[^A-Za-z0-9_-]+', '_', str(s.get('song', 'track')))[:40]

                    c_dl1, c_dl2 = st.columns(2)
                    with c_dl1:
                        st.download_button(
                            "💾 Download Song Audio (.wav)",
                            pkg["audio_wav"],
                            file_name=f"{safe_name}_full_song.wav",
                            mime="audio/wav",
                            key=f"dl_song_{ukey}",
                            use_container_width=True
                        )
                    with c_dl2:
                        st.download_button(
                            "⚡ Download Matching Pulse (.wav)",
                            pkg["stim_wav"],
                            file_name=f"{safe_name}_medical_pulse_150hz.wav",
                            mime="audio/wav",
                            key=f"dl_pulse_{ukey}",
                            use_container_width=True
                        )

                    with st.expander("📊 View All 12 Full Signal Graphs & 3-Zoom Viewer"):
                        st.image(pkg["analysis_png"], use_container_width=True)
                        st.image(pkg["viewer_png"], use_container_width=True)

                elif pkg and pkg.get("error"):
                    st.error(f"Generation error: {pkg['error']}")

            card_close()

        # --------------------------------------------------------------
        # SECTION: Upload Song WAV for Exact Full-Length Pulse
        # --------------------------------------------------------------
        st.divider()
        card_open()
        st.subheader("📤 Upload Audio File to Get Exact Same Length Medical Pulse")
        st.write("Upload any song file (`.wav`, `.mp3`) to derive a safe medical pulse (**20–150 Hz**) matching its **exact total runtime**.")

        uploaded_file = st.file_uploader("Choose a Song File", type=["wav", "mp3", "ogg", "flac"], key="custom_song_uploader")

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.read()
            if st.button("⚡ Process Audio & Generate Full-Length Matching Pulse", type="primary"):
                with st.spinner("Decoding song length and synthesizing matching length pulse under strict 150 Hz limit..."):
                    try:
                        full_audio = _decode_uploaded_audio(uploaded_bytes, uploaded_file.name, seconds=None)
                        if full_audio is None or len(full_audio) == 0:
                            st.error("Invalid audio format.")
                        else:
                            song_dur = len(full_audio) / ES_SR
                            res = _run_es_pipeline(uploaded_file.name, 90.0, "uploaded", (uploaded_bytes, uploaded_file.name), song_dur)
                            pkg = _build_pulse_package(res, uploaded_file.name)
                            pkg["song_dur"] = song_dur
                            st.session_state["uploaded_pulse_result"] = pkg
                    except Exception as exc:
                        st.error(f"Processing error: {exc}")

        up_pkg = st.session_state.get("uploaded_pulse_result")
        if up_pkg:
            st.success(f"✅ Generated! Medical pulse matches exact song length: **{up_pkg['song_dur']:.2f} seconds** ({up_pkg['song_dur']/60:.2f} minutes).")
            
            p1, p2, p3 = st.columns(3)
            p1.metric("Average Pulse Frequency", f"{up_pkg['stats']['avg_freq_hz']:.1f} Hz")
            p2.metric("Peak PSD Frequency", f"{up_pkg['stats']['psd_peak_hz']:.1f} Hz")
            p3.metric("Medical Freq Range", "20 Hz – 150 Hz Max")

            st.download_button(
                "💾 Download Exact Matching-Length Pulse (.wav)",
                up_pkg["stim_wav"],
                file_name=f"matching_pulse_{int(up_pkg['song_dur'])}s.wav",
                mime="audio/wav",
                use_container_width=True
            )

            with st.expander("📊 View All 12 Full Signal Analysis Graphs & 3-Zoom Viewer"):
                st.image(up_pkg["analysis_png"], use_container_width=True)
                st.image(up_pkg["viewer_png"], use_container_width=True)

        card_close()

# Fallback UI rendering for other pages
else:
    st.title("🎧 MuSync System Ready")
    st.button("Go to Recommendations Page ➜", on_click=_go_user_page, args=("Music Preference & Recommendation",))