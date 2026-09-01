"""
dataset.py
----------
Loads the real song catalog (Music_dataset2.csv, same format the
seniors' code expects) if present. If it is not present, generates a
small, clearly-labeled DEMO catalog so the app is still runnable for a
walkthrough - this demo catalog is never presented as real audio-feature
data; every place it's used shows the "DEMO DATA" badge.
"""

import os
import numpy as np
import pandas as pd
from .config import DATASET_PATH

REQUIRED_COLS_NOTE = "song, artist, genre, valence, energy, tempo, year"


def load_dataset():
    """Returns (df, is_demo: bool, note: str)."""
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns={
            "track name": "song", "track": "song",
            "artist name(s)": "artist", "artist name": "artist", "artists": "artist",
            "genres": "genre", "release date": "release_date", "release_date": "release_date",
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
        df.dropna(subset=[c for c in ["song", "artist"] if c in df.columns], inplace=True)
        for col in ["valence", "energy", "tempo"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5)
                if col != "tempo":
                    df[col] = df[col].clip(0.0, 1.0)
            else:
                df[col] = 0.5
        df.reset_index(drop=True, inplace=True)
        df["song_id"] = df.index.astype(int)
        return df, False, f"Loaded {len(df)} songs from Music_dataset2.csv"

    # ---- DEMO fallback (clearly labeled everywhere it's used) ----
    demo_rows = [
        ("Kal Ho Naa Ho", "Sonu Nigam", "Bollywood", 0.55, 0.35, 78, 2003),
        ("Tum Hi Ho", "Arijit Singh", "Bollywood", 0.30, 0.25, 68, 2013),
        ("Zinda", "Siddharth Mahadevan", "Bollywood", 0.75, 0.85, 140, 2011),
        ("Raabta", "Arijit Singh", "Bollywood", 0.60, 0.40, 90, 2017),
        ("Kabira", "Tochi Raina", "Bollywood", 0.45, 0.30, 84, 2013),
        ("Malang", "Ved Sharma", "Hindi Pop", 0.80, 0.78, 128, 2020),
        ("Kesariya", "Arijit Singh", "Hindi Pop", 0.55, 0.35, 96, 2022),
        ("Ghungroo", "Arijit Singh", "Hindi Pop", 0.85, 0.82, 122, 2019),
        ("Raga Yaman", "Ravi Shankar", "Classical", 0.50, 0.15, 60, 1968),
        ("Raga Bhairavi", "Ali Akbar Khan", "Classical", 0.40, 0.20, 55, 1972),
        ("Chalte Chalte", "Kishore Kumar", "Ghazal", 0.35, 0.20, 65, 1976),
        ("Chupke Chupke", "Jagjit Singh", "Ghazal", 0.45, 0.22, 62, 1982),
    ]
    df = pd.DataFrame(demo_rows, columns=["song", "artist", "genre", "valence", "energy", "tempo", "year"])
    df["song_id"] = df.index.astype(int)
    df["vibe"] = "Neutral"
    note = ("DATASET NOT FOUND: Music_dataset2.csv is missing. Using a small built-in "
            "12-song DEMO catalog so the app remains runnable. This is NOT real audio-feature "
            "data and must not be used for any research claim - place Music_dataset2.csv in "
            "the project root to use the real catalog.")
    return df, True, note
