"""
evidence.py
-----------
Static, human-curated evidence base for the "why this playlist"
explanation layer. Every entry below was checked against a real
bibliographic record before inclusion (titles, venues, years, and DOIs
verified via literature search on 2026-08-08). Still verify
independently before citing in a formal submission - this is a
software project's best-effort curation, not a systematic review.

Evidence categories are kept distinct per project requirement:
  A = evidence for a specific song            (NONE claimed here)
  B = evidence for a music characteristic      (tempo/mode/valence/arousal)
  C = evidence for a genre                     (used cautiously, see notes)
  D = evidence for a general music-response relationship
No entry in this file claims a specific song "treats" or "cures" anything.
"""

REFERENCES = {
    "gomez_danuser_2007": {
        "citation": "Gomez, P., & Danuser, B. (2007). Relationships between musical structure and psychophysiological measures of emotion. Emotion, 7(2), 377-387.",
        "doi": "10.1037/1528-3542.7.2.377",
        "studied": "11 structural features of 16 musical excerpts vs. self-reported valence/arousal and physiological measures (respiration, skin conductance, heart rate) in listeners.",
        "finding": "Mode, harmonic complexity and rhythmic articulation best distinguished negative vs. positive valence; tempo, accentuation and rhythmic articulation best distinguished high vs. low arousal.",
        "evidence_type": "B - music characteristic -> emotion/physiology relationship (not a specific song, not a specific genre).",
    },
    "trappe_2010": {
        "citation": "Trappe, H. J. (2010). The effects of music on the cardiovascular system and cardiovascular health. Heart, 96(23), 1868-1871.",
        "doi": "10.1136/hrt.2010.209858",
        "studied": "Narrative review of music-listening studies and their effects on heart rate, blood pressure, and cardiovascular autonomic measures; includes the author's own controlled comparisons of music genres (e.g. classical vs. heavy metal) on HR/BP.",
        "finding": "Reports that calmer, lower-tempo music genres were associated with smaller/more favorable HR and BP changes than high-tempo, high-intensity genres in the studies reviewed.",
        "evidence_type": "C/D - genre-level and general relationship; review-level evidence, not a randomized trial of your specific catalog.",
    },
    "russell_1980": {
        "citation": "Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161-1178.",
        "doi": "10.1037/h0077714",
        "studied": "Foundational psychological model placing emotions on two axes: valence (pleasant-unpleasant) and arousal (activated-deactivated).",
        "finding": "Established the valence-arousal framework this project uses to describe both the user's mood state and each song's audio features.",
        "evidence_type": "D - general theoretical framework underlying the whole valence/arousal design of this system.",
    },
    "schmidt_2018_wesad": {
        "citation": "Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction (ICMI), 400-408.",
        "doi": "10.1145/3242969.3242985",
        "studied": "15 subjects; chest (RespiBAN) and wrist (Empatica E4) physiological signals (ECG, EDA, EMG, respiration, temperature, BVP, acceleration) recorded during neutral, stress and amusement conditions.",
        "finding": "Provides labeled physiological data suitable for stress/affect classification and physiological feature-extraction research (used here for the HRV pipeline design).",
        "evidence_type": "Dataset - not a specific-song study.",
    },
    "zhang_2018_pmemo": {
        "citation": "Zhang, K., Zhang, H., Li, S., Yang, C., & Sun, L. (2018). The PMEmo dataset for music emotion recognition. Proceedings of the 2018 ACM International Conference on Multimedia Retrieval (ICMR), 135-142.",
        "doi": "10.1145/3206025.3206037",
        "studied": "794 songs with continuous/overall valence-arousal annotations from 457 subjects, plus simultaneous EDA signals for a subset of listeners.",
        "finding": "Links specific music/audio features to both emotion annotations and physiological (EDA) response, at the song level - the closest dataset in this project's evidence base to 'song-level' evidence, though limited to EDA, not HRV.",
        "evidence_type": "Dataset - song-level emotion + EDA evidence; does NOT include HRV.",
    },
    "koelstra_2012_deap": {
        "citation": "Koelstra, S., Muhl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., & Patras, I. (2012). DEAP: A database for emotion analysis using physiological signals. IEEE Transactions on Affective Computing, 3(1), 18-31.",
        "doi": "10.1109/T-AFFC.2011.15",
        "studied": "32 participants, EEG and peripheral physiological signals recorded while watching 40 one-minute music-video excerpts, with valence/arousal/dominance/liking ratings.",
        "finding": "Provides physiological-response labels for music-video stimuli; used here only as a secondary reference dataset for arousal/valence-physiology mapping, not merged directly with WESAD or PMEmo.",
        "evidence_type": "Dataset - music-video stimuli, not audio-only tracks; signals not directly comparable to WESAD's stress-induction protocol.",
    },
}

# Maps a mood/state target to the audio-feature direction it implies, each
# tagged with which reference(s) support that characteristic-level claim.
CHARACTERISTIC_EVIDENCE_MAP = {
    "Calm / low-arousal target": {
        "audio_characteristics": "Lower tempo, lower energy/intensity, consonant harmony",
        "supported_by": ["gomez_danuser_2007", "trappe_2010", "russell_1980"],
        "claim_level": "B/D - characteristic and general relationship level. NOT a claim that any specific track in the catalog has been clinically tested.",
    },
    "Positive valence target": {
        "audio_characteristics": "Major mode, lower harmonic complexity, smoother rhythmic articulation",
        "supported_by": ["gomez_danuser_2007", "russell_1980"],
        "claim_level": "B - characteristic level.",
    },
    "High-arousal / energetic target": {
        "audio_characteristics": "Higher tempo, stronger accentuation, more rhythmic articulation",
        "supported_by": ["gomez_danuser_2007", "russell_1980"],
        "claim_level": "B - characteristic level.",
    },
}

DISCLAIMER = (
    "Evidence links are shown at the level the underlying research actually "
    "supports (a music characteristic, a genre trend, or a general "
    "relationship) - never as proof that a specific catalog track has "
    "itself been clinically tested, and never as a therapeutic guarantee."
)
