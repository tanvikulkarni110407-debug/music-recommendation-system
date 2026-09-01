"""
bias.py
-------
Risk-of-Bias (RoB) module. Per project requirement, this is a
STRUCTURED CHECKLIST that a researcher fills in about the current
study/deployment configuration, not an automatically computed "bias
score" (which would itself be an unsupported claim).

Methodology basis: adapted from the domain structure of PROBAST
(Prediction model Risk Of Bias ASsessment Tool - Wolff et al., 2019,
Annals of Internal Medicine, "PROBAST: A Tool to Assess the Risk of
Bias and Applicability of Prediction Model Studies") for participant
selection, predictor measurement, outcome measurement, and analysis
domains, extended with the music-recommendation-specific bias sources
your mentor asked for (psychological/questionnaire, physiological
measurement, dataset, music-preference, and model bias).

Each domain records: risk level (Low/Some concerns/High/Not assessed),
justification text, and mitigation notes. The report is generated from
what the user actually enters - it is never pre-filled with a "Low"
default.
"""

BIAS_DOMAINS = {
    "questionnaire": {
        "title": "A. Psychological / Questionnaire Bias",
        "prompts": [
            "Response bias (social desirability, acquiescence)",
            "Misunderstanding of items (translation/literacy effects)",
            "Repeated-testing effects (practice, fatigue on re-administration)",
            "Environmental/timing effects at time of assessment",
        ],
    },
    "physiological": {
        "title": "B. Physiological Measurement Bias",
        "prompts": [
            "Device/sensor error or miscalibration",
            "Sensor placement inconsistency",
            "Motion artefacts in signal",
            "Signal quality / sampling rate limitations",
        ],
    },
    "dataset": {
        "title": "C. Dataset Bias",
        "prompts": [
            "Small sample size relative to population of interest",
            "Demographic imbalance (age/gender/culture)",
            "Class imbalance in labels",
            "Laboratory-induced stressors vs real-world stress",
        ],
    },
    "preference": {
        "title": "D. Music Preference Bias",
        "prompts": [
            "Age-related preference skew",
            "Cultural/linguistic background skew",
            "Familiarity/prior-exposure effects",
            "Environment at time of listening",
        ],
    },
    "model": {
        "title": "E. Model Bias",
        "prompts": [
            "Overfitting to training users/songs",
            "Data leakage between train/test",
            "Subject leakage (same user's data in both splits)",
            "Poor generalization to unseen users/songs (cold start)",
        ],
    },
}

RISK_LEVELS = ["Not assessed", "Low", "Some concerns", "High"]


def blank_assessment():
    """Structure to fill in via the UI; every field starts unassessed -
    nothing is pre-scored as 'Low risk' by default."""
    return {
        domain: {"risk": "Not assessed", "justification": "", "mitigation": ""}
        for domain in BIAS_DOMAINS
    }


def summarize(assessment):
    """Simple tally - NOT a weighted 'bias score'. Just a count per level,
    for the report table."""
    counts = {lvl: 0 for lvl in RISK_LEVELS}
    for domain in assessment.values():
        counts[domain.get("risk", "Not assessed")] += 1
    return counts
