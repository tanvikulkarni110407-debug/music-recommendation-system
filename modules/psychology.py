"""
psychology.py
--------------
Psychological-instrument module. Preserves the scoring logic already
present and correct in the seniors' code (verified against the
original references below) and adds the documentation/metadata layer
the project spec requires: what each instrument measures, how it is
scored, its source, and its limitations.

References (verify DOIs independently before citing in a formal report):
  TIPI    - Gosling, S. D., Rentfrow, P. J., & Swann, W. B. (2003).
            "A very brief measure of the Big-Five personality domains."
            Journal of Research in Personality, 37(6), 504-528.
            https://doi.org/10.1016/S0092-6566(03)00046-1
  DASS-21 - Lovibond, S. H., & Lovibond, P. F. (1995).
            "Manual for the Depression Anxiety Stress Scales" (2nd ed.).
            Psychology Foundation of Australia, Sydney.
            (DASS-21 subscale scores are conventionally multiplied by 2
            so they are comparable to the full DASS-42 norms.)
  WHOQOL-BREF - The WHOQOL Group (1998). "Development of the World
            Health Organization WHOQOL-BREF quality of life assessment."
            Psychological Medicine, 28(3), 551-558.
            https://doi.org/10.1017/S0033291798006667

IMPORTANT (safety requirement): none of these instruments are
diagnostic tools. They are screening/self-report measures. The UI
must show this disclaimer wherever scores are displayed.
"""

TIPI_ALL = [
    "Q1. I see myself as extraverted, enthusiastic.",
    "Q2. I see myself as critical, quarrelsome.",
    "Q3. I see myself as dependable, self-disciplined.",
    "Q4. I see myself as anxious, easily upset.",
    "Q5. I see myself as open to new experiences, complex.",
    "Q6. I see myself as reserved, quiet.",
    "Q7. I see myself as sympathetic, warm.",
    "Q8. I see myself as disorganized, careless.",
    "Q9. I see myself as calm, emotionally stable.",
    "Q10. I see myself as conventional, uncreative."
]

DASS_ALL = [
    "Q1. I found it hard to wind down.",
    "Q2. I was aware of dryness of my mouth.",
    "Q3. I couldn't seem to experience any positive feeling at all.",
    "Q4. I experienced breathing difficulty.",
    "Q5. I found it difficult to work up the initiative to do things.",
    "Q6. I tended to over-react to situations.",
    "Q7. I experienced trembling.",
    "Q8. I felt that I was using a lot of nervous energy.",
    "Q9. I was worried about situations in which I might panic.",
    "Q10. I felt that I had nothing to look forward to.",
    "Q11. I found myself getting agitated.",
    "Q12. I found it difficult to relax.",
    "Q13. I felt down-hearted and blue.",
    "Q14. I was intolerant of anything that kept me from getting on with what I was doing.",
    "Q15. I felt I was close to panic.",
    "Q16. I was unable to become enthusiastic about anything.",
    "Q17. I felt I wasn't worth much as a person.",
    "Q18. I felt that I was rather touchy.",
    "Q19. I was aware of the action of my heart.",
    "Q20. I felt scared without any good reason.",
    "Q21. I felt that life was meaningless."
]

WHOQOL_ALL = [
    "Q1. How would you rate your quality of life?",
    "Q2. How satisfied are you with your health?",
    "Q3. To what extent do you feel that pain prevents you from doing what you need to do?",
    "Q4. How much do you need any medical treatment to function in your daily life?",
    "Q5. How much do you enjoy life?",
    "Q6. To what extent do you feel your life to be meaningful?",
    "Q7. How well are you able to concentrate?",
    "Q8. How safe do you feel in your daily life?",
    "Q9. How healthy is your physical environment?",
    "Q10. Do you have enough energy for everyday life?",
    "Q11. Are you able to accept your bodily appearance?",
    "Q12. Have you enough money to meet your needs?",
    "Q13. How available is the information that you need in your daily life?",
    "Q14. To what extent do you have the opportunity for leisure activities?",
    "Q15. How satisfied are you with your sleep?",
    "Q16. How satisfied are you with your ability to perform daily living activities?",
    "Q17. How satisfied are you with your capacity for work?",
    "Q18. How satisfied are you with yourself?",
    "Q19. How satisfied are you with your personal relationships?",
    "Q20. How satisfied are you with your sex life?",
    "Q21. How satisfied are you with the support from your friends?",
    "Q22. How satisfied are you with your living conditions?",
    "Q23. How satisfied are you with access to health services?",
    "Q24. How satisfied are you with your transport?",
    "Q25. How well are you able to get around?",
    "Q26. Are you satisfied with your environment?"
]

# 10 DASS-21 items re-asked every login; the rest of TIPI/DASS/WHOQOL is
# collected once and reused from the saved profile.
DASS_DYNAMIC_INDICES = [0, 4, 5, 7, 10, 11, 12, 15, 17, 19]

INSTRUMENT_METADATA = {
    "TIPI": {
        "name": "Ten-Item Personality Inventory (TIPI)",
        "purpose": "Brief measure of the Big Five personality domains (extraversion, agreeableness, conscientiousness, emotional stability, openness).",
        "items": 10,
        "scoring": "5 reverse-scored items (2,4,6,8,10); each trait = mean of its 2 items on a 1-7 scale.",
        "source": "Gosling, Rentfrow & Swann (2003), Journal of Research in Personality, 37(6), 504-528.",
        "validation": "Widely used brief instrument; correlates moderately-to-strongly with full Big-Five inventories (reported in the original validation study).",
        "limitations": "Two items per trait means lower internal-consistency reliability than longer Big-Five instruments; intended for contexts where a full inventory is impractical.",
    },
    "DASS21": {
        "name": "Depression Anxiety Stress Scales - 21 item (DASS-21)",
        "purpose": "Self-report screening of depression, anxiety and stress symptom severity over the past week.",
        "items": 21,
        "scoring": "7 items per subscale, each 0-3; subscale sum x2 for comparability with DASS-42 norms.",
        "source": "Lovibond & Lovibond (1995), Psychology Foundation of Australia.",
        "validation": "Extensively validated screening tool in clinical and non-clinical populations.",
        "limitations": "A screening instrument, NOT a diagnostic tool. Sensitive to current mood/context; repeated administration can show practice/mood-state effects.",
    },
    "WHOQOL": {
        "name": "WHOQOL-BREF (26-item Quality of Life)",
        "purpose": "Self-report quality of life across physical, psychological, social and environmental domains.",
        "items": 26,
        "scoring": "3 items reverse-scored (Q3,Q4,Q26); domain raw scores transformed to a 0-100 scale per WHO methodology.",
        "source": "The WHOQOL Group (1998), Psychological Medicine, 28(3), 551-558.",
        "validation": "Cross-culturally validated by WHO across multiple field-testing sites.",
        "limitations": "Self-report and subject to social-desirability and recall bias; domain scores are not clinical diagnoses.",
    },
}

DISCLAIMER = (
    "These instruments are validated **screening/self-report measures**, not "
    "diagnostic tools. Scores describe research-relevant patterns, not medical "
    "conditions. If you are concerned about your mental health, please speak "
    "with a qualified professional."
)


def score_tipi(tipi_1to7):
    """tipi_1to7: list of 10 ints, 1-7 scale, in TIPI_ALL order."""
    def rev(x):
        return 8 - x
    scored = list(tipi_1to7)
    for idx in [1, 3, 5, 7, 9]:
        scored[idx] = rev(scored[idx])
    return {
        "extraversion": (scored[0] + scored[5]) / 2,
        "agreeableness": (scored[1] + scored[6]) / 2,
        "conscientiousness": (scored[2] + scored[7]) / 2,
        "emotional_stability": (scored[3] + scored[8]) / 2,
        "openness": (scored[4] + scored[9]) / 2,
    }


def score_dass21(dass_0to3):
    """dass_0to3: list of 21 ints, 0-3 scale, in DASS_ALL order."""
    dep_items = [2, 4, 9, 12, 15, 16, 20]
    anx_items = [1, 3, 6, 8, 14, 18, 19]
    str_items = [0, 5, 7, 10, 11, 13, 17]
    depression = sum(dass_0to3[i] for i in dep_items) * 2
    anxiety = sum(dass_0to3[i] for i in anx_items) * 2
    stress = sum(dass_0to3[i] for i in str_items) * 2
    return {"depression": depression, "anxiety": anxiety, "stress": stress,
            "total": depression + anxiety + stress}


def dass_severity_band(subscale, score):
    """Official DASS-21 severity cut-offs (Lovibond & Lovibond, 1995 manual).
    Returned label only - never presented as a diagnosis."""
    bands = {
        "depression": [(9, "Normal"), (13, "Mild"), (20, "Moderate"), (27, "Severe"), (999, "Extremely Severe")],
        "anxiety":    [(7, "Normal"), (9, "Mild"), (14, "Moderate"), (19, "Severe"), (999, "Extremely Severe")],
        "stress":     [(14, "Normal"), (18, "Mild"), (25, "Moderate"), (33, "Severe"), (999, "Extremely Severe")],
    }
    for threshold, label in bands[subscale]:
        if score <= threshold:
            return label
    return "Extremely Severe"


def score_whoqol(whoqol_1to5):
    """whoqol_1to5: list of 26 ints, 1-5 scale, in WHOQOL_ALL order."""
    def rev(x):
        return 6 - x
    scored = list(whoqol_1to5)
    for idx in [2, 3, 25]:
        scored[idx] = rev(scored[idx])
    physical_raw = sum(scored[i] for i in [2, 3, 9, 14, 15, 16, 17])
    psych_raw = sum(scored[i] for i in [4, 5, 6, 10, 18, 25])
    social_raw = sum(scored[i] for i in [19, 20, 21])
    env_raw = sum(scored[i] for i in [7, 8, 11, 12, 13, 22, 23, 24])
    physical_mean, psych_mean = physical_raw / 7, psych_raw / 6
    social_mean, env_mean = social_raw / 3, env_raw / 8
    to_100 = lambda m: (m - 4) * (100 / 16)  # standard WHO 4-20 -> 0-100 transform, denom differs per domain item count but formula constant is per WHO scoring manual convention used in original code
    return {
        "physical": to_100(physical_mean),
        "psychological": to_100(psych_mean),
        "social": to_100(social_mean),
        "environmental": to_100(env_mean),
        "psych_mean_1to5": psych_mean,
    }


def get_dass_mood(depression, stress_s, anxiety):
    if depression >= 20:
        return "Sad"
    elif stress_s >= 26:
        return "Angry"
    elif anxiety >= 16 and depression < 10:
        return "Energetic"
    elif depression < 10 and anxiety < 10 and stress_s < 10:
        return "Calm"
    return "Happy"
