from loudness_engine.level_diagnosis import diagnose_level


def diagnose(features, reference):
    """
    Interpret multiple audio measurements together.
    """

    diagnoses = []

    lra = features["loudness_range"]
    crest = features["crest_factor_db"]
    true_peak = features["true_peak_dbtp"]

    level = diagnose_level(features, reference)

    # --------------------------------------------------
    # Overall level + dynamics relationship
    # --------------------------------------------------

    if level["level_status"] == "SUBSTANTIALLY_BELOW_REFERENCE":

        if crest > reference["crest_max"]:
            diagnoses.append({
                "category": "dynamics",
                "priority": "HIGH",
                "diagnosis": "Low loudness is accompanied by high crest factor.",
                "interpretation": (
                    "The mix has substantial peak-to-average variation. "
                    "Investigate transient and sustained energy before "
                    "using compression or limiting to increase loudness."
                )
            })

        else:
            diagnoses.append({
                "category": "overall_level",
                "priority": "HIGH",
                "diagnosis": "Mix level is substantially below the reference.",
                "interpretation": (
                    "Integrated loudness is significantly below target "
                    "while crest factor is not unusually high. "
                    "Investigate gain staging and overall signal level "
                    "before making dynamics corrections."
                )
            })

    elif level["level_status"] == "BELOW_REFERENCE":

        diagnoses.append({
            "category": "overall_level",
            "priority": "MEDIUM",
            "diagnosis": "Mix level is below the reference.",
            "interpretation": (
                "Integrated loudness is below target. "
                "Review overall level in relation to the mix's dynamics."
            )
        })

    # --------------------------------------------------
    # Dynamics
    # --------------------------------------------------

    if crest > reference["crest_max"]:
        diagnoses.append({
            "category": "dynamics",
            "priority": "HIGH",
            "diagnosis": "High crest factor detected.",
            "interpretation": (
                "Peak energy is substantially greater than average energy. "
                "Investigate transient-to-sustain balance."
            )
        })

    elif crest < reference["crest_min"]:
        diagnoses.append({
            "category": "dynamics",
            "priority": "MEDIUM",
            "diagnosis": "Low crest factor detected.",
            "interpretation": (
                "Peak and average energy are relatively close. "
                "Check for excessive compression or limiting."
            )
        })

    # --------------------------------------------------
    # Loudness range
    # --------------------------------------------------

    if lra > reference["lra_max"]:
        diagnoses.append({
            "category": "loudness_range",
            "priority": "MEDIUM",
            "diagnosis": "Large loudness range detected.",
            "interpretation": (
                "Different sections of the mix vary substantially in loudness."
            )
        })

    elif lra < reference["lra_min"]:
        diagnoses.append({
            "category": "loudness_range",
            "priority": "MEDIUM",
            "diagnosis": "Very small loudness range detected.",
            "interpretation": (
                "Section-to-section loudness variation is limited."
            )
        })

    # --------------------------------------------------
    # True peak
    # --------------------------------------------------

    if true_peak is not None and true_peak > reference["true_peak_dbtp"]:
        diagnoses.append({
            "category": "true_peak",
            "priority": "HIGH",
            "diagnosis": "True peak exceeds the delivery target.",
            "interpretation": (
                "Peak level should be controlled before final delivery."
            )
        })

    return diagnoses
