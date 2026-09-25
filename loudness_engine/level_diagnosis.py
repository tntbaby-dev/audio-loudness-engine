def diagnose_level(features, reference):
    """
    Quantify the mix's level relative to the reference profile.
    """

    lufs = features["integrated_lufs"]
    true_peak = features["true_peak_dbtp"]

    target_lufs = reference["integrated_lufs"]
    target_peak = reference["true_peak_dbtp"]

    # LUFS deviation from reference
    lufs_deviation = float(lufs - target_lufs)

    # Distance between current true peak and delivery ceiling
    peak_headroom = None

    if true_peak is not None:
        peak_headroom = float(target_peak - true_peak)

    # --------------------------------------------------
    # Level classification
    # --------------------------------------------------

    if lufs_deviation <= -6.0:
        level_status = "SUBSTANTIALLY_BELOW_REFERENCE"

    elif lufs_deviation < -2.0:
        level_status = "BELOW_REFERENCE"

    elif lufs_deviation > 2.0:
        level_status = "ABOVE_REFERENCE"

    else:
        level_status = "WITHIN_REFERENCE"

    return {
        "level_status": level_status,
        "lufs_deviation": round(lufs_deviation, 3),
        "peak_headroom_db": (
            None
            if peak_headroom is None
            else round(peak_headroom, 3)
        ),
    }
