def build_report(features, diagnoses, recommendations):
    """
    Build a structured production report from
    measurements, diagnoses, and recommendations.
    """

    return {
        "file": features["file_path"],

        "measurements": {
            "integrated_lufs": features["integrated_lufs"],
            "loudness_range": features["loudness_range"],
            "sample_peak_dbfs": features["sample_peak_dbfs"],
            "true_peak_dbtp": features["true_peak_dbtp"],
            "rms_dbfs": features["rms_dbfs"],
            "crest_factor_db": features["crest_factor_db"],
            "zero_crossing_rate": features["zero_crossing_rate"],
        },

        "diagnoses": diagnoses,

        "recommendations": recommendations,
    }
