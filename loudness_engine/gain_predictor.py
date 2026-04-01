import numpy as np


def predict_safe_gain(current_lufs, current_true_peak, target_lufs=-16.0, peak_ceiling=-1.0):
    """
    Predict the maximum safe gain that reaches target LUFS
    without exceeding the peak ceiling.
    """

    loudness_gain = target_lufs - current_lufs
    peak_headroom = peak_ceiling - current_true_peak

    safe_gain = min(loudness_gain, peak_headroom)

    return {
        "loudness_gain_needed": round(loudness_gain, 3),
        "peak_headroom": round(peak_headroom, 3),
        "safe_gain": round(safe_gain, 3)
    }