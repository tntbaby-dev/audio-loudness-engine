from pathlib import Path

import numpy as np
import soundfile as sf

from loudness_engine.analyzer import analyze_file
from loudness_engine.true_peak import measure_true_peak


def dbfs(value: float) -> float:
    """
    Convert linear amplitude to dBFS.
    """
    return 20.0 * np.log10(max(value, 1e-12))


def extract_features(file_path):
    """
    Extract core audio features from one file.
    """
    file_path = Path(file_path)

    data, sample_rate = sf.read(file_path, always_2d=True)
    data = data.astype(np.float32)

    mono = np.mean(data, axis=1)

    duration_sec = len(mono) / sample_rate

    rms = float(np.sqrt(np.mean(mono ** 2)))
    rms_dbfs = float(dbfs(rms))

    peak_linear = float(np.max(np.abs(mono)))
    peak_dbfs = float(dbfs(peak_linear))

    crest_factor_db = float(peak_dbfs - rms_dbfs)

    zero_crossings = np.where(np.diff(np.signbit(mono)))[0]
    zero_crossing_rate = float(len(zero_crossings) / max(len(mono), 1))

    loudness_info = analyze_file(file_path)
    true_peak = measure_true_peak(file_path)

    return {
        "file_path": str(file_path),
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "duration_sec": round(float(duration_sec), 3),
        "integrated_lufs": loudness_info["integrated_lufs"],
        "loudness_range": loudness_info["loudness_range"],
        "sample_peak_dbfs": round(peak_dbfs, 3),
        "true_peak_dbtp": None if true_peak is None else round(float(true_peak), 3),
        "rms_dbfs": round(rms_dbfs, 3),
        "crest_factor_db": round(crest_factor_db, 3),
        "zero_crossing_rate": round(zero_crossing_rate, 6),
    }