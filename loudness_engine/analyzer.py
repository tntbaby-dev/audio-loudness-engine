import numpy as np
import soundfile as sf
import pyloudnorm as pyln


def analyze_file(file_path):
    """
    Analyze one audio file and return loudness metrics.
    """
    data, sample_rate = sf.read(file_path, always_2d=True)

    # Convert to mono by averaging channels
    mono = np.mean(data, axis=1).astype(np.float32)

    duration_sec = len(mono) / float(sample_rate)

    meter = pyln.Meter(sample_rate)

    integrated_lufs = float(meter.integrated_loudness(mono))
    loudness_range = float(meter.loudness_range(mono))

    peak_linear = float(np.max(np.abs(mono)))
    peak_dbfs = float(20.0 * np.log10(max(peak_linear, 1e-12)))

    return {
        "file_path": str(file_path),
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "duration_sec": round(duration_sec, 3),
        "integrated_lufs": round(integrated_lufs, 3),
        "loudness_range": round(loudness_range, 3),
        "peak_dbfs": round(peak_dbfs, 3),
    }