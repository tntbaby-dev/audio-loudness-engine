import soundfile as sf
import pyloudnorm as pyln

from .true_peak import measure_true_peak


def analyze_file(file_path):
    """
    Analyze one audio file and return loudness metrics.
    """
    data, sample_rate = sf.read(file_path, always_2d=True)

    duration_sec = len(data) / float(sample_rate)

    # Preserve the original channel layout for BS.1770 loudness measurement.
    meter = pyln.Meter(sample_rate)

    integrated_lufs = float(meter.integrated_loudness(data))
    loudness_range = float(meter.loudness_range(data))

    # Use the existing true-peak measurement implementation.
    true_peak_db = float(measure_true_peak(file_path))

    return {
        "file_path": str(file_path),
        "sample_rate": int(sample_rate),
        "channels": int(data.shape[1]),
        "duration_sec": round(duration_sec, 3),
        "integrated_lufs": round(integrated_lufs, 3),
        "loudness_range": round(loudness_range, 3),
        "true_peak_db": round(true_peak_db, 3),
    }