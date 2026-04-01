from pathlib import Path
import soundfile as sf
import numpy as np
import pyloudnorm as pyln

from loudness_engine.true_peak import measure_true_peak
from loudness_engine.gain_predictor import predict_safe_gain


def intelligent_normalize(file_path, output_path, target_lufs=-16, peak_ceiling=-1):

    file_path = Path(file_path)

    audio, sample_rate = sf.read(file_path, always_2d=True)
    audio = audio.astype(np.float32)

    mono = np.mean(audio, axis=1)

    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(mono)

    current_true_peak = measure_true_peak(file_path)

    gain_info = predict_safe_gain(
        current_lufs=current_lufs,
        current_true_peak=current_true_peak,
        target_lufs=target_lufs
    )

    gain_db = gain_info["safe_gain"]

    gain_linear = 10 ** (gain_db / 20)

    normalized_audio = audio * gain_linear

    peak = np.max(np.abs(normalized_audio))
    peak_db = 20 * np.log10(max(peak, 1e-12))

    if peak_db > peak_ceiling:
        ceiling_linear = 10 ** (peak_ceiling / 20)
        normalized_audio *= ceiling_linear / peak

    sf.write(output_path, normalized_audio, sample_rate)

    return {
        "input_file": str(file_path),
        "output_file": str(output_path),
        "original_lufs": round(current_lufs, 3),
        "true_peak": round(current_true_peak, 3),
        "safe_gain_applied": round(gain_db, 3)
    }