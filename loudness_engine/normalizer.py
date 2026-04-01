from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def dbfs_from_peak(peak_linear: float) -> float:
    return 20.0 * np.log10(max(peak_linear, 1e-12))


def peak_from_dbfs(dbfs: float) -> float:
    return 10.0 ** (dbfs / 20.0)


def normalize_file(file_path, output_path, target_lufs=-16.0, peak_ceiling_dbfs=-1.0):
    """
    Normalize one audio file to target LUFS while preserving original channel structure
    and respecting a peak ceiling.
    """
    data, sample_rate = sf.read(file_path, always_2d=True)
    data = data.astype(np.float32)

    # Mono analysis signal only
    mono = np.mean(data, axis=1)

    meter = pyln.Meter(sample_rate)
    original_lufs = float(meter.integrated_loudness(mono))

    # Compute gain needed in dB, then convert to linear
    gain_db = target_lufs - original_lufs
    gain_linear = 10.0 ** (gain_db / 20.0)

    # Apply loudness gain to original signal (preserve stereo)
    output_audio = data * gain_linear

    # Check peak after gain
    output_peak_linear = float(np.max(np.abs(output_audio)))
    output_peak_dbfs = float(dbfs_from_peak(output_peak_linear))

    clipping_risk = output_peak_dbfs > peak_ceiling_dbfs

    # If too high, scale down to ceiling
    if clipping_risk:
        ceiling_linear = peak_from_dbfs(peak_ceiling_dbfs)
        safety_scale = ceiling_linear / max(output_peak_linear, 1e-12)
        output_audio = output_audio * safety_scale

    # Re-measure final loudness using mono fold-down
    final_mono = np.mean(output_audio, axis=1)
    final_lufs = float(meter.integrated_loudness(final_mono))

    final_peak_linear = float(np.max(np.abs(output_audio)))
    final_peak_dbfs = float(dbfs_from_peak(final_peak_linear))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output_path, output_audio, sample_rate)

    return {
        "file_path": str(file_path),
        "output_path": str(output_path),
        "original_lufs": round(original_lufs, 3),
        "target_lufs": round(float(target_lufs), 3),
        "final_lufs": round(final_lufs, 3),
        "peak_ceiling_dbfs": round(float(peak_ceiling_dbfs), 3),
        "final_peak_dbfs": round(final_peak_dbfs, 3),
        "clipping_risk_detected": bool(clipping_risk),
        "channels": int(data.shape[1]),
    }
from pathlib import Path
import pandas as pd


def normalize_folder(input_dir, output_dir, target_lufs=-16.0, peak_ceiling_dbfs=-1.0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    results = []

    for file_path in input_dir.glob("*.wav"):
        output_path = output_dir / file_path.name

        result = normalize_file(
            file_path,
            output_path,
            target_lufs=target_lufs,
            peak_ceiling_dbfs=peak_ceiling_dbfs,
        )

        results.append(result)

    df = pd.DataFrame(results)

    report_path = output_dir / "normalization_report.csv"
    df.to_csv(report_path, index=False)

    return df