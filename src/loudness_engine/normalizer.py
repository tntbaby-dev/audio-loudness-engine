from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


def dbfs_from_peak(peak_linear: float) -> float:
    """
    Convert linear peak value to dBFS.
    """
    return 20.0 * np.log10(max(peak_linear, 1e-12))


def peak_from_dbfs(dbfs: float) -> float:
    """
    Convert dBFS to linear peak value.
    """
    return 10.0 ** (dbfs / 20.0)


def normalize_file(file_path, output_path, target_lufs=-16.0, peak_ceiling_dbfs=-1.0):
    """
    Normalize one audio file to a target LUFS while respecting a peak ceiling.
    This is safer than naive loudness normalization because it checks for clipping risk.
    """
    data, sample_rate = sf.read(file_path, always_2d=True)

    # Keep channel count
    channels = data.shape[1]

    # Create mono analysis signal
    mono = np.mean(data, axis=1).astype(np.float32)

    meter = pyln.Meter(sample_rate)
    original_lufs = float(meter.integrated_loudness(mono))

    # First pass: loudness normalization on mono
    normalized_mono = pyln.normalize.loudness(mono, original_lufs, target_lufs)

    # Measure resulting peak
    output_peak_linear = float(np.max(np.abs(normalized_mono)))
    output_peak_dbfs = float(dbfs_from_peak(output_peak_linear))

    applied_lufs_target = float(target_lufs)
    clipping_risk = output_peak_dbfs > peak_ceiling_dbfs

    # If peak exceeds ceiling, scale down
    if clipping_risk:
        ceiling_linear = peak_from_dbfs(peak_ceiling_dbfs)
        scale = ceiling_linear / max(output_peak_linear, 1e-12)
        normalized_mono = normalized_mono * scale

    # Re-measure after safety scaling
    final_peak_linear = float(np.max(np.abs(normalized_mono)))
    final_peak_dbfs = float(dbfs_from_peak(final_peak_linear))
    final_lufs = float(meter.integrated_loudness(normalized_mono))

    # Rebuild output channels
    if channels == 1:
        output_audio = normalized_mono
    else:
        output_audio = np.column_stack([normalized_mono] * channels)

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