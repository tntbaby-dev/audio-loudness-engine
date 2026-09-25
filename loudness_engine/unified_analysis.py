from typing import Any, Dict


def build_unified_analysis(
    file_path: str,
    sample_rate: int,
    channels: int,
    duration_sec: float,
    integrated_lufs: float,
    loudness_range: float,
    true_peak_db: float,
    energy_map: Dict[str, Any],
    dynamics: Dict[str, Any],
    transients: Dict[str, Any],
    frequency_dynamics: Dict[str, Any],
    tonal_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble all audio analysis layers into one consistent structure.

    This function only organizes existing measurements.
    It does not modify, normalize, average, or reinterpret them.
    """

    return {
        "schema_version": "1.0",

        "metadata": {
            "file_path": file_path,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_sec": duration_sec,
        },

        "loudness": {
            "integrated_lufs": integrated_lufs,
            "loudness_range": loudness_range,
            "true_peak_db": true_peak_db,
        },

        "energy": energy_map,

        "dynamics": dynamics,

        "transients": transients,

        "frequency_dynamics": frequency_dynamics,

        "tonal_analysis": tonal_analysis,
    }