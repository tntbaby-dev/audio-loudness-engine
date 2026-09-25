import soundfile as sf
import pyloudnorm as pyln

from .true_peak import measure_true_peak
from .energy_map import analyze_energy_map
from loudness_engine.dynamics import analyze_dynamics
from loudness_engine.transients import analyze_transients
from loudness_engine.frequency_dynamics import analyze_frequency_dynamics
from loudness_engine.tonal_analysis import analyze_tonal_analysis

def analyze_file(file_path):
    """
    Analyze one audio file and return loudness,
    true-peak, and spectral energy metrics.
    """

    data, sample_rate = sf.read(
        file_path,
        always_2d=True
    )

    duration_sec = len(data) / float(sample_rate)
    channels = int(data.shape[1])

    # Preserve original channel layout for loudness measurement.
    # pyloudnorm expects samples x channels.
    loudness_data = data

    meter = pyln.Meter(sample_rate)

    integrated_lufs = float(
        meter.integrated_loudness(loudness_data)
    )

    loudness_range = float(
        meter.loudness_range(loudness_data)
    )

    # FFmpeg-based true-peak analyzer operates on the original file.
    true_peak_db = float(
        measure_true_peak(file_path)
    )

    # Stereo-preserving spectral energy analysis.
    energy_map = analyze_energy_map(
        data,
        sample_rate
    )

    dynamics = analyze_dynamics(
        data,
        sample_rate
    )
    transients = analyze_transients(data, sample_rate)
    frequency_dynamics = analyze_frequency_dynamics(data, sample_rate)
    tonal_analysis = analyze_tonal_analysis(data, sample_rate)
    return {
        "file_path": str(file_path),
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_sec": round(duration_sec, 3),
        "integrated_lufs": round(integrated_lufs, 3),
        "loudness_range": round(loudness_range, 3),
        "true_peak_db": round(true_peak_db, 3),
        "energy_map": energy_map,
        "dynamics": dynamics,
        "transients": transients,
        "frequency_dynamics": frequency_dynamics,
        "tonal_analysis": tonal_analysis,
    }