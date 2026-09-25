import numpy as np
import soundfile as sf


# TNT descriptive frequency bands.
# These are descriptive engineering regions,
# not claims that each frequency range has one
# fixed perceptual function.

ENERGY_BANDS = [
    ("sub", 20, 60, "rumble"),
    ("bass", 60, 120, "weight"),
    ("low_mid", 120, 250, "warmth"),
    ("mid_low", 250, 500, "mud_boxiness"),
    ("mid", 500, 1000, "body_nasal"),
    ("high_mid", 1000, 2000, "intelligibility"),
    ("presence", 2000, 5000, "presence"),
    ("bright", 5000, 8000, "sibilance"),
    ("brilliance", 8000, 12000, "brightness"),
    ("air", 12000, 20000, "air"),
]


def _frame_audio(data, frame_size, hop_size):
    """Create overlapping audio frames."""

    if data.ndim == 1:
        data = data[:, None]

    length = data.shape[0]

    if length < frame_size:
        padding = frame_size - length
        data = np.pad(
            data,
            ((0, padding), (0, 0)),
            mode="constant",
        )
        length = frame_size

    frame_count = 1 + (length - frame_size) // hop_size

    frames = np.zeros(
        (frame_count, frame_size, data.shape[1]),
        dtype=np.float64,
    )

    for i in range(frame_count):
        start = i * hop_size
        end = start + frame_size
        frames[i] = data[start:end]

    return frames


def analyze_energy(
    file_path,
    frame_size=4096,
    hop_size=1024,
):
    """
    Analyze frequency energy distribution.

    Returns:
        Dictionary containing:
        - sample rate
        - duration
        - total analyzed energy
        - energy distribution across TNT bands
    """

    data, sample_rate = sf.read(
        file_path,
        always_2d=True,
        dtype="float64",
    )

    duration_sec = len(data) / float(sample_rate)

    # Preserve stereo/multichannel information during FFT.
    frames = _frame_audio(
        data,
        frame_size,
        hop_size,
    )

    # Hann window reduces spectral leakage.
    window = np.hanning(frame_size)

    windowed = frames * window[None, :, None]

    # FFT for every frame and channel.
    spectrum = np.fft.rfft(
        windowed,
        axis=1,
    )

    # Power spectrum.
    power = np.abs(spectrum) ** 2

    # Average channels AFTER calculating power.
    # This prevents left/right phase cancellation.
    power = np.mean(power, axis=2)

    frequencies = np.fft.rfftfreq(
        frame_size,
        d=1.0 / sample_rate,
    )

    # One-sided spectrum correction.
    if frame_size % 2 == 0:
        power[:, 1:-1] *= 2.0
    else:
        power[:, 1:] *= 2.0

    # Average energy across time.
    mean_power = np.mean(power, axis=0)

    # Frequency resolution.
    frequency_resolution = sample_rate / frame_size

    # Convert spectral power into approximate band energy.
    total_mask = (
        (frequencies >= 20)
        & (frequencies <= 20000)
    )

    total_energy = np.sum(
        mean_power[total_mask]
    ) * frequency_resolution

    bands = {}

    for name, low, high, description in ENERGY_BANDS:

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        band_energy = (
            np.sum(mean_power[mask])
            * frequency_resolution
        )

        if total_energy > 0:
            energy_fraction = (
                band_energy / total_energy
            )
        else:
            energy_fraction = 0.0

        if energy_fraction > 0:
            relative_db = (
                10.0
                * np.log10(energy_fraction)
            )
        else:
            relative_db = -np.inf

        bands[name] = {
            "frequency_low_hz": low,
            "frequency_high_hz": high,
            "description": description,
            "energy_fraction": float(
                energy_fraction
            ),
            "energy_percent": float(
                energy_fraction * 100.0
            ),
            "relative_energy_db": float(
                relative_db
            ),
        }

    return {
        "file_path": str(file_path),
        "sample_rate": int(sample_rate),
        "duration_sec": round(
            duration_sec,
            3,
        ),
        "frame_size": frame_size,
        "hop_size": hop_size,
        "frequency_resolution_hz": round(
            frequency_resolution,
            3,
        ),
        "bands": bands,
    }