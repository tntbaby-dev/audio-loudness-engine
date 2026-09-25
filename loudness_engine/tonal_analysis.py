import numpy as np
import soundfile as sf


FREQUENCY_BANDS = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 120.0),
    "low_mid": (120.0, 250.0),
    "mid_low": (250.0, 500.0),
    "mid": (500.0, 1000.0),
    "high_mid": (1000.0, 2000.0),
    "presence": (2000.0, 5000.0),
    "bright": (5000.0, 8000.0),
    "brilliance": (8000.0, 12000.0),
    "air": (12000.0, 20000.0),
}


def _safe_db(value, floor=1e-30):
    return 10.0 * np.log10(np.maximum(value, floor))


def _spectral_flatness(power):
    """
    Spectral flatness:
        geometric mean / arithmetic mean

    Values near 1:
        noise-like / flat spectrum

    Values near 0:
        tonal / peak-dominated spectrum
    """
    power = np.maximum(power, 1e-30)

    geometric_mean = np.exp(np.mean(np.log(power)))
    arithmetic_mean = np.mean(power)

    if arithmetic_mean <= 0.0:
        return 0.0

    return float(geometric_mean / arithmetic_mean)


def _find_spectral_peaks(power, frequencies, max_peaks=20):
    """
    Find the strongest local spectral peaks.

    No normalization is applied to the spectrum.
    Peak prominence is represented by the raw power value
    and its dB representation.
    """
    if len(power) < 3:
        return []

    local_peak_indices = np.where(
        (power[1:-1] > power[:-2])
        & (power[1:-1] >= power[2:])
    )[0] + 1

    if len(local_peak_indices) == 0:
        return []

    sorted_indices = local_peak_indices[
        np.argsort(power[local_peak_indices])[::-1]
    ]

    selected = sorted_indices[:max_peaks]

    peaks = []

    for index in selected:
        peaks.append(
            {
                "frequency_hz": float(frequencies[index]),
                "power": float(power[index]),
                "power_db": float(_safe_db(power[index])),
            }
        )

    return peaks


def _harmonicity_score(peaks):
    """
    Estimate harmonic relationships between spectral peaks.

    This is deliberately a simple structural measurement rather
    than a final musical 'harmonicity' judgment.

    For each strong peak, determine whether another strong peak
    exists near an integer multiple of its frequency.

    Returns:
        ratio of peaks participating in detectable harmonic
        relationships.
    """
    if len(peaks) < 2:
        return 0.0

    frequencies = np.array(
        [peak["frequency_hz"] for peak in peaks],
        dtype=float,
    )

    frequencies = frequencies[frequencies > 0.0]

    if len(frequencies) < 2:
        return 0.0

    related = 0

    tolerance = 0.03

    for frequency in frequencies:
        found_relationship = False

        for harmonic_number in range(2, 8):
            target = frequency * harmonic_number

            relative_error = np.abs(frequencies - target) / target

            if np.any(relative_error <= tolerance):
                found_relationship = True
                break

        if found_relationship:
            related += 1

    return float(related / len(frequencies))


def _analyze_channel(channel, sample_rate, frame_size, hop_size, n_fft):
    window = np.hanning(frame_size)

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sample_rate,
    )

    frequency_mask = (
        (frequencies >= 20.0)
        & (frequencies <= 20000.0)
    )

    frequencies = frequencies[frequency_mask]

    frames = []

    samples = len(channel)

    for start in range(
        0,
        samples - frame_size + 1,
        hop_size,
    ):
        frame = channel[start:start + frame_size]

        windowed = frame * window

        spectrum = np.fft.rfft(
            windowed,
            n=n_fft,
        )

        power = (
            np.abs(spectrum) ** 2
            / (
                sample_rate
                * np.sum(window ** 2)
            )
        )

        if n_fft % 2 == 0:
            power[1:-1] *= 2.0
        else:
            power[1:] *= 2.0

        power = power[frequency_mask]

        peaks = _find_spectral_peaks(
            power,
            frequencies,
            max_peaks=20,
        )

        flatness = _spectral_flatness(power)

        harmonicity = _harmonicity_score(peaks)

        frames.append(
            {
                "time_sec": float(start / sample_rate),
                "spectral_flatness": flatness,
                "harmonicity": harmonicity,
                "peaks": peaks,
            }
        )

    return frequencies, frames


def analyze_tonal_analysis(
    data,
    sample_rate,
    frame_size=4096,
    hop_size=1024,
    n_fft=4096,
):
    """
    Analyze tonal and harmonic characteristics while preserving
    the original stereo/multichannel structure.

    No mono downmixing is performed.

    Each channel is analyzed independently.

    Measurements:
        - spectral flatness
        - dominant spectral peaks
        - peak frequencies
        - peak power
        - simple harmonic relationships

    Args:
        data:
            Audio array shaped (samples, channels).

        sample_rate:
            Audio sample rate in Hz.

    Returns:
        Dictionary containing per-channel tonal analysis.
    """
    if data.ndim != 2:
        raise ValueError(
            "Audio data must have shape (samples, channels)."
        )

    samples, channels = data.shape

    if samples < frame_size:
        raise ValueError(
            "Audio is shorter than the selected frame size."
        )

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    if frame_size <= 0 or hop_size <= 0 or n_fft <= 0:
        raise ValueError(
            "frame_size, hop_size, and n_fft must be greater than zero."
        )

    if n_fft < frame_size:
        raise ValueError(
            "n_fft must be greater than or equal to frame_size."
        )

    channel_names = []

    if channels == 1:
        channel_names = ["mono"]
    elif channels == 2:
        channel_names = ["left", "right"]
    else:
        channel_names = [
            f"channel_{index + 1}"
            for index in range(channels)
        ]

    channel_results = {}

    for channel_index in range(channels):
        frequencies, frames = _analyze_channel(
            data[:, channel_index],
            sample_rate,
            frame_size,
            hop_size,
            n_fft,
        )

        channel_results[channel_names[channel_index]] = {
            "spectral_flatness": [
                frame["spectral_flatness"]
                for frame in frames
            ],
            "harmonicity": [
                frame["harmonicity"]
                for frame in frames
            ],
            "peaks": [
                frame["peaks"]
                for frame in frames
            ],
            "time_sec": [
                frame["time_sec"]
                for frame in frames
            ],
        }

    return {
        "frame_size": frame_size,
        "hop_size": hop_size,
        "n_fft": n_fft,
        "frame_duration_sec": float(
            frame_size / sample_rate
        ),
        "hop_duration_sec": float(
            hop_size / sample_rate
        ),
        "frequency_resolution_hz": float(
            sample_rate / n_fft
        ),
        "frequencies_hz": frequencies.tolist(),
        "channels": channel_results,
    }


if __name__ == "__main__":
    file_path = "Fp_Master.wav"

    data, sample_rate = sf.read(
        file_path,
        always_2d=True,
    )

    result = analyze_tonal_analysis(
        data,
        sample_rate,
    )

    print(
        "Channels:",
        list(result["channels"].keys()),
    )

    print(
        "Frequency bins:",
        len(result["frequencies_hz"]),
    )

    first_channel = next(
        iter(result["channels"].values())
    )

    print(
        "Total frames:",
        len(first_channel["time_sec"]),
    )

    print(
        "First spectral flatness:",
        first_channel["spectral_flatness"][0],
    )

    print(
        "First harmonicity:",
        first_channel["harmonicity"][0],
    )

    print(
        "First peaks:",
        first_channel["peaks"][0][:5],
    )