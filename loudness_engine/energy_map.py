import numpy as np


ENERGY_BANDS = {
    "sub": (20, 60),
    "bass": (60, 120),
    "low_mid": (120, 250),
    "mid_low": (250, 500),
    "mid": (500, 1000),
    "high_mid": (1000, 2000),
    "presence": (2000, 5000),
    "bright": (5000, 8000),
    "brilliance": (8000, 12000),
    "air": (12000, 20000),
}


def analyze_energy_map(data, sample_rate):
    """
    Stereo-preserving spectral energy analysis.

    Measurement convention:
    - Preserve original channel data.
    - Analyze left and right independently.
    - Do not downmix to mono.
    - Do not divide stereo measurements by 2.
    - Total stereo power = left power + right power.
    """

    if data.ndim != 2:
        raise ValueError(
            "Expected audio data with shape (samples, channels)."
        )

    samples, channels = data.shape

    if channels < 1:
        raise ValueError("Audio contains no channels.")

    n_fft = 4096
    hop_size = 1024

    if samples < n_fft:
        raise ValueError(
            f"Audio is too short for spectral analysis. "
            f"Need at least {n_fft} samples."
        )

    window = np.hanning(n_fft)

    channel_spectra = []

    for channel_index in range(channels):

        channel = data[:, channel_index]

        frames = []

        for start in range(
            0,
            samples - n_fft + 1,
            hop_size
        ):
            frame = channel[start:start + n_fft]

            windowed = frame * window

            fft_result = np.fft.rfft(windowed)

            power = (
    np.abs(fft_result) ** 2
    / (
        sample_rate
        * np.sum(window ** 2)
    )
)

            frames.append(power)

        channel_power = np.mean(
            np.asarray(frames),
            axis=0
        )

        channel_spectra.append(channel_power)

    # Preserve every channel.
    channel_spectra = np.asarray(channel_spectra)

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sample_rate
    )

    frequency_mask = (
        (frequencies >= 20) &
        (frequencies <= 20000)
    )

    frequencies = frequencies[frequency_mask]

    channel_spectra = channel_spectra[
        :,
        frequency_mask
    ]

    # Total stereo/multichannel energy.
    #
    # IMPORTANT:
    # No division by channel count.
    total_power = np.sum(
        channel_spectra,
        axis=0
    )

    total_energy = float(
        np.sum(total_power)
    )

    bands = {}

    for band_name, (low_hz, high_hz) in ENERGY_BANDS.items():

        band_mask = (
            (frequencies >= low_hz) &
            (frequencies < high_hz)
        )

        band_power = total_power[band_mask]

        band_energy = float(
            np.sum(band_power)
        )

        bandwidth_hz = high_hz - low_hz

        energy_percent = (
            (band_energy / total_energy) * 100.0
            if total_energy > 0
            else 0.0
        )

        energy_density = (
            band_energy / bandwidth_hz
            if bandwidth_hz > 0
            else 0.0
        )

        bands[band_name] = {
            "frequency_low_hz": low_hz,
            "frequency_high_hz": high_hz,
            "bandwidth_hz": bandwidth_hz,
            "energy": band_energy,
            "energy_percent": energy_percent,
            "energy_density": energy_density,
        }

    # Preserve channel-specific spectral information.
    channel_power_output = {}

    for channel_index, power in enumerate(channel_spectra):

        channel_name = (
            "left"
            if channel_index == 0
            else "right"
            if channel_index == 1
            else f"channel_{channel_index + 1}"
        )

        channel_power_output[channel_name] = {
            "power": power.tolist(),
            "power_db": (
                10.0
                * np.log10(
                    np.maximum(power, 1e-20)
                )
            ).tolist(),
        }

    total_power_db = (
        10.0
        * np.log10(
            np.maximum(total_power, 1e-20)
        )
    )

    return {
        "channels": channels,

        "bands": bands,

        "spectrum": {
            "frequencies_hz": frequencies.tolist(),

            "total_power": total_power.tolist(),

            "total_power_db": total_power_db.tolist(),

            "channels": channel_power_output,
        },
    }