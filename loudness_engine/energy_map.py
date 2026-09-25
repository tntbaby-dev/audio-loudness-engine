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
    - Analyze each channel independently.
    - Do not downmix to mono.
    - Do not divide stereo measurements by 2.
    - Total multichannel power = sum of channel powers.
    """

    if data.ndim != 2:
        raise ValueError(
            "Expected audio data with shape (samples, channels)."
        )

    samples, channels = data.shape

    if channels < 1:
        raise ValueError("Audio contains no channels.")

    n_fft = 16384
    hop_size = 4096

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

            # Convert to one-sided PSD.
            # Interior bins represent both positive and
            # negative frequency components.
            if n_fft % 2 == 0:
                power[1:-1] *= 2.0
            else:
                power[1:] *= 2.0

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

    # Keep the audible analysis range.
    frequency_mask = (
        (frequencies >= 20) &
        (frequencies <= 20000)
    )

    frequencies = frequencies[frequency_mask]

    channel_spectra = channel_spectra[
        :,
        frequency_mask
    ]

    # Preserve channel-specific spectral information
    # after applying the same frequency mask.
    left_power = (
        channel_spectra[0]
        if channels >= 1
        else None
    )

    right_power = (
        channel_spectra[1]
        if channels >= 2
        else None
    )

    # ---------------------------------------------------------
    # TOTAL STEREO / MULTICHANNEL POWER
    # ---------------------------------------------------------

    # IMPORTANT:
    # No division by channel count.
    total_power = np.sum(
        channel_spectra,
        axis=0
    )

    frequency_resolution_hz = sample_rate / n_fft

    total_energy = (
        float(np.sum(total_power))
        * frequency_resolution_hz
    )

    # ---------------------------------------------------------
    # STEREO SPECTRUM ANALYSIS
    # ---------------------------------------------------------

    stereo_analysis = {}

    if channels >= 2:

        stereo_sum = left_power + right_power

        stereo_difference = np.abs(
            left_power - right_power
        )

        stereo_balance = (
            (left_power - right_power)
            / np.maximum(stereo_sum, 1e-20)
        )

        stereo_analysis = {
            "left_power": left_power.tolist(),
            "right_power": right_power.tolist(),
            "difference": stereo_difference.tolist(),
            "balance": stereo_balance.tolist(),
        }

    # ---------------------------------------------------------
    # FREQUENCY BANDS
    # ---------------------------------------------------------

    bands = {}
    stereo_bands = {}

    for band_name, (low_hz, high_hz) in ENERGY_BANDS.items():

        band_mask = (
            (frequencies >= low_hz) &
            (frequencies < high_hz)
        )

        band_power = total_power[band_mask]

        # ---------------------------------------------
        # Stereo energy inside this frequency band
        # ---------------------------------------------

        if (
            channels >= 2
            and left_power is not None
            and right_power is not None
        ):

            left_band_energy = float(
                np.sum(left_power[band_mask])
                * frequency_resolution_hz
            )

            right_band_energy = float(
                np.sum(right_power[band_mask])
                * frequency_resolution_hz
            )

            stereo_band_total = (
                left_band_energy
                + right_band_energy
            )

            stereo_band_balance = (
                (
                    left_band_energy
                    - right_band_energy
                )
                / stereo_band_total
                if stereo_band_total > 0
                else 0.0
            )

            stereo_bands[band_name] = {
                "left_energy": left_band_energy,
                "right_energy": right_band_energy,
                "total_energy": stereo_band_total,
                "balance": stereo_band_balance,
            }

        # ---------------------------------------------
        # Total band energy
        # ---------------------------------------------

        band_energy = float(
            np.sum(band_power)
            * frequency_resolution_hz
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

    # ---------------------------------------------------------
    # CHANNEL SPECTRAL OUTPUT
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # TOTAL POWER IN dB
    # ---------------------------------------------------------

    total_power_db = (
        10.0
        * np.log10(
            np.maximum(total_power, 1e-20)
        )
    )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    result = {
        "channels": channels,

        "bands": bands,

        "spectrum": {
            "frequency_resolution_hz": float(
                frequency_resolution_hz
            ),
            "frequencies_hz": frequencies.tolist(),
            "total_power": total_power.tolist(),
            "total_power_db": total_power_db.tolist(),
            "channels": channel_power_output,
        },

        "stereo_analysis": {
            "spectrum": stereo_analysis,
            "bands": stereo_bands,
        },
    }

    return result