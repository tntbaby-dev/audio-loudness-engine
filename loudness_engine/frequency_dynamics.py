import numpy as np


FREQUENCY_BANDS = {
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


def analyze_frequency_dynamics(data, sample_rate):
    """
    Analyze how frequency-band energy changes over time.

    Measurement conventions:
    - Preserve original channel data.
    - Analyze channels independently.
    - No mono downmix.
    - No division by channel count.
    - Preserve raw band energy.
    - Preserve frame-to-frame energy change.
    """

    if data.ndim != 2:
        raise ValueError(
            "Expected audio data with shape (samples, channels)."
        )

    samples, channels = data.shape

    if channels < 1:
        raise ValueError("Audio contains no channels.")

    frame_size = 4096
    hop_size = 1024
    n_fft = 4096

    if samples < frame_size:
        raise ValueError(
            f"Audio is too short. Need at least {frame_size} samples."
        )

    window = np.hanning(frame_size)

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sample_rate
    )

    frequency_mask = (
        (frequencies >= 20.0)
        & (frequencies <= 20000.0)
    )

    frequencies = frequencies[frequency_mask]

    band_masks = {}

    for band_name, (low, high) in FREQUENCY_BANDS.items():
        band_masks[band_name] = (
            (frequencies >= low)
            & (frequencies < high)
        )

    channel_results = {}

    # ---------------------------------------------------------
    # CHANNEL ANALYSIS
    # ---------------------------------------------------------

    for channel_index in range(channels):

        channel = data[:, channel_index]

        frame_band_energy = {
            band_name: []
            for band_name in FREQUENCY_BANDS
        }

        time_frames = []

        for start in range(
            0,
            samples - frame_size + 1,
            hop_size
        ):

            frame = channel[
                start:start + frame_size
            ]

            windowed = frame * window

            fft_result = np.fft.rfft(
                windowed,
                n=n_fft
            )

            power = (
                np.abs(fft_result) ** 2
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

            frequency_resolution = (
                sample_rate / n_fft
            )

            for band_name, band_mask in band_masks.items():

                band_power = power[
                    band_mask
                ]

                band_energy = (
                    np.sum(band_power)
                    * frequency_resolution
                )

                frame_band_energy[
                    band_name
                ].append(
                    band_energy
                )

            time_frames.append(
                start / sample_rate
            )

        time_frames = np.asarray(
            time_frames,
            dtype=float
        )

        bands = {}

        for band_name in FREQUENCY_BANDS:

            energy = np.asarray(
                frame_band_energy[band_name],
                dtype=float
            )

            energy_db = (
                10.0
                * np.log10(
                    np.maximum(
                        energy,
                        1e-30
                    )
                )
            )

            energy_change_db = np.zeros_like(
                energy_db
            )

            energy_change_db[1:] = (
                energy_db[1:]
                - energy_db[:-1]
            )

            bands[band_name] = {
                "energy": energy.tolist(),
                "energy_db": energy_db.tolist(),
                "energy_change_db": (
                    energy_change_db.tolist()
                ),
            }

        channel_name = (
            "left"
            if channel_index == 0
            else "right"
            if channel_index == 1
            else f"channel_{channel_index + 1}"
        )

        channel_results[channel_name] = {
            "bands": bands,
            "time_sec": time_frames.tolist(),
        }

    # ---------------------------------------------------------
    # TOTAL MULTICHANNEL ANALYSIS
    # ---------------------------------------------------------

    total_band_energy = {
        band_name: []
        for band_name in FREQUENCY_BANDS
    }

    total_time_frames = []

    for start in range(
        0,
        samples - frame_size + 1,
        hop_size
    ):

        frame_band_energy = {
            band_name: 0.0
            for band_name in FREQUENCY_BANDS
        }

        for channel_index in range(channels):

            frame = data[
                start:start + frame_size,
                channel_index
            ]

            windowed = frame * window

            fft_result = np.fft.rfft(
                windowed,
                n=n_fft
            )

            power = (
                np.abs(fft_result) ** 2
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

            for band_name, band_mask in band_masks.items():

                band_power = power[
                    band_mask
                ]

                band_energy = (
                    np.sum(band_power)
                    * frequency_resolution
                )

                # Preserve total multichannel energy.
                frame_band_energy[
                    band_name
                ] += band_energy

        for band_name in FREQUENCY_BANDS:

            total_band_energy[
                band_name
            ].append(
                frame_band_energy[band_name]
            )

        total_time_frames.append(
            start / sample_rate
        )

    total_bands = {}

    for band_name in FREQUENCY_BANDS:

        energy = np.asarray(
            total_band_energy[band_name],
            dtype=float
        )

        energy_db = (
            10.0
            * np.log10(
                np.maximum(
                    energy,
                    1e-30
                )
            )
        )

        energy_change_db = np.zeros_like(
            energy_db
        )

        energy_change_db[1:] = (
            energy_db[1:]
            - energy_db[:-1]
        )

        total_bands[band_name] = {
            "energy": energy.tolist(),
            "energy_db": energy_db.tolist(),
            "energy_change_db": (
                energy_change_db.tolist()
            ),
        }

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------

    return {
        "frame_size": frame_size,
        "hop_size": hop_size,
        "n_fft": n_fft,
        "frame_duration_sec": (
            frame_size / sample_rate
        ),
        "hop_duration_sec": (
            hop_size / sample_rate
        ),
        "frequency_resolution_hz": (
            sample_rate / n_fft
        ),
        "frequencies_hz": (
            frequencies.tolist()
        ),
        "frequency_bands": {
            name: {
                "frequency_low_hz": low,
                "frequency_high_hz": high,
            }
            for name, (low, high)
            in FREQUENCY_BANDS.items()
        },
        "channels": channel_results,
        "total": {
            "bands": total_bands,
            "time_sec": (
                np.asarray(
                    total_time_frames
                ).tolist()
            ),
        },
    }