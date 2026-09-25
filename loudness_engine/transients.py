import numpy as np


def analyze_transients(data, sample_rate):
    """
    Time-domain transient analysis.

    Measurement convention:
    - Preserve original channel data.
    - Analyze channels independently.
    - No mono downmix.
    - No division by channel count.
    - Detect sudden increases in RMS energy.
    """

    if data.ndim != 2:
        raise ValueError(
            "Expected audio data with shape (samples, channels)."
        )

    samples, channels = data.shape

    if channels < 1:
        raise ValueError("Audio contains no channels.")

    frame_size = 1024
    hop_size = 256

    if samples < frame_size:
        raise ValueError(
            f"Audio is too short for transient analysis. "
            f"Need at least {frame_size} samples."
        )

    # ---------------------------------------------------------
    # CHANNEL ANALYSIS
    # ---------------------------------------------------------

    channel_results = {}

    for channel_index in range(channels):

        channel = data[:, channel_index]

        rms_frames = []
        peak_frames = []
        time_frames = []

        for start in range(
            0,
            samples - frame_size + 1,
            hop_size
        ):

            frame = channel[
                start:start + frame_size
            ]

            rms = np.sqrt(
                np.mean(
                    frame ** 2
                )
            )

            peak = np.max(
                np.abs(frame)
            )

            rms_frames.append(rms)
            peak_frames.append(peak)
            time_frames.append(
                start / sample_rate
            )

        rms_frames = np.asarray(
            rms_frames,
            dtype=float
        )

        peak_frames = np.asarray(
            peak_frames,
            dtype=float
        )

        time_frames = np.asarray(
            time_frames,
            dtype=float
        )

        # -----------------------------------------------------
        # ENERGY CHANGE
        # -----------------------------------------------------

        rms_db = (
            20.0
            * np.log10(
                np.maximum(
                    rms_frames,
                    1e-20
                )
            )
        )

        energy_change_db = np.zeros_like(
            rms_db
        )

        energy_change_db[1:] = (
            rms_db[1:]
            - rms_db[:-1]
        )

        # -----------------------------------------------------
        # TRANSIENT DETECTION
        # -----------------------------------------------------

        # A positive energy jump indicates a rapid increase.
        #
        # The threshold is deliberately conservative.
        # This is detection infrastructure, not classification.

        transient_threshold_db = 6.0

        transient_indices = np.where(
            energy_change_db
            >= transient_threshold_db
        )[0]

        transients = []

        for index in transient_indices:

            previous_index = index - 1

            if previous_index < 0:
                continue

            transients.append(
                {
                    "time_sec": float(
                        time_frames[index]
                    ),
                    "strength_db": float(
                        energy_change_db[index]
                    ),
                    "rms_before_dbfs": float(
                        rms_db[previous_index]
                    ),
                    "rms_after_dbfs": float(
                        rms_db[index]
                    ),
                    "peak_dbfs": float(
                        20.0
                        * np.log10(
                            max(
                                peak_frames[index],
                                1e-20
                            )
                        )
                    ),
                }
            )

        channel_name = (
            "left"
            if channel_index == 0
            else "right"
            if channel_index == 1
            else f"channel_{channel_index + 1}"
        )

        channel_results[channel_name] = {
            "rms_dbfs": rms_db.tolist(),
            "peak_dbfs": (
                20.0
                * np.log10(
                    np.maximum(
                        peak_frames,
                        1e-20
                    )
                )
            ).tolist(),
            "energy_change_db": (
                energy_change_db.tolist()
            ),
            "time_sec": (
                time_frames.tolist()
            ),
            "transients": transients,
        }

    # ---------------------------------------------------------
    # TOTAL MULTICHANNEL ENERGY
    # ---------------------------------------------------------

    total_power = np.sum(
        data ** 2,
        axis=1
    )

    total_rms_frames = []
    total_peak_frames = []
    total_time_frames = []

    for start in range(
        0,
        samples - frame_size + 1,
        hop_size
    ):

        frame_power = total_power[
            start:start + frame_size
        ]

        total_rms = np.sqrt(
            np.mean(
                frame_power
            )
        )

        total_peak = np.sqrt(
            np.max(
                frame_power
            )
        )

        total_rms_frames.append(
            total_rms
        )

        total_peak_frames.append(
            total_peak
        )

        total_time_frames.append(
            start / sample_rate
        )

    total_rms_frames = np.asarray(
        total_rms_frames,
        dtype=float
    )

    total_peak_frames = np.asarray(
        total_peak_frames,
        dtype=float
    )

    total_time_frames = np.asarray(
        total_time_frames,
        dtype=float
    )

    total_rms_db = (
        20.0
        * np.log10(
            np.maximum(
                total_rms_frames,
                1e-20
            )
        )
    )

    total_energy_change_db = np.zeros_like(
        total_rms_db
    )

    total_energy_change_db[1:] = (
        total_rms_db[1:]
        - total_rms_db[:-1]
    )

    # ---------------------------------------------------------
    # TOTAL TRANSIENT DETECTION
    # ---------------------------------------------------------

    transient_threshold_db = 6.0

    total_transient_indices = np.where(
        total_energy_change_db
        >= transient_threshold_db
    )[0]

    total_transients = []

    for index in total_transient_indices:

        previous_index = index - 1

        if previous_index < 0:
            continue

        total_transients.append(
            {
                "time_sec": float(
                    total_time_frames[index]
                ),
                "strength_db": float(
                    total_energy_change_db[index]
                ),
                "rms_before_dbfs": float(
                    total_rms_db[previous_index]
                ),
                "rms_after_dbfs": float(
                    total_rms_db[index]
                ),
                "rms_equivalent_peak_dbfs": float(
                    20.0
                    * np.log10(
                        max(
                            total_peak_frames[index],
                            1e-20
                        )
                    )
                ),
            }
        )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    return {
        "frame_size": frame_size,
        "hop_size": hop_size,
        "frame_duration_sec": (
            frame_size / sample_rate
        ),
        "hop_duration_sec": (
            hop_size / sample_rate
        ),
        "threshold_db": (
            transient_threshold_db
        ),

        "channels": channel_results,

        "total": {
            "rms_dbfs": (
                total_rms_db.tolist()
            ),
            "rms_equivalent_peak_dbfs": (
                20.0
                * np.log10(
                    np.maximum(
                        total_peak_frames,
                        1e-20
                    )
                )
            ).tolist(),
            "energy_change_db": (
                total_energy_change_db.tolist()
            ),
            "time_sec": (
                total_time_frames.tolist()
            ),
            "transients": total_transients,
        },
    }