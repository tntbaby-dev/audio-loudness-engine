import numpy as np


def analyze_dynamics(data, sample_rate):
    """
    Time-domain dynamics analysis.

    Measurement convention:
    - Preserve original channel data.
    - Analyze channels independently.
    - No mono downmix.
    - No division by channel count.
    - Preserve time-varying RMS and peak information.
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

    if samples < frame_size:
        raise ValueError(
            f"Audio is too short for dynamics analysis. "
            f"Need at least {frame_size} samples."
        )

    # ---------------------------------------------------------
    # ANALYZE EACH CHANNEL
    # ---------------------------------------------------------

    channel_results = {}

    for channel_index in range(channels):

        channel = data[:, channel_index]

        rms_frames = []
        peak_frames = []
        crest_frames = []

        frame_positions = []

        for start in range(
            0,
            samples - frame_size + 1,
            hop_size
        ):

            frame = channel[
                start:start + frame_size
            ]

            # RMS energy
            rms = np.sqrt(
                np.mean(
                    frame ** 2
                )
            )

            # Absolute peak
            peak = np.max(
                np.abs(frame)
            )

            # Crest factor
            if rms > 0:

                crest_factor_db = (
                    20.0
                    * np.log10(
                        peak / rms
                    )
                )

            else:

                crest_factor_db = None

            rms_frames.append(rms)
            peak_frames.append(peak)
            crest_frames.append(
                crest_factor_db
            )

            frame_positions.append(
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

        crest_frames = np.asarray(
            crest_frames,
            dtype=float
        )

        frame_positions = np.asarray(
            frame_positions,
            dtype=float
        )

        # -----------------------------------------------------
        # CONVERT RMS / PEAK TO dBFS
        # -----------------------------------------------------

        rms_dbfs = (
            20.0
            * np.log10(
                np.maximum(
                    rms_frames,
                    1e-20
                )
            )
        )

        peak_dbfs = (
            20.0
            * np.log10(
                np.maximum(
                    peak_frames,
                    1e-20
                )
            )
        )

        channel_name = (
            "left"
            if channel_index == 0
            else "right"
            if channel_index == 1
            else f"channel_{channel_index + 1}"
        )

        channel_results[channel_name] = {
            "rms": rms_frames.tolist(),
            "rms_dbfs": rms_dbfs.tolist(),
            "peak": peak_frames.tolist(),
            "peak_dbfs": peak_dbfs.tolist(),
            "crest_factor_db": crest_frames.tolist(),
            "time_sec": frame_positions.tolist(),
        }

    # ---------------------------------------------------------
    # TOTAL STEREO / MULTICHANNEL RMS
    # ---------------------------------------------------------

    # Preserve channel energy.
    #
    # Total power = sum of channel powers.
    #
    # No division by channel count.

    channel_power = data ** 2

    total_power = np.sum(
        channel_power,
        axis=1
    )

    total_rms_frames = []
    total_peak_frames = []
    total_crest_frames = []
    total_frame_positions = []

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

        if total_rms > 0:

            total_crest = (
                20.0
                * np.log10(
                    total_peak / total_rms
                )
            )

        else:

            total_crest = None

        total_rms_frames.append(
            total_rms
        )

        total_peak_frames.append(
            total_peak
        )

        total_crest_frames.append(
            total_crest
        )

        total_frame_positions.append(
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

    total_crest_frames = np.asarray(
        total_crest_frames,
        dtype=float
    )

    total_rms_dbfs = (
        20.0
        * np.log10(
            np.maximum(
                total_rms_frames,
                1e-20
            )
        )
    )

    total_peak_dbfs = (
        20.0
        * np.log10(
            np.maximum(
                total_peak_frames,
                1e-20
            )
        )
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

        "channels": channel_results,

        "total": {
            "rms": total_rms_frames.tolist(),
            "rms_dbfs": total_rms_dbfs.tolist(),
            "rms_equivalent_peak": total_peak_frames.tolist(),
            "rms_equivalent_peak_dbfs": total_peak_dbfs.tolist(),
            "crest_factor_db": (
                total_crest_frames.tolist()
            ),
            "time_sec": (
                np.asarray(
                    total_frame_positions
                ).tolist()
            ),
        },
    }