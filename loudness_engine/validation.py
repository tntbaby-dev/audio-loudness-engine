from typing import Any, Dict
import math


REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "metadata",
    "loudness",
    "energy",
    "dynamics",
    "transients",
    "frequency_dynamics",
    "tonal_analysis",
}


def _is_finite(value: Any) -> bool:
    """Return True if a numeric value is finite."""
    if isinstance(value, bool):
        return True

    if isinstance(value, (int, float)):
        return math.isfinite(value)

    return True


def _check_finite_recursive(value: Any, path: str = "") -> list[str]:
    """Find NaN or infinite numeric values anywhere in nested data."""
    errors = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            errors.extend(_check_finite_recursive(child, child_path))

    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            errors.extend(_check_finite_recursive(child, child_path))

    elif isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"Non-finite value at {path}: {value}")

    return errors


def _check_time_axis(time_values: Any, name: str) -> Dict[str, Any]:
    """Validate that a time axis is numeric and non-decreasing."""
    if not isinstance(time_values, list):
        return {
            "passed": False,
            "message": f"{name} is not a list.",
        }

    if not time_values:
        return {
            "passed": False,
            "message": f"{name} is empty.",
        }

    for index, value in enumerate(time_values):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return {
                "passed": False,
                "message": f"{name}[{index}] is not a finite number.",
            }

    for index in range(1, len(time_values)):
        if time_values[index] < time_values[index - 1]:
            return {
                "passed": False,
                "message": f"{name} is not monotonic at index {index}.",
            }

    return {
        "passed": True,
        "message": f"{name} is valid.",
        "length": len(time_values),
    }


def validate_unified_analysis(
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the structure and numerical integrity of unified audio analysis.

    This function does not modify or reinterpret analysis data.
    """

    errors = []
    warnings = []

    checks: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # 1. Top-level schema
    # ---------------------------------------------------------

    actual_keys = set(analysis.keys())
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - actual_keys

    checks["schema"] = {
        "passed": not missing_keys,
        "missing_keys": sorted(missing_keys),
    }

    if missing_keys:
        errors.append(
            f"Missing required top-level keys: {sorted(missing_keys)}"
        )

    # ---------------------------------------------------------
    # 2. Schema version
    # ---------------------------------------------------------

    schema_version = analysis.get("schema_version")

    checks["schema_version"] = {
        "passed": schema_version == "1.0",
        "value": schema_version,
    }

    if schema_version != "1.0":
        errors.append(
            f"Unsupported schema version: {schema_version}"
        )

    # ---------------------------------------------------------
    # 3. Metadata
    # ---------------------------------------------------------

    metadata = analysis.get("metadata", {})

    metadata_valid = True

    sample_rate = metadata.get("sample_rate")
    channels = metadata.get("channels")
    duration_sec = metadata.get("duration_sec")

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        metadata_valid = False
        errors.append("Invalid sample_rate.")

    if not isinstance(channels, int) or channels <= 0:
        metadata_valid = False
        errors.append("Invalid channel count.")

    if not isinstance(duration_sec, (int, float)):
        metadata_valid = False
        errors.append("Invalid duration_sec.")
    elif duration_sec <= 0 or not math.isfinite(duration_sec):
        metadata_valid = False
        errors.append("duration_sec must be finite and greater than zero.")

    checks["metadata"] = {
        "passed": metadata_valid,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": duration_sec,
    }

    # ---------------------------------------------------------
    # 4. Required analysis sections
    # ---------------------------------------------------------

    sections = [
        "loudness",
        "energy",
        "dynamics",
        "transients",
        "frequency_dynamics",
        "tonal_analysis",
    ]

    section_results = {}

    for section in sections:
        exists = section in analysis and isinstance(
            analysis.get(section), dict
        )

        section_results[section] = {
            "passed": exists,
        }

        if not exists:
            errors.append(
                f"Analysis section '{section}' is missing or invalid."
            )

    checks["sections"] = section_results

    # ---------------------------------------------------------
    # 5. Finite numerical values
    # ---------------------------------------------------------

    finite_errors = _check_finite_recursive(analysis)

    checks["finite_values"] = {
        "passed": not finite_errors,
        "error_count": len(finite_errors),
    }

    errors.extend(finite_errors)

    # ---------------------------------------------------------
    # 6. Channel structure
    # ---------------------------------------------------------

    channel_structure_valid = True

    energy = analysis.get("energy", {})
    dynamics = analysis.get("dynamics", {})
    frequency_dynamics = analysis.get("frequency_dynamics", {})
    tonal_analysis = analysis.get("tonal_analysis", {})

    energy_spectrum = energy.get("spectrum", {})
    energy_channels = energy_spectrum.get("channels", {})
    dynamics_channels = dynamics.get("channels", {})
    frequency_channels = frequency_dynamics.get("channels", {})
    tonal_channels = tonal_analysis.get("channels", {})

    expected_channel_names = ["left", "right"]

    if channels == 2:
        for name, channel_data in [
            ("energy", energy_channels),
            ("dynamics", dynamics_channels),
            ("frequency_dynamics", frequency_channels),
            ("tonal_analysis", tonal_channels),
        ]:
            if not all(
                channel_name in channel_data
                for channel_name in expected_channel_names
            ):
                channel_structure_valid = False
                errors.append(
                    f"{name} does not contain both left and right channels."
                )

    checks["channel_structure"] = {
        "passed": channel_structure_valid,
        "expected_channels": channels,
        "expected_names": expected_channel_names if channels == 2 else None,
    }

    # ---------------------------------------------------------
    # 7. Time-axis validation
    # ---------------------------------------------------------

    time_checks = {}

    dynamics_time = dynamics_channels.get("left", {}).get("time_sec")
    frequency_time = frequency_channels.get("left", {}).get("time_sec")
    tonal_time = tonal_channels.get("left", {}).get("time_sec")

    if dynamics_time is not None:
        time_checks["dynamics"] = _check_time_axis(
            dynamics_time,
            "dynamics.left.time_sec",
        )

    if frequency_time is not None:
        time_checks["frequency_dynamics"] = _check_time_axis(
            frequency_time,
            "frequency_dynamics.left.time_sec",
        )

    if tonal_time is not None:
        time_checks["tonal_analysis"] = _check_time_axis(
            tonal_time,
            "tonal_analysis.left.time_sec",
        )

    checks["time_axes"] = time_checks

    for result in time_checks.values():
        if not result["passed"]:
            errors.append(result["message"])

    # ---------------------------------------------------------
    # 8. Frame alignment
    # ---------------------------------------------------------

    frame_counts = {}

    if dynamics_time is not None:
        frame_counts["dynamics"] = len(dynamics_time)

    if frequency_time is not None:
        frame_counts["frequency_dynamics"] = len(frequency_time)

    if tonal_time is not None:
        frame_counts["tonal_analysis"] = len(tonal_time)

    frame_alignment_valid = len(set(frame_counts.values())) <= 1

    checks["frame_alignment"] = {
        "passed": frame_alignment_valid,
        "frame_counts": frame_counts,
    }

    if not frame_alignment_valid:
        errors.append(
            f"Frame counts are inconsistent: {frame_counts}"
        )

    # ---------------------------------------------------------
    # 9. Frequency-bin validation
    # ---------------------------------------------------------

    energy_spectrum = energy.get("spectrum", {})
    frequencies = energy_spectrum.get("frequencies_hz", [])

    frequency_bins_valid = (
        isinstance(frequencies, list)
        and len(frequencies) > 0
    )

    if frequency_bins_valid:
        for index in range(1, len(frequencies)):
            if frequencies[index] <= frequencies[index - 1]:
                frequency_bins_valid = False
                errors.append(
                    "Energy-map frequency bins are not strictly increasing."
                )
                break

    checks["frequency_bins"] = {
        "passed": frequency_bins_valid,
        "count": len(frequencies) if isinstance(frequencies, list) else 0,
    }

    # ---------------------------------------------------------
    # 10. M/S energy conservation
    # ---------------------------------------------------------

    stereo_analysis = energy.get("stereo_analysis", {})
    ms_analysis = stereo_analysis.get("ms", {})

    mid_power = ms_analysis.get("mid_power", [])
    side_power = ms_analysis.get("side_power", [])

    left_power = energy_spectrum.get("channels", {}).get(
        "left", {}
    ).get("power", [])

    right_power = energy_spectrum.get("channels", {}).get(
        "right", {}
    ).get("power", [])

    ms_valid = True
    max_error = 0.0

    if mid_power and side_power and left_power and right_power:
        if not (
            len(mid_power)
            == len(side_power)
            == len(left_power)
            == len(right_power)
        ):
            ms_valid = False
            errors.append("M/S and L/R spectrum lengths do not match.")
        else:
            for left, right, mid, side in zip(
                left_power,
                right_power,
                mid_power,
                side_power,
            ):
                lr_energy = left + right
                ms_energy = mid + side
                error = abs(lr_energy - ms_energy)

                if error > max_error:
                    max_error = error

            if max_error > 1e-9:
                ms_valid = False
                errors.append(
                    f"M/S energy conservation failed. "
                    f"Maximum error: {max_error}"
                )

    checks["ms_energy_conservation"] = {
        "passed": ms_valid,
        "max_error": max_error,
    }

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    return {
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }
