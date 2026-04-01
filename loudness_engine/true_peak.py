import subprocess
import json


def measure_true_peak(file_path):
    """
    Measure true peak using FFmpeg loudnorm analyzer.
    Returns peak in dBTP.
    """

    command = [
        "ffmpeg",
        "-i", str(file_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null",
        "-"
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = result.stderr

    start = output.find("{")
    end = output.rfind("}") + 1

    if start == -1 or end == -1:
        return None

    json_data = json.loads(output[start:end])

    return float(json_data["input_tp"])