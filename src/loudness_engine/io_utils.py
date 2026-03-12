from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


def list_audio_files(input_path: str):
    """
    Return a list of audio files from a file or folder path.
    Supports recursive scanning for folders.
    """
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if path.is_file():
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            return [path]
        return []

    return [
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in AUDIO_EXTENSIONS
    ]