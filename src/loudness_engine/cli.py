import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from loudness_engine.io_utils import list_audio_files
from loudness_engine.analyzer import analyze_file


def main():
    parser = argparse.ArgumentParser(description="Loudness Intelligence Engine")
    parser.add_argument("--input", required=True, help="Path to audio file or folder")
    parser.add_argument("--output", default="outputs", help="Folder to save reports")

    args = parser.parse_args()

    audio_files = list_audio_files(args.input)

    if not audio_files:
        print("No supported audio files found.")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for audio_file in tqdm(audio_files, desc="Analyzing files"):
        try:
            analysis = analyze_file(audio_file)
            results.append(analysis)
        except Exception as exc:
            errors.append({
                "file_path": str(audio_file),
                "error": str(exc)
            })

    report_df = pd.DataFrame(results)
    report_df.to_csv(output_dir / "report.csv", index=False)

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "errors": errors
            },
            f,
            indent=2
        )

    print(f"Finished. Analyzed {len(results)} files.")
    print(f"Encountered {len(errors)} errors.")
    print(f"Saved CSV report to: {output_dir / 'report.csv'}")
    print(f"Saved JSON report to: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
