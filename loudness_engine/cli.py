import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment

from loudness_engine.io_utils import list_audio_files
from loudness_engine.analyzer import analyze_file
from loudness_engine.intelligence import AudioFeatures, LoudnessIntelligence


# --- Step 1: Audio processor ---
def apply_gain (file_path: str, gain_db: float, output_path: str):
    audio = AudioSegment.from_file(file_path)
    processed = audio.apply_gain(gain_db)
    processed.export(output_path, format="wav")


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
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []
    dataset_summary = None  # safety initialization

    # Initialize intelligence engine ONCE (not inside loop)
    brain = LoudnessIntelligence(target_lufs=-20.0, max_peak=-6.0)

    for audio_file in tqdm(audio_files, desc="Analyzing files"):
        try:
            # 1️⃣ Run analyzer
            analysis = analyze_file(audio_file)

            print("DEBUG → analysis output:", analysis)

            # 2️⃣ Package into AudioFeatures (THIS is the correct place)
            
            features = AudioFeatures(
                integrated_lufs=analysis["integrated_lufs"],
                loudness_range=analysis["loudness_range"],
                peak=analysis["peak_dbfs"],
                duration=analysis["duration_sec"]
            )

            # 3️⃣ Run intelligence
            decisions = brain.evaluate(features)
            analysis["decisions"] = decisions

            # 4️⃣ Attach decisions to analysis
            output_path = processed_dir / Path(audio_file).name
            apply_gain(audio_file, decisions["recommended_gain_db"], str(output_path))
            analysis["processed_file"] = str(output_path)

            results.append(analysis)


        except Exception as exc:
            errors.append({
                "file_path": str(audio_file),
                "error": str(exc)
            })
    # --- DATASET INTELLIGENCE ---
    if results:
        lufs_values = [r["integrated_lufs"] for r in results]
        lra_values = [r["loudness_range"] for r in results]
        peak_values = [r["peak_dbfs"] for r in results]

        dataset_summary = {
            "avg_lufs": sum(lufs_values) / len(lufs_values),
            "avg_lra": sum(lra_values) / len(lra_values),
            "avg_peak": sum(peak_values) / len(peak_values),
            "min_lufs": min(lufs_values),
            "max_lufs": max(lufs_values),
            "file_count": len(results)
        }

        print("\n=== DATASET SUMMARY ===")
        print(dataset_summary)

    # Save CSV
    report_df = pd.DataFrame(results)
    report_df.to_csv(output_dir / "report.csv", index=False)

    # Save JSON
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": dataset_summary,
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