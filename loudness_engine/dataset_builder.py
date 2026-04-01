from pathlib import Path
import pandas as pd

from loudness_engine.feature_extractor import extract_features


def build_dataset(input_dir, output_csv="outputs/audio_dataset.csv"):

    input_dir = Path(input_dir)

    audio_files = list(input_dir.glob("*.wav"))

    rows = []

    for file in audio_files:
        try:
            features = extract_features(file)
            rows.append(features)
            print("Processed:", file.name)

        except Exception as e:
            print("Skipped:", file.name, "Reason:", e)

    df = pd.DataFrame(rows)

    Path("outputs").mkdir(exist_ok=True)

    df.to_csv(output_csv, index=False)

    print("\nDataset saved to:", output_csv)

    return df