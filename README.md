# Audio Loudness Intelligence Engine

An engine that analyzes audio datasets for loudness (LUFS), dynamic range, and peak levels, then applies **peak-safe gain adjustments** to normalize audio without clipping.

Designed to maintain **consistent loudness** across multiple audio files, and meet industry standard compliance.

## Features

- Integrated LUFS analysis
- True peak measurement
- Loudness range (LRA)
- RMS level analysis
- Intelligent gain recommendations with headroom protection
- Batch-processing via CLI
- Industry standard compliance (1.0 dBTP ceiling)
- Dataset-level analysis (average, min, max loudness)

## Project Structure

src/loudness_engine
├── analyzer.py
├── feature_extractor.py
├── dataset_builder.py
├── gain_predictor.py
├── intelligent_normalizer.py
├── normalizer.py
├── true_peak.py
└── io_utils.py

## Pipeline

Audio → Feature Extraction → Dataset → Gain Prediction → Normalization

## Example Output

{
“integrated_lufs”: -29.408,
“true_peak_dbtp”: -10.34,
“rms_dbfs”: -32.07,
“recommended_gain_db”: 12.73,
“projected_lufs”: -21.05,
"prjected_peak": -6.0
"headroom_ok": true,
"dynamics_warning": false
}

## Future Work

- Machine learning gain prediction
- Automatic mastering targets
- batch processing
- streaming audio support

## Requirements

Install dependencies: pip install pandas tqdm pydub soundfile numpy scipy pyloudnorm

brew install ffmpeg libsndfile


pip install -r requirements.txt

## License
TNT BABY PRODUCTIONS LTD


