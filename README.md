# Audio Loudness Engine

An experimental audio analysis and normalization engine built in Python.

The system analyzes audio files, extracts loudness-related DSP features, and applies safe gain adjustments to normalize audio without clipping.

## Features

- Integrated LUFS analysis
- True peak measurement
- Loudness range (LRA)
- RMS level analysis
- Crest factor calculation
- Zero-crossing rate
- Safe gain prediction
- Intelligent normalization

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
“crest_factor_db”: 19.88,
“zero_crossing_rate”: 0.422
}

## Future Work

- Machine learning gain prediction
- Automatic mastering targets
- batch processing
- streaming audio support

## Requirements

Install dependencies:

pip install -r requirements.txt

## Author

Built as part of an exploration into audio DSP and AI-driven loudness normalization.
