import sys

from loudness_engine.feature_extractor import extract_features
from loudness_engine.reference_profile import REFERENCE_PROFILE
from loudness_engine.diagnostic_engine import diagnose
from loudness_engine.production_recommendations import generate_recommendations
from loudness_engine.production_report import build_report


def analyze_mix(file_path):
    """
    Run the complete production analysis pipeline.
    """

    features = extract_features(file_path)

    diagnoses = diagnose(
        features,
        REFERENCE_PROFILE
    )

    recommendations = generate_recommendations(
        diagnoses
    )

    return build_report(
        features,
        diagnoses,
        recommendations
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python loudness_engine/run_analysis.py <audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    report = analyze_mix(file_path)

    print(report)
