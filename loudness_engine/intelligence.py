from dataclasses import dataclass

@dataclass
class AudioFeatures:
    integrated_lufs: float
    loudness_range: float
    peak: float
    duration: float

class LoudnessIntelligence:
    def __init__(self, target_lufs=-20.0, max_peak=-6.0):
        self.target_lufs = target_lufs
        self.max_peak = max_peak

    def evaluate(self, features: AudioFeatures) -> dict:
        # 1️⃣ Calculate raw gain
        gain_needed = self.target_lufs - features.integrated_lufs

        # 2️⃣ Predict peak after gain
        projected_peak = features.peak + gain_needed

        # 3️⃣ Apply peak safety constraint
        if projected_peak > self.max_peak:
            gain_needed = self.max_peak - features.peak

        # 4️⃣ Final projections
        projected_lufs = features.integrated_lufs + gain_needed
        projected_peak = features.peak + gain_needed

        # 5️⃣ Decisions
        decisions = {
            "recommended_gain_db": round(gain_needed, 2),
            "projected_lufs": round(projected_lufs, 2),
            "projected_peak": round(projected_peak, 2),
            "headroom_ok": projected_peak <= self.max_peak,
            "dynamics_warning": features.loudness_range > 18
        }

        return decisions
