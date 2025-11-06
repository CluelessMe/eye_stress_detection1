# stress_analyzer.py
import numpy as np
from collections import deque
import time

class StressAnalyzer:
    """
    Weighted heuristic model to compute Composite Stress Index (CSI)
    based on blink rate, EAR mean, and EAR variability.
    """

    def __init__(self, window_sec=10):
        self.window_sec = window_sec
        self.start_time = time.time()
        self.ear_values = deque(maxlen=150)
        self.blink_timestamps = []
        self.stress_indicator = "Low Stress"
        self.last_score = 0.0

    def update(self, ear, just_blinked=False):
        """Add new EAR value and blink event."""
        self.ear_values.append(ear)
        if just_blinked:
            self.blink_timestamps.append(time.time())

    def compute(self):
        """Compute stress level every few seconds."""
        elapsed = time.time() - self.start_time
        if elapsed < self.window_sec or len(self.ear_values) < 10:
            return None

        current_time = time.time()
        blink_rate = len([b for b in self.blink_timestamps if current_time - b <= 60])
        avg_ear = np.mean(self.ear_values)
        ear_std = np.std(self.ear_values)

        # Weighted heuristic composite score
        stress_score = (
            (blink_rate / 30) * 0.4 +
           ((self._baseline_ear - avg_ear) / 0.1) * 0.3 if hasattr(self, '_baseline_ear') else ((0.25 - avg_ear) / 0.1) * 0.3 +
            (ear_std / 0.002) * 0.3
        )
        stress_score = max(stress_score, 0)

        if stress_score < 0.8:
            stress_level = "Low Stress"
        elif stress_score < 1.6:
            stress_level = "Moderate Stress"
        else:
            stress_level = "High Stress / Fatigue"

        self.stress_indicator = stress_level
        self.last_score = stress_score
        self.start_time = time.time()

        return {
            "blink_rate": round(blink_rate, 2),
            "avg_ear": round(avg_ear, 3),
            "ear_std": round(ear_std, 4),
            "stress_score": round(stress_score, 2),
            "stress_indicator": stress_level
        }
