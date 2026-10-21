"""Explicit development-only stream for testing the web UI without a webcam."""

from time import monotonic

import numpy as np

from study_buddy_inference import InferenceResult
from study_buddy_logic import CLASS_NAMES, classify_probabilities


class SyntheticCamera:
    def isOpened(self) -> bool:
        return True

    def read(self):
        return True, np.zeros((224, 224, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class SyntheticClassifier:
    def infer_frame(self, frame) -> InferenceResult:
        scores = [0.02] * len(CLASS_NAMES)
        scores[int(monotonic() / 5) % len(CLASS_NAMES)] = 0.90
        return InferenceResult(
            tuple(scores), classify_probabilities(scores), "ok", (0, 0, 224, 224)
        )
