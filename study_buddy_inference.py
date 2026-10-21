"""The original six-class face inference path, shared by current and future UIs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from study_buddy_logic import CLASS_NAMES, classify_probabilities


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "study_buddy_resnet_final.keras"
CaptureStatus = Literal["ok", "no_face", "camera_error"]
FaceBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class InferenceResult:
    scores: tuple[float, ...] | None
    tentative_class: str | None
    capture_status: CaptureStatus
    face_box: FaceBox | None = None


def preprocess_face(frame: np.ndarray, face_box: FaceBox) -> np.ndarray:
    """Match the original OpenCV BGR crop -> RGB -> ResNet50 input path."""
    x, y, w, h = face_box
    rgb = cv2.cvtColor(
        cv2.resize(frame[y:y + h, x:x + w], (224, 224)),
        cv2.COLOR_BGR2RGB,
    )
    return preprocess_input(np.expand_dims(rgb.astype(np.float32), 0))


class StudentStateClassifier:
    """Load the saved model and Haar detector once, then infer from BGR frames."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, *, model=None, cascade=None):
        self.model = model if model is not None else tf.keras.models.load_model(
            model_path, compile=False
        )
        if self.model.input_shape[1:] != (224, 224, 3):
            raise ValueError("The selected model must accept 224x224 RGB face inputs")
        if self.model.output_shape[-1] != len(CLASS_NAMES):
            raise ValueError("The selected model must output the six Student Dataset classes")

        self.cascade = cascade if cascade is not None else cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.cascade.empty():
            raise RuntimeError("OpenCV's face detector could not be loaded")

    def infer_frame(self, frame: np.ndarray | None) -> InferenceResult:
        if frame is None or frame.size == 0:
            return InferenceResult(None, None, "camera_error")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return InferenceResult(None, None, "no_face")

        face_box = tuple(int(value) for value in max(
            faces, key=lambda face: face[2] * face[3]
        ))
        batch = preprocess_face(frame, face_box)
        scores = tuple(float(value) for value in self.model.predict(batch, verbose=0)[0])
        return InferenceResult(scores, classify_probabilities(scores), "ok", face_box)
